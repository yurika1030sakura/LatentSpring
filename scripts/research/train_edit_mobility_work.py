#!/usr/bin/env python3
"""Matched physical-work training of fixed, scalar and collective edit mobility."""
import argparse,json,math,time
from pathlib import Path
import torch
import torch.nn.functional as F
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.parity_refinement import evaluate_even_potential
from cfm_mol.edit_conditioned_bridge import edit_bridge,center
from cfm_mol.edit_mobility import MobilizedEditField,differentiable_oracle_potential,graph_penalty
from cfm_mol.joint_chemical_geometry import distinct_anchor_actions
from scripts.research.audit_masked_angular import ReplayOracle,equal,sha
from scripts.research.evaluate_chemical_policy import write


class ConditionalReplayOracle(ReplayOracle):
    def evaluate_chunked(self,x,max_request):
        assert self.condition==self.queries[self.index]['condition']
        return super().evaluate_chunked(x,max_request)


def query(oracle,positions,row,trace,phase):
    condition=dict(numbers=row['numbers'].tolist(),charge=int(row['electronic'][0]),spin_multiplicity=int(row['electronic'][1]))
    # The persistent worker reads this full condition on every synchronous RPC.
    oracle.condition=condition;before=oracle.evaluated
    record=dict(phase=phase,condition=condition,positions=positions.detach().clone(),raw_queries_before=before)
    trace.append(record)
    energy,force,parts=evaluate_even_potential(oracle,positions)
    record.update(**parts,
        raw_force_eV_A=force+parts['odd_force_eV_A'],inverted_force_eV_A=parts['odd_force_eV_A']-force,
        raw_queries_after=oracle.evaluated)
    return energy,force


def train(model,data,oracle,protocol,seed,traces=None):
    if traces is None:traces=[]
    rows=[];by_condition={i:[j for j,r in enumerate(data) if r['index']==i] for i in protocol['condition_indices']}
    for index,indices in by_condition.items():
        row=data[indices[0]];x=torch.stack([row['x'],row['y']]);energy,force=query(oracle,x,row,traces,f'condition_check_{index}')
        potential=energy+protocol['restraint_eV_A2']/2*x.square().sum((1,2))
        expected=potential.new_tensor([row['source_potential_eV'],row['destination_potential_eV']])
        assert float((potential-expected).abs().max())<=protocol['source_energy_tolerance_eV']
    optimizer=torch.optim.Adam(model.parameters(),lr=protocol['learning_rate']);g=torch.Generator().manual_seed(seed+1000)
    for step in range(protocol['steps']):
        index=protocol['condition_indices'][step%len(protocol['condition_indices'])];eligible=by_condition[index]
        optimizer.zero_grad(set_to_none=True);items=[];indices=[];directions=[]
        for _ in range(protocol['batch_size']):
            offset=eligible[int(torch.randint(len(eligible),(1,),generator=g))];row=data[offset];reverse=bool(torch.randint(2,(1,),generator=g))
            if reverse:x,b,c,action,U=row['y'],row['new_bonds'],row['bonds'],row['inverse_action'],row['destination_potential_eV']
            else:x,b,c,action,U=row['x'],row['bonds'],row['new_bonds'],row['action'],row['source_potential_eV']
            p=center(torch.randn(x.shape,dtype=x.dtype,generator=g));y,q,volume=edit_bridge(x,p,b,c,row['numbers'],row['electronic'],action,model,**protocol['bridge'])
            ratio=math.log(len(distinct_anchor_actions(row['numbers'],b))/len(distinct_anchor_actions(row['numbers'],c)))
            items.append(dict(y=y,q=q,p=p,volume=volume,source_potential=U,action_ratio=ratio,desired=c,row=row))
            indices.append(offset);directions.append(reverse)
        positions=torch.stack([v['y'] for v in items]);energy,force=query(oracle,positions,items[0]['row'],traces,f'training_{step}')
        potential=differentiable_oracle_potential(positions,energy,force,protocol['restraint_eV_A2'])
        works=[];penalties=[]
        for j,v in enumerate(items):
            works.append((potential[j]-v['source_potential'])/v['row']['electronic'][2]+.5*(v['q'].square().sum()-v['p'].square().sum())-v['volume']-v['action_ratio'])
            penalties.append(graph_penalty(v['y'],v['desired'],v['row']['numbers']))
        work=torch.stack(works);penalty=torch.stack(penalties)
        loss=F.softplus(work).mean()+protocol['graph_penalty_weight']*penalty.mean()
        if not torch.isfinite(loss):raise ValueError('Nonfinite physical-work objective')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),protocol['gradient_clip'],error_if_nonfinite=True);optimizer.step()
        rows.append(dict(step=step,index=index,indices=indices,reverse=directions,objective=float(loss),mean_work=float(work.mean()),
            mean_softplus_work=float(F.softplus(work).mean()),mean_graph_penalty=float(penalty.mean()),gradient_norm=float(norm)))
        if step%20==0:print(json.dumps(rows[-1]),flush=True)
    return dict(training=rows,query_trace=traces,generator_state=g.get_state(),state_dict=model.state_dict(),configuration=model.configuration)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','out']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['run','oracle-python','oracle-checkpoint']:p.add_argument('--'+name,type=Path)
    p.add_argument('--variant',choices=['fixed','scalar','collective'],required=True);p.add_argument('--replica',type=int,required=True)
    p.add_argument('--phase',choices=['train','audit'],required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2];pp=root/'research/evidence/edit_mobility_work_protocol_v1.json';protocol=json.loads(pp.read_text())
    assert protocol['frozen'] and a.replica in [0,1]
    dp=a.project/protocol['data_run'];assert sha(dp/'data.pt')==protocol['data_sha256'];data=torch.load(dp/'data.pt',map_location='cpu',weights_only=False)
    assert all(row['fit_only'] for row in data)
    seed=protocol['seeds'][a.replica];torch.manual_seed(seed);initial_info=protocol['initial_models'][str(a.replica)];initial_path=a.project/initial_info['path']
    assert sha(initial_path)==initial_info['sha256'];initial=torch.load(initial_path,map_location='cpu',weights_only=False)
    model=MobilizedEditField(dict(initial['configuration'],roots_only=False),dict(protocol['mobility'],variant=a.variant)).double()
    model.force_field.load_state_dict(initial['state_dict'])
    a.out.mkdir(parents=True,exist_ok=True);output=a.out/'results.json'
    if output.exists():raise FileExistsError(output)
    report=dict(complete=False,variant=a.variant,replica=a.replica,phase=a.phase,protocol_sha256=sha(pp),new_raw_queries=0,scientific_submission_ready=False,scope=protocol['interpretation'])
    write(output,report);oracle=None;traces=[];start=time.monotonic()
    try:
        if a.phase=='audit':
            producer=json.loads((a.run/'results.json').read_text());assert producer['complete'] and producer['protocol_sha256']==sha(pp)
            assert sha(a.run/'trace.pt')==producer['trace_sha256'];expected=torch.load(a.run/'trace.pt',map_location='cpu',weights_only=False)
            oracle=ConditionalReplayOracle(expected['query_trace'])
        else:
            assert sha(a.oracle_checkpoint)==protocol['oracle_sha256']
            row=data[0];oracle=EnergyOracle(a.oracle_python,root/'scripts/research/oracle_worker.py',a.oracle_checkpoint,numbers=row['numbers'].tolist(),
                charge=int(row['electronic'][0]),spin_multiplicity=int(row['electronic'][1]),device='cuda',batch_size=8)
            assert oracle.handshake['base_precision_dtype']=='torch.float32' and not oracle.handshake['tf32']
            torch.save(dict(configuration=model.configuration,state_dict=model.state_dict()),a.out/'initial_model.pt')
        actual=train(model,data,oracle,protocol,seed,traces=traces)
        assert oracle.evaluated==protocol['raw_queries_per_model']
        if a.phase=='audit':
            equal(actual,expected);assert oracle.index==len(oracle.queries)
            saved=torch.load(a.run/'model.pt',map_location='cpu',weights_only=False);equal(model.state_dict(),saved['state_dict'])
            report.update(complete=True,full_training_and_condition_replay=True,source_results_sha256=sha(a.run/'results.json'),trace_sha256=producer['trace_sha256'],
                model_sha256=producer['model_sha256'],raw_queries_in_producer=oracle.evaluated)
        else:
            assert oracle.evaluated==oracle.requested_evaluations
            torch.save(actual,a.out/'trace.pt');torch.save(dict(configuration=model.configuration,state_dict=model.state_dict(),seed=seed,protocol_sha256=sha(pp)),a.out/'model.pt')
            report.update(complete=True,new_raw_queries=oracle.evaluated,requested_raw_queries=oracle.requested_evaluations,model_sha256=sha(a.out/'model.pt'),trace_sha256=sha(a.out/'trace.pt'),
                training=actual['training'],oracle_runtime=oracle.handshake,oracle_evaluation_seconds=oracle.evaluation_seconds)
        report['elapsed_seconds']=time.monotonic()-start;write(output,report)
        print(json.dumps({k:v for k,v in report.items() if k not in ['training','oracle_runtime']}))
    except Exception as exc:
        torch.save(dict(configuration=model.configuration,state_dict=model.state_dict()),a.out/'failed_model.pt')
        torch.save(dict(query_trace=traces),a.out/'failed_trace.pt')
        report.update(failure=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start)
        if oracle is not None and a.phase=='train':report.update(new_raw_queries=oracle.evaluated,requested_raw_queries=oracle.requested_evaluations)
        write(output,report);raise
    finally:
        if oracle is not None and a.phase=='train':oracle.close()


if __name__=='__main__':main()
