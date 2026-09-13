#!/usr/bin/env python3
"""Matched conditional four-root kernels on preselected evaluation blocks."""
import argparse,json,math
from pathlib import Path
import torch
from cfm_mol.chemical_work import LinearBondWork
from cfm_mol.interaction_work_model import InteractionWorkModel
from cfm_mol.cooperative_edit_policy import block_catalogue,panel_catalogue,block_policy,reverse_index
from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.energy_oracle import EnergyOracle
from scripts.research.evaluate_edit_bridge import inputs
from scripts.research.audit_masked_angular import ReplayOracle,equal,sha
from scripts.research.evaluate_chemical_policy import write


def load_models(project,protocol):
    result={'uniform':(None,None,False)}
    for seed in [0,1]:
        spec=protocol['backbones'][str(seed)];path=project/spec['path'];assert sha(path)==spec['sha256']
        saved=torch.load(path,map_location='cpu',weights_only=False)
        assert saved['phase']=='full_fit' and set(saved['training_source_ids'])==set(range(36))
        backbone=LinearBondWork(**saved['configuration']).double();backbone.load_state_dict(saved['state_dict']);backbone.eval();backbone.requires_grad_(False)
        result[f'linear_s{seed}']=(backbone,None,False);result[f'restraint_s{seed}']=(backbone,None,True)
        for variant in ['linear','blind','environment']:
            spec=protocol['interaction_models'][f'{variant}_s{seed}'];path=project/spec['path'];assert sha(path)==spec['sha256']
            saved=torch.load(path,map_location='cpu',weights_only=False)
            assert saved['phase']=='full_fit' and set(saved['training_source_ids'])==set(range(9))|set(range(18,36))
            model=InteractionWorkModel(**saved['configuration']).double();model.load_state_dict(saved['state_dict']);model.eval();model.requires_grad_(False)
            result[f'{variant}_interaction_s{seed}']=(backbone,model,True)
    return result


def probabilities(target,old,options,spec,protocol):
    backbone,interaction,affinity=spec
    return block_policy(target,old,options,backbone,interaction,use_affinity=affinity,**protocol['policy'])


def run(target,models,sources,protocol,index):
    initial=[target.coordinate_state(old['positions']) for spec,old in sources];target.evaluate(initial,phase='sources')
    rows=[];action_rows=[];total_checks=0
    for old,(spec,previous) in zip(initial,sources):
        equal(old['graph'],previous['graph']);assert abs(float(old['potential_eV']-previous['potential_eV']))<1e-4
        blocks=protocol['blocks'][str(index)][str(spec['parent'])]
        plans=[];pending=[]
        use_panel=protocol.get('selection_scope')=='panel'
        candidates=[panel_catalogue(target,old,blocks)] if use_panel else [block_catalogue(target,old,roots) for roots in blocks]
        denominator=1 if use_panel else max(1,len(blocks))
        for options in candidates:
            forward={name:probabilities(target,old,options,model,protocol) for name,model in models.items()}
            plans.append((options,forward));pending.extend(item['candidate'] for item in options['valid'])
        # All forward policies are already frozen; only the final corners need E/F.
        target.evaluate(pending,phase=f'parent_{spec["parent"]}_joint_endpoints')
        totals={name:dict(utility_eV=0.,acceptance=0.,constitutional_flow=0.) for name in models}
        for block_id,(options,forward) in enumerate(plans):
            for choice,item in enumerate(options['valid']):
                new=item['candidate']
                reverse=panel_catalogue(target,new,blocks) if use_panel else block_catalogue(target,new,options['roots'])
                inv=reverse_index(item,reverse)
                delta=float(new['potential_eV']-old['potential_eV']);changed=new['graph']['connectivity_smiles']!=old['graph']['connectivity_smiles']
                for name,model in models.items():
                    rev=probabilities(target,new,reverse,model,protocol);fwd=forward[name]
                    # The affinity is undirected, whereas additive work reverses.
                    assert abs(float(fwd['total_interaction_eV'][choice]-rev['total_interaction_eV'][inv]))<1e-8
                    assert abs(float(fwd['predicted_linear_work_eV'][choice]+rev['predicted_linear_work_eV'][inv]))<1e-8
                    qf=float(fwd['log_probability'][choice]);qr=float(rev['log_probability'][inv])
                    ratio=-delta/target.kT+item['record']['log_volume']+qr-qf;assert math.isfinite(ratio)
                    alpha=math.exp(min(0.,ratio));probability=math.exp(qf)
                    totals[name]['utility_eV']-=probability*alpha*delta/denominator
                    totals[name]['acceptance']+=probability*alpha/denominator
                    totals[name]['constitutional_flow']+=probability*alpha*changed/denominator
                    action_rows.append(dict(parent=spec['parent'],block=block_id,method=name,roots=item['record']['roots'],
                        old_state_id=old['state_id'],new_state_id=new['state_id'],actions=item['record']['actions'],
                        log_volume=item['record']['log_volume'],forward_log_probability=qf,reverse_log_probability=qr,
                        predicted_electronic_interaction_eV=float(fwd['electronic_interaction_eV'][choice]),
                        predicted_linear_work_eV=float(fwd['predicted_linear_work_eV'][choice]),
                        potential_change_eV=delta,log_acceptance_ratio=ratio,probability=probability,acceptance_probability=alpha))
                    total_checks+=1
        empty=sum(c.get('empty_blocks',not c['valid']) for c,_ in plans)
        row=dict(parent=spec['parent'],blocks=len(blocks),empty_blocks=empty,
            valid_endpoints=len(pending),failed_matchings=sum(len(options['failed']) for options,_ in plans),methods=totals)
        if use_panel:row['selection_denominator']=denominator
        rows.append(row)
    return dict(rows=rows,action_rows=action_rows,states=target.states,query_trace=target.query_trace,symmetry_checks=total_checks)


def independent_checks(data,target):
    for row in data['action_rows']:
        energy=[]
        for key in ['old_state_id','new_state_id']:
            state=data['states'][row[key]];q=data['query_trace'][state['query_batch']];j=state['query_row']
            energy.append((float(q['raw_energy_eV'][j])+float(q['inverted_energy_eV'][j]))/2+target.restraint/2*float(state['positions'].square().sum()))
        expected=-(energy[1]-energy[0])/target.kT+row['log_volume']+row['reverse_log_probability']-row['forward_log_probability']
        assert abs(expected-row['log_acceptance_ratio'])<1e-7
    for parent in data['rows']:
        for method,totals in parent['methods'].items():
            selected=[r for r in data['action_rows'] if r['parent']==parent['parent'] and r['method']==method]
            for block in set(row['block'] for row in selected):
                assert abs(sum(row['probability'] for row in selected if row['block']==block)-1.)<1e-10
            value=sum(-row['probability']*row['acceptance_probability']*row['potential_change_eV'] for row in selected)/parent.get('selection_denominator',max(1,parent['blocks']))
            assert abs(value-totals['utility_eV'])<1e-10
    return len(data['action_rows'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','out','protocol']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['run','oracle-python','oracle-checkpoint']:p.add_argument('--'+name,type=Path)
    p.add_argument('--index',type=int,required=True);p.add_argument('--phase',choices=['evaluate','reevaluate','audit'],required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    pp,protocol,physical,_,condition,sources=inputs(root,a.project,a.index,root/a.protocol);models=load_models(a.project,protocol)
    a.out.mkdir(parents=True,exist_ok=True);output=a.out/'results.json'
    if output.exists():raise FileExistsError(output)
    report=dict(complete=False,index=a.index,phase=a.phase,protocol_sha256=sha(pp),new_raw_queries=0,scientific_submission_ready=False,scope=protocol['interpretation'])
    write(output,report);oracle=target=None
    try:
        if a.phase in ('audit','reevaluate'):
            producer=json.loads((a.run/'results.json').read_text());assert producer['complete'] and sha(a.run/'trace.pt')==producer['trace_sha256']
            if a.phase=='audit':assert producer['protocol_sha256']==sha(pp)
            else:
                prior_path=root/protocol['reused_evaluation_protocol'];assert sha(prior_path)==producer['protocol_sha256']==protocol['reused_evaluation_protocol_sha256']
                prior=json.loads(prior_path.read_text())
                for key in ['physical_protocol','physical_protocol_sha256','sources','blocks','counts','policy','backbones','interaction_models']:
                    assert protocol[key]==prior[key]
            expected=torch.load(a.run/'trace.pt',map_location='cpu',weights_only=False);oracle=ReplayOracle(expected['query_trace'])
        else:
            assert sha(a.oracle_checkpoint)==physical['raw_oracle_sha256']
            oracle=EnergyOracle(a.oracle_python,root/'scripts/research/oracle_worker.py',a.oracle_checkpoint,numbers=condition['numbers'],charge=condition['charge'],spin_multiplicity=condition['spin_multiplicity'],device='cuda',batch_size=32)
            assert oracle.handshake['base_precision_dtype']=='torch.float32' and not oracle.handshake['tf32']
        target=ChemicalTarget(oracle,condition,physical['kT_eV'],physical['restraint_eV_A2']);actual=run(target,models,sources,protocol,a.index)
        assert oracle.evaluated==protocol['counts'][str(a.index)]['raw_queries']
        checks=independent_checks(actual,target)
        if a.phase=='audit':
            equal(actual,expected);assert oracle.index==len(oracle.queries)
            report.update(complete=True,full_replay=True,source_results_sha256=sha(a.run/'results.json'),trace_sha256=producer['trace_sha256'],producer_raw_queries=oracle.evaluated,independent_MH_checks=checks)
        else:
            if a.phase=='reevaluate':
                assert oracle.index==len(oracle.queries);equal(actual['states'],expected['states']);equal(actual['query_trace'],expected['query_trace'])
                report.update(reused_raw_queries=oracle.evaluated,reused_trace_sha256=producer['trace_sha256'],reused_results_sha256=sha(a.run/'results.json'))
            else:assert oracle.evaluated==oracle.requested_evaluations
            torch.save(actual,a.out/'trace.pt');report.update(complete=True,trace_sha256=sha(a.out/'trace.pt'),
                new_raw_queries=0 if a.phase=='reevaluate' else oracle.evaluated,physical_calls_in_trace=oracle.evaluated,rows=actual['rows'],independent_MH_checks=checks)
        write(output,report);print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
    except Exception as exc:
        if target is not None:torch.save(dict(states=target.states,query_trace=target.query_trace),a.out/'failed_trace.pt')
        report['failure']=f'{type(exc).__name__}: {exc}'
        if oracle is not None and a.phase=='evaluate':report['new_raw_queries']=oracle.evaluated
        write(output,report);raise
    finally:
        if oracle is not None and a.phase=='evaluate':oracle.close()


if __name__=='__main__':main()
