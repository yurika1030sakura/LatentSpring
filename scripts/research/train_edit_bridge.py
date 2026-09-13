#!/usr/bin/env python3
"""Fast paired transport pretraining; molecular usefulness requires a separate pilot."""
import argparse,json,time
from pathlib import Path
import torch
from cfm_mol.edit_conditioned_bridge import EditBridgeField,edit_bridge,center
from cfm_mol.chemical_path_guide import exchanged_bond_graph
from scripts.research.audit_masked_angular import sha
from scripts.research.evaluate_chemical_policy import write
from scripts.research.mobility_relaxation_pilot import inputs


def build(root,project,out):
    rows=[];sources={}
    for index in [1,2,3,5]:
        _,_,physical,condition,fit=inputs(root,project,index)
        directory=project/f'runs/mobility_relaxation_continuation_v1/condition_{index:02d}'
        rp=directory/'results.json';ap=project/f'runs/mobility_relaxation_continuation_audit_v1/condition_{index:02d}/results.json'
        r=json.loads(rp.read_text());audit=json.loads(ap.read_text());assert r['complete'] and audit['complete'] and audit['full_replay']
        assert sha(rp)==audit['source_results_sha256'] and sha(directory/'trace.pt')==audit['trace_sha256']==r['trace_sha256']
        data=torch.load(directory/'trace.pt',map_location='cpu',weights_only=False);parents={spec['parent'] for spec,row in fit}
        assert {p['parent'] for p in data['pairs']}==parents
        for pair in data['pairs']:
            arms={v['endpoint']:v for v in data['arms'] if v['pair_id']==pair['pair_id'] and v['mobility']=='collective'}
            for cap in [0,8,16,32,64,128]:
                chosen={}
                for end,arm in arms.items():
                    ids=[arm['initial_state_id']]+[v['state_id'] for v in arm['events'] if v['queried'] and v['evaluation']<=cap]
                    chosen[end]=min((data['states'][sid] for sid in ids),key=lambda s:float(s['potential_eV']))
                x,y=chosen['source'],chosen['destination'];b=x['graph']['bond_orders'];c=y['graph']['bond_orders'];action=pair['action']
                torch.testing.assert_close(exchanged_bond_graph(b,action),c,atol=0,rtol=0)
                rows.append(dict(index=index,parent=pair['parent'],pair_id=pair['pair_id'],cap=cap,fit_only=True,
                    x=x['positions'],y=y['positions'],bonds=b,new_bonds=c,numbers=torch.tensor(condition['numbers']),
                    electronic=torch.tensor([condition['charge'],condition['spin_multiplicity'],physical['kT_eV']],dtype=torch.float64),
                    action=action,inverse_action=[action[0],action[1],action[3],action[2]],
                    source_state_id=x['state_id'],destination_state_id=y['state_id'],
                    source_potential_eV=float(x['potential_eV']),destination_potential_eV=float(y['potential_eV'])))
        sources[str(index)]=dict(results_sha256=sha(rp),audit_sha256=sha(ap),trace_sha256=r['trace_sha256'])
    assert len(rows)==192 and len({(r['index'],r['parent']) for r in rows})==32
    out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists():raise FileExistsError(out)
    torch.save(rows,out/'data.pt');write(out/'results.json',dict(complete=True,rows=192,fit_parents=32,conditions=4,data_sha256=sha(out/'data.pt'),sources=sources,
        new_physical_queries=0,scope='All selected FIT parents, both endpoints and six bounded readouts. Valid observed geometries, not necessarily stationary or equilibrium. No evaluated or reserved parents enter fitting.'))


def loss(model,row,p,reverse,protocol):
    x,y,b,c,action=(row['y'],row['x'],row['new_bonds'],row['bonds'],row['inverse_action']) if reverse else (row['x'],row['y'],row['bonds'],row['new_bonds'],row['action'])
    proposal,q,_=edit_bridge(x,p,b,c,row['numbers'],row['electronic'],action,model,**protocol['bridge'])
    position=(proposal-y).square().mean()/protocol['position_scale_A']**2
    kinetic=.5*q.square().sum()/(3*(len(x)-1))
    return position+protocol['momentum_weight']*kinetic,position,kinetic


@torch.no_grad()
def metrics(model,data,protocol):
    g=torch.Generator().manual_seed(protocol['metric_seed']);values=[]
    for row in data:
        for reverse in [False,True]:
            p=center(torch.randn(row['x'].shape,dtype=torch.float64,generator=g));v=loss(model,row,p,reverse,protocol)
            values.append([float(x) for x in v])
    means=torch.tensor(values,dtype=torch.float64).mean(0)
    return dict(objective=float(means[0]),scaled_position_MSE=float(means[1]),momentum_energy_per_dimension=float(means[2]),teacher_directions=len(values),fit_only=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--phase',choices=['build','train'],required=True);p.add_argument('--variant');p.add_argument('--replica',type=int)
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    if a.phase=='build':build(root,a.project,a.out);return
    pp=root/'research/evidence/edit_bridge_training_protocol_v1.json';protocol=json.loads(pp.read_text());assert protocol['frozen']
    assert a.variant in protocol['variants'] and a.replica in [0,1]
    directory=a.project/protocol['data_run'];header=json.loads((directory/'results.json').read_text())
    assert header['complete'] and sha(directory/'data.pt')==header['data_sha256']==protocol['data_sha256']
    data=torch.load(directory/'data.pt',map_location='cpu',weights_only=False);assert all(r['fit_only'] for r in data)
    a.out.mkdir(parents=True,exist_ok=True);output=a.out/'results.json'
    if output.exists():raise FileExistsError(output)
    seed=protocol['seeds'][a.replica];torch.manual_seed(seed)
    configuration=dict(protocol['model'],edit_conditioned=a.variant!='blind_collective',roots_only=a.variant=='edit_roots')
    model=EditBridgeField(**configuration).double();optimizer=torch.optim.Adam(model.parameters(),lr=protocol['learning_rate'])
    g=torch.Generator().manual_seed(seed+1000);initial=metrics(model,data,protocol);trace=[];start=time.monotonic()
    write(output,dict(complete=False,variant=a.variant,replica=a.replica,protocol_sha256=sha(pp),initial=initial,new_physical_queries=0))
    torch.save(dict(configuration=configuration,state_dict=model.state_dict()),a.out/'initial_model.pt')
    for step in range(protocol['steps']):
        optimizer.zero_grad(set_to_none=True);objective=0.;indices=[];reverse_flags=[]
        for _ in range(protocol['batch_size']):
            index=int(torch.randint(len(data),(1,),generator=g));reverse=bool(torch.randint(2,(1,),generator=g));row=data[index]
            p0=center(torch.randn(row['x'].shape,dtype=torch.float64,generator=g));objective=objective+loss(model,row,p0,reverse,protocol)[0]/protocol['batch_size']
            indices.append(index);reverse_flags.append(reverse)
        if not torch.isfinite(objective):raise ValueError('Nonfinite paired transport objective')
        objective.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),protocol['gradient_clip'],error_if_nonfinite=True);optimizer.step()
        trace.append(dict(step=step,objective=float(objective),gradient_norm=float(norm),indices=indices,reverse=reverse_flags))
        if step%50==0:print(json.dumps(trace[-1]),flush=True)
    final=metrics(model,data,protocol)
    torch.save(dict(configuration=configuration,state_dict=model.state_dict(),protocol_sha256=sha(pp),seed=seed),a.out/'model.pt')
    write(output,dict(complete=True,variant=a.variant,replica=a.replica,protocol_sha256=sha(pp),data_sha256=protocol['data_sha256'],model_sha256=sha(a.out/'model.pt'),
        initial=initial,final=final,trace=trace,elapsed_seconds=time.monotonic()-start,new_physical_queries=0,
        scientific_submission_ready=False,scope='Paired transport pretraining only. Fitting error is not molecular acceptance, useful generation, an equilibrium result or established AI novelty.'))


if __name__=='__main__':main()
