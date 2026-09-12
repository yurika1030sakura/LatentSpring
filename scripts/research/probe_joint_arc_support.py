#!/usr/bin/env python3
"""Geometry-only joint graph/radius/two-root screen on audited training starts."""
import argparse
import json
from pathlib import Path
import torch

from cfm_mol.chemical_moves import covalent_radii,infer_chemical_graph
from cfm_mol.chemical_path_guide import exchanged_bond_graph
from cfm_mol.joint_chemical_geometry import distinct_anchor_actions,joint_geometry_proposal
from cfm_mol.joint_arc_geometry import marginal_joint_arc_proposal
from scripts.research.evaluate_chemical_policy import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','out']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[2]
    pp=root/'research/evidence/joint_arc_support_protocol_v1.json';protocol=json.loads(pp.read_text())
    ap=args.project/'runs/proposal_training_preparation_audit_v1/results.json';audit=json.loads(ap.read_text())
    assert audit['complete'] and audit['full_producer_replay'] and sha(ap)==protocol['preparation_audit_sha256']
    rows=[];traces=[]
    for source in audit['rows']:
        index=source['index']
        if source['zero_support']:continue
        directory=args.project/f'runs/proposal_training_preparation_v1/condition_{index:02d}'
        assert sha(directory/'results.json')==source['results_sha256'] and sha(directory/'trace.pt')==source['trace_sha256']
        saved=torch.load(directory/'trace.pt',map_location='cpu',weights_only=False)
        numbers=torch.tensor(saved['condition']['numbers'],dtype=torch.long);radii=covalent_radii(numbers)
        electronic=torch.tensor([saved['condition']['charge'],saved['condition']['spin_multiplicity'],protocol['kT_eV']],dtype=torch.float64)
        for parent,sid in zip(saved['parent_ids'],saved['history_state_ids'][-1]):
            old=saved['states'][sid];actions=distinct_anchor_actions(numbers,old['graph']['bond_orders'])
            assert actions
            for trial in range(protocol['trials_per_parent']):
                seed=protocol['seed']+100000000*index+100000*parent+1000*trial
                rng=torch.Generator().manual_seed(seed)
                pick=int(torch.randint(len(actions),(1,),generator=rng));action=actions[pick]
                order=int(torch.randint(2,(1,),generator=rng));prefix=rng.get_state()
                i,j,k,l=action;inverse=(i,j,l,k);desired=exchanged_bond_graph(old['graph']['bond_orders'],action)
                for method in protocol['methods']:
                    rng=torch.Generator();rng.set_state(prefix)
                    kwargs=dict(order=order,radial_width=protocol['radial_width'],site_concentration=64.)
                    proposal=joint_geometry_proposal if method=='legacy_site64' else marginal_joint_arc_proposal
                    kind='site' if method=='legacy_site64' else method
                    y,q,forward=proposal(old['positions'],old['graph']['bond_orders'],numbers,electronic,radii,
                        action,kind=kind,generator=rng,**kwargs)
                    row=dict(condition=index,parent=parent,trial=trial,method=method,drawn=y is not None,
                        geometry_supported=False,reverse_density_positive=False,order_marginalization_rescued=False,validator_error=False)
                    trace=dict(condition=index,parent=parent,trial=trial,method=method,seed=seed,choice_index=pick,
                        action=action,order=order,prefix_generator_state=prefix,positions=y,log_forward=q,forward=forward,
                        generator_state=rng.get_state())
                    if y is None:row['failure']=forward['failure']
                    else:
                        try:
                            graph=infer_chemical_graph(y,numbers.tolist(),saved['condition']['charge'])
                            if not torch.equal(graph['bond_orders'],desired):raise ValueError('Different desired graph')
                            row['geometry_supported']=True
                            if inverse not in distinct_anchor_actions(numbers,graph['bond_orders']):raise ValueError('Inverse graph action ineligible')
                            _,qr,reverse=proposal(y,desired,numbers,electronic,radii,inverse,kind=kind,observed=old['positions'],**kwargs)
                            trace.update(log_reverse=qr,reverse=reverse)
                            row['reverse_density_positive']=bool(torch.isfinite(qr))
                            if method!='legacy_site64' and row['reverse_density_positive']:
                                row['order_marginalization_rescued']=not bool(torch.isfinite(reverse['order_log_densities'][order]))
                        except (ValueError,IndexError,RuntimeError) as exc:
                            row.update(failure=str(exc),error_type=type(exc).__name__,validator_error=not isinstance(exc,ValueError))
                    rows.append(row);traces.append(trace)
        print(json.dumps(dict(condition=index,parents=len(saved['parent_ids']),complete=True)),flush=True)
    summaries=[]
    for method in protocol['methods']:
        subset=[r for r in rows if r['method']==method]
        summaries.append(dict(method=method,attempts=len(subset),**{key:sum(r[key] for r in subset) for key in
            ['drawn','geometry_supported','reverse_density_positive','order_marginalization_rescued','validator_error']}))
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'results.json').exists() or (args.out/'trace.pt').exists():raise FileExistsError(args.out)
    torch.save(traces,args.out/'trace.pt')
    result=dict(complete=True,protocol_sha256=sha(pp),preparation_audit_sha256=sha(ap),trace_sha256=sha(args.out/'trace.pt'),
        rows=rows,summaries=summaries,new_physical_queries=0,new_model_fitted=False,zero_support_conditions_retained=[4,6],
        scientific_submission_ready=False,scope='Joint graph/radius/two-root training-source geometry screen, with shared action/order/radial-noise prefixes across methods. No physical energies, learned gain or equilibrium result.')
    write(args.out/'results.json',result);print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
