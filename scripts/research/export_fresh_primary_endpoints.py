#!/usr/bin/env python3
"""Export every primary paired endpoint and common start for independent energies."""
import argparse
import json
from pathlib import Path
import torch
from scripts.research.evaluate_chemical_policy import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();root=args.project
    protocol_path=root/'research/evidence/fresh_primary_xtb_protocol_v1.json'
    protocol=json.loads(protocol_path.read_text())
    sp=root/'runs/fresh_reuse_summary_v1/results.json';summary=json.loads(sp.read_text())
    assert sha(sp)==protocol['primary_summary_sha256'] and summary['complete']
    primary=summary['primary_comparison'];assert primary['available'] and primary['parents']==72
    wp=root/'runs/fresh_reuse_prepare_v1/trace.pt'
    warm_header=json.loads((wp.parent/'results.json').read_text())
    assert warm_header['complete'] and warm_header['trace_sha256']==sha(wp)
    warm=torch.load(wp,map_location='cpu',weights_only=False);condition=warm['condition'];parent_ids=warm['parent_ids']
    tasks=[];hashes={}
    def add(state,method,replica,parent,trace_path,state_id,query_budget,history_index):
        if trace_path not in hashes:hashes[trace_path]=sha(trace_path)
        tasks.append(dict(task_id=f'{method}_s{replica}_p{parent}',method=method,replica=replica,parent_id=parent,
            positions=state['positions'].tolist(),source_trace=str(trace_path.relative_to(root)),source_trace_sha256=hashes[trace_path],
            state_id=state_id,history_index=history_index,query_budget=query_budget,
            esen_even_energy_eV=float(state['energy_eV']),esen_potential_eV=float(state['potential_eV']),
            source_connectivity=state['graph']['connectivity_smiles'],inversion_check=False))
    for i,parent in enumerate(parent_ids):
        sid=warm['history_state_ids'][-1][i];add(warm['states'][sid],'warm',0,parent,wp,sid,None,len(warm['history_state_ids'])-1)
    endpoints={}
    for method in ['site','learned_vector']:
        for replica in [0,1]:
            run=root/f'runs/fresh_reuse_eval_v2/{method}_s{replica}'
            report=json.loads((run/'results.json').read_text())
            ap=root/f'runs/fresh_reuse_audit_v1/{method}_s{replica}/results.json';audit=json.loads(ap.read_text())
            assert audit['complete'] and audit['full_producer_replay'] and audit['results_sha256']==sha(run/'results.json')
            assert report['parent_ids']==parent_ids and report['condition']==condition
            for chunk in report['chunks']:
                path=run/chunk['file'];assert sha(path)==chunk['sha256']
                data=torch.load(path,map_location='cpu',weights_only=False)
                for i,parent in enumerate(data['parent_ids']):
                    position=parent_ids.index(parent)
                    budget=512 if method=='learned_vector' else primary['site_queries_per_parent'][position]
                    history_index=next(k for k,count in enumerate(data['query_count_history']) if count[i]==budget)
                    sid=data['history_state_ids'][history_index][i];state=data['states'][sid]
                    add(state,method,replica,parent,path,sid,budget,history_index)
                    endpoints[(method,replica,parent)]=float(state['potential_eV'])
            print(method,replica,'exported',flush=True)
    assert len(tasks)==360
    for replica in [0,1]:
        diff=[endpoints[('learned_vector',replica,p)]-endpoints[('site',replica,p)] for p in parent_ids]
        torch.testing.assert_close(torch.tensor(diff,dtype=torch.float64),
            torch.tensor(primary['paired_potential_differences_eV'][replica],dtype=torch.float64),atol=1e-12,rtol=0)
    for task in list(tasks[:protocol['inversion_checks']]):
        mirror=dict(task);mirror.update(task_id=task['task_id']+'_inverted',inversion_check=True,
            mirrored_task_id=task['task_id'],positions=[[-v for v in xyz] for xyz in task['positions']]);tasks.append(mirror)
    report=dict(complete=True,protocol_sha256=sha(protocol_path),primary_summary_sha256=sha(sp),condition=condition,
        parent_ids=parent_ids,source_attempts_retained=8192,primary_endpoints=288,common_starts=72,
        inversion_checks=protocol['inversion_checks'],tasks=tasks,reference_geometry_loaded=False,
        positions_optimized=False,new_physical_queries=0,scientific_submission_ready=False)
    if args.out.exists():raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
