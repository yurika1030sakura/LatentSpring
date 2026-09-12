#!/usr/bin/env python3
"""Diagnose audited transfer proposals without new physical calls or fitting."""
import argparse
import json
import math
from pathlib import Path
import torch
from scripts.research.evaluate_chemical_policy import sha


def summarize(rows):
    valid=[r for r in rows if r['valid']]
    return dict(attempts=len(rows),supported=len(valid),accepted=sum(r['accepted'] for r in rows),
        mean_acceptance_probability=sum(math.exp(min(0.,r['log_acceptance_ratio'])) for r in valid)/max(1,len(rows)),
        mean_proposal_potential_change_eV=sum(r['potential_change_eV'] for r in valid)/max(1,len(valid)),
        mean_accepted_potential_change_per_attempt_eV=sum(r['potential_change_eV'] for r in valid if r['accepted'])/max(1,len(rows)),
        mean_reverse_minus_forward_log_q=sum(r['coordinate_log_ratio'] for r in valid)/max(1,len(valid)),
        accepted_graph_changes=sum(r['graph_changed'] for r in valid if r['accepted']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();root=args.project
    master=json.loads((root/'research/evidence/transfer_reuse_protocol_v1.json').read_text())
    result=[]
    for index in master['conditions']:
        for method in master['methods']:
            for replica in master['replicas']:
                directory=root/f'runs/transfer_reuse_eval_v1/condition_{index:02d}/{method}_s{replica}'
                ap=root/f'runs/transfer_reuse_audit_v1/condition_{index:02d}/{method}_s{replica}/results.json'
                audit=json.loads(ap.read_text());report=json.loads((directory/'results.json').read_text())
                assert audit['complete'] and report['complete'] and audit['results_sha256']==sha(directory/'results.json')
                records=[];residuals=[]
                for chunk in report['chunks']:
                    path=directory/chunk['file'];assert sha(path)==chunk['sha256']
                    data=torch.load(path,map_location='cpu',weights_only=False)
                    for round_index,rows in enumerate(data['transitions']):
                        # Compare an identical prefix before either method can
                        # hit its query cap; this is diagnostic, not a new primary.
                        if round_index>=200:break
                        for row in rows:
                            if row['kind']!='joint_exchange':continue
                            r=dict(valid=row['valid'],accepted=row['accepted'],component=None)
                            if method=='learned_vector' and 'forward' in row:
                                f=row['forward'];r['component']=f['component']
                                for physical,learned in zip(f['components'][0]['steps'],f['components'][1]['steps']):
                                    eta0=physical['eta'];eta1=learned['parameters'][0]
                                    residuals.append(float((eta1-eta0).norm()))
                            if row['valid']:
                                old=data['states'][row['old_state_id']];new=data['states'][row['new_state_id']]
                                r.update(potential_change_eV=float(new['potential_eV']-old['potential_eV']),
                                    coordinate_log_ratio=float(row['coordinate_log_ratio']),log_acceptance_ratio=row['log_acceptance_ratio'],
                                    graph_changed=old['graph']['connectivity_smiles']!=new['graph']['connectivity_smiles'])
                            records.append(r)
                item=dict(condition_index=index,method=method,replica=replica,audit_sha256=sha(ap),prefix_microsteps=200,
                    overall=summarize(records),by_draw_component={str(k):summarize([r for r in records if r['component']==k]) for k in [0,1]} if method=='learned_vector' else {},
                    mean_residual_parameter_norm=sum(residuals)/len(residuals) if residuals else None)
                result.append(item);print(json.dumps(item),flush=True)
    out=dict(complete=True,rows=result,new_physical_queries=0,scientific_submission_ready=False,
        scope='Exploratory decomposition after the frozen transfer test; no new primary metric or training data. Component choice is randomized but contexts and proposal paths differ across methods.')
    if args.out.exists():raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':main()
