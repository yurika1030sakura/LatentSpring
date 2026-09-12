#!/usr/bin/env python3
"""Compare tuned physical widths with the frozen original neural endpoints."""
import argparse
import json
from pathlib import Path
import torch
from cfm_mol.hierarchical_uncertainty import hierarchical_paired_mean
from scripts.research.summarize_fresh_reuse import endpoint,bootstrap_mean
from scripts.research.summarize_transfer_reuse import curves_from_trace
from scripts.research.evaluate_chemical_policy import sha


def load_curves(directory,audit_path):
    report=json.loads((directory/'results.json').read_text());audit=json.loads(audit_path.read_text())
    assert report['complete'] and audit['complete'] and audit['full_producer_replay']
    assert audit['results_sha256']==sha(directory/'results.json')
    curves={}
    for chunk in report['chunks']:
        path=directory/chunk['file'];assert sha(path)==chunk['sha256']
        curves.update(curves_from_trace(torch.load(path,map_location='cpu',weights_only=False)))
    assert list(curves)==report['parent_ids']
    return report,curves


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','run','audits','out']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();root=args.project
    mp=root/'research/evidence/site_concentration_controls_protocol_v1.json';master=json.loads(mp.read_text())
    original={}
    for label in ['primary']+[f'transfer_{i:02d}' for i in range(6)]:
        for rep in [0,1]:
            if label=='primary':
                directory=root/f'runs/fresh_reuse_eval_v2/learned_vector_s{rep}';ap=root/f'runs/fresh_reuse_audit_v1/learned_vector_s{rep}/results.json'
            else:
                index=int(label[-2:]);directory=root/f'runs/transfer_reuse_eval_v1/condition_{index:02d}/learned_vector_s{rep}';ap=root/f'runs/transfer_reuse_audit_v1/condition_{index:02d}/learned_vector_s{rep}/results.json'
            original[(label,rep)]=load_curves(directory,ap)
    controls={};total=0
    for case in master['cases']:
        cp=root/case['protocol'];assert sha(cp)==case['sha256']
        for rep in [0,1]:
            suffix=f"kappa_{int(case['kappa'])}/{case['label']}/replica_{rep}"
            report,curves=load_curves(args.run/suffix,args.audits/suffix/'results.json')
            assert report['protocol_sha256']==sha(cp) and report['proposal_site_concentration']==case['kappa']
            assert report['parent_ids']==original[(case['label'],rep)][0]['parent_ids']
            controls[(case['kappa'],case['label'],rep)]=(report,curves);total+=report['new_raw_queries']
    rows=[]
    for kappa in master['concentrations']:
        for cohort,labels in [('primary',['primary']),('transfer',[f'transfer_{i:02d}' for i in range(6)])]:
            n=sum(len(original[(label,0)][0]['parent_ids']) for label in labels)
            scenarios=[('same_inference',None),('matched_total_incremental_calibration',0)]
            if kappa==400:scenarios.append(('matched_total_fully_charged_calibration',10108))
            for scenario,calibration in scenarios:
                q,r=divmod((9660-calibration)//2,n) if calibration is not None else (0,0)
                differences=[[],[]];missing=[];details=[];offset=0
                for label in labels:
                    ids=original[(label,0)][0]['parent_ids'];budgets=[512+2*q+2*(offset+i<r) for i in range(len(ids))];offset+=len(ids)
                    case_diff=[]
                    for rep in [0,1]:
                        learned=original[(label,rep)][1];control=controls[(kappa,label,rep)][1]
                        aa=[endpoint(learned[pid],512,None) for pid in ids]
                        bb=[endpoint(control[pid],budget,None) for pid,budget in zip(ids,budgets)]
                        if any(x is None for x in aa+bb):missing.append(dict(label=label,replica=rep))
                        else:
                            delta=[a['potential_eV']-b['potential_eV'] for a,b in zip(aa,bb)]
                            differences[rep].append(delta);case_diff.append(delta)
                    details.append(dict(label=label,physical_query_budgets=budgets,paired_differences_eV=case_diff))
                row=dict(kappa=kappa,cohort=cohort,scenario=scenario,physical_calibration_raw_calls=calibration,
                    available=not missing,missing=missing,details=details)
                if not missing:
                    if cohort=='primary':stats=bootstrap_mean([differences[rep][0] for rep in [0,1]],torch.Generator().manual_seed(24991),2000)
                    else:stats=hierarchical_paired_mean(differences,generator=torch.Generator().manual_seed(24991))
                    row['learned_minus_physical_potential_eV']=stats
                rows.append(row)
                print(json.dumps({k:v for k,v in row.items() if k!='details'}),flush=True)
    out=dict(complete=True,master_protocol_sha256=sha(mp),rows=rows,new_physical_queries_in_controls=total,
        new_physical_queries_in_summary=0,scientific_submission_ready=False,
        scope='Stronger physical angular-width controls; report both calibration-accounting regimes. Outcomes do not retrospectively change the old primary comparison or establish equilibrium or wall-time superiority.')
    if args.out.exists():raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':main()
