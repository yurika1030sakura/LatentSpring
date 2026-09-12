#!/usr/bin/env python3
"""Diagnose candidate energy ranking and confidence; not continuous target KL."""
import argparse
import json
from pathlib import Path
from statistics import mean
import torch
from cfm_mol.conditional_arc_energy import ConditionalArcEnergy
from scripts.research.audit_masked_angular import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','out']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[2]
    pp=root/'research/evidence/conditional_arc_training_protocol_v1.json';protocol=json.loads(pp.read_text())
    ap=args.project/'runs/conditional_arc_learning_audit_v1/results.json';audit=json.loads(ap.read_text())
    assert audit['complete'] and audit['protocol_sha256']==sha(pp)
    dp=args.project/protocol['data_run']/'data.pt';assert sha(dp)==protocol['data_sha256']
    records=torch.load(dp,map_location='cpu',weights_only=False)
    records=[r for r in records if r['role']!='fit'];assert len(records)==232
    models={}
    for name,gamma in [('physical_site',0.),('site_confinement',.1)]:
        models[name]=ConditionalArcEnergy(**dict(protocol['model'],restraint=gamma)).double().eval()
    for row in audit['rows']:
        path=args.project/f"runs/conditional_arc_train_v1/{row['name']}/model.pt";assert sha(path)==row['model_sha256']
        saved=torch.load(path,map_location='cpu',weights_only=False)
        model=ConditionalArcEnergy(**saved['configuration']).double();model.load_state_dict(saved['state_dict']);model.eval()
        models[row['name']]=model
    rows=[]
    with torch.no_grad():
        for r in records:
            selected=[0]+[i for i,name in enumerate(r['sample_roles']) if name.startswith('arc_')]
            assert len(selected)==5
            u=r['directions'][selected][None];work=r['work_eV'][selected];kT=r['electronic'][2]
            truth=(-work/kT).log_softmax(0);weights=truth.exp();best=int(work.argmin())
            for name,model in models.items():
                energy=model(r['positions'][None],r['bonds'][None],r['numbers'],r['electronic'],r['root'][None],u)[0]
                predicted=(-energy/kT).log_softmax(0);choice=int(energy.argmin())
                rows.append(dict(index=r['index'],parent=r['parent'],context=r['context'],role=r['role'],method=name,
                    finite_candidate_KL=float((weights*(truth-predicted)).sum()),
                    chosen_energy_regret_eV=float(work[choice]-work[best]),
                    best_candidate_selected=int(choice==best),oracle_best_probability=float(predicted[best].exp())))
    summaries={}
    keys=['finite_candidate_KL','chosen_energy_regret_eV','best_candidate_selected','oracle_best_probability']
    for role in ['withheld_parent','withheld_composition']:
        summaries[role]={}
        for name in models:
            group=[r for r in rows if r['role']==role and r['method']==name];cases={}
            for index in sorted({r['index'] for r in group}):
                current=[r for r in group if r['index']==index];parents=sorted({r['parent'] for r in current})
                cases[str(index)]={key:mean(mean(r[key] for r in current if r['parent']==parent) for parent in parents) for key in keys}
            summaries[role][name]=dict(per_condition=cases,**{key:mean(c[key] for c in cases.values()) for key in keys})
    result=dict(complete=True,learning_audit_sha256=sha(ap),data_sha256=sha(dp),rows=rows,summary=summaries,
        new_physical_queries=0,scientific_submission_ready=False,
        scope='Post-training diagnostic on the same232 internally withheld contexts. Each candidate set contains the center and four fixed arc draws. These finite-set probabilities omit quadrature/proposal-volume weights and are NOT a molecular conditional distribution or continuous-KL estimate. No model fitting or selection is performed.')
    if args.out.exists():raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({role:{m:{k:v for k,v in result.items() if k!='per_condition'} for m,result in values.items()}
        for role,values in summaries.items()},indent=2))


if __name__=='__main__':main()
