#!/usr/bin/env python3
"""Paired parent uncertainty for fixed whole-prefix screen expectations."""
import argparse,json,random
from pathlib import Path
from statistics import mean
from scripts.research.audit_masked_angular import sha


def parents(report):
    rows=report['prefixes'];out={}
    for key in sorted({(r['index'],r['parent']) for r in rows}):
        group=[r for r in rows if (r['index'],r['parent'])==key];assert len(group)==2
        out[key]={k:mean(r[k] for r in group) for k in ['expected_work_eV','expected_raw_calls']}
    return out


def rate(rows,selected):
    return (mean(mean(rows[i,p]['expected_work_eV'] for p in ids) for i,ids in selected.items())/
            mean(mean(rows[i,p]['expected_raw_calls'] for p in ids) for i,ids in selected.items()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','run','audit','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    pp=root/'research/evidence/source_force_screen_training_protocol_v1.json';protocol=json.loads(pp.read_text())
    reports={};models=[];provenance={}
    for variant in protocol['variants']:
        for replica in range(2):
            rel=Path(variant)/f'replica_{replica}'/'results.json';rp=a.run/rel;ap=a.audit/rel
            r=json.loads(rp.read_text());audit=json.loads(ap.read_text())
            assert r['complete'] and audit['complete'] and audit['all_metrics_replayed'] and audit['all_controls_replayed']
            assert audit['all1671_attempts_retained'] and audit['general_gate_balance_verified'] and audit['independent_supported_pairs']==1597
            assert r['protocol_sha256']==audit['protocol_sha256']==sha(pp)
            assert audit['results_sha256']==sha(rp) and audit['model_sha256']==r['model_sha256']==sha(rp.parent/'model.pt')
            assert r['data_sha256']==audit['data_sha256']==protocol['data_sha256']
            reports[variant,replica]=r;provenance[f'{variant}_{replica}']=dict(results_sha256=sha(rp),audit_sha256=sha(ap),model_sha256=r['model_sha256'])
            models.append(dict(variant=variant,replica=replica,fit=r['final_fit']['metrics'],internal=r['final_withheld_parent']['metrics'],
                per_condition=r['final_withheld_parent']['per_condition'],thinning=r['thinning_withheld_parent']['metrics'],
                fit_thinning_probability=r['fit_matched_thinning_probability'],trainable_parameters=r['trainable_parameters'],elapsed_seconds=r['elapsed_seconds']))
    comparisons=[]
    for replica in range(2):
        control=reports['linear',replica]
        values={name:parents(control[name+'_withheld_parent']) for name in ['zero','physical','work']}
        values.update({v:parents(reports[v,replica]['final_withheld_parent']) for v in protocol['variants']})
        values.update({v+'_thinning':parents(reports[v,replica]['thinning_withheld_parent']) for v in protocol['variants']})
        selected={i:sorted(p for ii,p in values['zero'] if ii==i) for i in sorted({ii for ii,p in values['zero']})}
        assert len(values['zero'])==12 and all(len(ids)==3 for ids in selected.values())
        pairs=([('physical','zero'),('work','zero')] if replica==0 else [])
        pairs += [(v,c) for v in protocol['variants'] for c in ['zero','physical','work',v+'_thinning']]+[('neural','linear')]
        for learned,baseline in pairs:
            after=rate(values[learned],selected);before=rate(values[baseline],selected)
            rng=random.Random(25931);draws=[]
            for _ in range(5000):
                sample={i:rng.choices(ids,k=len(ids)) for i,ids in selected.items()}
                draws.append(rate(values[learned],sample)-rate(values[baseline],sample))
            draws.sort();comparisons.append(dict(replica=replica,method=learned,control=baseline,method_rate=after,control_rate=before,
                difference_eV_per_expected_raw_call=after-before,relative_difference=after/before-1 if before>0 else None,
                fixed_composition_parent_bootstrap95=[draws[125],draws[4875]]))
    result=dict(complete=True,protocol_sha256=sha(pp),provenance=provenance,models=models,comparisons=comparisons,
        fixed_controls={name:reports['linear',0][name+'_withheld_parent']['metrics'] for name in ['zero','physical','work']},
        new_physical_queries=0,actual_queries_saved=0,scientific_submission_ready=False,
        scope='12 internal parents and their24 recorded physical prefixes. Initial/nonjoint terms remain fixed. These are whole-prefix expectations at recorded states, not changed screened trajectories, actual savings or final-test evidence. Parent intervals fix composition and are not multiplicity-adjusted.')
    if a.out.exists():raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(comparisons,indent=2))


if __name__=='__main__':main()
