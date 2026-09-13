#!/usr/bin/env python3
"""Audited matched-recipe comparisons with paired parent-cluster uncertainty."""
import argparse,json,random
from pathlib import Path
import numpy as np
from scripts.research.audit_masked_angular import sha
from scripts.research.summarize_source_force_screen import parents


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','run','audit','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    pp=root/'research/evidence/gate_distillation_training_protocol_v1.json';protocol=json.loads(pp.read_text())
    reports={};provenance={};models=[]
    for recipe in protocol['recipes']:
        for variant in protocol['variants']:
            for replica in range(2):
                rel=Path(recipe)/variant/f'replica_{replica}'/'results.json';rp=a.run/rel;ap=a.audit/rel
                r=json.loads(rp.read_text());audit=json.loads(ap.read_text())
                assert r['complete'] and audit['complete'] and audit['all_metrics_replayed'] and audit['all_controls_replayed']
                assert audit['independent_supported_pairs']==1597 and audit['general_gate_balance_verified'] and audit['index_stream_rebuilt']
                assert r['recipe']==audit['recipe']==recipe and r['variant']==audit['variant']==variant and r['replica']==audit['replica']==replica
                assert r['protocol_sha256']==audit['protocol_sha256']==sha(pp)
                assert audit['results_sha256']==sha(rp) and audit['model_sha256']==r['model_sha256']==sha(rp.parent/'model.pt')
                assert r['phase_boundary_model_sha256']==audit['phase_boundary_sha256']==sha(rp.parent/'phase_boundary_model.pt')
                assert r['index_stream_sha256']==audit['index_stream_sha256']
                assert [v['step'] for v in r['curves']]==protocol['log_steps']
                for v in r['curves']:assert v['stage']==('teacher' if recipe=='distill' and v['step']<=protocol['phase_boundary'] else 'utility')
                key=f'{recipe}_{variant}_{replica}';reports[recipe,variant,replica]=r
                provenance[key]=dict(results_sha256=sha(rp),audit_sha256=sha(ap),model_sha256=r['model_sha256'],index_stream_sha256=r['index_stream_sha256'])
                models.append(dict(recipe=recipe,variant=variant,replica=replica,fit=r['final_fit']['metrics'],internal=r['final_withheld_parent']['metrics'],
                    per_condition=r['final_withheld_parent']['per_condition'],thinning=r['thinning_withheld_parent']['metrics'],
                    fit_thinning_probability=r['fit_matched_thinning_probability'],elapsed_seconds=r['elapsed_seconds'],trainable_parameters=r['trainable_parameters']))
    comparisons=[]
    for replica in range(2):
        assert len({reports[recipe,v,replica]['index_stream_sha256'] for recipe in protocol['recipes'] for v in protocol['variants']})==1
        base=reports['direct','linear',replica]
        values={c:parents(base[c+'_withheld_parent']) for c in ['zero','physical','work']}
        for recipe in protocol['recipes']:
            for variant in protocol['variants']:
                name=recipe+'_'+variant;r=reports[recipe,variant,replica]
                values[name]=parents(r['final_withheld_parent']);values[name+'_thinning']=parents(r['thinning_withheld_parent'])
                for c in ['zero','physical','work']:
                    other=parents(r[c+'_withheld_parent']);assert set(other)==set(values[c])
                    for k in other:
                        for field in other[k]:assert abs(other[k][field]-values[c][k][field])<1e-10
        keys=sorted(values['zero']);assert len(keys)==12
        groups=[[j for j,(i,p) in enumerate(keys) if i==index] for index in sorted({i for i,p in keys})];assert all(len(g)==3 for g in groups)
        rng=random.Random(26031);draws=np.array([[j for group in groups for j in rng.choices(group,k=3)] for _ in range(5000)])
        rates={};bootstrap={}
        for name,rows in values.items():
            u=np.array([rows[k]['expected_work_eV'] for k in keys]);c=np.array([rows[k]['expected_raw_calls'] for k in keys])
            rates[name]=float(u.mean()/c.mean());bootstrap[name]=u[draws].mean(1)/c[draws].mean(1)
        pairs=[]
        for recipe in protocol['recipes']:
            for variant in protocol['variants']:
                name=recipe+'_'+variant;pairs += [(name,c) for c in ['zero','physical','work',name+'_thinning']]
            pairs.append((recipe+'_neural',recipe+'_linear'))
        pairs += [('distill_'+v,'direct_'+v) for v in protocol['variants']]
        for method,control in pairs:
            difference=np.sort(bootstrap[method]-bootstrap[control]);before=rates[control];after=rates[method]
            comparisons.append(dict(replica=replica,method=method,control=control,method_rate=after,control_rate=before,
                difference_eV_per_expected_raw_call=after-before,relative_difference=after/before-1 if before>0 else None,
                fixed_composition_parent_bootstrap95=[float(difference[125]),float(difference[4875])]))
    result=dict(complete=True,protocol_sha256=sha(pp),provenance=provenance,models=models,comparisons=comparisons,
        fixed_controls={c:reports['direct','linear',0][c+'_withheld_parent']['metrics'] for c in ['zero','physical','work']},
        matched_index_streams_verified=True,new_physical_queries=0,actual_queries_saved=0,scientific_submission_ready=False,
        scope='12 internal parents and24 recorded prefixes, all failures and fixed nonjoint/initial costs retained. Same total800 steps, seeds, index streams and optimizer reset across recipes. Descriptive paired parent intervals fix composition and are not multiplicity-adjusted. No actual screened-chain or final-test gain follows.')
    if a.out.exists():raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':main()
