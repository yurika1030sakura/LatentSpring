"""Audit recorded integration outputs without changing a model or protocol."""
import argparse,json
from pathlib import Path
import numpy as np
from cfm_mol.weighted_endpoints import WeightedEndpointStore,file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();pilot=a.root/'cartesian_diagnostic';audit=a.root/'reference_audit'
    native=json.loads((pilot/'summary.json').read_text())
    reference=json.loads((audit/'summary.json').read_text())
    assert native['complete'] and reference['complete']
    refs={r['name']:np.array(r['reference_probs']) for r in reference['rows']}
    protocol=json.loads((pilot/'protocol.json').read_text());rows=[];checks=[]
    for seed in protocol['seeds']:
        for arm in ['proposal_min','full_work_min']:
            store=WeightedEndpointStore.load(pilot/f'endpoints_{arm}_{seed}')
            for group in store.groups:
                probabilities=np.array([group.masses().get(str(j),0.) for j in range(len(refs[group.group_id]))])
                with np.load(pilot/f'teacher_{group.group_id}_{seed}.npz') as raw:
                    lw=raw['log_weight'] if arm=='full_work_min' else np.zeros(len(raw['log_weight']))
                    weights=np.exp(lw-lw.max());weights/=weights.sum()
                    p=np.bincount(raw['basin'],weights=weights,minlength=len(probabilities))
                    work_error=float(np.max(np.abs(raw['log_weight']-
                        (-(raw['energy_kcal_mol']-raw['energy_kcal_mol'].min())/(.00198720425864083*300)+raw['log_jacobian']))))
                err=float(abs(p-probabilities).max());assert err<1e-12 and work_error<1e-12
                rows.append(dict(seed=seed,arm=arm,molecule=group.group_id,
                    TV_reference=float(.5*abs(probabilities-refs[group.group_id]).sum()),
                    aggregation_mass_error=err,complete_work_formula_error=work_error,
                    manifest_sha256=file_sha256(pilot/f'endpoints_{arm}_{seed}'/'manifest.json')))
            # Sampling checks test LABEL mass before smoothing, not raw student mass.
            rng=np.random.default_rng(seed+2000000);draws=store.draw(20000,rng)
            counts={g.group_id:[] for g in store.groups}
            for d in draws:counts[d.group_id].append(d.basin_id)
            for g in store.groups:
                empirical=np.array([counts[g.group_id].count(str(j))/len(counts[g.group_id]) for j in range(len(refs[g.group_id]))])
                expected=np.array([g.masses().get(str(j),0.) for j in range(len(empirical))])
                checks.append(dict(seed=seed,arm=arm,group=g.group_id,draws=len(counts[g.group_id]),
                    label_TV_to_store=float(.5*abs(empirical-expected).sum())))
    aggregate={}
    for arm in ['proposal_min','full_work_min']:
        values=[np.mean([r['TV_reference'] for r in rows if r['arm']==arm and r['seed']==seed]) for seed in protocol['seeds']]
        aggregate[arm]=dict(mean=float(np.mean(values)),sample_sd=float(np.std(values,ddof=1)),
                           independent_teacher_run_means=list(map(float,values)))
    attempts=sum(e['attempted'] for r in native['runs'] for e in r['evaluation'])
    valid=sum(e['graph_valid'] for r in native['runs'] for e in r['evaluation'])
    finite=sum(e['finite'] for r in native['runs'] for e in r['evaluation'])
    report=dict(complete=True,teacher_TV_reference=aggregate,teacher_rows=rows,label_sampling_checks=checks,
        max_aggregation_error=max(r['aggregation_mass_error'] for r in rows),
        max_label_sampling_TV=max(r['label_TV_to_store'] for r in checks),
        original_checkpoint_loaded=False,main_backbone_trained=False,
        cpu_reference_model=dict(parameters=native['runs'][0]['trainable_parameters'],runs=len(native['runs']),
            optimizer_updates=sum(protocol['steps'] for r in native['runs']),attempted_outputs=attempts,
            finite_outputs=finite,graph_valid_outputs=valid,student_basin_TV=None,
            interpretation='Integration/gradient checks succeeded; molecular generation did NOT succeed at this diagnostic budget.'),
        fresh_teacher_queries=native['fresh_teacher_MMFF_queries'],
        independent_reference_queries=reference['total_reference_queries'],
        reused_fixed_basin_map=True,reference_rounding_audit=reference['rows'],
        new_confirmatory_paper_evidence=False)
    (a.root/'integration_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['teacher_rows','label_sampling_checks','reference_rounding_audit']},indent=2))


if __name__=='__main__':main()
