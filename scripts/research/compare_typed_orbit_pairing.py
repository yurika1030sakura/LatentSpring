#!/usr/bin/env python3
"""Audit the supplementary type-matching baseline and compare all five methods."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scripts.research.train_electronic_fm import sha
from scripts.research.audit_generator_output_support import assess
from scripts.research.evaluate_chemical_policy import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['primary','primary-audit','primary-training','supplement','protocol','out']:
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    protocol=json.loads(a.protocol.read_text())
    base=json.loads((a.primary/'results.json').read_text());audit=json.loads(a.primary_audit.read_text())
    result_path=a.supplement/'evaluation/results.json';extra=json.loads(result_path.read_text())
    assert base['complete'] and audit['complete'] and extra['complete']
    assert audit['source_results_sha256']==sha(a.primary/'results.json')
    assert extra['protocol_sha256']==sha(a.protocol)
    assert protocol['parent_protocol_sha256']==base['protocol_sha256']==audit['protocol_sha256']
    assert extra['supplementary_method']=='typed_rotation' and len(extra['rows'])==len(protocol['conditions'])
    directory=a.supplement/'training/typed_rotation'
    recipe=json.loads((directory/'protocol.json').read_text())
    validation=json.loads((directory/'validation.json').read_text())
    assert validation['complete'] and recipe['pairing_protocol_sha256']==sha(a.protocol)
    assert recipe['data_order_sha256']==base['sources']['independent']['protocol']['data_order_sha256']
    assert sha(directory/'last.ckpt')==extra['sources']['typed_rotation']['sha256']
    training=[json.loads(s) for s in (directory/'metrics.jsonl').read_text().splitlines()]
    original=[json.loads(s) for s in (a.primary_training/'independent/metrics.jsonl').read_text().splitlines()]
    assert len(training)==len(original)==protocol['training_steps']==validation['global_step']
    for old,new in zip(original,training):
        assert old['step']==new['step'] and old['processed_indices']==new['processed_indices']
        assert np.isfinite(new['fm_loss']) and np.isfinite(new['gradient_norm'])
        assert all(r['mode']=='typed_rotation' for r in new['pairing'])
    positions_checked=0;sources={};rows=base['rows']+extra['rows']
    for row in extra['rows']:
        index=row['condition_index'];source=a.supplement/f'evaluation/typed_rotation_c{index}.pt'
        assert sha(source)==row['sample_sha256']
        stored=torch.load(source,map_location='cpu',weights_only=False)
        reference=next(r for r in base['rows'] if r['method']=='warm' and r['condition_index']==index)
        assert stored['condition']==row['condition']==reference['condition']
        assert stored['seeds']==row['seeds']==reference['seeds']
        x=stored['positions'];assert len(x)==protocol['samples_per_condition']
        assert torch.isfinite(x).all() and x.mean(1).abs().max()<1e-8
        rebuilt=assess(x,row['condition'],list(range(len(x))))
        for k,v in rebuilt.items():assert row[k]==v
        positions_checked+=len(x);sources[source.name]=sha(source)
    methods={}
    names=['warm','independent','rotation','steric','typed_rotation']
    for method in names:
        selected=[r for r in rows if r['method']==method]
        assert len(selected)==len(protocol['conditions'])
        methods[method]=dict(attempted=sum(r['attempted'] for r in selected),
            graph_supported=sum(r['graph_supported'] for r in selected),
            geometrically_supported=sum(r['geometrically_supported'] for r in selected),
            validator_errors=sum(r['validator_errors'] for r in selected),
            distinct_connectivities_sum=sum(r['distinct_connectivity'] for r in selected),
            generation_seconds=sum(r['generation_seconds'] for r in selected),
            additional_training_seconds=0. if method=='warm' else validation['seconds'] if method=='typed_rotation' else audit['training'][method]['seconds'])
    comparisons={}
    for control in ['warm','independent','rotation','typed_rotation']:
        diffs=[]
        for index in protocol['conditions']:
            left=next(r for r in rows if r['method']=='steric' and r['condition_index']==index)
            right=next(r for r in rows if r['method']==control and r['condition_index']==index)
            diffs.append([int(x['graph_supported'])-int(y['graph_supported']) for x,y in zip(left['records'],right['records'])])
        d=np.array(diffs);g=np.random.default_rng(88411)
        idx=g.integers(d.shape[1],size=(10000,*d.shape))
        boot=np.take_along_axis(d[None],idx,axis=2).mean((1,2))
        comparisons['steric minus '+control]=dict(graph_support_fraction_difference=float(d.mean()),
            conditional_paired_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
            per_condition_count_difference=d.sum(1).tolist())
    write(a.out,dict(complete=True,primary_results_sha256=sha(a.primary/'results.json'),primary_audit_sha256=sha(a.primary_audit),
        supplementary_results_sha256=sha(result_path),supplementary_protocol_sha256=sha(a.protocol),sample_hashes=sources,
        supplementary_positions_audited=positions_checked,primary_positions_audited=audit['all_structural_outcomes_replayed'],
        matched_training_rows=len(training),methods=methods,comparisons=comparisons,
        supplementary_training=dict(mean_reassigned_atoms=float(np.mean([p['atoms_reassigned'] for r in training for p in r['pairing']])),
            mean_displacement_per_atom_A2=float(np.mean([p['displacement_per_atom'] for r in training for p in r['pairing']])),
            mean_overlap=float(np.mean([p['selected_overlap'] for r in training for p in r['pairing']])),
            metrics_sha256=sha(directory/'metrics.jsonl')),
        rows=[{k:v for k,v in r.items() if k!='records'} for r in rows],new_molecular_oracle_calls=0,scientific_submission_ready=False,
        limits=['One training seed; conditional Gaussian-draw bootstrap is not a training replication.','Typed assignment is existing symmetry-matching methodology, added as a stronger baseline during primary evaluation.','The typed target additionally permutes identical conditioning labels; it preserves physical molecular shape, not the original arbitrary atom-row order.','No thermal distribution or electronic-state validity certificate follows from graph perception.','Training wall times include diagnostic pairing work, so they are not optimized implementation speed comparisons.']))
    print(json.dumps(dict(methods=methods,comparisons=comparisons)),flush=True)


if __name__=='__main__':main()
