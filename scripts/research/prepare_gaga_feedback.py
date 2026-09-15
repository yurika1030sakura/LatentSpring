#!/usr/bin/env python3
"""Freeze a bounded geometry-feedback challenge before generating new outcomes."""
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path

import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    args = parser.parse_args()
    root = args.project.resolve()
    evidence = root/'research/evidence'
    original = json.loads((evidence/'matched_generators_harmonic_fm_s0_v1.json').read_text())
    assert sha(root/original['data']) == original['data_sha256']
    data = torch.load(root/original['data'], map_location='cpu', weights_only=False)
    train = {r['condition']['composition_hex'] for r in data['training']}
    validation = {r['condition']['composition_hex'] for r in data['validation']}
    assert not train & validation
    unique = {}
    for row in data['validation']:
        unique.setdefault(row['condition']['composition_hex'], row['condition'])
    bins = [(17,28),(29,40),(41,52),(53,64)]
    validation_rows = []
    for low, high in bins:
        candidates = [c for c in unique.values() if low <= c['n_atoms'] <= high]
        candidates.sort(key=lambda c:hashlib.sha256(('gaga-feedback-validation-v1:'+c['composition_hex']).encode()).hexdigest())
        assert len(candidates) >= 8
        validation_rows.extend(copy.deepcopy(candidates[:8]))
    pool_name = 'runs/benchmark_expansion_v1/pool_v2.json'
    audit_name = 'runs/benchmark_expansion_v1/panel_v3/audit.json'
    old_name = 'research/evidence/wide_generalization_panel_v1.json'
    pool = json.loads((root/pool_name).read_text())
    audit = json.loads((root/audit_name).read_text())
    old = json.loads((root/old_name).read_text())
    assert pool['complete'] and audit['complete']
    assert audit['pool_sha256'] == sha(root/pool_name)
    excluded = train | validation | {r['composition_hex'] for r in old['rows']}
    eligible = []
    for decision in audit['decisions']:
        if not decision['qualified']:
            continue
        index = decision['pool_index']
        c = pool['rows'][index]['condition']
        if c['composition_hex'] in excluded:
            continue
        assert all(c['composition_hex'] in corpus['matches'] and not corpus['matches'][c['composition_hex']]
                   for corpus in audit['corpora'].values())
        eligible.append(dict(c, pool_index=index))
    test_rows = []
    for low, high in bins:
        candidates = [c for c in eligible if low <= c['n_atoms'] <= high]
        candidates.sort(key=lambda c:hashlib.sha256(('gaga-feedback-confirmation-v1:'+c['composition_hex']).encode()).hexdigest())
        assert len(candidates) >= 8, (low,high,len(candidates))
        test_rows.extend(candidates[:8])
    panel = dict(frozen=True,validation_rows=validation_rows,test_rows=test_rows,
        selection='8 per fixed atom-count bin by salted composition hash; no new model outcomes used.',
        validation_role='Internal validation compositions; never used for gradient updates.',
        test_role='Previously qualified, previously ungenerated pool compositions; disjoint from the old64 panel and both complete processed corpora.',
        excluded_old_panel=old_name,excluded_old_panel_sha256=sha(root/old_name),
        source_pool=pool_name,source_pool_sha256=sha(root/pool_name),
        qualification_audit=audit_name,qualification_audit_sha256=sha(root/audit_name),
        reserved_outcomes_allowed=False,new_physical_queries=0)
    panel_path = evidence/'gaga_feedback_panel_v1.json'
    write_new(panel_path,panel)
    campaign = dict(format='gaga_feedback_v1',frozen=True,
        at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        panel=str(panel_path.relative_to(root)),panel_sha256=sha(panel_path),
        seeds=[0,1],challengers=['distance','tree'],gaga_sampling_max_t=[350,500,650],
        validation_samples=16,test_samples=32,evaluation_batch=8,calls_per_sample=128,
        challenger_selection='Highest pooled validation graph validity across both seeds; tie favors distance. One shared choice for both seeds.',
        gaga_selection='Highest pooled validation graph validity across both seeds; ties favor larger maximum timestep. The training checkpoint is always the original fixed650 model.',
        confirmation='Always report both feedback challengers, original harmonic FM, selected GAGA and original GAGA650, deduplicating identical GAGA settings.',
        primary_gate='Selected challenger minus validation-selected GAGA has positive raw graph-validity difference in each seed and a positive lower bound of the paired composition bootstrap95 interval.',
        statistics=dict(bootstrap_repetitions=20000,bootstrap_seed=43123,unit='composition, with both model seeds kept together',interval_scope='Conditional on these fitted models; secondary contrasts descriptive.'),
        training_budget='Each feedback model has15000 updates x32 examples x2 supervised backbone passes =960000 backbone example passes, equal to original30000 x32 one-pass models. Data presentations and optimizer updates differ; report measured runtime and never claim equal wall time.',
        architecture='Same2381566 trainable parameters and initial tensor state; replace the existing second invariant edge channel with detached provisional-geometry context.',
        tree_scope='Effective-resistance inclusion probabilities of a regularized auxiliary weighted-tree law; not supplied chemical bonds or the exact posterior of the unchanged source.',
        source_and_inference='Original harmonic source, raw coordinates and128 backbone calls; no energy ranking, optimization or bond labels.',
        expected_new_training_steps=60000,expected_training_presentations=1920000,
        expected_training_backbone_example_passes=3840000,expected_validation_outputs=6144,
        maximum_confirmation_outputs=10240,new_physical_queries=0,reserved_outcomes_allowed=False)
    for seed in campaign['seeds']:
        baseline_paths = {}
        for kind in ['harmonic_fm','gaga']:
            checkpoint = f'runs/matched_generators_v1/training/s{seed}/{kind}/last.ckpt'
            baseline_paths[kind] = dict(path=checkpoint,sha256=sha(root/checkpoint))
        for context in campaign['challengers']:
            spec = json.loads((evidence/f'matched_generators_harmonic_fm_s{seed}_v1.json').read_text())
            spec.update(format='gaga_feedback_training_v1',kind='harmonic_fm',context=context,two_pass=True,
                tree_regularization=1e-3,training_steps=15000,batch_schedule_total_steps=30000,
                checkpoint_every=500,validation_steps=[15000],final_checkpoint='EMA at fixed15000 updates; same selection across seeds on validation only.',
                panel=campaign['panel'],panel_sha256=campaign['panel_sha256'],
                campaign='research/evidence/gaga_feedback_campaign_v1.json',
                validation_samples=16,evaluation_batch=8,validation_seed=43001+seed,
                evaluation_seed=43101+seed,baselines=baseline_paths)
            write_new(evidence/f'gaga_feedback_{context}_s{seed}_v1.json',spec)
    write_new(evidence/'gaga_feedback_campaign_v1.json',campaign)
    print(json.dumps(dict(validation_compositions=len(validation_rows),test_compositions=len(test_rows),training_rows=len(data['training']),panel_sha256=sha(panel_path))))


if __name__ == '__main__':
    main()
