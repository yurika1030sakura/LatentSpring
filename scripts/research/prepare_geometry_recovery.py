"""Freeze a small paired geometry-recovery experiment before generating outputs."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import torch
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, required=True)
    a = p.parse_args(); root = a.project.resolve()
    path = root / 'research/evidence/geometry_recovery_v1.json'
    assert not path.exists()
    campaign_file = root / 'research/evidence/seed_replication_v1.json'
    campaign = json.loads(campaign_file.read_text())
    data_file = root / campaign['data']
    assert sha(data_file) == campaign['data_sha256']
    data = torch.load(data_file, map_location='cpu', weights_only=False)
    train_keys = {r['condition']['composition_hex'] for r in data['training']}
    assert len(data['training']) == 20000
    previous = json.loads((root / 'research/evidence/round2_illustration_protocol_v1.json').read_text())
    exclude = {c['composition_hex'] for c in previous['conditions']}
    ranked = sorted(enumerate(data['validation']), key=lambda item: hashlib.sha256(
        ('geometry-recovery-dev-20260924:' + item[1]['condition']['composition_hex']).encode()).hexdigest())
    selected = []; used = set()
    for lower, upper in [(17, 28), (29, 40)]:
        candidates = []
        for i, row in ranked:
            c = row['condition']; key = c['composition_hex']
            if key in used or key in exclude or not lower <= c['n_atoms'] <= upper:
                continue
            assert key not in train_keys
            candidates.append(dict(validation_row=i, condition=c)); used.add(key)
            if len(candidates) == 12:
                break
        assert len(candidates) == 12, (lower, upper, len(candidates))
        selected.extend(candidates)
    head_file = root / campaign['old_hydrogen_protocol']
    heads = json.loads(head_file.read_text())['physical_heads']
    parents = []; physical_heads = []
    for fit in [0, 1]:
        arm = campaign['fits'][fit]['parents']['fm']
        assert sha(root / arm['checkpoint']) == arm['checkpoint_sha256']
        info = heads[str(fit)]['fm']; assert sha(root / info['path']) == info['sha256']
        parents.append(arm); physical_heads.append(info)
    spec = dict(format='geometry_recovery_v1', frozen=True,
        at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        campaign=str(campaign_file.relative_to(root)), campaign_sha256=sha(campaign_file),
        parent_protocol=campaign['old_connection_protocol'],
        parent_protocol_sha256=campaign['old_connection_protocol_sha256'],
        data=campaign['data'], data_sha256=campaign['data_sha256'],
        parents=parents, physical_heads=physical_heads,
        fits=[0, 1], variants=['replay', 'recovery', 'recovery_local'],
        steps=4000, batch_size=32, learning_rate=1e-5, ema_decay=.999,
        checkpoint_every=500, batch_seeds=[73101, 73102], noise_seeds=[73201, 73202],
        evaluation_seeds=[73301, 73302], conditions=selected,
        evaluation_batch=8, samples_per_condition=16, backbone_calls=128,
        physical_strength=4., hydrogen_readout=False,
        objectives=dict(odd_updates='Original two-pass CFM objective, identical in every arm.',
            even_updates='Two-pass late endpoint regression to the same original training coordinates.',
            replay='Unperturbed interpolation; controls for continuation, data exposure and late-time weighting.',
            recovery='Masked atom noise plus a displaced spatial region, applied to the late training interpolation.',
            recovery_local='Same corrupted states plus reference-relative local distance and neighbor-angle losses.',
            weights=dict(endpoint=4.,local_distance=.5,local_angle=.1),
            progress=[.60,.95],atom_noise_scales_A=[.05,.15,.30],atom_mask_probability=.25,
            region_shift_std_A=.35,region_probability=.5),
        model_selection='Final EMA after4000 updates; no intermediate checkpoint selection.',
        primary='All-attempt graph-valid, zero-radical PoseBusters geometry-subset yield; compare each recovery arm with frozen AND replay in each fit.',
        secondary='Geometry plus GFN2 RMS force<=5, graph validity, force distribution, per-composition distinct valid connectivity, heavy connectivity and individual geometry failures.',
        advancement='Require positive geometry and geometry-plus-force point gains over both controls in both fits, a positive paired composition97.5% bootstrap lower bound for geometry (two candidate corrections), and no loss of graph-valid distinct-connectivity yield greater than2 percentage points. A development gate, not confirmatory evidence.',
        scope='Same20k OMol25 references, no supplied or pseudo bond labels, original source/sampler, frozen existing force head. Recovery changes the learned backbone. This24-composition validation panel is held out from training but is a development panel; no claim it was untouched by all earlier studies. Existing64-composition primary benchmark is not used for this pilot.',
        budget=dict(new_backbone_updates=24000,backbone_training_example_forwards=1536000,
            new_generation_outputs=3072,new_gfn2_attempts=3072,new_esen_queries=0,
            max_gpu_hours=18,per_array_task_time_hours=3),
        statistics=dict(bootstrap_seed=73401,bootstrap_repetitions=10000,
            unit='composition, preserving paired draws and retaining both fits; all fit-specific contrasts reported'))
    write(path,spec)
    print(json.dumps(dict(protocol=str(path),sha256=sha(path),conditions=len(selected),
        sizes=[s['condition']['n_atoms'] for s in selected],budget=spec['budget'])),flush=True)


if __name__ == '__main__': main()
