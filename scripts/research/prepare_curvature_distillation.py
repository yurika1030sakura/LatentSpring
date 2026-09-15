#!/usr/bin/env python3
"""Freeze full local teachers and a controlled transfer into the FM generator."""
import datetime
import json
from pathlib import Path
import torch
from scripts.research.train_electronic_fm import sha


def main():
    root=Path(__file__).resolve().parents[2];torch.set_num_threads(1)
    audit=root/'research/evidence/curvature_escort_audit_v1.json'
    assert json.loads(audit.read_text())['primary_gate']
    for seed in [0,1]:
        pilot=json.loads((root/f'research/evidence/curvature_escort_s{seed}_v1.json').read_text())
        original=json.loads((root/f'research/evidence/thermal_distillation_s{seed}_v1.json').read_text())
        fresh=json.loads((root/f'research/evidence/fresh_physics_s{seed}_v1.json').read_text())
        full=dict(pilot);full['teacher_files']=[dict(r) for r in pilot['teacher_files']]
        count=0
        for ref in full['teacher_files']:
            saved=torch.load(root/ref['path'],weights_only=False,map_location='cpu')
            ref['anchor_indices']=torch.where(saved['record']['eligible'])[0].tolist();count+=len(ref['anchor_indices'])
        full.update(format='curvature_full_teacher_v1',methods=['translation','secant'],
            production_particles=dict(translation=24,secant=16),production_seed=41503+seed,
            maximum_new_esen_queries=count*40*2,selection='All original349 eligible teacher anchors, across both continuations, in unchanged order. Fresh production draws independent of the64-anchor pilot.',
            primary='Fixed full teacher for a neural distillation comparison; no further map tuning.',
            at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pilot_audit_sha256=sha(audit))
        target=root/f'research/evidence/curvature_full_teacher_s{seed}_v1.json';assert not target.exists()
        target.write_text(json.dumps(full,indent=2)+'\n')
        spec=dict(original)
        for key in ['condition_manifest','condition_manifest_sha256','conditions','samples_per_condition',
            'evaluation_batch','evaluation_seed','terminal_noise_std_A','xtb_binary','xtb_binary_sha256','xtb',
            'xtb_energy_inversion_tolerance_eV','xtb_force_inversion_tolerance_eV_A']:
            spec[key]=fresh[key]
        replay=root/f'runs/thermal_distillation_v1/s{seed}/study/replay/last.ckpt'
        selection=root/f'runs/thermal_distillation_v1/s{seed}/study/escort/metrics.jsonl'
        spec.update(format='curvature_distillation_v1',at_utc=full['at_utc'],methods=['force24','work24','curvature_work16'],
            teacher_protocol=str(target.relative_to(root)),teacher_protocol_sha256=sha(target),
            teacher_run=f'runs/curvature_distillation_v1/teacher/s{seed}',
            replay_checkpoint=str(replay.relative_to(root)),replay_checkpoint_sha256=sha(replay),
            selection_log=str(selection.relative_to(root)),selection_log_sha256=sha(selection),
            oracle_worker='scripts/research/oracle_worker.py',
            objective='Three1000-update physical students reuse the identical original reference/anchor/source/time schedule and original matched replay weights. Compare uniform24 translated particles, complete-work24 translated particles, and complete-work16 affine particles fitted with up to8 independent pilot queries.',
            primary='Curvature-work paired update versus translation-work paired update: raw jointly-valid energy and force on the24 unfitted compositions; each uses at most24 particle energy queries per anchor. Force24 is the cheaper anchor-force-only control.',
            primary_gate='Curvature-work lowers jointly-valid energy relative to work24 in each continuation and has a pooled paired95 upper bound below zero under both eSEN and independent GFN2. Observed graph-validity loss <=2pp. Report all-output and joint-quality yield curves too.',
            claim_boundary='Reused evaluation panel, with no fitting on its structures or outcomes. Curvature settings fixed using FIT-only mechanism pilot. This tests whether improved local work sampling transfers into raw neural outputs. No global Boltzmann or exact task-arithmetic claim.',
            matched_replay_reuse='Reuse only if every original eligible anchor has at least one supported particle under both new maps, so all original replay targets, pool indices and mixture weights remain identical.',
            new_neural_outputs=len(fresh['conditions'])*fresh['samples_per_condition']*3,
            new_training_steps=3000,teacher_anchors=count,
            esen_queries_per_seed=2*len(fresh['conditions'])*fresh['samples_per_condition']*3)
        target=root/f'research/evidence/curvature_distillation_s{seed}_v1.json';assert not target.exists()
        target.write_text(json.dumps(spec,indent=2)+'\n')
        print(json.dumps(dict(seed=seed,teacher_anchors=count,teacher_new_queries=count*80,
            new_training_steps=3000,new_nn_outputs=spec['new_neural_outputs'])))


if __name__=='__main__':main()
