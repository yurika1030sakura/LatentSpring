#!/usr/bin/env python3
"""Freeze a FIT-only, equal-budget curvature-escort pilot."""
import datetime
import json
from pathlib import Path
import torch
from scripts.research.train_electronic_fm import sha


def main():
    root = Path(__file__).resolve().parents[2]
    torch.set_num_threads(1)
    for seed in [0,1]:
        original = json.loads((root/f'research/evidence/thermal_distillation_s{seed}_v1.json').read_text())
        files = []
        for i in range(8):
            file = root/f'runs/thermal_distillation_v1/s{seed}/study/teacher/refined_c{i}.pt'
            saved = torch.load(file, weights_only=False, map_location='cpu')
            indices = torch.where(saved['record']['eligible'])[0][:4].tolist()
            assert len(indices) == 4
            files.append(dict(path=str(file.relative_to(root)), sha256=sha(file), condition_index=i,
                anchor_indices=indices, composition_hex=saved['condition']['composition_hex']))
        spec = {k:original[k] for k in ['oracle_checkpoint','oracle_interpreter','oracle_sha256']}
        worker = root/'scripts/research/oracle_worker.py'
        spec.update(frozen=True, format='curvature_escort_pilot_v1', model_seed=seed,
            at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), teacher_files=files,
            oracle_worker=str(worker.relative_to(root)), oracle_worker_sha256=sha(worker),
            methods=['translation','isotropic','secant'], production_particles=dict(translation=24,isotropic=16,secant=16),
            pilot_candidates_per_anchor=8, ridge_fraction=.01, temperature_K=300,
            kT=original['thermal_teacher']['kT'], production_seed=41303+seed,
            maximum_new_esen_queries=32*(24+16+16)*2,
            selection='First four existing eligible anchors in each of eight FIT compositions per continuation; no energy or ESS ranking. All original pilot particles and force queries are charged to fitted escorts.',
            target='Exactly the original anchor-centered Gaussian times exp(-DeltaE/kT), restricted to the same chemical graph. Original anchor, sigma and temperature unchanged.',
            map='Translation baseline: original force displacement. Isotropic and secant maps: mean and covariance of the Gaussian under a PSD local quadratic energy approximation, fitted from independent cached pilot forces. Complete work includes full Gaussian ratio and intrinsic log determinant.',
            cost='Translation uses24 fresh production queries per anchor. Fitted maps use at most8 pilot queries plus16 fresh production queries. Anchor queries common; mirrored eSEN queries counted twice. No pilot particles enter fitted-map production estimators. All fresh proposals are queried, even when graph-invalid.',
            primary='Absolute local production ESS under complete work per equal maximum24 particle-query budget. Translation has24 production particles, fitted maps16. Report per-seed means, paired anchor differences, all graph retention and zero-support cases.',
            gate='Secant mean ESS exceeds translation in each seed, and pooled paired-anchor bootstrap95 lower bound for the ESS difference is positive. Graph-retention loss no more than2 percentage points in observed rate. Isotropic control tests whether a full curvature fit adds value.',
            statistical_scope='FIT-only mechanism pilot. ESS is a finite-sample diagnostic, not proof of target accuracy or neural improvement. Promotion to molecular generation requires a separate student comparison.',
            new_training_steps=0,new_neural_outputs=0,reserved_outcomes_allowed=False)
        out = root/f'research/evidence/curvature_escort_s{seed}_v1.json'
        assert not out.exists()
        out.write_text(json.dumps(spec,indent=2)+'\n')
        print(json.dumps(dict(seed=seed,anchors=32,new_queries=spec['maximum_new_esen_queries'])))


if __name__ == '__main__':
    main()
