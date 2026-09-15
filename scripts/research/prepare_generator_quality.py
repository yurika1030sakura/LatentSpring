#!/usr/bin/env python3
"""Freeze physical readout of the existing matched-generator outputs."""
import datetime
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[2]
    panel_path = root/'research/evidence/wide_generalization_panel_v1.json'
    panel = json.loads(panel_path.read_text())
    original = json.loads((root/'research/evidence/fresh_physics_s0_v1.json').read_text())
    models = ['gaussian_fm', 'harmonic_fm', 'edm', 'gaga']
    for seed in [0, 1]:
        outputs = []
        for method in models:
            folder = root/f'runs/matched_generators_v1/training/s{seed}/{method}'
            report_path = folder/f'evaluation/{method}_128_results.json'
            report = json.loads(report_path.read_text())
            assert report['complete'] and report['calls_per_sample'] == 128
            for i, row in enumerate(report['rows']):
                sample_path = folder/f'evaluation/{method}_128_c{i}.pt'
                assert row['condition_index'] == i and sha(sample_path) == row['sample_sha256']
                assert row['condition']['composition_hex'] == panel['rows'][i]['composition_hex']
                outputs.append(dict(method=method, condition_index=i, sample=str(sample_path.relative_to(root)),
                    sample_sha256=sha(sample_path), report=str(report_path.relative_to(root)),
                    report_sha256=sha(report_path), graph_valid=[r['graph_supported'] for r in row['records']]))
        spec = {k:original[k] for k in ['oracle_checkpoint', 'oracle_interpreter', 'oracle_sha256',
            'xtb_binary', 'xtb_binary_sha256', 'xtb', 'xtb_energy_inversion_tolerance_eV',
            'xtb_force_inversion_tolerance_eV_A']}
        worker = root/'scripts/research/oracle_worker.py'
        spec.update(frozen=True, format='generator_quality_v1', model_seed=seed,
            at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), methods=models,
            condition_manifest=str(panel_path.relative_to(root)), condition_manifest_sha256=sha(panel_path),
            source_pool=panel['source_pool'], source_pool_sha256=panel['source_pool_sha256'],
            oracle_worker=str(worker.relative_to(root)), oracle_worker_sha256=sha(worker),
            samples_per_condition=32, outputs=outputs, chunks=4,
            raw_esen_queries=2*(64+4*64*32),
            xtb_attempts=2*64+sum(sum(r['graph_valid']) for r in outputs),
            energy_thresholds_eV_atom=[0., .025, .05, .1, .2, .5, 1., 2.],
            force_thresholds_eV_A=[1., 2., 5., 10., 20., 50., 100.],
            primary='Physical-quality yield curves at fixed128 network evaluations: fraction of ALL attempts passing graph validity and each predeclared physical threshold. Compare harmonic FM with Gaussian FM, EDM and GAGA at identical training budgets.',
            energy_reference='Single OMol25 geometry of the same composition. Excess energy is relative to this geometry, not an isomer-specific strain or free energy.',
            secondary='Per-composition conditional valid-output energy and force; every method own valid subset, with missing cells and counts explicit. All-output eSEN scores also retained. No arbitrary cross-objective draw pairing.',
            xtb_selection='Every graph-valid raw output, independent of its energy, plus both reference parities. Graph-invalid outputs remain failures in all yield denominators. GFN2 failures also fail joint yields.',
            inference='No optimization, reranking, rejection or extra network generation. Evaluation-only oracle calls; these never modify outputs.',
            statistical_scope='Two existing independent initialization seeds,64 previously evaluated but unfitted compositions. Freeze before any new physical readout. Report size strata and composition-bootstrap intervals; not new training replication.',
            new_training_steps=0, new_neural_outputs=0, reserved_outcomes_allowed=False)
        target = root/f'research/evidence/generator_quality_s{seed}_v1.json'
        assert not target.exists()
        target.write_text(json.dumps(spec, indent=2)+'\n')
        print(json.dumps(dict(seed=seed, outputs=len(outputs)*32,
            esen_queries=spec['raw_esen_queries'], xtb_attempts=spec['xtb_attempts'])))


if __name__ == '__main__':
    main()
