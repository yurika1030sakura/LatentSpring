#!/usr/bin/env python3
"""Audit all frozen source-model outputs on the prospectively selected panel."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.research.tree_prior_fm import load_prior, sample_source, geometry_counts
from scripts.research.audit_generator_output_support import assess
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def uncertainty(difference, strata):
    difference = np.asarray(difference)
    generator = np.random.default_rng(30891)
    indices = generator.integers(difference.shape[1], size=(10000, *difference.shape))
    bootstrap = np.take_along_axis(difference[None], indices, axis=2).mean((1, 2))
    groups = []
    for stratum in sorted(set(strata)):
        group = np.flatnonzero(np.array(strata) == stratum)
        assert len(group) == 4
        groups.append(group)
    composition_indices = np.concatenate([group[generator.integers(len(group), size=(10000, len(group)))] for group in groups], axis=1)
    means = difference.mean(1)
    composition_bootstrap = means[composition_indices].mean(1)
    return dict(mean=float(difference.mean()), within_composition_paired95=np.quantile(bootstrap, [.025, .975]).tolist(),
        descriptive_stratified_composition95=np.quantile(composition_bootstrap, [.025, .975]).tolist(),
        per_condition_counts=difference.sum(1).tolist(), positive_conditions=int((means > 0).sum()),
        zero_conditions=int((means == 0).sum()), negative_conditions=int((means < 0).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['project', 'run', 'out']:
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(2)
    studies, sources = [], {}
    checked = 0
    for seed_index in [0, 1]:
        protocol_path = args.project/f'research/evidence/tree_source_transfer_s{seed_index}_v1.json'
        spec = json.loads(protocol_path.read_text())
        manifest_path = args.project/spec['condition_manifest']
        assert sha(manifest_path) == spec['condition_manifest_sha256']
        panel = json.loads(manifest_path.read_text())
        assert panel['complete'] and panel['role'] == 'new_development' and not panel['reserved_outcomes_allowed']
        assert len(panel['rows']) == 32 and len({r['composition_hex'] for r in panel['rows']}) == 32
        assert not set(panel['excluded_compositions']).intersection(row['composition_hex'] for row in panel['rows'])
        for filename, digest in panel['excluded_panels'].items():
            assert sha(args.project/'research/evidence'/filename) == digest
        strata = [str(r['panel_stratum']) for r in panel['rows']]
        evaluation = args.run/f's{seed_index}/evaluation'
        completion = json.loads((evaluation/'complete.json').read_text())
        assert completion['complete'] and completion['protocol_sha256'] == sha(protocol_path)
        assert completion['methods'] == spec['methods']
        rows, records = [], {}
        for method in spec['methods']:
            checkpoint = args.project/spec['frozen_references'][method]['checkpoint']
            assert sha(checkpoint) == spec['frozen_references'][method]['checkpoint_sha256']
            recipe = json.loads((checkpoint.parent/'protocol.json').read_text())
            assert recipe['source_prior_kind'] == method and 'latent_tree_context' not in recipe
            for key in ['data_seed', 'fm_seed', 'fm_steps', 'fm_lr', 'target_domain', 'warm_checkpoint_sha256']:
                assert recipe[key] == spec[key]
            prior = load_prior(method, None, spec, sha(protocol_path))
            path = evaluation/f'{method}_results.json'
            report = json.loads(path.read_text())
            assert report['complete'] and report['protocol_sha256'] == sha(protocol_path)
            assert report['checkpoint_sha256'] == sha(checkpoint)
            assert sorted(row['condition_index'] for row in report['rows']) == list(range(32))
            records[method] = {}
            for row in report['rows']:
                index = row['condition_index']
                sample_path = evaluation/f'{method}_c{index}.pt'
                assert sha(sample_path) == row['sample_sha256']
                stored = torch.load(sample_path, map_location='cpu', weights_only=False)
                condition, original = stored['condition'], panel['rows'][index]
                assert row['condition'] == condition
                assert condition['numbers'] == original['atomic_numbers']
                assert condition['charge'] == original['charge'] and condition['spin_multiplicity'] == original['spin_multiplicity']
                assert condition['requested_kT_eV'] == 1.
                x, x0 = stored['positions'], stored['initial_positions']
                assert x.shape == x0.shape == (32, len(condition['numbers']), 3)
                assert torch.isfinite(x).all() and torch.isfinite(x0).all()
                assert x.mean(1).abs().max() < 1e-8 and x0.mean(1).abs().max() < 1e-8
                assert len(stored['seeds']) == len(stored['auxiliary_tree_edges']) == 32
                for j, seed in enumerate(stored['seeds']):
                    assert seed == spec['evaluation_seed']*1000003 + index*100003 + j
                    replay, edges = sample_source(prior, condition['numbers'], condition['charge'], condition['spin_multiplicity'], seed)
                    torch.testing.assert_close(replay, x0[j], atol=1e-10, rtol=1e-10)
                    assert edges == stored['auxiliary_tree_edges'][j]
                    checked += 1
                for key, value in assess(x, condition, list(range(len(x)))).items():
                    assert row[key] == value, (seed_index, method, index, key)
                assert geometry_counts(x, condition['numbers']) == row['final_geometry']
                assert geometry_counts(x0, condition['numbers']) == row['initial_geometry']
                flags = [geometry_counts(value[None], condition['numbers']) for value in x]
                records[method][index] = dict(graph=np.array([int(r['graph_supported']) for r in row['records']]),
                    geometry=np.array([int(r['geometrically_supported']) for r in row['records']]),
                    disconnected=np.array([r['disconnected'] for r in flags]))
                rows.append({key: value for key, value in row.items() if key != 'records'})
            sources[f's{seed_index}/{method}'] = dict(checkpoint_sha256=sha(checkpoint), results_sha256=sha(path))
        methods = {}
        for method in records:
            chosen = [row for row in rows if row['method'] == method]
            methods[method] = {key: sum(row[key] for row in chosen) for key in
                ['attempted', 'graph_supported', 'geometrically_supported', 'validator_errors', 'generation_seconds']}
            methods[method].update({key: sum(row['final_geometry'][key] for row in chosen) for key in ['disconnected', 'overlap']})
            methods[method]['distinct_connectivities_sum'] = sum(row['distinct_connectivity'] for row in chosen)
        comparisons = {}
        for left, right in [('fixed', 'gaussian'), ('harmonic_tree', 'gaussian'), ('fixed', 'harmonic_tree')]:
            comparisons[left+' minus '+right] = {metric: uncertainty([
                records[left][index][metric]-records[right][index][metric] for index in range(32)], strata)
                for metric in ['graph', 'geometry', 'disconnected']}
        studies.append(dict(seed_index=seed_index, protocol_sha256=sha(protocol_path), methods=methods, comparisons=comparisons, rows=rows))
    assert checked == 6144
    write(args.out, dict(complete=True, source_and_structural_outputs_replayed=checked, sources=sources, studies=studies,
        panel_sha256=sha(manifest_path), auditor_sha256=sha(Path(__file__)), new_molecular_oracle_calls=0, scientific_submission_ready=False,
        limits=['Additional32 development compositions; full composition overlap with warm pretraining is not audited.',
                'Two frozen continuation checkpoints per method; no new training or model selection by condition/seed.',
                'Within-composition intervals condition on fitted models and the selected panel.',
                'Stratified composition bootstrap is descriptive heterogeneity analysis, not a population or pretraining-unseen confidence certificate.',
                'No multiplicity correction, independent neural-generation replay, energy evaluation or Boltzmann claim.']))
    print(json.dumps({f's{s["seed_index"]}': dict(methods=s['methods'], comparisons=s['comparisons']) for s in studies}), flush=True)


if __name__ == '__main__':
    main()
