#!/usr/bin/env python3
"""Replay both moment-control studies and retain every structural outcome."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from cfm_mol.tree_prior_controls import TreePriorControl
from scripts.research.tree_prior_fm import load_prior, sample_source, geometry_counts
from scripts.research.audit_generator_output_support import assess
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def interval(difference):
    difference = np.asarray(difference)
    rng = np.random.default_rng(30391)
    indices = rng.integers(difference.shape[1], size=(10000, *difference.shape))
    bootstrap = np.take_along_axis(difference[None], indices, axis=2).mean((1, 2))
    return dict(mean=float(difference.mean()), conditional_paired_bootstrap95=np.quantile(bootstrap, [.025, .975]).tolist(),
                per_condition_counts=difference.sum(1).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['project', 'run', 'out']:
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(2)
    studies = []
    total = 0
    all_sources = {}
    for seed_index in [0, 1]:
        protocol_path = args.project / f'research/evidence/tree_moment_controls_s{seed_index}_v1.json'
        spec = json.loads(protocol_path.read_text())
        assert spec['frozen']
        root = args.run / f's{seed_index}'
        selection = json.loads((root/'study/selection.json').read_text())
        progress = json.loads((root/'study/progress.json').read_text())
        assert progress['complete'] and progress['completed'] == spec['methods']
        assert progress['protocol_sha256'] == sha(protocol_path)
        calibration_path = args.project/'research/evidence/tree_covariance_calibration_v1.json'
        assert sha(calibration_path) == spec['covariance_calibration_sha256']
        rows = []
        records = {}
        times = {}
        for method in ['gaussian', 'fixed'] + spec['methods']:
            trained = method in spec['methods']
            evaluation = root / ('study/evaluation' if trained else 'references')
            result_path = evaluation / f'{method}_results.json'
            report = json.loads(result_path.read_text())
            assert report['complete'] and report['protocol_sha256'] == sha(protocol_path)
            assert sorted(row['condition_index'] for row in report['rows']) == sorted(spec['conditions'])
            if trained:
                directory = root / 'study' / method
                checkpoint = directory/'last.ckpt'
                validation = json.loads((directory/'validation.json').read_text())
                assert validation['complete'] and validation['global_step'] == spec['fm_steps']
                assert validation['checkpoint_sha256'] == sha(checkpoint)
                times[method] = validation['seconds']
            else:
                reference = spec['frozen_references'][method]
                checkpoint = args.project/reference['checkpoint']
                directory = checkpoint.parent
                assert sha(checkpoint) == reference['checkpoint_sha256']
            assert sha(checkpoint) == report['checkpoint_sha256']
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            recipe = saved['research_protocol']
            assert saved['global_step'] == spec['fm_steps'] and recipe['source_prior_kind'] == method
            for key in ['data_seed', 'fm_seed', 'fm_steps', 'fm_lr', 'target_domain', 'warm_checkpoint_sha256']:
                assert recipe[key] == spec[key]
            if trained:
                assert recipe['tree_protocol_sha256'] == sha(protocol_path)
                prior = TreePriorControl(**saved['source_prior']['configuration']).double()
                prior.load_state_dict(saved['source_prior']['state_dict'], strict=True)
                assert prior.configuration == load_prior(method, None, spec, sha(protocol_path)).configuration
            else:
                prior = load_prior(method, None, spec, sha(protocol_path))
            del saved
            metrics_path = directory/'metrics.jsonl'
            metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
            actual = [r['processed_index'] for r in metrics]
            assert actual == selection['selected'][:spec['fm_steps']]
            assert len(set(actual)) == len(actual) == spec['fm_steps']
            assert [r['step'] for r in metrics] == list(range(1, spec['fm_steps']+1))
            assert all(np.isfinite(r['fm_loss']) and np.isfinite(r['gradient_norm']) for r in metrics)
            records[method] = {}
            for row in report['rows']:
                index = row['condition_index']
                sample_path = evaluation / f'{method}_c{index}.pt'
                assert sha(sample_path) == row['sample_sha256']
                stored = torch.load(sample_path, map_location='cpu', weights_only=False)
                condition = stored['condition']
                assert condition == row['condition']
                x, x0 = stored['positions'], stored['initial_positions']
                assert x.shape == x0.shape == (spec['samples_per_condition'], len(condition['numbers']), 3)
                assert len(stored['seeds']) == len(stored['auxiliary_tree_edges']) == len(x)
                assert torch.isfinite(x).all() and torch.isfinite(x0).all()
                assert x.mean(1).abs().max() < 1e-8 and x0.mean(1).abs().max() < 1e-8
                for j, seed in enumerate(stored['seeds']):
                    assert seed == spec['evaluation_seed']*1000003 + index*100003 + j
                    replay, edges = sample_source(prior, condition['numbers'], condition['charge'], condition['spin_multiplicity'], seed)
                    torch.testing.assert_close(replay, x0[j], atol=1e-10, rtol=1e-10)
                    assert edges == stored['auxiliary_tree_edges'][j]
                    if prior is not None:
                        assert torch.isfinite(prior.log_prob(x0[j], condition['numbers'], condition['charge'], condition['spin_multiplicity']))
                    total += 1
                for key, value in assess(x, condition, list(range(len(x)))).items():
                    assert row[key] == value, (seed_index, method, index, key)
                assert geometry_counts(x, condition['numbers']) == row['final_geometry']
                assert geometry_counts(x0, condition['numbers']) == row['initial_geometry']
                flags = [geometry_counts(value[None], condition['numbers']) for value in x]
                records[method][index] = dict(condition=condition,
                    graph=np.array([int(r['graph_supported']) for r in row['records']]),
                    geometry=np.array([int(r['geometrically_supported']) for r in row['records']]),
                    disconnected=np.array([r['disconnected'] for r in flags]))
                rows.append({key: value for key, value in row.items() if key != 'records'})
            all_sources[f's{seed_index}/{method}'] = dict(results_sha256=sha(result_path),
                checkpoint_sha256=sha(checkpoint), metrics_sha256=sha(metrics_path))
        methods = {}
        for method in records:
            chosen = [r for r in rows if r['method'] == method]
            methods[method] = {key: sum(r[key] for r in chosen) for key in
                ['attempted', 'graph_supported', 'geometrically_supported', 'validator_errors', 'generation_seconds']}
            methods[method].update({key: sum(r['final_geometry'][key] for r in chosen) for key in ['disconnected', 'overlap']})
            methods[method]['distinct_connectivities_sum'] = sum(r['distinct_connectivity'] for r in chosen)
            methods[method]['new_training_seconds'] = times.get(method, 0.)
        comparisons = {}
        for left, right in [('fixed', 'covariance_gaussian'), ('fixed', 'harmonic_tree'),
                            ('fixed', 'gaussian'), ('covariance_gaussian', 'gaussian'), ('harmonic_tree', 'gaussian')]:
            differences = {}
            for metric in ['graph', 'geometry', 'disconnected']:
                arrays = []
                for index in spec['conditions']:
                    l, r = records[left][index], records[right][index]
                    assert l['condition'] == r['condition']
                    arrays.append(l[metric]-r[metric])
                differences[metric] = interval(arrays)
            comparisons[left+' minus '+right] = differences
        studies.append(dict(seed_index=seed_index, protocol_sha256=sha(protocol_path),
            selection_sha256=sha(root/'study/selection.json'), methods=methods, comparisons=comparisons, rows=rows))
    assert total == 4096
    write(args.out, dict(complete=True, source_and_structural_outputs_replayed=total,
        sources=all_sources, studies=studies, new_molecular_oracle_calls=0,
        scientific_submission_ready=False, auditor_sha256=sha(Path(__file__)),
        limits=['Same eight development compositions and warm checkpoint; two continuation/data-order seeds.',
                'All paired intervals condition on fitted models and these compositions; no multiplicity adjustment.',
                'Single Gaussian covariance is a fixed Monte Carlo approximation; harmonic-tree covariance matching is analytic.',
                'Saved source draws and structural readouts replayed; no optimizer or final neural-generation replay.',
                'No final density, energy-distribution or learned-affinity-utility claim.']))
    print(json.dumps({f's{x["seed_index"]}': x['methods'] for x in studies}), flush=True)


if __name__ == '__main__':
    main()
