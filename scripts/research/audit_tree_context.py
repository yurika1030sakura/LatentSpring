#!/usr/bin/env python3
"""Audit actual/sham/no-context transport, including the common source draws."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from flowmol.model_utils.load import read_config_file, model_from_config

from cfm_mol.radial_reference import prepare_research_backbone
from cfm_mol.latent_tree_context import context_tree, tree_features
from scripts.research.tree_prior_fm import load_prior, sample_source, geometry_counts
from scripts.research.audit_generator_output_support import assess
from scripts.research.audit_tree_moment_controls import interval
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


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
        protocol_path = args.project/f'research/evidence/tree_context_s{seed_index}_v1.json'
        spec = json.loads(protocol_path.read_text())
        assert spec['frozen']
        root = args.run/f's{seed_index}'
        selection = json.loads((root/'study/selection.json').read_text())
        progress = json.loads((root/'study/progress.json').read_text())
        assert progress['complete'] and progress['completed'] == spec['methods']
        assert progress['protocol_sha256'] == sha(protocol_path)
        config_path = args.project/spec['config']
        assert sha(config_path) == spec['config_sha256']
        cfg = read_config_file(config_path)
        cfg['mol_fm'].pop('bgfm', None)
        cfg['mol_fm']['prior_config']['x']['align'] = False
        initial_adapter = None
        common_sources = {}
        rows, records, training = [], {}, {}
        for method in ['fixed'] + spec['methods']:
            trained = method in spec['methods']
            mode = spec['context_modes'].get(method)
            directory = root/'study'/method if trained else (args.project/spec['frozen_references']['fixed']['checkpoint']).parent
            checkpoint = directory/'last.ckpt'
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            recipe = saved['research_protocol']
            assert saved['global_step'] == spec['fm_steps'] and recipe['source_prior_kind'] == 'fixed'
            for key in ['data_seed', 'fm_seed', 'fm_steps', 'fm_lr', 'target_domain', 'warm_checkpoint_sha256']:
                assert recipe[key] == spec[key]
            if trained:
                assert recipe['model_variant'] == method and recipe['context_mode'] == mode
                assert recipe['tree_protocol_sha256'] == sha(protocol_path)
                assert recipe['latent_tree_context'] == spec['context_adapter']
                assert recipe['context_seed_offset'] == spec['context_seed_offset']
                validation = json.loads((directory/'validation.json').read_text())
                assert validation['complete'] and validation['checkpoint_sha256'] == sha(checkpoint)
                initial = torch.load(directory/'adapter_initial.pt', map_location='cpu', weights_only=False)
                assert initial['2.weight'].count_nonzero() == 0 and initial['2.bias'].count_nonzero() == 0
                if initial_adapter is None:
                    initial_adapter = initial
                else:
                    assert initial.keys() == initial_adapter.keys()
                    for key in initial:
                        torch.testing.assert_close(initial[key], initial_adapter[key], atol=0, rtol=0)
                assert [group['lr'] for group in saved['optimizer_state_dict']['param_groups']] == [spec['fm_lr'], spec['context_lr']]
            else:
                assert sha(checkpoint) == spec['frozen_references']['fixed']['checkpoint_sha256']
                assert 'latent_tree_context' not in recipe
            # Actual full-sized checkpoint restoration, without optimizer retraining.
            model = model_from_config(cfg)
            prepare_research_backbone(model, recipe)
            model.load_state_dict(saved['state_dict'], strict=True)
            assert all(torch.isfinite(value).all() for value in model.state_dict().values() if value.is_floating_point())
            if trained:
                final = model.vector_field.latent_tree_adapter.state_dict()
                change = sum(float((final[k]-initial[k]).square().sum()) for k in initial)**.5
                assert change > 0
                training[method] = dict(seconds=validation['seconds'], adapter_parameters=sum(v.numel() for v in final.values()),
                    adapter_change_norm=change, initial_adapter_sha256=sha(directory/'adapter_initial.pt'),
                    model_parameters=sum(p.numel() for p in model.parameters()))
            del model, saved
            metrics_path = directory/'metrics.jsonl'
            metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
            actual = [row['processed_index'] for row in metrics]
            assert actual == selection['selected'][:spec['fm_steps']]
            assert len(set(actual)) == len(actual) == spec['fm_steps']
            assert [r['step'] for r in metrics] == list(range(1, spec['fm_steps']+1))
            assert all(np.isfinite(r['fm_loss']) and np.isfinite(r['gradient_norm']) for r in metrics)
            if trained:
                assert all(np.isfinite(r['adapter_gradient_norm']) for r in metrics)
                assert any(r['adapter_gradient_norm'] > 0 for r in metrics)
                for row in metrics:
                    mapping = row['pairing'][0]['source_permutation']
                    assert sorted(mapping) == list(range(row['n_atoms']))
                training[method]['mean_fm_loss'] = float(np.mean([r['fm_loss'] for r in metrics]))
                training[method]['mean_clipped_adapter_gradient_norm'] = float(np.mean([r['adapter_gradient_norm'] for r in metrics]))
            prior = load_prior(method, None, spec, sha(protocol_path))
            evaluation = root/('study/evaluation' if trained else 'references')
            result_path = evaluation/f'{method}_results.json'
            report = json.loads(result_path.read_text())
            assert report['complete'] and report['protocol_sha256'] == sha(protocol_path)
            assert report['checkpoint_sha256'] == sha(checkpoint)
            assert sorted(row['condition_index'] for row in report['rows']) == sorted(spec['conditions'])
            records[method] = {}
            for row in report['rows']:
                index = row['condition_index']
                path = evaluation/f'{method}_c{index}.pt'
                assert sha(path) == row['sample_sha256']
                stored = torch.load(path, map_location='cpu', weights_only=False)
                c = stored['condition']
                assert c == row['condition']
                x, x0 = stored['positions'], stored['initial_positions']
                assert x.shape == x0.shape == (spec['samples_per_condition'], len(c['numbers']), 3)
                assert len(stored['seeds']) == len(stored['auxiliary_tree_edges']) == len(x)
                assert torch.isfinite(x).all() and torch.isfinite(x0).all()
                assert x.mean(1).abs().max() < 1e-8 and x0.mean(1).abs().max() < 1e-8
                if trained:
                    assert stored['context_mode'] == mode and len(stored['context_tree_edges']) == len(x)
                for j, seed in enumerate(stored['seeds']):
                    assert seed == spec['evaluation_seed']*1000003 + index*100003 + j
                    replay, edges = sample_source(prior, c['numbers'], c['charge'], c['spin_multiplicity'], seed)
                    torch.testing.assert_close(replay, x0[j], atol=1e-10, rtol=1e-10)
                    assert edges == stored['auxiliary_tree_edges'][j]
                    if trained:
                        context = context_tree(prior, c['numbers'], c['charge'], c['spin_multiplicity'], edges,
                            seed+spec['context_seed_offset'], mode)
                        assert context == stored['context_tree_edges'][j]
                        assert torch.isfinite(tree_features(len(c['numbers']), context)).all()
                    checked += 1
                if method == 'fixed':
                    common_sources[index] = (x0.clone(), stored['auxiliary_tree_edges'], c)
                else:
                    reference_x0, reference_tree, reference_c = common_sources[index]
                    assert c == reference_c and stored['auxiliary_tree_edges'] == reference_tree
                    torch.testing.assert_close(x0, reference_x0, atol=0, rtol=0)
                for key, value in assess(x, c, list(range(len(x)))).items():
                    assert row[key] == value, (seed_index, method, index, key)
                assert geometry_counts(x, c['numbers']) == row['final_geometry']
                assert geometry_counts(x0, c['numbers']) == row['initial_geometry']
                flags = [geometry_counts(value[None], c['numbers']) for value in x]
                records[method][index] = dict(graph=np.array([int(r['graph_supported']) for r in row['records']]),
                    geometry=np.array([int(r['geometrically_supported']) for r in row['records']]),
                    disconnected=np.array([r['disconnected'] for r in flags]))
                rows.append({key: value for key, value in row.items() if key != 'records'})
            sources[f's{seed_index}/{method}'] = dict(checkpoint_sha256=sha(checkpoint),
                results_sha256=sha(result_path), metrics_sha256=sha(metrics_path))
        assert training['tree_actual']['adapter_parameters'] == training['tree_sham']['adapter_parameters']
        assert training['tree_actual']['model_parameters'] == training['tree_sham']['model_parameters']
        methods = {}
        for method in records:
            chosen = [row for row in rows if row['method'] == method]
            methods[method] = {key: sum(row[key] for row in chosen) for key in
                ['attempted', 'graph_supported', 'geometrically_supported', 'validator_errors', 'generation_seconds']}
            methods[method].update({key: sum(row['final_geometry'][key] for row in chosen) for key in ['disconnected', 'overlap']})
            methods[method]['distinct_connectivities_sum'] = sum(row['distinct_connectivity'] for row in chosen)
        comparisons = {}
        for left, right in [('tree_actual', 'tree_sham'), ('tree_actual', 'fixed'), ('tree_sham', 'fixed')]:
            comparisons[left+' minus '+right] = {metric: interval([
                records[left][index][metric]-records[right][index][metric] for index in spec['conditions']])
                for metric in ['graph', 'geometry', 'disconnected']}
        studies.append(dict(seed_index=seed_index, protocol_sha256=sha(protocol_path), selection_sha256=sha(root/'study/selection.json'),
            methods=methods, training=training, comparisons=comparisons, rows=rows))
    assert checked == 3072
    write(args.out, dict(complete=True, source_and_structural_outputs_replayed=checked, sources=sources, studies=studies,
        auditor_sha256=sha(Path(__file__)), new_molecular_oracle_calls=0, scientific_submission_ready=False,
        limits=['Same eight development compositions, two continuation/data-order seeds and shared warm checkpoint.',
                'Conditional paired intervals with no multiplicity correction; no composition-generalization inference.',
                'Source positions, source/context trees, final structural readouts and full checkpoint restoration replayed.',
                'Training source permutations checked for bijectivity; complete optimizer trajectories and final neural generation are not independently rerun.',
                'No final conditional/marginal density or Boltzmann-distribution claim.']))
    print(json.dumps({f's{s["seed_index"]}': dict(methods=s['methods'], comparisons=s['comparisons']) for s in studies}), flush=True)


if __name__ == '__main__':
    main()
