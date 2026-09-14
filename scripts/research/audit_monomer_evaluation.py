#!/usr/bin/env python3
"""Audit all prospective monomer outputs and matched frozen-source comparisons."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cfm_mol.source_checkpoint import prior_from_checkpoint
from scripts.research.tree_prior_fm import sample_source, geometry_counts
from scripts.research.audit_generator_output_support import assess
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def uncertainty(difference, strata):
    difference = np.asarray(difference)
    rng = np.random.default_rng(31091)
    draw_indices = rng.integers(difference.shape[1], size=(10000, *difference.shape))
    draws = np.take_along_axis(difference[None], draw_indices, axis=2).mean((1, 2))
    groups = [np.flatnonzero(np.array(strata) == label) for label in sorted(set(strata))]
    ids = np.concatenate([group[rng.integers(len(group), size=(10000, len(group)))] for group in groups], 1)
    composition_draws = difference.mean(1)[ids].mean(1)
    return dict(mean=float(difference.mean()), within_composition_paired95=np.quantile(draws, [.025, .975]).tolist(),
        descriptive_size_stratified95=np.quantile(composition_draws, [.025, .975]).tolist(),
        per_condition_counts=difference.sum(1).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['project', 'run', 'out']:
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(2)
    panel_path = args.project/'research/evidence/monomer_panel_v1.json'
    panel = json.loads(panel_path.read_text())
    assert panel['complete'] and panel['role'] == 'new_development'
    pool_path = args.project/'research/evidence/monomer_reference_pool_v1.json'
    assert sha(pool_path) == panel['source_pool_sha256']
    pool = json.loads(pool_path.read_text())
    qualification_path = args.project/'research/evidence/monomer_qualification_v1.json'
    assert sha(qualification_path) == panel['qualification_audit_sha256']
    qualification = json.loads(qualification_path.read_text())
    accepted = {row['candidate_index'] for row in qualification['decisions'] if row['qualified']}
    overlap_path = args.project/'research/evidence/monomer_overlap_v1.json'
    overlap = json.loads(overlap_path.read_text())
    assert overlap['complete'] and overlap['panel_sha256'] == sha(panel_path)
    assert all(row['overlapping_compositions'] == 0 for row in overlap['corpora'].values())
    assert len({row['composition_hex'] for row in panel['rows']}) == len(panel['rows']) == 21
    assert not set(panel['excluded_compositions']).intersection(row['composition_hex'] for row in panel['rows'])
    for reference in panel['rows']:
        assert reference['candidate_index'] in accepted
        assert 'positions' not in reference and 'reference_positions' not in reference and 'energy_eV' not in reference
        original = next(row for row in pool['rows'] if row['condition']['candidate_index'] == reference['candidate_index'])
        c = dict(reference, numbers=reference['atomic_numbers'])
        x = torch.tensor(original['reference_positions'], dtype=torch.float64)[None]
        assert assess(x, c, [0])['graph_supported'] == 1
        assert geometry_counts(x, c['numbers']) == dict(disconnected=0, overlap=0)
    strata = [str(row['panel_stratum']) for row in panel['rows']]
    checked, studies, sources = 0, [], {}
    for seed_index in [0, 1]:
        protocol_path = args.project/f'research/evidence/monomer_evaluation_s{seed_index}_v1.json'
        spec = json.loads(protocol_path.read_text())
        assert spec['frozen'] and spec['condition_manifest_sha256'] == sha(panel_path)
        assert spec['overlap_audit_sha256'] == sha(overlap_path) and spec['qualification_audit_sha256'] == sha(qualification_path)
        evaluation = args.run/f's{seed_index}/evaluation'
        complete = json.loads((evaluation/'complete.json').read_text())
        assert complete['complete'] and complete['methods'] == spec['methods'] and complete['protocol_sha256'] == sha(protocol_path)
        rows, records, training_indices = [], {}, None
        for method in spec['methods']:
            checkpoint = args.project/spec['frozen_references'][method]['checkpoint']
            assert sha(checkpoint) == spec['frozen_references'][method]['checkpoint_sha256']
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            recipe = saved['research_protocol']
            assert recipe['source_prior_kind'] == method and not recipe.get('latent_tree_context')
            for key in ['data_seed', 'fm_seed', 'fm_steps', 'fm_lr', 'target_domain', 'warm_checkpoint_sha256']:
                assert recipe[key] == spec[key]
            prior = prior_from_checkpoint(saved)
            if prior is not None:
                assert prior.configuration == recipe['source_prior_configuration']
            del saved
            metrics_path = checkpoint.parent/'metrics.jsonl'
            metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
            actual = [row['processed_index'] for row in metrics]
            assert len(actual) == len(set(actual)) == spec['fm_steps']
            if training_indices is None:
                training_indices = actual
            else:
                assert actual == training_indices
            path = evaluation/f'{method}_results.json'
            report = json.loads(path.read_text())
            assert report['complete'] and report['protocol_sha256'] == sha(protocol_path)
            assert report['checkpoint_sha256'] == sha(checkpoint)
            assert sorted(row['condition_index'] for row in report['rows']) == list(range(21))
            records[method] = {}
            for row in report['rows']:
                index = row['condition_index']
                sample_path = evaluation/f'{method}_c{index}.pt'
                assert sha(sample_path) == row['sample_sha256']
                stored = torch.load(sample_path, map_location='cpu', weights_only=False)
                c = stored['condition']
                assert c == row['condition'] and c['numbers'] == panel['rows'][index]['atomic_numbers']
                assert c['charge'] == 0 and c['spin_multiplicity'] == 1 and c['requested_kT_eV'] == 1.
                x, x0 = stored['positions'], stored['initial_positions']
                assert x.shape == x0.shape == (64, len(c['numbers']), 3)
                assert torch.isfinite(x).all() and torch.isfinite(x0).all()
                assert x.mean(1).abs().max() < 1e-8 and x0.mean(1).abs().max() < 1e-8
                assert len(stored['seeds']) == len(stored['auxiliary_tree_edges']) == 64
                for j, seed in enumerate(stored['seeds']):
                    assert seed == spec['evaluation_seed']*1000003 + index*100003 + j
                    replay, tree = sample_source(prior, c['numbers'], 0, 1, seed)
                    torch.testing.assert_close(replay, x0[j], atol=1e-10, rtol=1e-10)
                    assert tree == stored['auxiliary_tree_edges'][j]
                    checked += 1
                for key, value in assess(x, c, list(range(len(x)))).items():
                    assert row[key] == value, (seed_index, method, index, key)
                assert geometry_counts(x, c['numbers']) == row['final_geometry']
                assert geometry_counts(x0, c['numbers']) == row['initial_geometry']
                flags = [geometry_counts(value[None], c['numbers']) for value in x]
                records[method][index] = dict(graph=np.array([int(r['graph_supported']) for r in row['records']]),
                    geometry=np.array([int(r['geometrically_supported']) for r in row['records']]),
                    disconnected=np.array([r['disconnected'] for r in flags]))
                rows.append({key: value for key, value in row.items() if key != 'records'})
            sources[f's{seed_index}/{method}'] = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                source_results_sha256=sha(path), training_metrics_sha256=sha(metrics_path))
        methods = {}
        for method in records:
            chosen = [row for row in rows if row['method'] == method]
            methods[method] = {key: sum(row[key] for row in chosen) for key in
                ['attempted', 'graph_supported', 'geometrically_supported', 'validator_errors', 'generation_seconds']}
            methods[method].update({key: sum(row['final_geometry'][key] for row in chosen) for key in ['disconnected', 'overlap']})
            methods[method]['distinct_connectivities_sum'] = sum(row['distinct_connectivity'] for row in chosen)
        comparisons = {}
        pairs = [('fixed', 'gaussian'), ('harmonic_tree', 'gaussian'), ('fixed', 'harmonic_tree')]
        if seed_index == 0:
            pairs += [(m, r) for m in ['node', 'pair'] for r in ['fixed', 'gaussian', 'harmonic_tree']]
        for left, right in pairs:
            comparisons[left+' minus '+right] = {metric: uncertainty([
                records[left][index][metric]-records[right][index][metric] for index in range(21)], strata)
                for metric in ['graph', 'geometry', 'disconnected']}
        studies.append(dict(seed_index=seed_index, protocol_sha256=sha(protocol_path), methods=methods, comparisons=comparisons, rows=rows))
    assert checked == 10752
    write(args.out, dict(complete=True, sources_and_structural_outcomes_replayed=checked, reference_assays_replayed=21,
        panel_sha256=sha(panel_path), overlap_audit_sha256=sha(overlap_path), sources=sources, studies=studies,
        auditor_sha256=sha(Path(__file__)), new_molecular_oracle_calls=0, scientific_submission_ready=False,
        limits=['Prospective neutral singlet organic monomers with reference qualification; not unrestricted OMol25.',
                'No exact composition overlap with the two checksum-verified processed corpora; unrecorded pretraining not certified.',
                'Node/pair learned-source comparisons use only the first continuation, and remain exploratory.',
                'Conditional paired draw and descriptive size-stratified composition intervals; no multiplicity correction.',
                'Source laws and structural readouts replayed; optimizer trajectories and final neural generation not independently replayed.',
                'No new energy evaluation, absolute final density or Boltzmann claim.']))
    print(json.dumps({f's{s["seed_index"]}': dict(methods=s['methods'], comparisons=s['comparisons']) for s in studies}), flush=True)


if __name__ == '__main__':
    main()
