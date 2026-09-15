#!/usr/bin/env python3
"""Replay raw structural results and report prespecified, size-stratified contrasts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from cfm_mol.source_checkpoint import prior_from_checkpoint
from scripts.research.tree_prior_fm import sample_source, geometry_counts
from scripts.research.audit_generator_output_support import assess
from scripts.research.audit_source_utility import flags, intervals
from scripts.research.run_matched_generators import batches
from scripts.research.train_electronic_fm import sha


def dictionary_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def totals(rows):
    result = {k:sum(r[k] for r in rows) for k in ['attempted', 'graph_supported',
        'geometrically_supported', 'validator_errors', 'distinct_connectivity', 'generation_seconds']}
    result['graph_rate'] = result['graph_supported']/result['attempted']
    return result


def check_rows(directory, label, report, panel, count, source=None, source_seed=None,
               source_atol=0., source_errors=None):
    assert report['complete'] and len(report['rows']) == len(panel)
    assert [r['condition_index'] for r in report['rows']] == list(range(len(panel)))
    for index, row in enumerate(report['rows']):
        file = directory/f'{label}_c{index}.pt'
        assert sha(file) == row['sample_sha256']
        saved = torch.load(file, weights_only=False, map_location='cpu')
        c = saved['condition']
        assert c == row['condition']
        for key in ['composition_hex', 'atomic_numbers', 'charge', 'spin_multiplicity', 'n_atoms']:
            assert c[key] == panel[index][key]
        assert 'reference_positions' not in c and 'energy_eV' not in c
        x, x0 = saved['positions'], saved['initial_positions']
        assert x.shape == x0.shape == (count, c['n_atoms'], 3)
        assert torch.isfinite(x).all() and torch.isfinite(x0).all()
        assert x.mean(1).abs().max() < 2e-4
        result = assess(x, c, list(range(count)))
        assert all(result[k] == row[k] for k in result)
        assert geometry_counts(x, c['numbers']) == row['final_geometry']
        if source_seed is not None:
            expected = [source_seed*1000003+index*100003+j for j in range(count)]
            assert saved['seeds'] == expected
            for j, seed in enumerate(expected):
                replay, edges = sample_source(source, c['numbers'], c['charge'], c['spin_multiplicity'], seed)
                if source_errors is not None:
                    source_errors.append(float((replay-x0[j]).abs().max()))
                torch.testing.assert_close(replay, x0[j], rtol=0, atol=source_atol)
                assert edges == saved['auxiliary_tree_edges'][j]
        else:
            assert saved['model_state_sha256'] == report['model_state_sha256']
            assert saved['calls_per_sample'] == report['calls_per_sample']
    return report['rows']


def wide(project, run, panel, panel_hash):
    results, artifacts = {}, []
    for seed in [0, 1]:
        file = project/f'research/evidence/wide_generalization_s{seed}_v1.json'
        spec = json.loads(file.read_text())
        assert spec['frozen'] and spec['condition_manifest_sha256'] == panel_hash
        folder = run/f's{seed}/evaluation'
        done = json.loads((folder/'complete.json').read_text())
        assert done['complete'] and done['protocol_sha256'] == sha(file)
        results[seed] = {}
        for method, reference in spec['frozen_references'].items():
            checkpoint = project/reference['checkpoint']
            assert sha(checkpoint) == reference['checkpoint_sha256']
            state = torch.load(checkpoint, weights_only=False, map_location='cpu')
            source = prior_from_checkpoint(state)
            del state
            report_file = folder/f'{method}_results.json'
            report = json.loads(report_file.read_text())
            assert report['protocol_sha256'] == sha(file) and report['checkpoint_sha256'] == reference['checkpoint_sha256']
            results[seed][method] = check_rows(folder, method, report, panel, spec['samples_per_condition'], source, spec['evaluation_seed'])
            artifacts.append(dict(seed=seed, method=method, report_sha256=sha(report_file),
                protocol_sha256=sha(file), checkpoint_sha256=reference['checkpoint_sha256']))
            print(json.dumps(dict(seed=seed, method=method, **totals(report['rows']))), flush=True)
    return results, artifacts, [('harmonic_tree', 'gaussian')], {}


def matched(project, run, panel, panel_hash):
    results, artifacts, initialization, training = {}, [], {}, {}
    for seed in [0, 1]:
        results[seed], initialization[seed], training[seed] = {}, {}, {}
        schedules = []
        for kind in ['gaussian_fm', 'harmonic_fm', 'edm', 'gaga']:
            file = project/f'research/evidence/matched_generators_{kind}_s{seed}_v1.json'
            spec = json.loads(file.read_text())
            assert spec['frozen'] and not spec['pretrained'] and spec['condition_manifest_sha256'] == panel_hash
            folder = run/f's{seed}/{kind}'
            done = json.loads((folder/'complete.json').read_text())
            assert done['complete'] and done['protocol_sha256'] == sha(file)
            init = json.loads((folder/'initialization.json').read_text())
            assert not init['pretrained'] and init['protocol_sha256'] == sha(file)
            assert sha(folder/'batch_indices.npy') == init['batch_schedule_sha256']
            schedule = np.load(folder/'batch_indices.npy')
            assert schedule.shape == (spec['training_steps'], spec['batch_size'])
            if not schedules:
                assert sha(project/spec['data']) == spec['data_sha256']
                data = torch.load(project/spec['data'], weights_only=False, map_location='cpu')
                expected = batches(data['training'], spec['training_steps'], spec['batch_size'], spec['batch_seed'])
                np.testing.assert_array_equal(schedule, expected)
                excluded = {r['composition_hex'] for r in panel}
                assert not excluded & {r['condition']['composition_hex'] for rows in data.values() for r in rows}
                del data, expected
            else:
                np.testing.assert_array_equal(schedule, schedules[0])
            schedules.append(schedule)
            initialization[seed][kind] = init
            record = json.loads((folder/'training.json').read_text())
            assert record['complete'] and record['steps'] == spec['training_steps']
            assert record['examples_seen'] == spec['training_steps']*spec['batch_size']
            assert record['validation_used_for_model_selection'] is False
            assert sha(folder/'last.ckpt') == record['checkpoint_sha256']
            checkpoint = torch.load(folder/'last.ckpt', weights_only=False, map_location='cpu')
            assert checkpoint['global_step'] == spec['training_steps'] and checkpoint['protocol_sha256'] == sha(file)
            assert checkpoint['initial_state_sha256'] == init['state_sha256']
            final_hash = dictionary_hash(checkpoint['ema_state_dict'])
            del checkpoint
            training[seed][kind] = record
            for calls in spec['inference_calls']:
                label = f'{kind}_{calls}'
                report_file = folder/'evaluation'/f'{label}_results.json'
                report = json.loads(report_file.read_text())
                assert report['model_state_sha256'] == final_hash
                assert report['calls_per_sample'] == calls
                results[seed][label] = check_rows(folder/'evaluation', label, report, panel, spec['samples_per_condition'])
                artifacts.append(dict(seed=seed, method=label, report_sha256=sha(report_file),
                    checkpoint_sha256=record['checkpoint_sha256'], protocol_sha256=sha(file)))
                print(json.dumps(dict(seed=seed, method=label, **totals(report['rows']))), flush=True)
        assert len({r['state_sha256'] for r in initialization[seed].values()}) == 1
        assert len({r['parameter_count'] for r in initialization[seed].values()}) == 1
    assert initialization[0]['edm']['state_sha256'] != initialization[1]['edm']['state_sha256']
    comparisons = [('harmonic_fm_128', m) for m in ['gaussian_fm_128', 'edm_128', 'gaga_128']]
    return results, artifacts, comparisons, dict(initialization=initialization, training=training,
        identical_capacity_and_initialization_within_seed=True, independently_initialized_seeds=True,
        identical_minibatch_streams_within_seed=True, no_pretraining=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['project', 'run', 'out']:
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--kind', choices=['wide', 'matched'], required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    torch.set_num_threads(2)
    panel_file = args.project/'research/evidence/wide_generalization_panel_v1.json'
    panel = json.loads(panel_file.read_text())['rows']
    overlap_file = args.project/'research/evidence/wide_generalization_overlap_v1.json'
    overlap = json.loads(overlap_file.read_text())
    assert overlap['complete'] and overlap['panel_sha256'] == sha(panel_file)
    assert all(c['overlapping_compositions'] == 0 for c in overlap['corpora'].values())
    strata = [r['panel_stratum'][0] for r in panel]
    assert len(panel) == 64 and all(strata.count(k) == 16 for k in set(strata))
    results, artifacts, contrasts, extra = (wide if args.kind == 'wide' else matched)(args.project, args.run, panel, sha(panel_file))
    summary = {s:{m:totals(rows) for m,rows in models.items()} for s,models in results.items()}
    by_size = {s:{m:{str(k):totals([r for r,b in zip(rows,strata) if b == k]) for k in sorted(set(strata))}
                     for m,rows in models.items()} for s,models in results.items()}
    comparisons = {}
    rng = np.random.default_rng(41191)
    arrays = {}
    for candidate, baseline in contrasts:
        name = candidate+'__minus__'+baseline
        comparisons[name] = {}
        for metric in ['graph_supported', 'geometrically_supported']:
            d = np.stack([np.stack([flags(a, metric)-flags(b, metric) for a,b in zip(results[s][candidate], results[s][baseline])]) for s in [0,1]])
            comparisons[name][metric] = intervals(d, strata, rng)
            arrays[name+'__'+metric] = d
    archive = args.out.with_suffix('.npz')
    assert not archive.exists()
    np.savez_compressed(archive, **arrays)
    report = dict(complete=True, kind=args.kind, panel_sha256=sha(panel_file), overlap_sha256=sha(overlap_file),
        summary=summary, by_size=by_size, comparisons=comparisons, artifacts=artifacts,
        arrays_sha256=sha(archive), structural_assays_replayed=sum(v['attempted'] for models in summary.values() for v in models.values()),
        new_molecular_oracle_calls=0, intervals_condition_on_fitted_models=True,
        strata='16 compositions in each17–28/29–40/41–52/53–64 atom bin',
        scientific_submission_ready=False, **extra)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(comparisons), flush=True)


if __name__ == '__main__':
    main()
