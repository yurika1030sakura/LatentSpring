#!/usr/bin/env python3
"""Check selected targets, paired weights, raw outputs and force-shift contrasts."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from flowmol.model_utils.load import model_from_config, read_config_file

from cfm_mol.radial_reference import prepare_research_backbone
from cfm_mol.source_checkpoint import prior_from_checkpoint
from scripts.research.audit_expanded_generators import check_rows, totals
from scripts.research.audit_source_utility import flags, intervals
from scripts.research.audit_source_sc_energy import masked_intervals
from scripts.research.train_electronic_fm import sha
from scripts.research.audit_generator_output_support import assess


def read(p):
    return json.loads(p.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['project', 'out']:
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    root = args.project
    torch.set_num_threads(2)
    reports, energies, forces, artifacts, source_errors = {}, {}, {}, [], []
    for seed in [0, 1]:
        specfile = root/f'research/evidence/force_shift_ablation_s{seed}_v1.json'
        spec = read(specfile)
        for key in ['data', 'config', 'warm_checkpoint', 'selection_log', 'replay_checkpoint', 'condition_manifest']:
            assert sha(root/spec[key]) == spec[key+'_sha256']
        folder = root/f'runs/component_ablations_v1/shift/s{seed}'
        done = read(folder/'complete.json')
        assert done['complete'] and done['protocol_sha256'] == sha(specfile) and done['raw_queries'] == 1280
        training = read(folder/'training.json')
        old = [json.loads(line) for line in (root/spec['selection_log']).read_text().splitlines()]
        new = [json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
        assert len(old) == len(new) == 1000
        pools = []
        for ref in spec['teacher_files']:
            assert sha(root/ref['path']) == ref['sha256']
            pool = torch.load(root/ref['path'], weights_only=False, map_location='cpu')
            if len(pool['raw_positions']):
                pools.append(pool)
        valid_targets = 0
        for a,b in zip(old,new):
            assert all(a[k] == b[k] for k in ['step', 'selection', 'composition'])
            sel = a['selection']
            if sel['kind'] == 'generated_fit':
                pool = pools[sel['pool']]
                original_indices = torch.where(pool['record']['eligible'])[0]
                i, j = int(original_indices[sel['row']]), sel['particle']
                x = pool['record']['source'][i,j]
                torch.testing.assert_close(x+pool['record']['shift'][i], pool['proposals'][sel['row'],j], atol=1e-12, rtol=0)
                valid_targets += assess(x[None], pool['condition'], [0])['graph_supported']
        assert valid_targets == training['graph_valid_before_shift'] and training['generated_targets'] == 500
        states = {k:torch.load(root/spec[k], weights_only=False, map_location='cpu') for k in ['warm_checkpoint', 'replay_checkpoint']}
        warm, replay = states['warm_checkpoint'], states['replay_checkpoint']
        student = torch.load(folder/'student.ckpt', weights_only=False, map_location='cpu')
        final = torch.load(folder/'last.ckpt', weights_only=False, map_location='cpu')
        assert sha(folder/'last.ckpt') == training['checkpoint_sha256']
        assert student['ablation_protocol_sha256'] == sha(specfile)
        cfg = read_config_file(root/spec['config'])
        cfg['mol_fm'].pop('bgfm', None)
        model = model_from_config(cfg)
        prepare_research_backbone(model, warm['research_protocol'])
        model.load_state_dict(warm['state_dict'], strict=True)
        names = {n for n,_ in model.named_parameters()}
        for name,value in warm['state_dict'].items():
            expected = value+(student['state_dict'][name]-replay['state_dict'][name]) if name in names else value
            torch.testing.assert_close(final['state_dict'][name], expected, atol=0, rtol=0)
        del model, student, final
        prior = prior_from_checkpoint(warm)
        panel = read(root/spec['condition_manifest'])['rows']
        reports[seed], energies[seed], forces[seed] = {}, {}, {}
        for name,directory,label in [('no_shift',folder,'no_force_shift'),
                ('force',root/f'runs/thermal_transfer_v1/s{seed}/study','escort_delta')]:
            reportfile = directory/'evaluation'/f'{label}_results.json'
            report = read(reportfile)
            reports[seed][name] = check_rows(reportfile.parent,label,report,panel,64,prior,
                spec['evaluation_seed'],source_atol=1e-12,source_errors=source_errors)
            quality = read(directory/'physical_eval/results.json')
            metadata = {r['condition_index']:r for r in quality['rows'] if r['method'] == label}
            ee, ff = [], []
            for i,row in enumerate(report['rows']):
                file = directory/'physical_eval'/f'{label}_c{i}.pt'
                assert sha(file) == metadata[i]['artifact_sha256']
                data = torch.load(file, weights_only=False, map_location='cpu')
                assert data['source_sample_sha256'] == row['sample_sha256']
                original = torch.load(reportfile.parent/f'{label}_c{i}.pt', weights_only=False, map_location='cpu')
                torch.testing.assert_close(data['positions'], original['positions'], atol=0, rtol=0)
                e, f = data['raw_energy_eV'], data['raw_force_eV_A']
                assert e.shape == (128,) and f.shape == (128, row['condition']['n_atoms'], 3)
                assert torch.isfinite(e).all() and torch.isfinite(f).all()
                even, force = (e[:64]+e[64:])/2, (f[:64]-f[64:])/2
                torch.testing.assert_close(data['even_energy_eV'], even, atol=0, rtol=0)
                torch.testing.assert_close(data['even_force_eV_A'], force, atol=0, rtol=0)
                ee.append((even/row['condition']['n_atoms']).numpy())
                ff.append(force.square().sum(-1).mean(-1).sqrt().numpy())
            energies[seed][name], forces[seed][name] = np.stack(ee), np.stack(ff)
            artifacts.append(dict(seed=seed, method=name, report_sha256=sha(reportfile)))
    masks = np.stack([np.stack([flags(a,'graph_supported').astype(bool)&flags(b,'graph_supported').astype(bool)
        for a,b in zip(reports[s]['force'],reports[s]['no_shift'])]) for s in [0,1]])
    rng = np.random.default_rng(42591)
    comparisons = {}
    for metric, values in [('energy_per_atom_eV',energies),('force_rms_eV_A',forces)]:
        d = np.stack([values[s]['force']-values[s]['no_shift'] for s in [0,1]])
        comparisons[metric] = dict(all_outputs=intervals(d,[0]*10,rng),jointly_valid=masked_intervals(d,masks),
            by_seed=[masked_intervals(d[s:s+1],masks[s:s+1]) for s in [0,1]])
    d = np.stack([np.stack([flags(a,'graph_supported')-flags(b,'graph_supported')
        for a,b in zip(reports[s]['force'],reports[s]['no_shift'])]) for s in [0,1]])
    comparisons['graph_supported'] = intervals(d,[0]*10,rng)
    result = dict(complete=True, summary={s:{m:totals(v) for m,v in rr.items()} for s,rr in reports.items()},
        force_minus_no_shift=comparisons, artifacts=artifacts, exact_training_selection_replay=True,
        parameter_transfer_reconstructed=True, source_replay_max_error_A=max(source_errors),
        new_neural_outputs=1280, new_raw_esen_queries=2560, new_teacher_queries=0,
        scope='Only the displacement is removed. Both arms retain original candidate indices, force-dependent widths, selection rule and paired parameter transfer; this is not a completely physics-free control.')
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(summary=result['summary'],comparisons=comparisons)),flush=True)


if __name__ == '__main__':
    main()
