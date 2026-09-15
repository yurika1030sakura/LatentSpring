#!/usr/bin/env python3
"""Audit self-conditioning and update-direction ablations against cached controls."""
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


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['project', 'out']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    torch.set_num_threads(2)
    root = args.project
    sc_rows, sc_summary, norms, rows, energies, forces = {}, {}, {}, {}, {}, {}
    artifacts, source_errors = [], []
    for seed in [0, 1]:
        sc_rows[seed], sc_summary[seed] = {}, {}
        specfile = root/f'research/evidence/sc_component_ablation_s{seed}_v1.json'
        spec = read(specfile)
        folder = root/f'runs/component_ablations_v1/sc/s{seed}'
        assert read(folder/'complete.json')['protocol_sha256'] == sha(specfile)
        panel = read(root/spec['condition_manifest'])['rows']
        reference = spec['sc_reference']
        file = root/reference['report']
        assert sha(file) == reference['report_sha256']
        reference_state = torch.load(root/reference['checkpoint']['path'], map_location='cpu', weights_only=False)
        assert sha(root/reference['checkpoint']['path']) == reference['checkpoint']['sha256']
        prior = prior_from_checkpoint(reference_state)
        assert reference_state['research_protocol']['geometry_self_conditioning']['deep_supervision']
        report = read(file)
        sc_rows[seed]['sc_3000'] = check_rows(file.parent, 'harmonic_tree', report, panel,
            spec['samples_per_condition'], prior, spec['evaluation_seed'], source_atol=1e-12, source_errors=source_errors)
        for steps in [3000, 6000]:
            stage = folder/f'step{steps}'
            training = read(stage/'training.json')
            assert training['complete'] and training['steps'] == steps and not training['self_conditioning']
            assert training['primitive_denoiser_training_forwards'] == steps
            assert sha(stage/'last.ckpt') == training['checkpoint_sha256']
            state = torch.load(stage/'last.ckpt', map_location='cpu', weights_only=False)
            assert 'geometry_self_conditioning' not in state['research_protocol']
            assert state['research_protocol']['sc_ablation_protocol_sha256'] == sha(specfile)
            assert state['global_step'] == steps
            reportfile = stage/'evaluation/no_sc_results.json'
            report = read(reportfile)
            assert report['protocol_sha256'] == sha(specfile)
            assert report['checkpoint_sha256'] == training['checkpoint_sha256']
            sc_rows[seed][f'no_sc_{steps}'] = check_rows(reportfile.parent, 'no_sc', report, panel,
                spec['samples_per_condition'], prior, spec['evaluation_seed'], source_atol=1e-12, source_errors=source_errors)
            artifacts.append(dict(kind='sc', seed=seed, steps=steps, report_sha256=sha(reportfile)))
        sc_summary[seed] = {k:totals(v) for k,v in sc_rows[seed].items()}
        print(json.dumps(dict(seed=seed, sc=sc_summary[seed])), flush=True)

        specfile = root/f'research/evidence/update_norm_control_s{seed}_v1.json'
        spec = read(specfile)
        folder = root/f'runs/component_ablations_v1/norm/s{seed}'
        assert read(folder/'complete.json')['protocol_sha256'] == sha(specfile)
        norms[seed] = read(folder/'update.json')
        assert norms[seed]['coefficient_uses_training_weights_only']
        assert sha(folder/'last.ckpt') == norms[seed]['checkpoint_sha256']
        states = {}
        for name, ref in spec['checkpoints'].items():
            assert sha(root/ref['path']) == ref['sha256']
            states[name] = torch.load(root/ref['path'], map_location='cpu', weights_only=False)
        cfg = read_config_file(root/spec['config'])
        cfg['mol_fm'].pop('bgfm', None)
        model = model_from_config(cfg)
        prepare_research_backbone(model, states['frozen']['research_protocol'])
        model.load_state_dict(states['frozen']['state_dict'], strict=True)
        names = {n for n,_ in model.named_parameters()}
        merged = torch.load(folder/'last.ckpt', map_location='cpu', weights_only=False)
        base = states['frozen']['state_dict']
        direct = states['physical']['state_dict']
        paired = states['paired']['state_dict']
        norm = lambda a:sum(float((a[n]-base[n]).double().square().sum()) for n in sorted(names))**.5
        alpha = norm(paired)/norm(direct)
        assert abs(alpha-norms[seed]['coefficient']) < 1e-12
        for name, value in base.items():
            expected = value+alpha*(direct[name]-value) if name in names else value
            torch.testing.assert_close(merged['state_dict'][name], expected, atol=0, rtol=0)
        assert abs(norm(merged['state_dict'])/norm(paired)-1) < 1e-5
        del model, states, merged
        rows[seed], energies[seed], forces[seed] = {}, {}, {}
        panel = read(root/spec['condition_manifest'])['rows']
        locations = {
            'frozen': (root/f'runs/thermal_distillation_v1/s{seed}/study', 'frozen'),
            'replay': (root/f'runs/thermal_distillation_v1/s{seed}/study', 'replay'),
            'direct': (root/f'runs/thermal_distillation_v1/s{seed}/study', 'escort'),
            'paired': (root/f'runs/thermal_transfer_v1/s{seed}/study', 'escort_delta'),
            'norm_direct': (folder, 'norm_direct')}
        for name, (directory, label) in locations.items():
            reportfile = directory/'evaluation'/f'{label}_results.json'
            report = read(reportfile)
            rows[seed][name] = check_rows(reportfile.parent, label, report, panel,
                spec['samples_per_condition'], prior, spec['evaluation_seed'], source_atol=1e-12, source_errors=source_errors)
            quality = read(directory/'physical_eval/results.json')
            assert quality['complete']
            metadata = {r['condition_index']:r for r in quality['rows'] if r['method'] == label}
            ee, ff = [], []
            for i, row in enumerate(report['rows']):
                p = directory/'physical_eval'/f'{label}_c{i}.pt'
                assert sha(p) == metadata[i]['artifact_sha256']
                data = torch.load(p, map_location='cpu', weights_only=False)
                assert data['source_sample_sha256'] == row['sample_sha256']
                sample = torch.load(reportfile.parent/f'{label}_c{i}.pt', map_location='cpu', weights_only=False)
                torch.testing.assert_close(data['positions'], sample['positions'], atol=0, rtol=0)
                energy, force = data['raw_energy_eV'], data['raw_force_eV_A']
                n = spec['samples_per_condition']
                assert energy.shape == (2*n,) and force.shape == (2*n, row['condition']['n_atoms'], 3)
                assert torch.isfinite(energy).all() and torch.isfinite(force).all()
                even, even_force = (energy[:n]+energy[n:])/2, (force[:n]-force[n:])/2
                torch.testing.assert_close(data['even_energy_eV'], even, atol=0, rtol=0)
                torch.testing.assert_close(data['even_force_eV_A'], even_force, atol=0, rtol=0)
                ee.append((even/row['condition']['n_atoms']).numpy())
                ff.append(even_force.square().sum(-1).mean(-1).sqrt().numpy())
            energies[seed][name], forces[seed][name] = np.stack(ee), np.stack(ff)
            artifacts.append(dict(kind='update', seed=seed, method=name, report_sha256=sha(reportfile)))
    rng = np.random.default_rng(42191)
    sc_comparisons = {}
    for steps in [3000, 6000]:
        d = np.stack([np.stack([flags(a, 'graph_supported')-flags(b, 'graph_supported')
            for a,b in zip(sc_rows[s]['sc_3000'], sc_rows[s][f'no_sc_{steps}'])]) for s in [0,1]])
        sc_comparisons[f'sc_minus_no_sc_{steps}'] = intervals(d, [0]*24, rng)
    comparisons = {}
    for left, right in [('direct','frozen'),('paired','frozen'),('paired','direct'),
                        ('norm_direct','frozen'),('paired','norm_direct')]:
        masks = np.stack([np.stack([flags(a,'graph_supported').astype(bool)&flags(b,'graph_supported').astype(bool)
            for a,b in zip(rows[s][left], rows[s][right])]) for s in [0,1]])
        result = {}
        for metric, values in [('energy_per_atom_eV', energies), ('force_rms_eV_A', forces)]:
            d = np.stack([values[s][left]-values[s][right] for s in [0,1]])
            result[metric] = dict(all_outputs=intervals(d, [0]*10, rng), jointly_valid=masked_intervals(d, masks),
                by_seed=[masked_intervals(d[s:s+1], masks[s:s+1]) for s in [0,1]])
        d = np.stack([np.stack([flags(a,'graph_supported')-flags(b,'graph_supported')
            for a,b in zip(rows[s][left], rows[s][right])]) for s in [0,1]])
        result['graph_supported'] = intervals(d, [0]*10, rng)
        comparisons[left+' minus '+right] = result
    result = dict(complete=True, self_conditioning=dict(summary=sc_summary, comparisons=sc_comparisons),
        update_direction=dict(summary={s:{k:totals(v) for k,v in rr.items()} for s,rr in rows.items()},
            controls=norms, comparisons=comparisons), artifacts=artifacts,
        new_neural_outputs=4352, new_raw_esen_queries=2560, audit_new_queries=0,
        maximum_source_replay_error_A=max(source_errors), source_replay_atol_A=1e-12,
        source_replay_note='Cross-CPU double-precision matrix multiplication may differ in final bits. Seeds and auxiliary tree edges must still match exactly; raw samples and readout hashes are unchanged.',
        scope='Two explicit component tests on existing evaluation panels; all prescribed controls retained. Intervals condition on fitted models.')
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(sc=sc_comparisons, paired_vs_norm=comparisons['paired minus norm_direct'])), flush=True)


if __name__ == '__main__':
    main()
