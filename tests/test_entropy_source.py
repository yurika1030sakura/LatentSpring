import copy
import json
from pathlib import Path

import pytest
import torch

from cfm_mol.entropy_source import file_sha, load_entropy_source


@pytest.fixture
def source(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    protocol_path = repo/'research/evidence/species_breadth_source_protocol_v2.json'
    manifest_path = repo/'research/evidence/development_panel_v1.json'
    protocol = json.loads(protocol_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    condition = dict(manifest['rows'][0])
    condition.update(manifest_sha256=file_sha(manifest_path), manifest_index=0,
        manifest_role='new_development', requested_kT_eV=1., numbers=condition['atomic_numbers'])
    directory = tmp_path/'condition_00'
    directory.mkdir()
    fields = {key: protocol[key] for key in ['checkpoint_sha256', 'manifest_sha256', 'oracle_sha256',
        'model_input_kT_eV', 'physical_target_kT_eV', 'restraint_eV_A2', 'source_terminal_noise_std_A']}
    sampler = '64-step midpoint displacement FM at T=1 plus independent COM Gaussian noise'
    report = {**fields, 'complete': True, 'condition': condition, 'reference_geometry_loaded': False,
        'source_sampler': sampler, 'prefix_replay_max_error_A': 0., 'prefix_input_changed': False,
        'prefix_cpu_rng_changed': False, 'prefix_cuda_rng_changed': False,
        'oracle_evaluations': 64, 'streams': {}, 'artifacts': {}}
    for number, stream in enumerate(['training', 'development']):
        seeds = [1000000000*9181+100003*condition['candidate_index']+100000003*number+i for i in range(2)]
        generator = torch.Generator().manual_seed(number)
        x = torch.randn(32, condition['n_atoms'], 3, dtype=torch.float64, generator=generator)
        x -= x.mean(1, keepdim=True)
        data = {'positions': x, 'energy_eV': torch.zeros(32, dtype=torch.float64), 'condition': condition,
            'stream': stream, 'sample_ids': list(range(32)), 'batch_seeds': seeds}
        path = directory/f'{stream}_samples.pt'
        torch.save(data, path)
        report['artifacts'][path.name] = file_sha(path)
        report['streams'][stream] = {'batch_seeds': seeds, 'samples': 32, 'maximum_force_norm_eV_A': 1.}
    path = directory/'results.json'
    path.write_text(json.dumps(report))
    panel = {**fields, 'source_protocol_sha256': file_sha(protocol_path), 'config_sha256': protocol['config_sha256'],
        'deterministic_runtime': True, 'reference_geometry_loaded': False, 'source_sampler': sampler,
        'configuration': {'train_count': 32, 'eval_count': 32, 'batch': 16, 'seed': 9181,
            'replay_diagnostic': False, 'deterministic_runtime': True}, 'complete': False,
        'rows': [{'index': 0, 'success': True, 'results_sha256': file_sha(path)}]}
    (tmp_path/'panel.json').write_text(json.dumps(panel))
    return tmp_path, {'protocol_path': protocol_path, 'manifest_path': manifest_path, 'engineering': True}


def refresh(root, mutate):
    directory = root/'condition_00'
    path = directory/'results.json'
    report = json.loads(path.read_text())
    mutate(directory, report)
    path.write_text(json.dumps(report))
    panel = json.loads((root/'panel.json').read_text())
    panel['rows'][0]['results_sha256'] = file_sha(path)
    (root/'panel.json').write_text(json.dumps(panel))


def test_completed_condition_can_load_while_later_sources_run(source):
    root, args = source
    result = load_entropy_source(root, 0, **args)
    assert result['source_kind'] == 'finite_fm_gaussian'
    assert len(result['training']['positions']) == 32
    assert 'work' not in result['development']
    assert result['source']['physical_target_kT_eV'] != result['source']['model_input_kT_eV']
    with pytest.raises(ValueError, match='production role'):
        load_entropy_source(root, 0, **{**args, 'engineering': False})


@pytest.mark.parametrize('corruption', ['charge', 'seed', 'work', 'nan', 'com'])
def test_repacked_artifacts_cannot_hide_semantic_corruption(source, corruption):
    root, args = source
    def mutate(directory, report):
        path = directory/'training_samples.pt'
        data = torch.load(path, weights_only=False)
        if corruption == 'charge':
            data['condition'] = copy.deepcopy(data['condition'])
            data['condition']['charge'] += 1
        elif corruption == 'seed':
            data['batch_seeds'][0] += 1
        elif corruption == 'work':
            data['work'] = torch.zeros(32)
        elif corruption == 'nan':
            data['energy_eV'][0] = float('nan')
        else:
            data['positions'][0, 0, 0] += 1
        torch.save(data, path)
        report['artifacts'][path.name] = file_sha(path)
    refresh(root, mutate)
    with pytest.raises(ValueError):
        load_entropy_source(root, 0, **args)


@pytest.mark.parametrize('field,value', [('physical_target_kT_eV', 1.), ('reference_geometry_loaded', True),
    ('prefix_replay_max_error_A', 1e-6), ('prefix_cuda_rng_changed', True)])
def test_target_and_replay_contracts_fail_closed(source, field, value):
    root, args = source
    refresh(root, lambda directory, report: report.update({field: value}))
    with pytest.raises(ValueError):
        load_entropy_source(root, 0, **args)
