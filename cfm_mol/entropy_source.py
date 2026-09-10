"""Strict source-law and electronic-state contract for broader entropy refinement."""
import hashlib
import json
import math
from pathlib import Path

import torch


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_entropy_source(source_root, index, *, protocol_path, manifest_path, engineering=False):
    """Load one completed condition, even while later source conditions are running.

    The parent panel may be incomplete; the selected condition must have a
    complete immutable record and a matching parent-row checksum. No path work
    or source density is manufactured for the finite-FM-plus-noise sampler.
    """
    root = Path(source_root)
    protocol_path, manifest_path = Path(protocol_path), Path(manifest_path)
    protocol = json.loads(protocol_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    panel = json.loads((root/'panel.json').read_text())
    if not (protocol['frozen'] and protocol['version'] == 2 and protocol['deterministic_runtime']
            and not protocol['reserved_outcomes_allowed'] and manifest['complete']
            and manifest['role'] == 'new_development' and len(manifest['rows']) == 8):
        raise ValueError('Require frozen deterministic eight-condition development source')
    if not isinstance(index, int) or not 0 <= index < 8:
        raise ValueError('Invalid prescribed condition index')
    if (file_sha(manifest_path) != protocol['manifest_sha256']
            or panel['source_protocol_sha256'] != file_sha(protocol_path)):
        raise ValueError('Source protocol or condition manifest changed')
    for key in ['checkpoint_sha256', 'config_sha256', 'manifest_sha256', 'oracle_sha256']:
        if panel[key] != protocol[key]:
            raise ValueError(f'Source header differs: {key}')
    if not panel['deterministic_runtime'] or panel['reference_geometry_loaded']:
        raise ValueError('Source runtime or reference-coordinate contract failed')
    expected_counts = (32, 32, 16, 9181) if engineering else (4096, 512, 64, 9182)
    recipe = panel['configuration']
    if tuple(recipe[k] for k in ['train_count', 'eval_count', 'batch', 'seed']) != expected_counts:
        raise ValueError('Source stream counts or seed differ from prescribed production role')
    if recipe['replay_diagnostic'] or not recipe['deterministic_runtime']:
        raise ValueError('A diagnostic prefix is not a qualified source')
    directory = root/f'condition_{index:02d}'
    path = directory/'results.json'
    source = json.loads(path.read_text())
    rows = [row for row in panel['rows'] if row['index'] == index]
    if len(rows) != 1 or not rows[0]['success'] or not source['complete'] or rows[0]['results_sha256'] != file_sha(path):
        raise ValueError('Condition source is incomplete or changed')
    condition = dict(manifest['rows'][index])
    condition.update(manifest_sha256=protocol['manifest_sha256'], manifest_index=index,
        manifest_role='new_development', requested_kT_eV=protocol['model_input_kT_eV'], numbers=condition['atomic_numbers'])
    if source['condition'] != condition or source['reference_geometry_loaded']:
        raise ValueError('Condition metadata or reference-coordinate contract differs')
    for key in ['checkpoint_sha256', 'manifest_sha256', 'oracle_sha256', 'model_input_kT_eV',
            'physical_target_kT_eV', 'restraint_eV_A2', 'source_terminal_noise_std_A']:
        if source[key] != protocol[key] or panel[key] != protocol[key]:
            raise ValueError(f'Source law or physical target differs: {key}')
    sampler = '64-step midpoint displacement FM at T=1 plus independent COM Gaussian noise'
    if source['source_sampler'] != sampler or panel['source_sampler'] != sampler:
        raise ValueError('Unknown finite source sampler')
    if source['prefix_replay_max_error_A'] != 0 or any(source[k] for k in
            ['prefix_input_changed', 'prefix_cpu_rng_changed', 'prefix_cuda_rng_changed']):
        raise ValueError('Source prefix replay is not qualified')
    loaded = {}
    all_seeds = []
    for stream_index, (label, count) in enumerate(zip(['training', 'development'], expected_counts[:2])):
        sample_path = directory/f'{label}_samples.pt'
        if file_sha(sample_path) != source['artifacts'][sample_path.name]:
            raise ValueError('Source sample artifact changed')
        data = torch.load(sample_path, map_location='cpu', weights_only=False)
        if data['condition'] != condition or data['stream'] != label or data['sample_ids'] != list(range(count)):
            raise ValueError('Sample identities, stream or electronic condition differ')
        expected_seeds = [1000000000*expected_counts[3]+100003*condition['candidate_index']
            +100000003*stream_index+i for i in range(count//expected_counts[2])]
        detail = source['streams'][label]
        if data['batch_seeds'] != expected_seeds or detail['batch_seeds'] != expected_seeds or detail['samples'] != count:
            raise ValueError('Sample stream random-seed provenance differs')
        all_seeds.extend(expected_seeds)
        x, energy = data['positions'], data['energy_eV']
        if (x.shape != (count, condition['n_atoms'], 3) or energy.shape != (count,)
                or not torch.isfinite(x).all() or not torch.isfinite(energy).all()
                or float(x.mean(1).abs().max()) > 1e-8):
            raise ValueError('Invalid source coordinates, energies or COM')
        if not math.isfinite(detail['maximum_force_norm_eV_A']):
            raise ValueError('Physical oracle force qualification failed')
        if any(key in data for key in ['work', 'log_q', 'importance_weights']):
            raise ValueError('No path weight or source density is qualified for this source')
        loaded[label] = data
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError('Training and development streams overlap')
    if source['oracle_evaluations'] != sum(expected_counts[:2]):
        raise ValueError('Source physical-query accounting differs')
    return {**loaded, 'condition': condition, 'recipe': recipe, 'source': source,
        'source_kind': 'finite_fm_gaussian', 'source_protocol_sha256': file_sha(protocol_path),
        'source_results_sha256': file_sha(path), 'engineering_only': engineering,
        'source_checkpoint_sha256': protocol['checkpoint_sha256'],
        'training_sha256': source['artifacts']['training_samples.pt'],
        'evaluation_sha256': source['artifacts']['development_samples.pt']}
