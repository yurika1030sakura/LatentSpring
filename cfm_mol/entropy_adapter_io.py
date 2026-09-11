"""Strict loading of the versioned exact-entropy refinement experiment families."""
import hashlib
import json
from pathlib import Path

import torch

from cfm_mol.linear_entropy_adapter import LinearEntropyAdapter
from cfm_mol.species_coupling_adapter import SpeciesCouplingAdapter
from cfm_mol.affine_species_adapter import AffineSpeciesCouplingAdapter


def build_species_adapter(numbers, config, *, affine=False):
    """Decode constructor options separately from validated layout metadata.

    Legacy whole-group checkpoints have no layout metadata. The first split
    implementation recorded blocks but omitted the opt-in flag, so infer that
    specific historical format from its stored blocks, never from atom count.
    """
    options = {'charge', 'spin_multiplicity', 'kT', 'sweeps', 'hidden', 'radial', 'split_groups'}
    derived = {'minimum_active', 'internal_blocks', 'permutation_equivariant'}
    if set(config)-options-derived:
        raise ValueError('Unknown species-adapter configuration fields')
    kwargs = {key: value for key, value in config.items() if key in options}
    if 'split_groups' not in kwargs and 'internal_blocks' in config:
        kwargs['split_groups'] = any(block is not None for block in config['internal_blocks'])
    model_class = AffineSpeciesCouplingAdapter if affine else SpeciesCouplingAdapter
    model = model_class(numbers, **kwargs)
    for key in derived & config.keys():
        if model.configuration[key] != config[key]:
            raise ValueError(f'Species-adapter derived layout differs: {key}')
    return model


def load_entropy_adapter(directory, device='cpu'):
    directory = Path(directory)
    report = json.loads((directory/'results.json').read_text())
    checkpoint = directory/'adapter.ckpt'
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if not report['complete'] or report['artifacts']['adapter.ckpt'] != digest:
        raise ValueError('Adapter checkpoint lacks complete matching provenance')
    state = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
    if state['source_checkpoint_sha256'] != report['source_checkpoint_sha256'] or state['condition'] != report['condition']:
        raise ValueError('Checkpoint condition or base generator differs from report')
    if 'source_kind' in state or 'source_kind' in report:
        if (state.get('source_kind') not in ['finite_fm_gaussian', 'finite_fm_gaussian_inversion_mixture']
                or state.get('source_kind') != report.get('source_kind')
                or state.get('source_protocol_sha256') != report.get('source_protocol_sha256')
                or not state.get('source_protocol_sha256')):
            raise ValueError('Finite source-law checkpoint provenance differs')
        if state['source_kind'] == 'finite_fm_gaussian_inversion_mixture':
            if (state.get('target_kind') != 'inversion_energy_average' or state['target_kind'] != report.get('target_kind')
                    or not state.get('refinement_protocol_sha256')
                    or state['refinement_protocol_sha256'] != report.get('refinement_protocol_sha256')):
                raise ValueError('Inversion source/target checkpoint provenance differs')
    kind = state['kind']
    if kind != report['kind']:
        raise ValueError('Checkpoint and report architecture differ')
    if kind in ['species_convex', 'species_affine']:
        config = state['adapter_configuration']
        if (config != report['adapter_configuration'] or config['kT'] != report['kT_eV']
                or config['charge'] != report['condition']['charge']
                or config['spin_multiplicity'] != report['condition']['spin_multiplicity']):
            raise ValueError('Species-adapter configuration differs')
        model = build_species_adapter(state['condition']['numbers'], config, affine=kind == 'species_affine')
    elif kind in ['typed', 'scalar']:
        model = LinearEntropyAdapter(state['condition']['numbers'], kind=kind, maximum_weight=report['maximum_pair_weight'])
    else:
        raise ValueError('Unknown adapter architecture; do not reinterpret checkpoints')
    model = model.to(device).double().eval()
    expected_numbers = model.numbers.clone()
    expected_electronic = model.electronic.clone() if hasattr(model, 'electronic') else None
    model.load_state_dict(state['state_dict'], strict=True)
    if not torch.equal(model.numbers, expected_numbers):raise ValueError('Checkpoint atom labels differ from metadata')
    if expected_electronic is not None and not torch.equal(model.electronic, expected_electronic):
        raise ValueError('Checkpoint electronic conditioning differs from metadata')
    for parameter in model.parameters():parameter.requires_grad_(False)
    return model, report, digest
