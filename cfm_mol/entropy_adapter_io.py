"""Strict loading of the versioned exact-entropy refinement experiment families."""
import hashlib
import json
from pathlib import Path

import torch

from cfm_mol.linear_entropy_adapter import LinearEntropyAdapter
from cfm_mol.species_coupling_adapter import SpeciesCouplingAdapter
from cfm_mol.affine_species_adapter import AffineSpeciesCouplingAdapter


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
    kind = state['kind']
    if kind != report['kind']:
        raise ValueError('Checkpoint and report architecture differ')
    if kind in ['species_convex', 'species_affine']:
        config = state['adapter_configuration']
        if (config != report['adapter_configuration'] or config['kT'] != report['kT_eV']
                or config['charge'] != report['condition']['charge']
                or config['spin_multiplicity'] != report['condition']['spin_multiplicity']):
            raise ValueError('Species-adapter configuration differs')
        model_class = SpeciesCouplingAdapter if kind == 'species_convex' else AffineSpeciesCouplingAdapter
        model = model_class(state['condition']['numbers'], **config)
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
