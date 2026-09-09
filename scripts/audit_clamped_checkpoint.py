#!/usr/bin/env python3
"""Smoke-check corrected density on one archived checkpoint, fixed geometries.

This checks execution and numerical sensitivity, not an arm effect. No energy
oracle is called and no checkpoint is modified. The output is q_T of the
memoryless clamped ODE, NOT the sampling distribution of the joint generator.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import dgl
import torch

from cfm_mol.clamped_density import log_density_clamped_flow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--steps', type=int, nargs='+', default=[8, 16, 32])
    parser.add_argument('--terminal-time', type=float, default=0.95)
    parser.add_argument('--max-atoms', type=int, default=12)
    parser.add_argument('--n-hutchinson', type=int, default=2)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--gradient-steps', type=int, default=0,
                        help='Also check parameter gradients of a centered log-q diagnostic; 0 disables')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    from flowmol.model_utils.load import read_config_file, model_from_config
    from flowmol.data_processing.dataset import MoleculeDataset
    from flowmol.data_processing.utils import get_batch_idxs, get_upper_edge_mask
    cfg = read_config_file(args.config)
    cfg['mol_fm'].pop('bgfm', None)
    model = model_from_config(cfg)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)['state_dict']
    if any(not torch.isfinite(v).all() for v in state.values() if v.is_floating_point()):
        raise ValueError('Non-finite checkpoint')
    model.load_state_dict(state, strict=True)
    model.to(args.device).eval()
    ds_cfg = dict(cfg['dataset'], fake_atom_p=0.0, fake_atom_std=1.0,
                  explicit_aromaticity=cfg['mol_fm'].get('explicit_aromaticity', False))
    dataset = MoleculeDataset('val', ds_cfg, prior_config=cfg['mol_fm']['prior_config'])
    counts = dataset.node_idx_array[:, 1]-dataset.node_idx_array[:, 0]
    candidates = torch.where((counts >= 3) & (counts <= args.max_atoms))[0]
    if not candidates.numel():
        raise ValueError('No suitable validation molecule')
    parent_index = int(candidates[0])
    base = dataset[parent_index]
    # Geometry and probe RNG are separate. Reusing the same probe at all times
    # permits consistent quadrature comparison without changing geometries.
    gen = torch.Generator().manual_seed(31415)
    graphs = []
    for k in range(3):
        g = base.clone()
        x = base.ndata['x_1_true'].clone()
        if k:
            x += 0.05*torch.randn(x.shape, generator=gen)
        g.ndata['x_1_true'] = x-x.mean(0)
        graphs.append(g)
    graph = dgl.batch(graphs).to(args.device)
    nbi, _ = get_batch_idxs(graph)
    uem = get_upper_edge_mask(graph)
    probe_gen = torch.Generator().manual_seed(27182)
    probes = [(torch.randint(0, 2, graph.ndata['x_1_true'].shape, generator=probe_gen)*2-1)
              for _ in range(args.n_hutchinson)]
    def xi_fn(step, k, x):
        return probes[k].to(x)
    results = []
    for steps in args.steps:
        start = time.monotonic()
        logp = log_density_clamped_flow(model, graph, nbi, uem, n_ode_steps=steps,
            n_hutchinson=args.n_hutchinson, terminal_time=args.terminal_time, xi_fn=xi_fn)
        row = {'steps': steps, 'log_q_T': logp.cpu().tolist(),
               'centered_log_q_T': (logp-logp.mean()).cpu().tolist(),
               'seconds': time.monotonic()-start}
        print(json.dumps(row), flush=True)
        results.append(row)
    gradient_check = None
    if args.gradient_steps:
        model.zero_grad(set_to_none=True)
        start = time.monotonic()
        logp = log_density_clamped_flow(model, graph, nbi, uem,
            n_ode_steps=args.gradient_steps, n_hutchinson=args.n_hutchinson,
            terminal_time=args.terminal_time, xi_fn=xi_fn, for_training=True)
        # This checks the likelihood gradient path, not a molecular energy loss
        # or an optimizer update. No checkpoint or dataset is modified.
        objective = (logp-logp.mean()).square().mean()
        objective.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads):
            raise FloatingPointError('Missing or non-finite checkpoint gradients')
        norm = sum(g.double().square().sum().item() for g in grads)**0.5
        if norm == 0:
            raise RuntimeError('Unexpected zero gradient in checkpoint diagnostic')
        gradient_check = {'steps': args.gradient_steps, 'diagnostic': 'variance of centered log q; no optimizer update',
            'objective': objective.item(), 'parameter_tensors_with_grad': len(grads),
            'gradient_l2_norm': norm, 'all_gradients_finite': True,
            'seconds': time.monotonic()-start}
        print(json.dumps(gradient_check), flush=True)
    checkpoint_hash = hashlib.sha256()
    with args.checkpoint.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            checkpoint_hash.update(chunk)
    report = {'definition': 'memoryless COM-free clamped positional ODE; endpoint head converted with schedule; full-state midpoint',
        'claim': 'execution/numerical diagnostic only, no molecular performance conclusion',
        'terminal_time': args.terminal_time, 'checkpoint': str(args.checkpoint),
        'config': str(args.config), 'parent_index': parent_index, 'n_atoms': int(counts[parent_index]),
        'n_hutchinson': args.n_hutchinson, 'torch': torch.__version__,
        'checkpoint_sha256': checkpoint_hash.hexdigest(),
        'gradient_check': gradient_check,
        'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
        'geometry_sha256': hashlib.sha256(graph.ndata['x_1_true'].cpu().numpy().tobytes()).hexdigest(),
        'results': results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
