#!/usr/bin/env python3
"""Resolution and trace-noise panel on fixed, archived molecular geometries.

This is a numerical development panel, not a held-out performance endpoint.
All resolutions share geometry and probes; trace replicas are independent,
while each replica's probe is fixed across time for consistent quadrature.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from flowmol.model_utils.load import read_config_file, model_from_config
from cfm_mol.clamped_density import log_density_clamped_flow
from cfm_mol.perturbation_loader import PerturbationLoader


def write_json(path, data):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--shard', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--parents', type=int, default=8)
    p.add_argument('--max-atoms', type=int, default=12)
    p.add_argument('--steps', nargs='+', type=int, default=[16, 32, 64, 128])
    p.add_argument('--replicas', type=int, default=8)
    p.add_argument('--exact-parent-count', type=int, default=1)
    p.add_argument('--terminal-time', type=float, default=0.95)
    p.add_argument('--device', default='cuda')
    p.add_argument('--composition-split', type=Path)
    p.add_argument('--perturbation-indices', type=int, nargs='+')
    p.add_argument('--geometry-softening',type=float)
    args = p.parse_args()
    if args.replicas < 2:
        raise ValueError('At least two replicas needed for noise diagnostics')
    cfg = read_config_file(args.config)
    cfg['mol_fm'].pop('bgfm', None)
    model = model_from_config(cfg)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    protocol=checkpoint.get('research_protocol',{})
    parameterization=protocol.get('position_parameterization','endpoint')
    if protocol and protocol['data_endpoint_time']!=args.terminal_time:
        raise ValueError('Evaluation T differs from position training endpoint')
    from cfm_mol.smooth_geometry import patch_smooth_geometry
    softening=checkpoint.get('research_protocol',{}).get('geometry_softening',0.) if args.geometry_softening is None else args.geometry_softening
    patch_smooth_geometry(model,softening)
    model.to(args.device).eval()
    loader = PerturbationLoader([args.shard], n_atom_types=model.n_atom_types,
        n_charge_classes=6, n_bond_types=4, b_parents=1, device=args.device,
        max_atoms_per_parent=args.max_atoms, seed=9002,
        perturbation_indices=args.perturbation_indices)
    if args.composition_split:
        split=json.loads(args.composition_split.read_text())
        held={item['parent_id'] for item in split['perturbation_parents']}
        loader._eligible_parents=[i for i in loader._eligible_parents if i in held]
        if len(loader._eligible_parents)<args.parents:
            raise ValueError('Insufficient composition-disjoint perturbation parents')
        loader._order=loader._fresh_order();loader._ptr=0
    args.out.mkdir(parents=True, exist_ok=True)
    report = {'claim':'numerical development panel; no held-out or sampling claim',
        'checkpoint':str(args.checkpoint), 'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),
        'shard':str(args.shard), 'shard_sha256':hashlib.sha256(args.shard.read_bytes()).hexdigest(),
        'terminal_time':args.terminal_time, 'replicas':args.replicas,
        'perturbation_indices':args.perturbation_indices,
        'geometry_softening':softening,
        'position_parameterization':parameterization,
        'probe_policy':'independent Rademacher replicas, fixed over time and across resolutions',
        'rows':[], 'complete':False}
    if args.composition_split:
        report['composition_split_sha256']=hashlib.sha256(args.composition_split.read_bytes()).hexdigest()
        report['composition_disjoint_development']=True
    for i in range(args.parents):
        graph, energy, pid, nbi, uem = loader.next_batch()
        x = graph.ndata['x_1_true']
        gen = torch.Generator().manual_seed(9100+i)
        probes = [(2*torch.randint(0,2,x.shape,generator=gen)-1).to(x) for _ in range(args.replicas)]
        row = {'parent_id':int(loader._last_parent_indices[0]),
            'n_atoms':int(graph.batch_num_nodes()[0]),
            'geometry_sha256':hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest(),
            'energies_eV':[float(e) if torch.isfinite(e) else None for e in energy],
            'resolutions':[]}
        previous = None
        for steps in args.steps:
            start = time.monotonic()
            q = log_density_clamped_flow(model, graph, nbi, uem,
                n_ode_steps=steps, n_hutchinson=1, n_trace_replicates=args.replicas,
                terminal_time=args.terminal_time, xi_fn=lambda step,k,x:probes[k],parameterization=parameterization)
            centered = (q.double()-q.double().mean(-1,keepdim=True)).cpu()
            item = {'steps':steps, 'seconds':time.monotonic()-start,
                'log_q':q.cpu().tolist(), 'centered_log_q':centered.tolist(),
                'mean_probe_variance':float(centered.var(0,unbiased=True).mean())}
            if previous is not None:
                item['max_centered_change_from_previous'] = float((centered-previous).abs().max())
                item['max_mean_centered_change_from_previous'] = float((centered.mean(0)-previous.mean(0)).abs().max())
            valid = torch.isfinite(energy).cpu()
            if valid.sum() >= 2:
                r = q[:,valid.to(q.device)].double().cpu()
                e = energy[valid.to(energy.device)].double().cpu()
                r = r-r.mean(-1,keepdim=True)+(e-e.mean())  # kT=1 eV diagnostic
                plug_in = r.mean(0).square().mean()
                bias = r.var(0,unbiased=True).mean()/args.replicas
                item.update(squared_mean_residual=float(plug_in),
                    estimated_noise_bias=float(bias),
                    unbiased_replica_product_residual=float(plug_in-bias))
            row['resolutions'].append(item)
            previous = centered
            print(json.dumps({'parent':row['parent_id'], 'steps':steps,
                'seconds':item['seconds'], 'probe_variance':item['mean_probe_variance']}), flush=True)
        if i < args.exact_parent_count:
            start = time.monotonic()
            q = log_density_clamped_flow(model,graph,nbi,uem,n_ode_steps=32,
                n_hutchinson=0,terminal_time=args.terminal_time,parameterization=parameterization)
            row['exact_trace_32'] = {'log_q':q.cpu().tolist(),
                'centered_log_q':(q-q.mean()).cpu().tolist(), 'seconds':time.monotonic()-start}
        report['rows'].append(row)
        write_json(args.out/'panel.json',report)
    report['complete'] = True
    write_json(args.out/'panel.json',report)


if __name__ == '__main__':
    main()
