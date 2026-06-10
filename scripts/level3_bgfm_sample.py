"""Level 3 BGFM sampler: K-step forward ODE on positions, atom types fixed.

For Level 3 efficiency comparison we generate N geometries of a *fixed*
molecule via BGFM with K << 100 ODE steps, and compare them statistically
to the MD ground-truth ensemble (level3_md_reference.py).

Why a custom sampler vs flowmol's .sample():
  .sample() samples positions AND discrete channels from prior; the
  discrete channels (atom types, charges, edges) get noised back during
  integration via CTMC. For Level 3 we want to compare to MD which fixes
  the molecule's composition -- so the comparison must hold composition
  fixed too. We hold a_t = a_1_true, c_t = c_1_true, e_t = e_1_true
  throughout and only flow x_t (= positions).

Output: <out>/level3_bgfm_<mol_id>.npz
  positions   : (N_samples, n_atoms, 3) float32
  energies    : (N_samples,) float32 eV   (filled in by Stage-2 OMol25 eval)
  atomic_numbers: (n_atoms,)
  charge, spin, K_steps, n_samples

Two-stage: this script (flowmol env) only generates geometries. A Stage-2
script (omol25 env) computes their OMol25 energies, mirroring the Boltzmann
correlation eval pipeline.

Usage (envs/flowmol):
  conda activate envs/flowmol
  python scripts/level3_bgfm_sample.py \\
      --checkpoint <bgfm_ckpt> --config configs/omol25_4m_bgfm_energy.yaml \\
      --eval_data /n/netscratch/.../val_data_processed.pt \\
      --mol_indices 0,7,42,100,2000 \\
      --K_steps 12 --n_samples 100 \\
      --out_dir runs/eval/level3_bgfm
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import dgl
import numpy as np
import torch
from torch.nn.functional import one_hot


def _build_molecule_template_graph(
    atom_type_idx: np.ndarray, atom_charges_raw: np.ndarray,
    n_atom_types_total: int, n_charge_classes: int = 6, n_bond_types: int = 4,
    device: str = "cuda",
):
    """Build one molecule graph with fully-connected edges + fixed discrete
    channels (a_1_true / c_1_true / e_1_true), no positions yet."""
    n_atoms = len(atom_type_idx)
    src, dst = torch.meshgrid(
        torch.arange(n_atoms), torch.arange(n_atoms), indexing="ij")
    mask = src != dst
    g = dgl.graph((src[mask], dst[mask]), num_nodes=n_atoms)
    a_oh = one_hot(torch.from_numpy(atom_type_idx).long(),
                   num_classes=n_atom_types_total).float()
    c_oh = one_hot(torch.from_numpy(atom_charges_raw).long() + 2,
                   num_classes=n_charge_classes).float()
    g.ndata["a_1_true"] = a_oh
    g.ndata["a_t"] = a_oh.clone()
    g.ndata["c_1_true"] = c_oh
    g.ndata["c_t"] = c_oh.clone()
    n_e = g.num_edges()
    e_oh = torch.zeros(n_e, n_bond_types)
    e_oh[:, 0] = 1.0     # bond-free all-none
    g.edata["e_1_true"] = e_oh
    g.edata["e_t"] = e_oh.clone()
    return g.to(device)


def sample_positions_via_flow(
    model, g_batched, node_batch_idx, upper_edge_mask,
    K_steps: int = 12, prior_std: float = 1.0,
) -> torch.Tensor:
    """Forward Euler integration of the position channel only.

      x_0 ~ N(0, prior_std^2 I), COM-removed.
      for k = 0 .. K-1:
          v = v_theta^x(x_t, t = (k + 0.5) / K | a_1, c_1, e_1 fixed)
          x_{t+1/K} = x_t + (1/K) * v
      return x_1

    The discrete channels are held at their true labels throughout (set by
    _build_molecule_template_graph). Same pattern as bgfm_density's reverse
    integrator, except FORWARD in time.
    """
    device = g_batched.device
    n_nodes = g_batched.num_nodes()
    B = int(node_batch_idx.max().item()) + 1

    # COM-free Gaussian prior on positions, per-molecule
    x = torch.randn(n_nodes, 3, device=device) * prior_std
    # Remove per-molecule COM
    # Use scatter to compute mean per molecule
    com_sum = torch.zeros(B, 3, device=device)
    com_sum.scatter_add_(0, node_batch_idx.unsqueeze(-1).expand(-1, 3), x)
    counts = torch.zeros(B, device=device)
    counts.scatter_add_(0, node_batch_idx, torch.ones(n_nodes, device=device))
    com = com_sum / counts.unsqueeze(-1).clamp(min=1)
    x = x - com[node_batch_idx]

    dt = 1.0 / K_steps
    with torch.no_grad():
        for k in range(K_steps):
            t_val = (k + 0.5) * dt
            t_scalar = torch.full((B,), t_val, device=device, dtype=torch.float32)
            g_batched.ndata["x_t"] = x
            vf_out = model.vector_field(
                g_batched, t_scalar,
                node_batch_idx=node_batch_idx,
                upper_edge_mask=upper_edge_mask,
            )
            v = vf_out["x"]
            x = x + dt * v
            # Re-remove COM (numerical hygiene)
            com_sum.zero_()
            com_sum.scatter_add_(0, node_batch_idx.unsqueeze(-1).expand(-1, 3), x)
            com = com_sum / counts.unsqueeze(-1).clamp(min=1)
            x = x - com[node_batch_idx]
    return x


_PERIODIC_SYMBOLS = [
    "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Ga","Ge","As","Se","Br","Kr",
    "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
    "In","Sn","Sb","Te","I","Xe",
    "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy",
    "Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt",
    "Au","Hg","Tl","Pb","Bi",
]
SYMBOL_TO_Z = {sym: idx + 1 for idx, sym in enumerate(_PERIODIC_SYMBOLS)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--eval_data", type=Path, required=True)
    ap.add_argument("--mol_indices", required=True,
                    help="Comma-sep indices into val shard")
    ap.add_argument("--K_steps", type=int, default=12)
    ap.add_argument("--n_samples", type=int, default=100,
                    help="Samples per molecule")
    ap.add_argument("--batch_per_pass", type=int, default=50,
                    help="How many sample copies per integration call")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--no_patch", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    from flowmol.model_utils.load import model_from_config, read_config_file
    from flowmol.data_processing.utils import get_batch_idxs, get_upper_edge_mask

    cfg = read_config_file(args.config)
    cfg.get("mol_fm", {}).pop("bgfm", None)
    atom_map = cfg["dataset"]["atom_map"]
    n_real = len(atom_map)
    has_fake = cfg["mol_fm"].get("fake_atom_p", 0.0) > 0
    has_mask = cfg["mol_fm"].get("parameterization", "") == "ctmc"
    n_total_atom_types = n_real + int(has_fake) + int(has_mask)

    model = model_from_config(cfg)
    if not args.no_patch:
        from cfm_mol.domain import default_d_min_table
        from cfm_mol.flow_model import patch_flowmol
        e_weight = cfg["mol_fm"].get("total_loss_weights", {}).get("e", 2.0)
        bond_free = float(e_weight) == 0.0
        d_min = torch.zeros(n_total_atom_types, n_total_atom_types)
        d_min[:n_real, :n_real] = default_d_min_table(
            n_atom_types=n_real, atom_map=atom_map)
        patch_flowmol(model, d_min, tangent=True, retract=True, gluing=True,
                      discrete_projection=not bond_free,
                      train_time_discrete=not bond_free, atom_map=atom_map)

    state = torch.load(str(args.checkpoint), map_location="cpu")
    sd = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[lvl3-bgfm] loaded ckpt: missing={len(missing)} unexpected={len(unexpected)}",
          flush=True)
    model = model.to(device).eval()

    print(f"[lvl3-bgfm] loading val shard: {args.eval_data}", flush=True)
    src = torch.load(args.eval_data, map_location="cpu", weights_only=False)
    positions_shard = src["positions"].numpy()
    atom_types = src["atom_types"].numpy().astype(np.int64)
    atom_charges = src["atom_charges"].numpy().astype(np.int64)
    nia = src["node_idx_array"].numpy()
    M_total = nia.shape[0]
    print(f"[lvl3-bgfm] val shard: M={M_total}", flush=True)

    indices = [int(x) for x in args.mol_indices.split(",")]

    for mi in indices:
        if mi >= M_total:
            print(f"[lvl3-bgfm] mol {mi} >= M, skip", flush=True)
            continue
        out_path = args.out_dir / f"level3_bgfm_mol{mi:05d}.npz"
        if out_path.exists():
            print(f"[lvl3-bgfm] mol {mi} exists at {out_path}, skip", flush=True)
            continue

        s, e = int(nia[mi, 0]), int(nia[mi, 1])
        n_atoms = e - s
        at_idx = atom_types[s:e]
        ac = atom_charges[s:e]
        Z = np.array([SYMBOL_TO_Z[atom_map[int(i)]] for i in at_idx], dtype=np.int64)

        print(f"\n[lvl3-bgfm] mol {mi}: {n_atoms} atoms, generating "
              f"{args.n_samples} samples with K={args.K_steps} steps", flush=True)

        # Generate in batches of batch_per_pass copies
        all_pos = []
        n_remaining = args.n_samples
        while n_remaining > 0:
            bs = min(args.batch_per_pass, n_remaining)
            # Build a batched graph of bs copies of this molecule template
            template_graphs = []
            for _ in range(bs):
                g = _build_molecule_template_graph(
                    at_idx, ac,
                    n_atom_types_total=n_total_atom_types,
                    n_charge_classes=6, n_bond_types=4, device=device,
                )
                template_graphs.append(g)
            g_batched = dgl.batch(template_graphs)
            nbi, _ = get_batch_idxs(g_batched)
            uem = get_upper_edge_mask(g_batched)

            x_samples = sample_positions_via_flow(
                model, g_batched, nbi, uem,
                K_steps=args.K_steps, prior_std=1.0)
            x_cpu = x_samples.detach().cpu().numpy().reshape(bs, n_atoms, 3)
            all_pos.append(x_cpu)
            n_remaining -= bs
            print(f"  [{args.n_samples - n_remaining}/{args.n_samples}] generated", flush=True)

        all_pos_arr = np.concatenate(all_pos, axis=0).astype(np.float32)
        np.savez(
            out_path,
            positions=all_pos_arr,
            energies=np.full(args.n_samples, np.nan, dtype=np.float32),
            atomic_numbers=Z,
            charge=int(ac[0]) if ac.size > 0 else 0,
            spin=1,
            K_steps=args.K_steps,
            n_samples=args.n_samples,
        )
        print(f"[lvl3-bgfm] wrote {out_path}", flush=True)

    print("\n[lvl3-bgfm] all done. Next: stage-2 OMol25 energies via "
          "scripts/level3_compute_energies.py", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
