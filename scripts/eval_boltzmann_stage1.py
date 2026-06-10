"""Boltzmann-correlation eval, Stage 1 (envs/flowmol): perturbation log-density.

The oral-defining experiment for BGFM. Tests whether the learned density is
Boltzmann by checking, FOR EACH MOLECULE, whether log p_theta tracks -E/kT
across small geometric perturbations of that molecule.

Why per-molecule perturbations (not cross-molecule):
  Cross-molecule energies are dominated by SIZE (more atoms => more negative
  total DFT energy), which confounds the correlation. Perturbing a single
  molecule holds composition fixed, so any log p_theta vs -E relationship is
  a clean test of local Boltzmann structure:
      Boltzmann  =>  log p_theta(x* + d) = -E(x* + d)/kT + const.

Procedure:
  - load K held-out molecules from val data
  - for each, make M perturbations x* + sigma * noise (COM-removed)
  - compute log p_theta for original + perturbations via FFJORD density
  - export geometries + log_p_theta to JSON for Stage 2 (OMol25 energies)

Output: <out_dir>/boltzmann_samples.json with records
  {group_id, pert_id, atomic_numbers, positions, charge, spin, log_p_theta}

Usage:
  conda activate envs/flowmol
  python scripts/eval_boltzmann_stage1.py \\
      --checkpoint <ckpt> --config <config> \\
      --eval_data /n/netscratch/ryl_lab/Lab/yulili_cfm_mol/omol25_4m_processed \\
      --n_molecules 60 --n_perturb 16 --sigma 0.15 \\
      --out_dir runs/eval/boltzmann/<tag>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


_PERIODIC_SYMBOLS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
    "Au", "Hg", "Tl", "Pb", "Bi",
]
SYMBOL_TO_Z = {sym: idx + 1 for idx, sym in enumerate(_PERIODIC_SYMBOLS)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--eval_data", type=Path, required=True,
                    help="Processed dir with val_data_processed.pt (+ atom_map in config).")
    ap.add_argument("--n_molecules", type=int, default=60)
    ap.add_argument("--n_perturb", type=int, default=16,
                    help="Perturbations per molecule (plus the original).")
    ap.add_argument("--sigma", type=float, default=0.15,
                    help="Perturbation noise std in Angstrom.")
    ap.add_argument("--n_ode_steps", type=int, default=12)
    ap.add_argument("--n_hutchinson", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--no_patch", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_atoms_filter", type=int, default=None,
                    help="Optional: only pick val molecules with <= this many atoms")
    ap.add_argument("--chemistry_slice", type=str, default=None,
                    choices=[None, "organic_chnofs", "transition_metal",
                             "halogenated", "heavy_main_group", "charged"],
                    help="Optional: filter val to a chemistry slice for per-slice R²")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    from flowmol.model_utils.load import model_from_config, read_config_file
    from flowmol.data_processing.dataset import MoleculeDataset
    from flowmol.data_processing.utils import get_batch_idxs, get_upper_edge_mask
    import dgl
    from cfm_mol.bgfm_density import log_density_via_flow

    cfg = read_config_file(args.config)
    cfg.get("mol_fm", {}).pop("bgfm", None)
    cfg["dataset"]["processed_data_dir"] = str(args.eval_data)
    atom_map = cfg["dataset"]["atom_map"]

    # Build model + patch (geometry hooks only; same as training/eval).
    model = model_from_config(cfg)
    if not args.no_patch:
        from cfm_mol.domain import default_d_min_table
        from cfm_mol.flow_model import patch_flowmol
        n_real = len(atom_map)
        has_fake = cfg["mol_fm"].get("fake_atom_p", 0.0) > 0
        has_mask = cfg["mol_fm"].get("parameterization", "") == "ctmc"
        n_total = n_real + int(has_fake) + int(has_mask)
        d_min = torch.zeros(n_total, n_total)
        d_min[:n_real, :n_real] = default_d_min_table(n_atom_types=n_real, atom_map=atom_map)
        e_weight = cfg["mol_fm"].get("total_loss_weights", {}).get("e", 2.0)
        bond_free = float(e_weight) == 0.0
        patch_flowmol(model, d_min, tangent=True, retract=True, gluing=True,
                      discrete_projection=not bond_free,
                      train_time_discrete=not bond_free, atom_map=atom_map)

    state = torch.load(str(args.checkpoint), map_location="cpu")
    sd = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[boltz1] loaded ckpt: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model = model.to(device).eval()

    # Val dataset (single molecules with known geometry).
    ds_cfg = dict(cfg["dataset"])
    ds_cfg["fake_atom_p"] = 0.0
    ds_cfg["fake_atom_std"] = 1.0
    ds_cfg["explicit_aromaticity"] = cfg["mol_fm"].get("explicit_aromaticity", False)
    val = MoleculeDataset("val", ds_cfg, prior_config=cfg["mol_fm"]["prior_config"])
    # Optional size filter -- the eval batches (1 + n_perturb) copies of one
    # molecule through n_ode_steps with create_graph=False, but the activation
    # memory still scales as N_atoms^2 * n_ode_steps. For 20GB MIG slices,
    # filter to <=60-80 atoms to fit.
    # Chemistry slice filter (per-slice R² for universal-coverage claim)
    # Slice definitions (indices into atom_map, which lists elements 0..N-1):
    #   organic_chnofs    : only H, C, N, O, F, S
    #   transition_metal  : at least one Sc-Zn / Y-Cd / Hf-Hg
    #   halogenated       : at least one Cl, Br, I (besides organic backbone)
    #   heavy_main_group  : at least one element with Z >= 14 (Si and beyond, p-block)
    #   charged           : nonzero atom_charges
    chemistry_slice = getattr(args, 'chemistry_slice', None)
    elem_indices = None
    if chemistry_slice is not None:
        # Map atom_map symbol -> dataset atom_type index
        sym_to_data_idx = {sym: i for i, sym in enumerate(atom_map)}
        def _sym_idx(sym):
            return sym_to_data_idx.get(sym, -1)
        SLICE_DEFS = {
            "organic_chnofs": dict(
                require_only_in={"H","C","N","O","F","S"}),
            "transition_metal": dict(
                require_any_of=set(
                    ["Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
                     "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
                     "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg"])),
            "halogenated": dict(
                require_any_of={"Cl","Br","I"}),
            "heavy_main_group": dict(
                require_any_of={"Si","P","Cl","Ge","As","Se","Br","Sn","Sb","Te","I","Pb","Bi"}),
        }
        max_atoms = getattr(args, 'max_atoms_filter', None)
        if max_atoms is None: max_atoms = 200
        # Translate per-molecule into eligible mask using val data
        eligible = []
        atom_types_all = val.atom_types
        for mi in range(len(val)):
            ns, ne = int(val.node_idx_array[mi, 0]), int(val.node_idx_array[mi, 1])
            n_atoms_i = ne - ns
            if n_atoms_i > max_atoms:
                continue
            at_idx = atom_types_all[ns:ne]
            if at_idx.dim() == 2:   # one-hot
                at_idx = at_idx.argmax(dim=-1)
            present_syms = {atom_map[int(i)] for i in at_idx if 0 <= int(i) < len(atom_map)}
            if chemistry_slice == "charged":
                ac = val.atom_charges[ns:ne]
                if ac.abs().sum() > 0: eligible.append(mi)
            elif chemistry_slice in SLICE_DEFS:
                d = SLICE_DEFS[chemistry_slice]
                if "require_only_in" in d:
                    if present_syms.issubset(d["require_only_in"]):
                        eligible.append(mi)
                elif "require_any_of" in d:
                    if present_syms & d["require_any_of"]:
                        eligible.append(mi)
        print(f"[boltz1] chemistry slice '{chemistry_slice}' (+max_atoms={max_atoms}): "
              f"{len(eligible)}/{len(val)} val molecules eligible", flush=True)
        if not eligible:
            raise RuntimeError(f"No val molecules in slice '{chemistry_slice}'")
        import random
        random.seed(args.seed); random.shuffle(eligible)
        n_mol = min(args.n_molecules, len(eligible))
        idxs = eligible[:n_mol]
    else:
        max_atoms = getattr(args, 'max_atoms_filter', None)
        if max_atoms is not None and max_atoms > 0:
            n_atoms_arr = val.node_idx_array[:, 1] - val.node_idx_array[:, 0]
            eligible = (n_atoms_arr <= max_atoms).nonzero(as_tuple=True)[0].tolist()
            print(f"[boltz1] size filter <= {max_atoms} atoms: "
                  f"{len(eligible)}/{len(val)} val molecules eligible", flush=True)
            if not eligible:
                raise RuntimeError(f"No val molecules with <= {max_atoms} atoms")
            import random
            random.seed(args.seed); random.shuffle(eligible)
            n_mol = min(args.n_molecules, len(eligible))
            idxs = eligible[:n_mol]
        else:
            n_mol = min(args.n_molecules, len(val))
            idxs = torch.randperm(len(val))[:n_mol].tolist()

    records = []
    for gi, di in enumerate(idxs):
        g0 = val[di]
        g0 = g0.to(device)
        n_atoms = g0.num_nodes()
        base_pos = g0.ndata['x_1_true'].detach().clone()
        atom_idx = g0.ndata['a_1_true'].argmax(dim=-1).cpu().numpy()
        # Map atom indices to Z (atom_map may have fewer than periodic length)
        z = np.array([SYMBOL_TO_Z[atom_map[i]] for i in atom_idx], dtype=np.int64)

        # Build batched graph of (1 + n_perturb) copies of this molecule.
        graphs = []
        pert_positions = []
        for p in range(args.n_perturb + 1):
            gp = g0.clone()
            if p == 0:
                pos = base_pos.clone()
            else:
                noise = torch.randn_like(base_pos) * args.sigma
                pos = base_pos + noise
                pos = pos - pos.mean(dim=0, keepdim=True)
            gp.ndata['x_1_true'] = pos
            graphs.append(gp)
            pert_positions.append(pos.detach().cpu().numpy())
        gbatch = dgl.batch(graphs).to(device)

        # endpoint aux: x_t = x_1_true, discrete = labels
        gbatch.ndata['x_t'] = gbatch.ndata['x_1_true']
        gbatch.ndata['a_t'] = gbatch.ndata['a_1_true']
        gbatch.ndata['c_t'] = gbatch.ndata['c_1_true']
        gbatch.edata['e_t'] = gbatch.edata['e_1_true']
        nbi, _ = get_batch_idxs(gbatch)
        uem = get_upper_edge_mask(gbatch)

        with torch.enable_grad():
            logp = log_density_via_flow(
                model, gbatch, nbi, uem,
                n_ode_steps=args.n_ode_steps,
                n_hutchinson=args.n_hutchinson, prior_std=1.0)
        logp = logp.detach().cpu().numpy()

        for p in range(args.n_perturb + 1):
            records.append({
                "group_id": gi,
                "pert_id": p,
                "atomic_numbers": z.tolist(),
                "positions": pert_positions[p].astype(float).tolist(),
                "charge": 0,
                "spin": 1,
                "log_p_theta": float(logp[p]),
            })
        if (gi + 1) % 10 == 0:
            print(f"[boltz1] {gi+1}/{n_mol} molecules ({len(records)} records)", flush=True)

    out_json = args.out_dir / "boltzmann_samples.json"
    with open(out_json, "w") as f:
        json.dump(records, f)
    print(f"[boltz1] wrote {out_json} ({len(records)} records, "
          f"{n_mol} groups x {args.n_perturb+1})", flush=True)
    print(f"[boltz1] next: scripts/eval_boltzmann_stage2.py --samples_json {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
