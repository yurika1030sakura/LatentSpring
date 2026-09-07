"""Re-score published Boltzmann-eval geometries at a different solver resolution.

The point is an EXACTLY matched comparison. The published stage-1 run draws its
displacements with `torch.randn_like` interleaved with `log_density_via_flow`,
whose Hutchinson probes consume a step-count-dependent number of RNG draws, so
re-running stage 1 at a different `--n_ode_steps` silently redraws the
geometries. This script instead reads the published geometries back out of
`boltzmann_samples.json` and recomputes only `log_p_theta`. Same checkpoint,
same parents, same geometries, same energies -- only the solver differs.

The molecule order is recovered the way stage 1 chose it (torch.manual_seed then
torch.randperm) and then VERIFIED against the stored pert_id == 0 positions; the
script refuses to run if any base geometry disagrees.

Usage:
  python scripts/rescore/rescore_published.py \
      --samples <wide_arm>/boltz1/boltzmann_samples.json \
      --checkpoint <ckpt> --config <config> --eval_data <processed> \
      --n_ode_steps 48 --out <out.json>
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--eval_data", type=Path, required=True)
    ap.add_argument("--n_ode_steps", type=int, required=True)
    ap.add_argument("--n_hutchinson", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rng_seed", type=int, default=12345,
                    help="seed for the Hutchinson probes only; fixed across "
                         "resolutions so the trace estimator is comparable")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pos_tol", type=float, default=1e-4)
    ap.add_argument("--max_groups", type=int, default=None,
                    help="smoke-test only: stop after this many groups")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    from flowmol.model_utils.load import model_from_config, read_config_file
    from flowmol.data_processing.dataset import MoleculeDataset
    from flowmol.data_processing.utils import get_batch_idxs, get_upper_edge_mask
    import dgl
    from cfm_mol.bgfm_density import log_density_via_flow

    recs = json.load(open(args.samples))
    groups: dict[int, list[dict]] = {}
    for r in recs:
        groups.setdefault(int(r["group_id"]), []).append(r)
    for gid in groups:
        groups[gid].sort(key=lambda r: int(r["pert_id"]))
    n_groups_all = len(groups)
    n_groups = min(n_groups_all, args.max_groups) if args.max_groups else n_groups_all
    print(f"[rescore] {len(recs)} records, {n_groups} groups from {args.samples}", flush=True)

    cfg = read_config_file(args.config)
    cfg.get("mol_fm", {}).pop("bgfm", None)
    cfg["dataset"]["processed_data_dir"] = str(args.eval_data)

    model = model_from_config(cfg)
    from cfm_mol.domain import default_d_min_table
    from cfm_mol.flow_model import patch_flowmol
    # Same patch construction as scripts/eval_boltzmann_stage1.py:96-111, so the
    # model this script scores is the model that produced the published numbers.
    atom_map = cfg["dataset"]["atom_map"]
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
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    bad = sum(int((~torch.isfinite(v)).sum()) for v in sd.values() if v.is_floating_point())
    if bad:
        sys.exit(f"ABORT: checkpoint has {bad} non-finite params")
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()

    # Recover stage 1's molecule order, then verify it against stored geometry.
    ds_cfg = dict(cfg["dataset"])
    ds_cfg["fake_atom_p"] = 0.0
    ds_cfg["fake_atom_std"] = 1.0
    ds_cfg["explicit_aromaticity"] = cfg["mol_fm"].get("explicit_aromaticity", False)
    val = MoleculeDataset("val", ds_cfg, prior_config=cfg["mol_fm"]["prior_config"])

    # Match each published group to its val molecule BY GEOMETRY rather than by
    # replaying stage 1's RNG stream. Stage 1 seeds once and then constructs the
    # model, whose weight initialisation consumes draws, so the permutation is
    # not recoverable from the seed alone -- and would break again on any change
    # to model construction. Bucketing by atom count keeps this near-linear.
    n_atoms_arr = (val.node_idx_array[:, 1] - val.node_idx_array[:, 0])
    by_size: dict[int, list[int]] = {}
    for mi, na in enumerate(n_atoms_arr.tolist()):
        by_size.setdefault(int(na), []).append(mi)

    idxs: list[int] = []
    for gid in range(n_groups):
        base = np.asarray(groups[gid][0]["positions"], dtype=np.float64)
        cands = by_size.get(base.shape[0], [])
        hit = -1
        for mi in cands:
            pos = val[mi].ndata["x_1_true"].detach().cpu().numpy().astype(np.float64)
            if pos.shape == base.shape and np.abs(pos - base).max() <= args.pos_tol:
                hit = mi
                break
        if hit < 0:
            sys.exit(f"ABORT: group {gid} ({base.shape[0]} atoms) has no geometry "
                     f"match among {len(cands)} val molecules of that size")
        idxs.append(hit)
    if len(set(idxs)) != len(idxs):
        sys.exit("ABORT: two published groups matched the same val molecule")
    print(f"[rescore] matched all {n_groups} groups to val molecules by geometry", flush=True)

    out = []
    mismatched = 0
    for gid in range(n_groups):
        grecs = groups[gid]
        g0 = val[idxs[gid]].to(device)
        base_stored = torch.tensor(grecs[0]["positions"], dtype=torch.float32, device=device)
        base_ds = g0.ndata["x_1_true"]
        if base_ds.shape != base_stored.shape or \
           (base_ds - base_stored).abs().max().item() > args.pos_tol:
            mismatched += 1
            if mismatched <= 3:
                d = (base_ds - base_stored).abs().max().item() if base_ds.shape == base_stored.shape else float("nan")
                print(f"[rescore] group {gid}: base geometry mismatch (max|d|={d:.3e}, "
                      f"shapes {tuple(base_ds.shape)} vs {tuple(base_stored.shape)})", flush=True)
            continue

        graphs = []
        for r in grecs:
            gp = g0.clone()
            gp.ndata["x_1_true"] = torch.tensor(r["positions"], dtype=torch.float32, device=device)
            graphs.append(gp)
        gb = dgl.batch(graphs).to(device)
        gb.ndata["x_t"] = gb.ndata["x_1_true"]
        gb.ndata["a_t"] = gb.ndata["a_1_true"]
        gb.ndata["c_t"] = gb.ndata["c_1_true"]
        gb.edata["e_t"] = gb.edata["e_1_true"]
        nbi, _ = get_batch_idxs(gb)
        uem = get_upper_edge_mask(gb)

        # Fix the probe stream per group so the only thing that varies between
        # resolutions is the number of steps, not the trace draws.
        torch.manual_seed(args.rng_seed + gid)
        with torch.enable_grad():
            logp = log_density_via_flow(
                model, gb, nbi, uem,
                n_ode_steps=args.n_ode_steps,
                n_hutchinson=args.n_hutchinson, prior_std=1.0)
        logp = logp.detach().cpu().numpy()

        for j, r in enumerate(grecs):
            out.append({"group_id": gid, "pert_id": int(r["pert_id"]),
                        "log_p_theta_new": float(logp[j]),
                        "log_p_theta_published": float(r["log_p_theta"])})
        if (gid + 1) % 20 == 0:
            print(f"[rescore] {gid+1}/{n_groups} groups", flush=True)

    if mismatched:
        sys.exit(f"ABORT: {mismatched}/{n_groups} groups failed the geometry check; "
                 f"the recovered molecule order does not match the published run")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n_ode_steps": args.n_ode_steps,
               "n_hutchinson": args.n_hutchinson,
               "samples": str(args.samples),
               "checkpoint": str(args.checkpoint),
               "records": out}, open(args.out, "w"))
    print(f"[rescore] wrote {len(out)} records -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
