"""EXPERIMENT A -- the t-sweep.

Diagnostic, explanatory only.  Trains nothing, writes nothing to the paper.

On ONE fixed flow-matching checkpoint and a FIXED set of held-out minibatches,
with the SAME x_0 and x_1 at every t (so the sweep is paired), measure at
t in {0.80, 0.85, 0.90, 0.92, 0.95, 0.97}:

  (a) LOCATION MISMATCH  ||F(x_t) - F(x_1)||, per atom and per molecule.
      The precomputed shards hold teacher labels only at x_1, so forces at the
      intermediate point x_t are obtained from the live eSEN teacher through
      the cross-environment XML-RPC worker (scripts/omol25_worker.py) with
      cfm_mol/physics_drift.py::OMol25DriftClient as the client -- exactly the
      path the on-policy training branch uses.  F(x_1) is taken from the SAME
      teacher so the difference is a pure location effect; the dataset's DFT
      label force at x_1 is reported alongside as a calibration check.
      If the worker is unreachable, this block is recorded as unavailable with
      the error text and the rest of the sweep still runs.  Intermediate forces
      are NEVER fabricated and F(x_1) is NEVER substituted for F(x_t).

  (b) GRADIENT NORM ||grad_theta L_force|| per minibatch, and its
      across-minibatch variance (both var of the norm and the total gradient
      covariance trace mean_b ||g_b - gbar||^2, which is the quantity an
      optimiser actually feels).

  (c) the analytic amplification t/(1-t) alongside.

Both score read-outs are swept (see scripts/force_diag/_common.py):
'as_implemented' (what the reported baseline optimised) and
'endpoint_correct'.

Usage (GPU):
  $FLOWMOL_PY -u scripts/force_diag/exp_a_t_sweep.py \
      --checkpoint <ckpt> --config configs/sweep/p0_A_fmonly_s1.yaml \
      --out <json> --rpc_url http://127.0.0.1:28950
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    load_model, build_val_batches, score_readout, named_trainable, grad_flat,
    summarize, dump_json,
)


def _quantiles(t: torch.Tensor, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    if t.numel() == 0:
        return {f"q{int(q*100)}": None for q in qs}
    tt = t.detach().float().flatten()
    return {f"q{int(q*100)}": float(torch.quantile(tt, q).item()) for q in qs}


def _stats(t: torch.Tensor):
    if t.numel() == 0:
        return {"n": 0, "mean": None, "sd": None, "max": None}
    tt = t.detach().float().flatten()
    return {
        "n": int(tt.numel()),
        "mean": float(tt.mean().item()),
        "sd": float(tt.std(unbiased=False).item()),
        "max": float(tt.max().item()),
        **_quantiles(tt),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--t_values", type=float, nargs="+",
                    default=[0.80, 0.85, 0.90, 0.92, 0.95, 0.97])
    ap.add_argument("--n_minibatches", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_atoms", type=int, default=60)
    ap.add_argument("--kT", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no_patch", action="store_true",
                    help="skip the Layer-1 geometric hooks (default: apply, "
                         "matching training)")
    ap.add_argument("--score_variants", nargs="+",
                    default=["as_implemented", "endpoint_correct"])
    ap.add_argument("--rpc_url", default="http://127.0.0.1:28950")
    ap.add_argument("--clip_force", type=float, default=1e9,
                    help="teacher force clip; huge = effectively unclipped, so "
                         "the mismatch is measured on raw teacher output")
    ap.add_argument("--skip_teacher", action="store_true")
    ap.add_argument("--teacher_minibatches", type=int, default=None,
                    help="how many of the minibatches to send to the teacher "
                         "(default: all)")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    torch.manual_seed(args.seed)

    from cfm_mol.bgfm_loss import force_loss, score_force_cosine
    from cfm_mol.bgfm_train_hook import (
        _build_path_aux_graph, _mol_total_charge,
    )
    from flowmol.data_processing.utils import get_batch_idxs, get_upper_edge_mask

    model, cfg, bgfm_cfg, health = load_model(
        args.config, args.checkpoint, device, patch=not args.no_patch)
    print(f"[expA] checkpoint health: {json.dumps(health)[:400]}", flush=True)

    batches, batch_meta = build_val_batches(
        cfg, args.n_minibatches, args.batch_size, args.max_atoms,
        args.seed, device)
    print(f"[expA] {len(batches)} fixed held-out minibatches, "
          f"atoms/batch={[int(g.num_nodes()) for g in batches]}", flush=True)

    named = named_trainable(model)

    # ---- teacher client (cross-env eSEN worker) --------------------------
    teacher = None
    teacher_error = None
    if not args.skip_teacher:
        try:
            from cfm_mol.physics_drift import OMol25DriftClient, DriftConfig
            atom_map = list(cfg["dataset"]["atom_map"])
            n_channels = int(batches[0].ndata["a_1_true"].shape[-1])
            atom_map = atom_map + ["X"] * max(0, n_channels - len(atom_map))
            teacher = OMol25DriftClient(
                DriftConfig(rpc_url=args.rpc_url, clip_force=args.clip_force),
                atom_map)
            print(f"[expA] teacher up at {args.rpc_url}", flush=True)
        except Exception as exc:  # noqa: BLE001
            teacher_error = f"{type(exc).__name__}: {exc}"
            print(f"[expA] TEACHER UNAVAILABLE: {teacher_error}", flush=True)
            traceback.print_exc()

    n_tb = args.teacher_minibatches or len(batches)

    # ---- teacher force at the ENDPOINT x_1 (t-independent, computed once) --
    endpoint = {}   # mb -> dict(F1_teacher, valid1, F1_dft, nbi, uem, charge)
    for b, g in enumerate(batches):
        nbi, ebi = get_batch_idxs(g)
        uem = get_upper_edge_mask(g)
        rec = {"nbi": nbi, "ebi": ebi, "uem": uem,
               "F1_dft": g.ndata["force_1_true"].detach()}
        if teacher is not None and b < n_tb:
            q = _mol_total_charge(g, nbi)
            F1, v1 = teacher.compute_forces_dgl(
                g.ndata["x_1_true"].detach(), g.ndata["a_1_true"], nbi,
                mol_charge=q, return_valid=True)
            rec["F1_teacher"] = F1.detach()
            rec["valid1"] = v1
            print(f"[expA] teacher F(x_1) mb{b}: {int(v1.sum())}/{v1.numel()} "
                  f"atoms valid", flush=True)
        endpoint[b] = rec

    results = {
        "meta": {
            "experiment": "A_t_sweep",
            "checkpoint": args.checkpoint,
            "config": args.config,
            "checkpoint_health": health,
            "batches": batch_meta,
            "kT": args.kT,
            "t_values": args.t_values,
            "score_variants": args.score_variants,
            "patched": not args.no_patch,
            "device": device,
            "teacher_rpc_url": args.rpc_url,
            "teacher_available": teacher is not None,
            "teacher_error": teacher_error,
            "teacher_clip_force": args.clip_force,
            "teacher_minibatches": n_tb if teacher is not None else 0,
            "note": ("F(x_1) used for the mismatch is the SAME eSEN teacher as "
                     "F(x_t); the dataset DFT label force at x_1 is reported "
                     "separately as calibration."),
        },
        "per_t": [],
    }

    # ---- calibration: teacher F(x_1) vs dataset DFT label F(x_1) ----------
    if teacher is not None:
        d_all, c_all, n_dft, n_tea = [], [], [], []
        for b in range(n_tb):
            r = endpoint[b]
            if "F1_teacher" not in r:
                continue
            m = r["valid1"]
            a, c = r["F1_teacher"][m], r["F1_dft"][m]
            d_all.append((a - c).norm(dim=-1))
            n_tea.append(a.norm(dim=-1))
            n_dft.append(c.norm(dim=-1))
            cos = (a * c).sum(-1) / (a.norm(dim=-1) * c.norm(dim=-1) + 1e-12)
            c_all.append(cos)
        if d_all:
            results["teacher_vs_dft_at_x1"] = {
                "per_atom_abs_diff_norm_eVA": _stats(torch.cat(d_all)),
                "per_atom_cosine": _stats(torch.cat(c_all)),
                "per_atom_norm_teacher_eVA": _stats(torch.cat(n_tea)),
                "per_atom_norm_dft_label_eVA": _stats(torch.cat(n_dft)),
            }

    # ---- the sweep --------------------------------------------------------
    for t_val in args.t_values:
        amp = t_val / (1.0 - t_val)
        entry = {"t": t_val, "amplification_t_over_1mt": amp,
                 "amplification_1_over_1mt": 1.0 / (1.0 - t_val)}

        # (a) location mismatch, per minibatch, teacher at x_t
        mism_atom, mism_cos, Ft_norm, F1_norm, disp_atom = [], [], [], [], []
        mism_mol_rms, mism_mol_rel = [], []
        n_fail_mol = 0
        n_mol_total = 0

        # cache x_t per minibatch so (a) and (b) see the SAME geometry
        x_t_cache = {}
        for b, g in enumerate(batches):
            r = endpoint[b]
            tt = torch.full((g.batch_size,), t_val, device=device,
                            dtype=torch.float32)
            g_aux = _build_path_aux_graph(
                g.clone(), model.vector_field, tt,
                node_batch_idx=r["nbi"], edge_batch_idx=r["ebi"],
                upper_edge_mask=r["uem"])
            x_t_cache[b] = g_aux
            if teacher is None or b >= n_tb or "F1_teacher" not in r:
                continue
            x_t = g_aux.ndata["x_t"].detach()
            q = _mol_total_charge(g_aux, r["nbi"])
            Ft, vt = teacher.compute_forces_dgl(
                x_t, g_aux.ndata["a_1_true"], r["nbi"], mol_charge=q,
                return_valid=True)
            m = vt & r["valid1"]
            n_mol_total += int(g.batch_size)
            if m.any():
                d = (Ft[m] - r["F1_teacher"][m])
                mism_atom.append(d.norm(dim=-1))
                Ft_norm.append(Ft[m].norm(dim=-1))
                F1_norm.append(r["F1_teacher"][m].norm(dim=-1))
                cos = ((Ft[m] * r["F1_teacher"][m]).sum(-1)
                       / (Ft[m].norm(dim=-1)
                          * r["F1_teacher"][m].norm(dim=-1) + 1e-12))
                mism_cos.append(cos)
                disp_atom.append((x_t - g.ndata["x_1_true"]).detach()[m]
                                 .norm(dim=-1))
                # per molecule (only molecules whose atoms are all valid)
                nbi = r["nbi"]
                for mi in range(int(g.batch_size)):
                    sel = (nbi == mi) & m
                    if not bool(sel.any()):
                        n_fail_mol += 1
                        continue
                    dm = (Ft[sel] - r["F1_teacher"][sel])
                    ref = r["F1_teacher"][sel]
                    mism_mol_rms.append(float(
                        dm.pow(2).sum(-1).mean().sqrt().item()))
                    denom = float(ref.norm().item())
                    mism_mol_rel.append(
                        float(dm.norm().item() / denom) if denom > 1e-9
                        else float("nan"))
            else:
                n_fail_mol += int(g.batch_size)

        if mism_atom:
            entry["location_mismatch"] = {
                "available": True,
                "per_atom_mismatch_norm_eVA": _stats(torch.cat(mism_atom)),
                "per_atom_cos_Fxt_Fx1": _stats(torch.cat(mism_cos)),
                "per_atom_norm_F_xt_eVA": _stats(torch.cat(Ft_norm)),
                "per_atom_norm_F_x1_eVA": _stats(torch.cat(F1_norm)),
                "per_atom_displacement_xt_minus_x1_A": _stats(torch.cat(disp_atom)),
                "per_mol_rms_mismatch_eVA": summarize(mism_mol_rms),
                "per_mol_relative_mismatch": summarize(mism_mol_rel),
                "n_molecules_scored": len(mism_mol_rms),
                "n_molecules_teacher_failed": n_fail_mol,
                "n_molecules_attempted": n_mol_total,
            }
        else:
            entry["location_mismatch"] = {
                "available": False,
                "reason": ("teacher unavailable" if teacher is None
                           else "teacher returned no valid atoms"),
                "teacher_error": teacher_error,
            }

        # (b) gradient of L_force at this t, per score variant
        entry["gradient"] = {}
        for variant in args.score_variants:
            grads = []
            per_mb = []
            for b, g in enumerate(batches):
                # clone per forward: the vector field may stash self-conditioning
                # state on the graph, which must not leak across score variants.
                g_aux = x_t_cache[b].clone()
                r = endpoint[b]
                tt = torch.full((g.batch_size,), t_val, device=device,
                                dtype=torch.float32)
                model.zero_grad(set_to_none=True)
                vf = model.vector_field(
                    g_aux, tt, node_batch_idx=r["nbi"],
                    upper_edge_mask=r["uem"])
                s = score_readout(vf["x"], g_aux.ndata["x_t"],
                                  tt[r["nbi"]], variant)
                L = force_loss(s, g_aux.ndata["force_1_true"], kT=args.kT)
                with torch.no_grad():
                    cosdiag = float(score_force_cosine(
                        s.detach(), g_aux.ndata["force_1_true"],
                        kT=args.kT).item())
                    snorm = float(s.detach().norm(dim=-1).mean().item())
                    capped = float((s.detach().norm(dim=-1) >= 999.0)
                                   .float().mean().item())
                gflat, spans = grad_flat(L, named)
                if not torch.isfinite(gflat).all():
                    per_mb.append({"mb": b, "loss": float(L.item()),
                                   "grad_norm": None, "nonfinite_grad": True})
                    continue
                grads.append(gflat)
                per_mb.append({
                    "mb": b, "loss": float(L.item()),
                    "grad_norm": float(gflat.norm().item()),
                    "score_force_cosine": cosdiag,
                    "mean_score_norm": snorm,
                    "frac_atoms_score_capped": capped,
                })
            g_norms = [d["grad_norm"] for d in per_mb]
            blk = {
                "per_minibatch": per_mb,
                "grad_norm": summarize(g_norms),
                "loss": summarize([d["loss"] for d in per_mb]),
                "score_force_cosine": summarize(
                    [d.get("score_force_cosine") for d in per_mb]),
                "mean_score_norm": summarize(
                    [d.get("mean_score_norm") for d in per_mb]),
                "frac_atoms_score_capped": summarize(
                    [d.get("frac_atoms_score_capped") for d in per_mb]),
            }
            if len(grads) >= 2:
                G = torch.stack(grads)                    # (B, P)
                gbar = G.mean(0)
                dev = G - gbar
                trcov = float((dev.pow(2).sum(1)).mean().item()
                              * len(grads) / (len(grads) - 1))
                blk["grad_mean_norm"] = float(gbar.norm().item())
                blk["grad_total_variance_trace_cov"] = trcov
                blk["grad_variance_over_mean_sq"] = (
                    trcov / float(gbar.pow(2).sum().item())
                    if float(gbar.pow(2).sum().item()) > 0 else None)
                # mean pairwise cosine between per-minibatch gradients
                Gn = G / G.norm(dim=1, keepdim=True).clamp(min=1e-20)
                C = Gn @ Gn.T
                n = C.shape[0]
                off = (C.sum() - C.diag().sum()) / (n * (n - 1))
                blk["mean_pairwise_minibatch_cosine"] = float(off.item())
                del G, gbar, dev, Gn, C
            del grads
            if device == "cuda":
                torch.cuda.empty_cache()
            entry["gradient"][variant] = blk

        results["per_t"].append(entry)
        print(f"[expA] t={t_val} done ({time.time()-t0:.0f}s)", flush=True)
        dump_json(results, args.out)   # incremental, teardown-proof

    results["meta"]["wall_seconds"] = time.time() - t0
    dump_json(results, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
