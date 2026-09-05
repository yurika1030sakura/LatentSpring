"""Calibrate lambda_1 for the corrected score read-out.

The reported force cells used a read-out that fed the endpoint prediction into
score_from_fm_velocity, which expects a velocity, so the score they matched was
(1-t) times the intended one.  Correcting it multiplies the score by 1/(1-t) --
33x at t = 0.97 -- and the loss is squared, so rerunning at the old
lambda_1 = 0.1 would almost certainly diverge immediately and tell us nothing.

This measures, on the SHARED WARM-START CHECKPOINT and under the SAME multi-t
force loss training actually forms (the sum over t_eval_values, not a single t),
    ||g_force_old||, ||g_force_corrected||, ||g_FM||
and reports
    lambda_1_new = lambda_1_old * ||g_force_old|| / ||g_force_corrected||
so the weighted force update keeps the magnitude the old experiment had.  It
also reports lambda_1_new * ||g_force_corrected|| / ||g_FM|| so we can see
whether that magnitude is sane relative to the flow-matching gradient.

Gradients are taken of the SUMMED multi-t loss, because per-t norms do not
compose: ||a+b+c|| is not ||a||+||b||+||c||.

No training.  Forward/backward only.
"""
from __future__ import annotations
import argparse, json, math, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.force_diag._common import (            # noqa: E402
    load_model, build_val_batches, named_trainable, grad_flat, summarize,
)
from cfm_mol.bgfm_loss import score_from_fm_velocity, force_loss  # noqa: E402


def multi_t_force_grad(model, named, g0, t_values, corrected, kT, device):
    """Gradient of sum_t L_force(t), which is what training forms.

    Uses the model's own sample_conditional_path, so x_t and the discrete
    channels a_t / c_t / e_t are populated exactly as in training.  The caller
    pins the RNG so the two read-outs see identical x_0 draws.
    """
    from flowmol.data_processing.utils import get_batch_idxs, get_upper_edge_mask
    total = None
    for tv in t_values:
        g = g0.clone()
        nbi, ebi = get_batch_idxs(g)
        uem = get_upper_edge_mask(g)
        t = torch.full((g.batch_size,), float(tv), device=device)
        g = model.vector_field.sample_conditional_path(g, t, nbi, ebi, uem)
        x_t = g.ndata['x_t']
        t_atom = t[nbi]
        v = model.vector_field(g, t, node_batch_idx=nbi,
                               upper_edge_mask=uem)['x']
        if corrected:
            v = (v - x_t) / (1.0 - t_atom).clamp_min(1e-4).unsqueeze(-1)
        sc = score_from_fm_velocity(v, x_t, t_atom, prior_std=1.0)
        L = force_loss(sc, g.ndata['force_1_true'], kT=kT)
        total = L if total is None else total + L
    gf, _ = grad_flat(total, named)
    return total, gf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--t_values", type=float, nargs="+", default=[0.85, 0.92, 0.97])
    ap.add_argument("--lambda1_old", type=float, default=0.1)
    ap.add_argument("--n_minibatches", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_atoms", type=int, default=60)
    ap.add_argument("--kT", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, bgfm_cfg, health = load_model(a.config, a.checkpoint, dev)
    print(f"[calib] checkpoint health: {json.dumps(health)[:300]}", flush=True)
    batches, batch_meta = build_val_batches(
        cfg, a.n_minibatches, a.batch_size, a.max_atoms, a.seed, dev)
    named = named_trainable(model)
    from flowmol.data_processing.utils import get_batch_idxs, get_upper_edge_mask

    rows = []
    for i, g in enumerate(batches):
        torch.manual_seed(a.seed * 1000 + i)          # same x_0 for both arms
        st = torch.cuda.get_rng_state_all() if dev.startswith('cuda') else None
        cst = torch.get_rng_state()
        L_old, g_old = multi_t_force_grad(model, named, g, a.t_values,
                                          False, a.kT, dev)
        torch.set_rng_state(cst)
        if st is not None: torch.cuda.set_rng_state_all(st)
        L_new, g_new = multi_t_force_grad(model, named, g, a.t_values,
                                          True, a.kT, dev)
        rows.append({"mb": i,
                     "L_old": float(L_old), "L_corrected": float(L_new),
                     "g_old": float(g_old.norm()), "g_corrected": float(g_new.norm()),
                     "cos_old_corrected": float(
                         torch.dot(g_old, g_new) / (g_old.norm() * g_new.norm() + 1e-12))})
        print(f"[calib] mb {i}: |g_old|={rows[-1]['g_old']:.4g} "
              f"|g_corr|={rows[-1]['g_corrected']:.4g} "
              f"cos={rows[-1]['cos_old_corrected']:+.3f}", flush=True)

    go = summarize([r["g_old"] for r in rows])
    gc = summarize([r["g_corrected"] for r in rows])
    ratio = go["mean"] / gc["mean"] if gc["mean"] else float("nan")
    lam_new = a.lambda1_old * ratio
    out = {"meta": {"checkpoint": a.checkpoint, "config": a.config,
                    "t_values": a.t_values, "lambda1_old": a.lambda1_old,
                    "n_minibatches": a.n_minibatches, "kT": a.kT,
                    "note": "gradients of the SUMMED multi-t force loss"},
           "per_minibatch": rows,
           "g_force_old": go, "g_force_corrected": gc,
           "ratio_old_over_corrected": ratio,
           "lambda_1_new": lam_new,
           "lambda_1_new_over_10": lam_new / 10.0,
           "cos_old_vs_corrected": summarize([r["cos_old_corrected"] for r in rows])}
    json.dump(out, open(a.out, "w"), indent=2)
    print("\n=== CALIBRATION ===")
    print(f"  |g_force| old       = {go['mean']:.6g}  (sd {go['sd']:.3g})")
    print(f"  |g_force| corrected = {gc['mean']:.6g}  (sd {gc['sd']:.3g})")
    print(f"  ratio               = {ratio:.6g}")
    print(f"  lambda_1_new        = {lam_new:.6g}      (old {a.lambda1_old})")
    print(f"  second weight       = {lam_new/10:.6g}   (pre-specified, = new/10)")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
