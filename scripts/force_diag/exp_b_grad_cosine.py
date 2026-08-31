"""EXPERIMENT B -- gradient cosine between the training objectives.

Diagnostic, explanatory only.  Trains nothing, writes nothing to the paper.

On the SAME checkpoint and the SAME fixed held-out minibatches as Experiment A,
flatten the parameter gradients of

    L_FM     flow matching (FlowMol3's own multi-channel loss, weighted by
             total_loss_weights exactly as training_step does)
    L_force  the BGFM force loss at probe time t (score read out of the vector
             field, matched to F(x_1)/kT from the dataset's DFT labels)
    L_value  the energy/Boltzmann-consistency loss: within-parent variance of
             (log p_theta + E/kT) over K perturbations, via the FFJORD
             reverse-time density (cfm_mol.bgfm_density)

and report cos(g_FM, g_force) and cos(g_FM, g_value) as distributions over
minibatches and over t, plus a per-parameter-block breakdown.

BATCH SOURCES -- read this before interpreting the numbers.
  L_FM and L_force are computed on the SAME minibatch of held-out molecules,
  so cos(g_FM, g_force) is a within-batch quantity.
  L_value cannot be: it is only defined on a group of K geometries of ONE
  parent, which is what the precomputed perturbation shards hold.  So the
  value arm uses perturbation minibatches from the HELD-OUT shard
  (perturbation_val_*.pt) and L_FM is recomputed on that same perturbation
  graph, making cos(g_FM, g_value) a within-batch quantity too -- just on a
  different batch source.  Both are labelled as such in the output.
  The perturbation shards carry ENERGIES only, not forces, so the force arm
  cannot be run on them; this experiment therefore needs no teacher worker.

Usage (GPU):
  $FLOWMOL_PY -u scripts/force_diag/exp_b_grad_cosine.py \
      --checkpoint <ckpt> --config configs/sweep/p0_A_fmonly_s1.yaml \
      --shards <perturbation_val_*.pt> --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    load_model, build_val_batches, score_readout, named_trainable, grad_flat,
    cosine, per_block_cosine, summarize, dump_json,
)


# ---------------------------------------------------------------------------
# FlowMol3's own FM loss, but with the timepoint under our control.
# ---------------------------------------------------------------------------
def fm_losses(model, g, t=None):
    """Replica of flowmol.models.flowmol.FlowMol.forward with an optional
    caller-supplied `t`.

    With t=None it draws `t = torch.rand(batch_size)` as the first random call,
    exactly as the original does, so under an identical torch seed this function
    reproduces `model(g)` bit-for-bit (asserted by --selfcheck).
    """
    from flowmol.data_processing.utils import get_batch_idxs, get_upper_edge_mask
    from flowmol.models.flowmol import _apply_loss_weight
    import torch.nn.functional as fn

    device = g.device
    batch_size = g.batch_size
    if not hasattr(model, "loss_fn_dict"):
        model.configure_loss_fns(device=device)

    node_batch_idx, edge_batch_idx = get_batch_idxs(g)
    upper_edge_mask = get_upper_edge_mask(g)

    if t is None:
        t = torch.rand(batch_size, device=device).float()

    g = model.vector_field.sample_conditional_path(
        g, t, node_batch_idx, edge_batch_idx, upper_edge_mask)

    if model.distort_p > 0.0:
        t_mask = (t > model.distort_t)[node_batch_idx]
        distort_mask = torch.rand(g.num_nodes(), 1, device=device) < model.distort_p
        distort_mask = distort_mask & t_mask.unsqueeze(-1)
        g.ndata['x_t'] = g.ndata['x_t'] + torch.randn_like(g.ndata['x_t']) * distort_mask * 0.5

    vf_output = model.vector_field(
        g, t, node_batch_idx=node_batch_idx, upper_edge_mask=upper_edge_mask)

    targets = {}
    alpha_t_prime = model.interpolant_scheduler.alpha_t_prime(t)
    for feat_idx, feat in enumerate(model.canonical_feat_order):
        if float(model.total_loss_weights.get(feat, 1.0)) == 0.0:
            continue
        data_src = g.edata if feat == 'e' else g.ndata
        if model.parameterization in ['endpoint', 'dirichlet', 'ctmc']:
            target = data_src[f'{feat}_1_true']
            if feat == "e":
                target = target[upper_edge_mask]
            if feat in ['a', 'c', 'e']:
                if model.target_blur == 0.0:
                    target = target.argmax(dim=-1)
                else:
                    target = target + torch.randn_like(target) * model.target_blur
                    target = fn.softmax(target, dim=-1)
        elif model.parameterization == 'vector-field':
            alpha_t_prime_i = alpha_t_prime[:, feat_idx]
            x_1 = data_src[f'{feat}_1_true']
            x_0 = data_src[f'{feat}_0']
            if feat == 'e':
                alpha_t_prime_i = alpha_t_prime_i[edge_batch_idx][upper_edge_mask].unsqueeze(-1)
                x_1 = x_1[upper_edge_mask]
                x_0 = x_0[upper_edge_mask]
            else:
                alpha_t_prime_i = alpha_t_prime_i[node_batch_idx].unsqueeze(-1)
            target = alpha_t_prime_i * (x_1 - x_0)
        if model.parameterization == 'ctmc' and feat in ['a', 'c', 'e']:
            if feat == 'e':
                xt_idxs = data_src[f'{feat}_t'][upper_edge_mask].argmax(-1)
            else:
                xt_idxs = data_src[f'{feat}_t'].argmax(-1)
            target[xt_idxs != model.n_cat_dict[feat]] = -100
        targets[feat] = target

    if model.time_scaled_loss:
        time_weights = model.interpolant_scheduler.loss_weights(t)

    losses = {}
    for feat_idx, feat in enumerate(model.canonical_feat_order):
        if float(model.total_loss_weights.get(feat, 1.0)) == 0.0:
            losses[feat] = torch.zeros((), device=device)
            continue
        if model.time_scaled_loss:
            weight = time_weights[:, feat_idx]
            weight = (weight[edge_batch_idx][upper_edge_mask] if feat == 'e'
                      else weight[node_batch_idx]).unsqueeze(-1)
        else:
            weight = 1.0
        losses[feat] = _apply_loss_weight(
            model.loss_fn_dict[feat](vf_output[feat], targets[feat]), weight)
        if model.time_scaled_loss:
            losses[feat] = losses[feat].mean()
    return losses


def fm_total(model, g, t=None):
    losses = fm_losses(model, g, t)
    total = torch.zeros((), device=g.device)
    for feat in model.canonical_feat_order:
        total = total + model.total_loss_weights[feat] * losses[feat]
    return total, {k: float(v.detach().item()) for k, v in losses.items()}


def _prep_perturbation_graph_for_fm(g_pert, node_batch_idx, prior_std=1.0):
    """Give a perturbation graph the prior endpoint x_0 the FM loss needs.
    COM-removed per virtual molecule, matching the centred-normal prior."""
    device = g_pert.device
    n = g_pert.num_nodes()
    B = int(node_batch_idx.max().item()) + 1 if node_batch_idx.numel() else 0
    x0 = torch.randn(n, 3, device=device) * prior_std
    com = torch.zeros(B, 3, device=device)
    com.scatter_add_(0, node_batch_idx.unsqueeze(-1).expand(-1, 3), x0)
    cnt = torch.zeros(B, device=device)
    cnt.scatter_add_(0, node_batch_idx, torch.ones(n, device=device))
    x0 = x0 - (com / cnt.unsqueeze(-1).clamp(min=1))[node_batch_idx]
    g_pert.ndata['x_0'] = x0
    return g_pert


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
    ap.add_argument("--no_patch", action="store_true")
    ap.add_argument("--score_variants", nargs="+",
                    default=["as_implemented", "endpoint_correct"])
    ap.add_argument("--per_block", action="store_true", default=True)
    # value-loss (perturbation) arm
    ap.add_argument("--shards", nargs="*", default=None)
    ap.add_argument("--value_minibatches", type=int, default=8)
    ap.add_argument("--b_parents", type=int, default=8)
    ap.add_argument("--value_max_atoms", type=int, default=50)
    ap.add_argument("--n_ode_steps", type=int, default=4)
    ap.add_argument("--n_hutchinson", type=int, default=2)
    ap.add_argument("--skip_value", action="store_true")
    ap.add_argument("--selfcheck", action="store_true", default=True)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    torch.manual_seed(args.seed)

    from cfm_mol.bgfm_loss import force_loss
    from cfm_mol.bgfm_train_hook import _build_path_aux_graph
    from cfm_mol.bgfm_density import energy_consistency_loss_per_mol
    from flowmol.data_processing.utils import get_batch_idxs, get_upper_edge_mask

    model, cfg, bgfm_cfg, health = load_model(
        args.config, args.checkpoint, device, patch=not args.no_patch)
    print(f"[expB] checkpoint health: {json.dumps(health)[:400]}", flush=True)

    batches, batch_meta = build_val_batches(
        cfg, args.n_minibatches, args.batch_size, args.max_atoms,
        args.seed, device)
    named = named_trainable(model)

    results = {
        "meta": {
            "experiment": "B_gradient_cosine",
            "checkpoint": args.checkpoint,
            "config": args.config,
            "checkpoint_health": health,
            "molecule_batches": batch_meta,
            "kT": args.kT,
            "t_values": args.t_values,
            "score_variants": args.score_variants,
            "patched": not args.no_patch,
            "device": device,
            "batch_source_note": (
                "cos(g_FM, g_force): both from the SAME held-out molecule "
                "minibatch. cos(g_FM, g_value): both from the SAME held-out "
                "PERTURBATION minibatch (the value loss needs K geometries per "
                "parent, which only the perturbation shards hold)."),
        },
    }

    # ---- self-check: fm_losses replicates FlowMol3's own forward ----------
    if args.selfcheck:
        g = batches[0].clone()
        torch.manual_seed(12345)
        with torch.no_grad():
            ref = model(g.clone())
        torch.manual_seed(12345)
        with torch.no_grad():
            mine = fm_losses(model, g.clone(), t=None)
        chk = {k: [float(ref[k].item()), float(mine[k].item())] for k in ref}
        maxdiff = max(abs(a - b) for a, b in chk.values())
        results["meta"]["fm_loss_selfcheck"] = {
            "per_channel_ref_vs_replica": chk, "max_abs_diff": maxdiff}
        print(f"[expB] fm_losses self-check max|diff| = {maxdiff:.3e}", flush=True)

    # =====================================================================
    # ARM 1: molecule batches -> cos(g_FM, g_force)
    # =====================================================================
    # g_FM as TRAINING sees it (t ~ U[0,1]); one draw per minibatch, reused for
    # every t so that only g_force moves.
    g_fm_train = {}
    fm_train_diag = []
    for b, g in enumerate(batches):
        model.zero_grad(set_to_none=True)
        torch.manual_seed(1000 + b)
        L, parts = fm_total(model, g.clone(), t=None)
        gf, spans = grad_flat(L, named)
        g_fm_train[b] = gf
        fm_train_diag.append({"mb": b, "loss": float(L.item()), "channels": parts,
                              "grad_norm": float(gf.norm().item())})
    results["fm_random_t"] = {"per_minibatch": fm_train_diag,
                              "grad_norm": summarize([d["grad_norm"] for d in fm_train_diag])}
    print(f"[expB] FM(random t) gradients done ({time.time()-t0:.0f}s)", flush=True)

    per_t_out = []
    for t_val in args.t_values:
        entry = {"t": t_val, "amplification_t_over_1mt": t_val / (1.0 - t_val)}

        # g_FM with every molecule pinned at this t
        g_fm_at_t = {}
        fm_at_t_diag = []
        for b, g in enumerate(batches):
            model.zero_grad(set_to_none=True)
            torch.manual_seed(2000 + b)
            tt = torch.full((g.batch_size,), t_val, device=device, dtype=torch.float32)
            L, parts = fm_total(model, g.clone(), t=tt)
            gf, spans = grad_flat(L, named)
            g_fm_at_t[b] = gf
            fm_at_t_diag.append({"mb": b, "loss": float(L.item()),
                                 "channels": parts,
                                 "grad_norm": float(gf.norm().item())})
        entry["fm_at_t"] = {"per_minibatch": fm_at_t_diag,
                            "grad_norm": summarize([d["grad_norm"] for d in fm_at_t_diag])}

        entry["force"] = {}
        for variant in args.score_variants:
            cos_train, cos_at_t, per_mb, blocks = [], [], [], []
            for b, g in enumerate(batches):
                nbi, ebi = get_batch_idxs(g)
                uem = get_upper_edge_mask(g)
                tt = torch.full((g.batch_size,), t_val, device=device,
                                dtype=torch.float32)
                g_aux = _build_path_aux_graph(
                    g.clone(), model.vector_field, tt,
                    node_batch_idx=nbi, edge_batch_idx=ebi, upper_edge_mask=uem)
                model.zero_grad(set_to_none=True)
                vf = model.vector_field(g_aux, tt, node_batch_idx=nbi,
                                        upper_edge_mask=uem)
                s = score_readout(vf["x"], g_aux.ndata["x_t"], tt[nbi], variant)
                Lf = force_loss(s, g_aux.ndata["force_1_true"], kT=args.kT)
                gfo, spans = grad_flat(Lf, named)
                c1 = cosine(g_fm_train[b], gfo)
                c2 = cosine(g_fm_at_t[b], gfo)
                cos_train.append(c1)
                cos_at_t.append(c2)
                per_mb.append({"mb": b, "L_force": float(Lf.item()),
                               "grad_norm": float(gfo.norm().item()),
                               "cos_with_fm_random_t": c1,
                               "cos_with_fm_at_t": c2})
                if args.per_block and b < 3:
                    blocks.append({"mb": b,
                                   "vs_fm_random_t": per_block_cosine(
                                       g_fm_train[b], gfo, spans)})
                del gfo
            entry["force"][variant] = {
                "per_minibatch": per_mb,
                "cos_with_fm_random_t": summarize(cos_train),
                "cos_with_fm_at_t": summarize(cos_at_t),
                "grad_norm": summarize([d["grad_norm"] for d in per_mb]),
                "L_force": summarize([d["L_force"] for d in per_mb]),
                "per_block_cosine_first_minibatches": blocks,
            }
        for b in list(g_fm_at_t):
            del g_fm_at_t[b]
        if device == "cuda":
            torch.cuda.empty_cache()
        per_t_out.append(entry)
        results["per_t"] = per_t_out
        dump_json(results, args.out)
        print(f"[expB] t={t_val} force arm done ({time.time()-t0:.0f}s)", flush=True)

    # =====================================================================
    # ARM 2: perturbation batches -> cos(g_FM, g_value)
    # =====================================================================
    if not args.skip_value:
        shards = args.shards or bgfm_cfg.get("energy_perturbation_shards", [])
        try:
            from cfm_mol.perturbation_loader import PerturbationLoader
            atom_map = cfg["dataset"]["atom_map"]
            loader = PerturbationLoader(
                shard_paths=shards,
                n_atom_types=int(getattr(model, "n_atom_types", len(atom_map))),
                n_extra_atom_classes=int(bgfm_cfg.get("n_extra_atom_classes", 0)),
                n_charge_classes=6,
                n_bond_types=int(bgfm_cfg.get("n_bond_types", 4)),
                b_parents=args.b_parents, device=device, seed=args.seed,
                max_atoms_per_parent=args.value_max_atoms)
            value_rows = []
            for vb in range(args.value_minibatches):
                g_p, E, pid, nbi, uem = loader.next_batch()
                # --- value gradient
                model.zero_grad(set_to_none=True)
                Lv, diag = energy_consistency_loss_per_mol(
                    model, g_p.clone(), nbi, uem, E, pid, kT=args.kT,
                    n_ode_steps=args.n_ode_steps, n_hutchinson=args.n_hutchinson)
                if not torch.isfinite(Lv):
                    value_rows.append({"vb": vb, "L_value": None,
                                       "nonfinite": True})
                    continue
                gv, spans_v = grad_flat(Lv, named)
                # --- FM gradient on the SAME perturbation graph
                row = {"vb": vb, "L_value": float(Lv.item()),
                       "value_grad_norm": float(gv.norm().item()),
                       "n_virtual_mols": int(g_p.batch_size),
                       "n_atoms": int(g_p.num_nodes()),
                       "diag": {k: (float(v) if isinstance(v, (int, float)) else str(v))
                                for k, v in diag.items()},
                       "fm": {}}
                for label, tv in ([("random_t", None)]
                                  + [(f"t={t}", t) for t in args.t_values]):
                    g_fmp = _prep_perturbation_graph_for_fm(g_p.clone(), nbi)
                    model.zero_grad(set_to_none=True)
                    torch.manual_seed(3000 + vb)
                    tt = (None if tv is None else
                          torch.full((g_fmp.batch_size,), tv, device=device,
                                     dtype=torch.float32))
                    Lfm, parts = fm_total(model, g_fmp, t=tt)
                    gfm, spans_f = grad_flat(Lfm, named)
                    row["fm"][label] = {
                        "L_FM": float(Lfm.item()),
                        "grad_norm": float(gfm.norm().item()),
                        "cos_fm_value": cosine(gfm, gv),
                    }
                    if args.per_block and vb < 3 and label == "random_t":
                        row["per_block_cosine_fm_random_t_vs_value"] = \
                            per_block_cosine(gfm, gv, spans_v)
                    del gfm
                del gv
                if device == "cuda":
                    torch.cuda.empty_cache()
                value_rows.append(row)
                print(f"[expB] value minibatch {vb} done "
                      f"({time.time()-t0:.0f}s)", flush=True)
                results["value_arm"] = {"shards": shards, "rows": value_rows}
                dump_json(results, args.out)
            agg = {}
            for label in (["random_t"] + [f"t={t}" for t in args.t_values]):
                agg[label] = summarize(
                    [r["fm"][label]["cos_fm_value"] for r in value_rows
                     if "fm" in r and label in r["fm"]])
            results["value_arm"] = {
                "shards": shards, "rows": value_rows,
                "cos_fm_value_summary": agg,
                "L_value": summarize([r.get("L_value") for r in value_rows]),
                "value_grad_norm": summarize(
                    [r.get("value_grad_norm") for r in value_rows]),
                "n_ode_steps": args.n_ode_steps,
                "n_hutchinson": args.n_hutchinson,
            }
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results["value_arm"] = {"available": False,
                                    "error": f"{type(exc).__name__}: {exc}",
                                    "shards": shards}

    # ---- headline comparison ---------------------------------------------
    head = {}
    for variant in args.score_variants:
        head[variant] = {
            f"t={e['t']}": e["force"][variant]["cos_with_fm_random_t"]["mean"]
            for e in per_t_out}
    results["headline"] = {
        "cos_fm_force_mean_by_t": head,
        "cos_fm_value_mean": (results.get("value_arm", {})
                              .get("cos_fm_value_summary", {})
                              .get("random_t", {}).get("mean")
                              if isinstance(results.get("value_arm"), dict) else None),
    }
    results["meta"]["wall_seconds"] = time.time() - t0
    dump_json(results, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
