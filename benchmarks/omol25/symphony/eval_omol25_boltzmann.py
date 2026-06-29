"""Boltzmann-consistency Stage-1 for Symphony on OMol25.

Symphony is autoregressive: log p(molecule) = -(sum of per-fragment generation
losses over the molecule's fragment sequence). For each held-out OMol25 molecule
we make K perturbations, fragment each, run the TRAINED model, sum the per-fragment
loss, and write the Stage-2 JSON. Run from baselines/symphony in the symphony env.

Loads the trained params/config from --workdir (via analyses.analysis.load_model_at_step),
NOT a fresh init. With --ref_json it reuses the EXACT molecules/perturbations/
charge/spin from a FlowMol stage1 JSON so the R2 is apples-to-apples across models.
"""
import argparse, json
import numpy as np
import jax, jax.numpy as jnp
import jraph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--step", default="best", help="checkpoint step ('best' -> params_best.pkl)")
    ap.add_argument("--npz", default=None, help="preprocessed val npz (only when --ref_json not given)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n_molecules", type=int, default=4)
    ap.add_argument("--n_perturb", type=int, default=5)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--ref_json", default=None, help="reuse molecules+perturbations+charge/spin from a FlowMol stage1 JSON; only recompute log_p_theta")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from symphony.data import input_pipeline, fragments
    from symphony import datatypes, loss
    from analyses import analysis

    # Build the model from the saved training config and restore TRAINED params.
    # run_in_evaluation_mode=False: generation_loss reads the training-mode predictions.
    model, params, config = analysis.load_model_at_step(
        a.workdir, a.step, run_in_evaluation_mode=False)
    params = jax.tree_util.tree_map(jnp.asarray, params)
    apply_fn = jax.jit(model.apply)
    num_species = 83
    rc = float(config.radial_cutoff)
    nn_tol = float(config.get("nn_tolerance", 0.5))
    lk = dict(
        ignore_position_loss_for_small_fragments=config.loss_kwargs.ignore_position_loss_for_small_fragments,
        discretized_loss=config.loss_kwargs.discretized_loss,
    )
    print(f"[eval] restored params step={a.step} rc={rc} nn_tol={nn_tol} mode={config.fragment_logic}", flush=True)

    # Assemble the list of groups (each = one molecule + K perturbations).
    rng0 = jax.random.PRNGKey(0)
    groups = []   # {"gid", "species"(np int32), "Z"(list), "charge", "spin", "xs"(list of (N,3)), "pert_ids"}
    if a.ref_json:
        from collections import OrderedDict
        by_g = OrderedDict()
        for rec in json.load(open(a.ref_json)):
            by_g.setdefault(rec["group_id"], []).append(rec)
        for gid, recs in by_g.items():
            recs = sorted(recs, key=lambda r: r["pert_id"])
            Z = recs[0]["atomic_numbers"]
            groups.append({"gid": gid, "species": (np.asarray(Z) - 1).astype(np.int32), "Z": Z,
                           "charge": recs[0].get("charge", 0), "spin": recs[0].get("spin", 1),
                           "xs": [np.asarray(r["positions"], np.float32) for r in recs],
                           "pert_ids": [r["pert_id"] for r in recs]})
        print(f"[eval] reusing {len(groups)} groups from {a.ref_json}", flush=True)
    else:
        z = np.load(a.npz)
        pos_all, sp_all, ni = z[f"{a.split}_pos"], z[f"{a.split}_species"], z[f"{a.split}_nodeidx"]
        np_rng = np.random.default_rng(0)
        for g in range(min(a.n_molecules, ni.shape[0])):
            s, e = int(ni[g, 0]), int(ni[g, 1])
            pos0 = pos_all[s:e]; species = sp_all[s:e].astype(np.int32)
            xs = [pos0.copy() if p == 0 else pos0 + np_rng.normal(0, a.sigma, pos0.shape) for p in range(a.n_perturb)]
            groups.append({"gid": g, "species": species, "Z": (species + 1).tolist(),
                           "charge": 0, "spin": 1, "xs": xs, "pert_ids": list(range(a.n_perturb))})
        print(f"[eval] {len(groups)} molecules from npz {a.split}", flush=True)

    def fragment(xs, species, gid):
        struct = datatypes.Structures(
            nodes=datatypes.NodesInfo(positions=jnp.asarray(xs, jnp.float32),
                                      species=jnp.asarray(species)),
            edges=None, receivers=None, senders=None, globals=None,
            n_node=jnp.asarray([len(species)]), n_edge=None)
        struct = input_pipeline.infer_edges_with_radial_cutoff_on_positions(struct, radial_cutoff=rc)
        return list(fragments.generate_fragments(
            rng=jax.random.fold_in(rng0, gid), graph=struct, num_species=num_species,
            nn_tolerance=nn_tol, max_radius=None, mode=config.fragment_logic,
            heavy_first=False, max_targets_per_graph=int(config.max_targets_per_graph),
            transition_first=False))

    records = []
    for grp in groups:
        gid, species, Z = grp["gid"], grp["species"], grp["Z"]
        # Fragment all perturbations first, then pad each to the group's max shape so
        # jax.jit recompiles at most once per group (not once per perturbation).
        frag_lists = [fragment(xs, species, gid) for xs in grp["xs"]]
        nonempty = [fl for fl in frag_lists if fl]
        if nonempty:
            pad_n = max(int(jraph.batch(fl).n_node.sum()) for fl in nonempty) + 1
            pad_e = max(int(jraph.batch(fl).n_edge.sum()) for fl in nonempty) + 1
            pad_g = max(len(fl) for fl in nonempty) + 1
        for fl, xs, pid in zip(frag_lists, grp["xs"], grp["pert_ids"]):
            if not fl:
                logp = float("nan")
            else:
                padded = jraph.pad_with_graphs(jraph.batch(fl), n_node=pad_n, n_edge=pad_e, n_graph=pad_g)
                preds = apply_fn(params, None, padded)
                total_loss, _ = loss.generation_loss(preds=preds, graphs=padded, **lk)
                mask = jraph.get_graph_padding_mask(padded)
                logp = -float(jnp.sum(jnp.where(mask, total_loss, 0.0)))
            records.append({"group_id": gid, "pert_id": pid, "atomic_numbers": Z,
                            "positions": np.asarray(xs).tolist(),
                            "charge": grp["charge"], "spin": grp["spin"], "log_p_theta": logp})
        finite = [r["log_p_theta"] for r in records[-len(frag_lists):] if np.isfinite(r["log_p_theta"])]
        rng_str = f"[{min(finite):.1f},{max(finite):.1f}]" if finite else "all-nan"
        print(f"  group {gid}: N={len(species)} logp {rng_str}", flush=True)
    json.dump(records, open(a.out, "w"))
    print(f"wrote {a.out}: {len(records)} records, {len(groups)} groups")


if __name__ == "__main__":
    main()
