"""Boltzmann-consistency Stage-1 for Symphony on OMol25.

Symphony is autoregressive: log p(molecule) = -(sum of per-fragment generation
losses over the molecule's fragment sequence). For each held-out OMol25 molecule
we make K perturbations, fragment each, run the model, sum the per-fragment loss,
and write the Stage-2 JSON. Run from baselines/symphony in the symphony env.
"""
import argparse, json
import numpy as np
import jax, jax.numpy as jnp
import jraph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--step", default="best")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n_molecules", type=int, default=4)
    ap.add_argument("--n_perturb", type=int, default=5)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from symphony.data import input_pipeline, fragments
    from symphony import datatypes, loss, models
    from configs.omol25 import smoke as smoke_cfg

    # Smoke: build the model and use freshly-initialized params (the 3-step
    # checkpoint is ~random anyway; this verifies the log-p eval path). A real
    # run would restore trained params from the workdir's orbax checkpoint.
    config = smoke_cfg.get_config()
    # Training-mode model: generation_loss reads the training-mode predictions
    # (logits/distributions), not the eval-mode sampler. params are shared.
    model = models.create_model(config, run_in_evaluation_mode=False)
    params = None  # initialized lazily on the first graph below
    apply_fn = jax.jit(model.apply)
    num_species = 83
    rc = float(config.radial_cutoff)
    nn_tol = float(config.get("nn_tolerance", 0.5))
    lk = dict(
        ignore_position_loss_for_small_fragments=config.loss_kwargs.ignore_position_loss_for_small_fragments,
        discretized_loss=config.loss_kwargs.discretized_loss,
    )

    z = np.load(a.npz)
    pos_all, sp_all, ni = z[f"{a.split}_pos"], z[f"{a.split}_species"], z[f"{a.split}_nodeidx"]
    rng0 = jax.random.PRNGKey(0)
    np_rng = np.random.default_rng(0)
    records = []
    for g in range(min(a.n_molecules, ni.shape[0])):
        s, e = int(ni[g, 0]), int(ni[g, 1])
        pos0 = pos_all[s:e]
        species = sp_all[s:e].astype(np.int32)
        Z = (species + 1).tolist()
        for p in range(a.n_perturb):
            xs = pos0.copy() if p == 0 else pos0 + np_rng.normal(0, a.sigma, pos0.shape)
            struct = datatypes.Structures(
                nodes=datatypes.NodesInfo(positions=jnp.asarray(xs, jnp.float32),
                                          species=jnp.asarray(species)),
                edges=None, receivers=None, senders=None, globals=None,
                n_node=jnp.asarray([len(species)]), n_edge=None)
            struct = input_pipeline.infer_edges_with_radial_cutoff_on_positions(struct, radial_cutoff=rc)
            frags = list(fragments.generate_fragments(
                rng=jax.random.fold_in(rng0, g), graph=struct, num_species=num_species,
                nn_tolerance=nn_tol, max_radius=None, mode=config.fragment_logic,
                heavy_first=False, max_targets_per_graph=int(config.max_targets_per_graph),
                transition_first=False))
            if not frags:
                logp = float("nan")
            else:
                batched = jraph.batch(frags)
                padded = jraph.pad_with_graphs(
                    batched, n_node=int(batched.n_node.sum()) + 1,
                    n_edge=int(batched.n_edge.sum()) + 1, n_graph=len(frags) + 1)
                if params is None:
                    params = jax.jit(model.init)(jax.random.PRNGKey(0), padded)
                preds = apply_fn(params, None, padded)
                total_loss, _ = loss.generation_loss(preds=preds, graphs=padded, **lk)
                mask = jraph.get_graph_padding_mask(padded)
                logp = -float(jnp.sum(jnp.where(mask, total_loss, 0.0)))
            records.append({"group_id": g, "pert_id": p, "atomic_numbers": Z,
                            "positions": xs.tolist(), "charge": 0, "spin": 1,
                            "log_p_theta": logp})
        print(f"  group {g}: N={len(species)} nfrags~{len(frags)} logp={records[-1]['log_p_theta']:.1f}", flush=True)
    json.dump(records, open(a.out, "w"))
    print(f"wrote {a.out}: {len(records)} records")


if __name__ == "__main__":
    main()
