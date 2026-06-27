"""Boltzmann-consistency Stage-1 for EDM / GeoLDM on OMol25.

Loads a checkpoint, takes held-out OMol25 molecules, makes K geometric
perturbations of each, computes the model's per-molecule log-density
(log p = -NLL from the diffusion forward), and writes the Stage-2 JSON
({group_id, pert_id, atomic_numbers, positions, charge, spin, log_p_theta}).
Run from baselines/edm (or baselines/geoldm) in the edm env. I/O / eval only.
"""
import argparse, json, pickle
from pathlib import Path
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="outputs/<exp_name> dir")
    ap.add_argument("--datadir", required=True, help="processed OMol25 dir")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n_molecules", type=int, default=4)
    ap.add_argument("--n_perturb", type=int, default=5)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--n_t_avg", type=int, default=8, help="average NLL over this many random t (reduce variance)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from configs.datasets_config import get_dataset_info
    from qm9 import dataset as qm9dataset
    from qm9.models import get_model
    from qm9.omol25_data import OMol25EDMDataset
    from equivariant_diffusion.utils import remove_mean_with_mask

    device = torch.device(a.device)
    with open(Path(a.model_path) / "args.pickle", "rb") as f:
        margs = pickle.load(f)
    margs.datadir = a.datadir
    if not hasattr(margs, "normalization_factor"): margs.normalization_factor = 1
    if not hasattr(margs, "aggregation_method"): margs.aggregation_method = "sum"

    dinfo = get_dataset_info(margs.dataset, margs.remove_h)
    n_species = len(dinfo["atom_decoder"])
    dataloaders, _ = qm9dataset.retrieve_dataloaders(margs)
    if getattr(margs, "train_diffusion", False):   # GeoLDM latent diffusion
        from qm9.models import get_latent_diffusion as _build
    else:                                          # EDM (or GeoLDM AE)
        _build = get_model
    model, nodes_dist, _ = _build(margs, device, dinfo, dataloaders["train"])
    fn = "generative_model_ema.npy" if getattr(margs, "ema_decay", 0) > 0 else "generative_model.npy"
    model.load_state_dict(torch.load(Path(a.model_path) / fn, map_location=device))
    model = model.to(device).eval()

    ds = OMol25EDMDataset(a.datadir, a.split, n_species=n_species)
    rng = np.random.default_rng(0)
    records = []
    for g in range(min(a.n_molecules, len(ds))):
        it = ds[g]
        pos0 = it["positions"].numpy()
        N = pos0.shape[0]
        Z = (it["one_hot"].argmax(1).numpy() + 1).tolist()
        xs = [pos0] + [pos0 + rng.normal(0, a.sigma, pos0.shape) for _ in range(a.n_perturb - 1)]
        xs = np.stack(xs)                                    # (K,N,3)
        M = a.n_perturb
        x = torch.tensor(xs, dtype=torch.float32, device=device)
        oh = it["one_hot"].unsqueeze(0).repeat(M, 1, 1).to(device).float()
        ch = it["charges"].unsqueeze(0).repeat(M, 1).unsqueeze(2).to(device).float()
        node_mask = torch.ones(M, N, 1, device=device)
        x = remove_mean_with_mask(x, node_mask)
        em = node_mask.squeeze(2).unsqueeze(1) * node_mask.squeeze(2).unsqueeze(2)
        em = (em * (~torch.eye(N, dtype=torch.bool, device=device)).unsqueeze(0)).view(M * N * N, 1)
        h = {"categorical": oh, "integer": ch}
        # average -NLL over several random t to reduce the diffusion-loss variance
        acc = torch.zeros(M, device=device)
        with torch.no_grad():
            for _ in range(a.n_t_avg):
                acc += -model(x, h, node_mask, em, context=None)
        logp = (acc / a.n_t_avg).cpu().numpy()
        for p in range(M):
            records.append({"group_id": g, "pert_id": p, "atomic_numbers": Z,
                            "positions": xs[p].tolist(), "charge": 0, "spin": 1,
                            "log_p_theta": float(logp[p])})
        print(f"  group {g}: N={N}  logp range [{logp.min():.1f},{logp.max():.1f}]", flush=True)
    json.dump(records, open(a.out, "w"))
    print(f"wrote {a.out}: {len(records)} records, {min(a.n_molecules,len(ds))} groups")


if __name__ == "__main__":
    main()
