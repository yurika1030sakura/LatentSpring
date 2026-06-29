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
    ap.add_argument("--max_atoms", type=int, default=50, help="only eval molecules with <= this many atoms (memory + parity with FlowMol eval)")
    ap.add_argument("--ref_json", default=None, help="reuse the exact molecules+perturbations+charge/spin from a FlowMol stage1 JSON; only recompute log_p_theta with this model (apples-to-apples across models)")
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

    # Build the list of groups to evaluate. Each group = one held-out molecule with
    # K perturbations. Two sources:
    #   --ref_json : reuse the EXACT molecules/perturbations/charge/spin from a FlowMol
    #                stage1 JSON (apples-to-apples cross-model comparison); recompute logp.
    #   else       : draw n_molecules (<= max_atoms) from the dataset, perturb in-script.
    groups = []   # each: {"gid", "Z"(list), "charge", "spin", "xs"(K,N,3), "pert_ids"(list)}
    if a.ref_json:
        ref = json.load(open(a.ref_json))
        from collections import OrderedDict
        by_g = OrderedDict()
        for rec in ref:
            by_g.setdefault(rec["group_id"], []).append(rec)
        for gid, recs in by_g.items():
            recs = sorted(recs, key=lambda r: r["pert_id"])
            xs = np.stack([np.asarray(r["positions"], dtype=np.float32) for r in recs])
            groups.append({"gid": gid, "Z": recs[0]["atomic_numbers"],
                           "charge": recs[0].get("charge", 0), "spin": recs[0].get("spin", 1),
                           "xs": xs, "pert_ids": [r["pert_id"] for r in recs]})
        print(f"[eval] reusing {len(groups)} groups from {a.ref_json}", flush=True)
    else:
        ds = OMol25EDMDataset(a.datadir, a.split, n_species=n_species)
        rng = np.random.default_rng(0)
        picked = 0
        for g in range(len(ds)):
            if picked >= a.n_molecules: break
            it = ds[g]
            pos0 = it["positions"].numpy()
            if pos0.shape[0] > a.max_atoms: continue
            Z = (it["one_hot"].argmax(1).numpy() + 1).tolist()
            xs = np.stack([pos0] + [pos0 + rng.normal(0, a.sigma, pos0.shape) for _ in range(a.n_perturb - 1)])
            groups.append({"gid": g, "Z": Z, "charge": 0, "spin": 1,
                           "xs": xs.astype(np.float32), "pert_ids": list(range(a.n_perturb))})
            picked += 1
        print(f"[eval] picked {len(groups)} molecules (<= {a.max_atoms} atoms) from {a.split}", flush=True)

    records = []
    for grp in groups:
        gid, Z = grp["gid"], grp["Z"]
        xs = grp["xs"]                                       # (K,N,3)
        M, N = xs.shape[0], xs.shape[1]
        oh1 = torch.zeros(N, n_species)
        oh1[torch.arange(N), torch.tensor([z - 1 for z in Z])] = 1.0
        x = torch.tensor(xs, dtype=torch.float32, device=device)
        oh = oh1.unsqueeze(0).repeat(M, 1, 1).to(device).float()
        ch = torch.tensor(Z, dtype=torch.float32).view(1, N, 1).repeat(M, 1, 1).to(device)
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
        for j, p in enumerate(grp["pert_ids"]):
            records.append({"group_id": gid, "pert_id": p, "atomic_numbers": Z,
                            "positions": xs[j].tolist(), "charge": grp["charge"], "spin": grp["spin"],
                            "log_p_theta": float(logp[j])})
        print(f"  group {gid}: N={N}  logp range [{logp.min():.1f},{logp.max():.1f}]", flush=True)
    json.dump(records, open(a.out, "w"))
    print(f"wrote {a.out}: {len(records)} records, {len(groups)} groups")


if __name__ == "__main__":
    main()
