"""Sample molecules from an EDM/GeoLDM OMol25 checkpoint and export to a common
JSON ([{atomic_numbers, positions}]) for the shared xyz2mol validity metric.
Run from baselines/edm (or baselines/geoldm) in the edm env."""
import argparse, json, pickle
from pathlib import Path
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--datadir", required=True)
    ap.add_argument("--n_samples", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from configs.datasets_config import get_dataset_info
    from qm9 import dataset as qm9dataset
    from qm9.models import get_model
    from qm9.sampling import sample

    device = torch.device(a.device)
    with open(Path(a.model_path) / "args.pickle", "rb") as f:
        margs = pickle.load(f)
    margs.datadir = a.datadir
    if not hasattr(margs, "normalization_factor"): margs.normalization_factor = 1
    if not hasattr(margs, "aggregation_method"): margs.aggregation_method = "sum"
    dinfo = get_dataset_info(margs.dataset, margs.remove_h)
    dataloaders, _ = qm9dataset.retrieve_dataloaders(margs)
    if getattr(margs, "train_diffusion", False):   # GeoLDM latent diffusion
        from qm9.models import get_latent_diffusion as _build
    else:
        _build = get_model
    model, nodes_dist, prop_dist = _build(margs, device, dinfo, dataloaders["train"])
    fn = "generative_model_ema.npy" if getattr(margs, "ema_decay", 0) > 0 else "generative_model.npy"
    model.load_state_dict(torch.load(Path(a.model_path) / fn, map_location=device))
    model = model.to(device).eval()

    nodesxsample = nodes_dist.sample(a.n_samples)
    one_hot, charges, x, node_mask = sample(margs, device, model, dinfo,
                                            prop_dist=None, nodesxsample=nodesxsample)
    one_hot = one_hot.cpu(); x = x.cpu(); node_mask = node_mask.cpu().squeeze(2).bool()
    records = []
    for i in range(a.n_samples):
        m = node_mask[i]
        Z = (one_hot[i][m].argmax(1).numpy() + 1).tolist()
        pos = x[i][m].numpy().tolist()
        records.append({"atomic_numbers": Z, "positions": pos, "charge": 0})
    json.dump(records, open(a.out, "w"))
    print(f"wrote {a.out}: {len(records)} sampled molecules")


if __name__ == "__main__":
    main()
