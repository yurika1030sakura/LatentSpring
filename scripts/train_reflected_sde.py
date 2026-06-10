"""Train the reflected-SDE coordinate-channel baseline on QM9 for E2.

Loads QM9 processed coords + atom types, discards bond info (coord-only).
Trains a CoordScoreNet with denoising score matching + retraction on the
steric fibre. Checkpoints go under runs/reflected_sde/.

Usage:
    PYTHONPATH=. python scripts/train_reflected_sde.py \\
        --processed_dir data/qm9_processed \\
        --epochs 50 --batch_size 64 \\
        --out runs/reflected_sde
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from cfm_mol.baselines.reflected_sde import CoordScoreNet, ReflectedSDEModel
from cfm_mol.domain import default_d_min_table


class QM9CoordDataset(Dataset):
    """Minimal dataset for E2: positions + one-hot atom types. We pad each
    molecule to a fixed N_max so batching is trivial. d_min is only applied
    on real atoms (padding gets a sentinel atom_type we set to 0 with
    zero-d_min later)."""

    def __init__(self, processed_dir: Path, n_atom_types: int = 5,
                 n_max: int = 30, split: str = "train"):
        data = torch.load(processed_dir / f"{split}_data_processed.pt")
        positions = data["positions"]                     # (N_total, 3)
        atom_types = data["atom_types"]                   # (N_total, A)
        if atom_types.dim() == 2:
            atom_types = atom_types.argmax(dim=-1)        # (N_total,)
        node_idx = data["node_idx_array"]                 # (n_mols, 2)

        self.n_max = n_max
        self.n_atom_types = n_atom_types
        self.positions = []
        self.atom_types = []
        self.lengths = []
        for m in range(node_idx.shape[0]):
            s, e = int(node_idx[m, 0]), int(node_idx[m, 1])
            n = e - s
            if n > n_max:
                continue
            pos = positions[s:e] - positions[s:e].mean(dim=0, keepdim=True)
            a = atom_types[s:e]
            pad_pos = torch.zeros(n_max, 3)
            pad_a = torch.zeros(n_max, dtype=torch.long)
            pad_pos[:n] = pos
            pad_a[:n] = a
            self.positions.append(pad_pos)
            self.atom_types.append(pad_a)
            self.lengths.append(n)

        self.positions = torch.stack(self.positions)      # (M, n_max, 3)
        self.atom_types = torch.stack(self.atom_types)    # (M, n_max)
        self.lengths = torch.tensor(self.lengths)

    def __len__(self):
        return self.positions.shape[0]

    def __getitem__(self, idx):
        return {
            "r": self.positions[idx],
            "a": self.atom_types[idx],
            "n": self.lengths[idx],
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n_atom_types", type=int, default=5)
    ap.add_argument("--n_max", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"loading QM9 from {args.processed_dir} ...")
    ds_train = QM9CoordDataset(args.processed_dir, args.n_atom_types,
                                args.n_max, split="train")
    ds_val = QM9CoordDataset(args.processed_dir, args.n_atom_types,
                              args.n_max, split="val")
    print(f"  train: {len(ds_train)} mols, val: {len(ds_val)} mols")

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=2)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                         num_workers=2)

    d_min = default_d_min_table(args.n_atom_types)
    score_net = CoordScoreNet(n_atom_types=args.n_atom_types)
    model = ReflectedSDEModel(score_net, d_min).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"training {args.epochs} epochs on {args.device} ...")
    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total, n = 0.0, 0
        for batch in dl_train:
            r = batch["r"].to(args.device)
            a = batch["a"].to(args.device)
            t = torch.rand(r.shape[0], device=args.device)
            loss = model.training_loss(r, a, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * r.shape[0]
            n += r.shape[0]
        train_loss = total / n

        model.eval()
        vtotal, vn = 0.0, 0
        with torch.no_grad():
            for batch in dl_val:
                r = batch["r"].to(args.device)
                a = batch["a"].to(args.device)
                t = torch.rand(r.shape[0], device=args.device)
                loss = model.training_loss(r, a, t)
                vtotal += float(loss.item()) * r.shape[0]
                vn += r.shape[0]
        val_loss = vtotal / vn

        print(f"  epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.out / "best.pt")

    torch.save(model.state_dict(), args.out / "last.pt")
    print(f"done. best_val={best_val:.4f}. checkpoints in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
