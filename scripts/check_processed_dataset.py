"""Check FlowMol3/BGFM processed datasets before launching GPU training.

Examples:
    PYTHONPATH=. python scripts/check_processed_dataset.py \\
        --config configs/omol25_4m_bgfm.yaml --split train --batches 2

    PYTHONPATH=. python scripts/check_processed_dataset.py \\
        --config configs/geom_bgfm.yaml --split train --batches 1
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import yaml


REPO = Path(__file__).resolve().parents[1]
FLOWMOL = REPO / "baselines" / "flowmol3"
if str(FLOWMOL) not in sys.path:
    sys.path.insert(0, str(FLOWMOL))


def _load_cfg(path: Path) -> dict:
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def _torch_load_metadata(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _single_file_summary(data_path: Path) -> dict:
    data = _torch_load_metadata(data_path)
    node_idx = data["node_idx_array"]
    n_atoms = (node_idx[:, 1] - node_idx[:, 0]).long()
    edge_idx = data.get("edge_idx_array")
    if edge_idx is not None:
        stored_edges = (edge_idx[:, 1] - edge_idx[:, 0]).long()
    else:
        stored_edges = torch.zeros_like(n_atoms)
    return {
        "n_molecules": int(node_idx.shape[0]),
        "n_atoms_total": int(data["positions"].shape[0]),
        "n_atoms_mean": float(n_atoms.float().mean().item()),
        "n_atoms_max": int(n_atoms.max().item()),
        "stored_edges_total": int(data.get("bond_idxs", torch.empty(0, 2)).shape[0]),
        "stored_edges_mean": float(stored_edges.float().mean().item()),
        "has_forces": "forces" in data,
        "forces_shape": tuple(data["forces"].shape) if "forces" in data else None,
        "has_energies": "energies" in data,
        "energies_shape": tuple(data["energies"].shape) if "energies" in data else None,
        "atom_types_shape": tuple(data["atom_types"].shape),
        "atom_charges_shape": tuple(data["atom_charges"].shape),
    }


def _print_summary(summary: dict) -> None:
    for key, value in summary.items():
        print(f"[dataset] {key}: {value}", flush=True)


def _check_static_files(processed_dir: Path) -> None:
    for name in (
        "train_data_marginal_dists.pt",
        "train_data_n_atoms_histogram.pt",
    ):
        path = processed_dir / name
        status = "OK" if path.exists() else "MISSING"
        print(f"[dataset] {name}: {status}", flush=True)
    valencies = sorted(p.name for p in processed_dir.glob("train_data_valencies_*.json"))
    print(f"[dataset] valencies files: {valencies if valencies else 'MISSING'}", flush=True)


def _inspect_first_batches(cfg: dict, split: str, batches: int, batch_size: int | None) -> None:
    import dgl
    from torch.utils.data import DataLoader
    from flowmol.data_processing.adaptive_sampler import AdaptiveEdgeSampler
    from flowmol.data_processing.dataset import MoleculeDataset

    dataset_config = copy.deepcopy(cfg["dataset"])
    dataset_config["fake_atom_p"] = cfg["mol_fm"].get("fake_atom_p", 0.0)
    dataset_config["fake_atom_std"] = cfg["mol_fm"].get("fake_atom_std", 1.0)
    dataset_config["explicit_aromaticity"] = cfg["mol_fm"].get("explicit_aromaticity", False)

    dataset = MoleculeDataset(split, dataset_config, prior_config=cfg["mol_fm"]["prior_config"])
    max_num_edges = cfg["training"].get("max_num_edges")
    if max_num_edges is not None:
        sampler = AdaptiveEdgeSampler(dataset, max_num_edges)
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=dgl.batch, num_workers=0)
        print(f"[dataset] dataloader: AdaptiveEdgeSampler(max_num_edges={max_num_edges})", flush=True)
    else:
        bs = int(batch_size or cfg["training"].get("batch_size", 1))
        loader = DataLoader(dataset, batch_size=bs, shuffle=False, collate_fn=dgl.batch, num_workers=0)
        print(f"[dataset] dataloader: batch_size={bs}", flush=True)

    for bi, graph in zip(range(batches), loader):
        nodes = graph.batch_num_nodes().detach().cpu().tolist()
        edges = graph.batch_num_edges().detach().cpu().tolist()
        print(
            f"[dataset] batch {bi}: graphs={len(nodes)} max_atoms={max(nodes)} "
            f"mean_atoms={sum(nodes)/max(len(nodes), 1):.1f} total_nodes={sum(nodes)} "
            f"max_edges={max(edges)} total_edges={sum(edges)}",
            flush=True,
        )
        print(
            f"[dataset] batch {bi}: has_force_1_true={'force_1_true' in graph.ndata} "
            f"has_energy_1_true={'energy_1_true' in graph.ndata} "
            f"has_e_1_true={'e_1_true' in graph.edata}",
            flush=True,
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--batches", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--no_first_batch", action="store_true",
                    help="Only inspect files; do not instantiate MoleculeDataset/DataLoader.")
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    processed_dir = Path(cfg["dataset"]["processed_data_dir"])
    print(f"[dataset] config: {args.config}", flush=True)
    print(f"[dataset] processed_dir: {processed_dir}", flush=True)
    print(f"[dataset] split: {args.split}", flush=True)
    if not processed_dir.exists():
        raise FileNotFoundError(f"processed_dir does not exist: {processed_dir}")

    _check_static_files(processed_dir)

    manifest_path = processed_dir / "train_data_shards_manifest.json"
    if args.split == "train" and manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        n_mol = sum(int(s.get("n_molecules", 0)) for s in manifest["shards"])
        n_atoms = sum(int(s.get("n_atoms", 0)) for s in manifest["shards"])
        print(f"[dataset] sharded train: n_shards={manifest.get('n_shards')} "
              f"n_molecules={n_mol} n_atoms={n_atoms} "
              f"mean_atoms={n_atoms / max(n_mol, 1):.2f}", flush=True)
        print("[dataset] note: current MoleculeDataset concatenates shards in memory; "
              "use --no_first_batch for 100M until lazy sharding is implemented.",
              flush=True)
    else:
        data_path = processed_dir / f"{args.split}_data_processed.pt"
        if not data_path.exists():
            raise FileNotFoundError(f"processed split file missing: {data_path}")
        _print_summary(_single_file_summary(data_path))

    if not args.no_first_batch:
        _inspect_first_batches(cfg, args.split, args.batches, args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
