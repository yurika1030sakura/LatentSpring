"""OMol25 dataset adapter: AseDBDataset -> FlowMol3-style DGL graph.

OMol25 stores molecules as `ase.Atoms` objects in an ASE LMDB DB. Each entry
provides positions, atomic_numbers, and metadata (charge/spin/energy/forces).
It does NOT store bond orders, so we infer connectivity from covalent radii
via `ase.neighborlist.natural_cutoffs` (tolerance 1.2x). All inferred bonds
are single bonds (order 1). Multi-bond perception could be added with
openbabel/rdkit but is slow at 4M-100M scale; the geometry itself encodes
multi-bond character which the position/energy flow learns.

Each __getitem__ returns a DGL graph whose layout matches
MoleculeDataset (baselines/flowmol3/flowmol/data_processing/dataset.py):
    g.ndata['x_1_true']   positions (N, 3), COM-centered
    g.ndata['a_1_true']   atom type one-hot (N, n_atom_types[+1 if fake])
    g.ndata['c_1_true']   charge one-hot (N, 6) -- charge + 2 in [0, 5]
    g.ndata['{x,a,c}_0']  prior samples (coupled_node_prior)
    g.edata['e_1_true']   bond type one-hot (2E_full, n_bond_types)
    g.edata['e_0']        bond prior sample

We use the same fully-connected-pair edge layout (upper + lower triangle,
0 = no-bond).

This lets training/val loops from FlowMol3 run unchanged.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import dgl
import pytorch_lightning as pl
from torch.utils.data import DataLoader


# OMol25 paper: 83 unique elements in coverage (H..Bi).
DEFAULT_ATOM_MAP_OMOL25: List[str] = [
    "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca",
    "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi",
]
_SYM_LIST = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi",
]
_SYM_TO_Z = {s: i + 1 for i, s in enumerate(_SYM_LIST)}


def _build_z_to_map_idx(atom_map: List[str]) -> dict:
    return {_SYM_TO_Z[s]: i for i, s in enumerate(atom_map) if s in _SYM_TO_Z}


class OMol25Dataset(torch.utils.data.Dataset):
    """Streaming AseDBDataset -> FlowMol3-style DGL graph.

    Lazy: loads one `ase.Atoms` at a time. Memory stays O(batch), independent
    of dataset size (works for 4M or 100M).

    Args
    ----
    src : path or list of paths to LMDB shards (string or list).
    atom_map : list[str], optional
        Defaults to DEFAULT_ATOM_MAP_OMOL25 (83 elements).
    max_atoms : int
        Skip molecules with more atoms than this (default 120 for 4M; 200 for
        100M). Also excludes N < 2.
    fake_atom_p : float
    fake_atom_std : float
        Set >0 to match FlowMol3's fake-atom augmentation.
    explicit_aromaticity : bool
    bond_tolerance : float
        Multiplier for natural_cutoffs (default 1.2).
    prior_config : dict
        FlowMol3 prior config (same key set as `MoleculeDataset.prior_config`).
    """

    def __init__(
        self,
        src,
        atom_map: List[str] = None,
        max_atoms: int = 120,
        fake_atom_p: float = 0.0,
        fake_atom_std: float = 1.0,
        explicit_aromaticity: bool = False,
        bond_tolerance: float = 1.2,
        prior_config: Optional[dict] = None,
    ):
        try:
            from fairchem.core.datasets import AseDBDataset
        except Exception as e:
            raise ImportError(
                "fairchem-core is required to read OMol25 LMDB shards. "
                "Install it in your training env (envs/flowmol) via: "
                "pip install fairchem-core==2.19  (torch 2.5+ required)."
            ) from e
        self._src = src
        self._ase_db = AseDBDataset({"src": str(src) if not isinstance(src, list) else src})
        self.atom_map = list(atom_map or DEFAULT_ATOM_MAP_OMOL25)
        self.n_atom_types = len(self.atom_map)
        self.max_atoms = int(max_atoms)
        self.fake_atom_p = float(fake_atom_p)
        self.fake_atom_std = float(fake_atom_std)
        self.use_fake_atoms = self.fake_atom_p > 0
        self.explicit_aromaticity = bool(explicit_aromaticity)
        self.n_bond_types = 5 if self.explicit_aromaticity else 4
        self.bond_tolerance = float(bond_tolerance)
        self.prior_config = prior_config or {}
        self._z_to_idx = _build_z_to_map_idx(self.atom_map)

    def __len__(self) -> int:
        return len(self._ase_db)

    # ------------------------------------------------------------------
    # FlowMol3 graph builder
    # ------------------------------------------------------------------
    def _bonds_from_positions(self, atoms) -> np.ndarray:
        """Return (E, 2) int64 array of UNIQUE undirected bonds (i < j) using
        covalent-radius connectivity. All bonds are order=1 (single)."""
        from ase.neighborlist import natural_cutoffs, neighbor_list
        cutoffs = natural_cutoffs(atoms, mult=self.bond_tolerance)
        i, j = neighbor_list("ij", atoms, cutoffs)
        # Deduplicate by keeping i < j only.
        mask = i < j
        if not mask.any():
            return np.zeros((0, 2), dtype=np.int64)
        return np.stack([i[mask], j[mask]], axis=1).astype(np.int64)

    def __getitem__(self, idx: int):
        # Import here so workers pick up their own ASE handles.
        from flowmol.data_processing.priors import coupled_node_prior, edge_prior

        try:
            atoms = self._ase_db.get_atoms(idx)
        except Exception:
            return self._empty_graph()

        n = len(atoms)
        if n < 2 or n > self.max_atoms:
            return self._empty_graph()

        z = atoms.get_atomic_numbers()
        type_idx = np.array(
            [self._z_to_idx.get(int(zi), -1) for zi in z], dtype=np.int64,
        )
        if (type_idx < 0).any():
            return self._empty_graph()

        pos = atoms.get_positions().astype(np.float32)
        pos = pos - pos.mean(axis=0, keepdims=True)

        charge_total = int(atoms.info.get("charge", 0))

        bond_idxs = self._bonds_from_positions(atoms)
        if bond_idxs.shape[0] == 0:
            return self._empty_graph()

        # Per-atom formal charge: OMol25 stores only total charge. We don't
        # have per-atom partitioning, so default to 0 with total shifted to
        # atom 0. This is a minor approximation; FlowMol3 uses formal charges
        # only as a conditioning signal.
        per_atom_charge = np.zeros(n, dtype=np.int64)
        if charge_total != 0:
            per_atom_charge[0] = charge_total

        # Tensors.
        positions = torch.from_numpy(pos)
        atom_types = F.one_hot(torch.from_numpy(type_idx),
                                num_classes=self.n_atom_types).float()
        per_atom_charge = np.clip(per_atom_charge, -2, 3)
        atom_charges = F.one_hot(torch.from_numpy(per_atom_charge) + 2,
                                  num_classes=6).float()

        # Fake atoms (optional, matches FlowMol3).
        if self.use_fake_atoms:
            max_n_fake = int(np.ceil(n * self.fake_atom_p))
            n_fake = int(torch.randint(0, max_n_fake + 1, (1,)).item()) if max_n_fake > 0 else 0
            if n_fake > 0:
                anchors = torch.randint(0, n, (n_fake,))
                fake_pos = positions[anchors] + torch.randn(n_fake, 3) * self.fake_atom_std
                positions = torch.cat([positions, fake_pos], dim=0)
                atom_types = torch.cat([
                    atom_types,
                    torch.zeros(n_fake, self.n_atom_types),
                ], dim=0)
                # extend with a fake-atom column
                atom_types = torch.cat([
                    atom_types, torch.zeros(atom_types.shape[0], 1),
                ], dim=1)
                atom_types[-n_fake:, -1] = 1
                atom_charges = torch.cat([
                    atom_charges,
                    F.one_hot(torch.full((n_fake,), 2, dtype=torch.long),
                              num_classes=6).float(),
                ], dim=0)
            else:
                # still add the fake-atom column for consistency
                atom_types = torch.cat([
                    atom_types, torch.zeros(atom_types.shape[0], 1),
                ], dim=1)
            n_total = positions.shape[0]
        else:
            n_total = n

        # Edge construction: fully connected upper+lower triangle, labeled
        # with bond type at each (i,j) and 0 elsewhere.
        adj = torch.zeros((n_total, n_total), dtype=torch.int32)
        bi = torch.from_numpy(bond_idxs)
        # Only accept bonds among real atoms (fake atoms are unbonded).
        valid = (bi[:, 0] < n) & (bi[:, 1] < n)
        bi = bi[valid]
        if bi.shape[0] > 0:
            # Bond order = 1 (single). FlowMol3 index: 1 = single in 4-class,
            # 1 = single in 5-class (0=none,1=single,2=double,3=triple,4=aromatic).
            adj[bi[:, 0], bi[:, 1]] = 1

        upper_edge_idxs = torch.triu_indices(n_total, n_total, offset=1)
        upper_edge_labels = adj[upper_edge_idxs[0], upper_edge_idxs[1]]
        lower_edge_idxs = torch.stack([upper_edge_idxs[1], upper_edge_idxs[0]])
        edges = torch.cat([upper_edge_idxs, lower_edge_idxs], dim=1)
        edge_labels = torch.cat([upper_edge_labels, upper_edge_labels])

        edge_labels = F.one_hot(edge_labels.to(torch.int64),
                                num_classes=self.n_bond_types).float()

        g = dgl.graph((edges[0], edges[1]), num_nodes=n_total)
        g.edata["e_1_true"] = edge_labels
        g.ndata["x_1_true"] = positions
        g.ndata["a_1_true"] = atom_types
        g.ndata["c_1_true"] = atom_charges

        # Priors.
        dst = {"x": positions, "a": atom_types, "c": atom_charges}
        prior_node_feats = coupled_node_prior(dst_dict=dst, prior_config=self.prior_config)
        for feat, val in prior_node_feats.items():
            g.ndata[f"{feat}_0"] = val
        upper_mask = torch.zeros(g.num_edges(), dtype=torch.bool)
        upper_mask[:upper_edge_idxs.shape[1]] = True
        g.edata["e_0"] = edge_prior(upper_mask, self.prior_config["e"],
                                     explicit_aromaticity=self.explicit_aromaticity)
        return g

    def _empty_graph(self) -> dgl.DGLGraph:
        """Return a minimal valid 2-node graph (collated but discarded)."""
        n = 2
        n_a = self.n_atom_types + (1 if self.use_fake_atoms else 0)
        g = dgl.graph(([0, 1], [1, 0]), num_nodes=n)
        g.ndata["x_1_true"] = torch.zeros(n, 3)
        g.ndata["a_1_true"] = F.one_hot(torch.zeros(n, dtype=torch.long),
                                         num_classes=n_a).float()
        g.ndata["c_1_true"] = F.one_hot(torch.full((n,), 2, dtype=torch.long),
                                         num_classes=6).float()
        g.ndata["x_0"] = torch.randn(n, 3)
        g.ndata["a_0"] = F.one_hot(torch.zeros(n, dtype=torch.long),
                                    num_classes=n_a).float()
        g.ndata["c_0"] = F.one_hot(torch.full((n,), 2, dtype=torch.long),
                                    num_classes=6).float()
        g.edata["e_1_true"] = F.one_hot(torch.zeros(2, dtype=torch.long),
                                         num_classes=self.n_bond_types).float()
        g.edata["e_0"] = F.one_hot(torch.zeros(2, dtype=torch.long),
                                    num_classes=self.n_bond_types).float()
        return g

    @functools.cached_property
    def n_atoms_per_graph(self):
        """Required by AdaptiveEdgeSampler. Approximate with a constant --
        actual n_atoms is variable but AdaptiveEdgeSampler is used only when
        max_num_edges is set (we disable it for OMol25 via config)."""
        return torch.full((len(self),), self.max_atoms // 2, dtype=torch.long)

    @functools.cached_property
    def n_edges_per_graph(self):
        return self.n_atoms_per_graph.square()


class OMol25DataModule(pl.LightningDataModule):
    """LightningDataModule around OMol25Dataset. Mirrors MoleculeDataModule's
    interface so run_train.py can swap it in."""

    def __init__(
        self,
        dataset_config: dict,
        dm_prior_config: dict,
        batch_size: int,
        num_workers: int = 0,
        distributed: bool = False,
        max_num_edges: int = None,
    ):
        super().__init__()
        self.dataset_config = dataset_config
        self.prior_config = dm_prior_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed = distributed
        self.max_num_edges = max_num_edges  # ignored -- OMol25 can't adaptive-batch
        self.save_hyperparameters()

    def _make_dataset(self, src_key: str):
        src = self.dataset_config[src_key]
        return OMol25Dataset(
            src=src,
            atom_map=self.dataset_config["atom_map"],
            max_atoms=self.dataset_config.get("max_atoms", 120),
            fake_atom_p=self.dataset_config.get("fake_atom_p", 0.0),
            fake_atom_std=self.dataset_config.get("fake_atom_std", 1.0),
            explicit_aromaticity=self.dataset_config.get("explicit_aromaticity", False),
            bond_tolerance=self.dataset_config.get("bond_tolerance", 1.2),
            prior_config=self.prior_config,
        )

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = self._make_dataset("train_src")
            self.val_dataset = self._make_dataset("val_src")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=dgl.batch,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=max(1, self.num_workers // 2),
            collate_fn=dgl.batch,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )
