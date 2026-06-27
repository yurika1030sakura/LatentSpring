"""OMol25 dataset for Symphony (bond-free, 83 elements incl. transition metals).

Reads a torch-free .npz (produced by omol25_to_npz.py) and yields
datatypes.Structures with NodesInfo(positions, species). species index =
OMol25 atom-type index 0..82; atomic number = index + 1 (contiguous Z 1..83).
I/O only — no model/architecture change.
"""
from typing import Dict, Iterable, List
import os
import numpy as np

from symphony.data import datasets
from symphony import datatypes

ATOM_MAP: List[str] = [
    "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca",
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr",
    "Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd",
    "Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "Tl","Pb","Bi",
]
N_SPECIES = len(ATOM_MAP)


class OMol25Dataset(datasets.InMemoryDataset):
    """In-memory OMol25 dataset backed by a precomputed .npz."""

    def __init__(self, npz_path: str, num_train_molecules: int,
                 num_val_molecules: int, num_test_molecules: int):
        super().__init__()
        self.npz_path = npz_path
        self.nt = num_train_molecules
        self.nv = num_val_molecules
        self.nte = num_test_molecules
        self.all_structures = None
        self._splits = None

    @property
    def num_species(self) -> int:
        return N_SPECIES

    @staticmethod
    def get_atomic_numbers() -> np.ndarray:
        return np.arange(1, N_SPECIES + 1)

    @staticmethod
    def species_to_atomic_numbers() -> Dict[int, int]:
        return {i: i + 1 for i in range(N_SPECIES)}

    @staticmethod
    def atoms_to_species() -> Dict[str, int]:
        return {s: i for i, s in enumerate(ATOM_MAP)}

    def _load(self):
        if self.all_structures is not None:
            return
        if not os.path.exists(self.npz_path):
            raise FileNotFoundError(
                f"{self.npz_path} not found — run omol25_to_npz.py first.")
        z = np.load(self.npz_path)
        self.all_structures = []
        self._splits = {}
        offset = 0
        for split in ["train", "val", "test"]:
            pos = z[f"{split}_pos"]
            sp = z[f"{split}_species"]
            ni = z[f"{split}_nodeidx"]
            idxs = []
            for s, e in ni:
                s, e = int(s), int(e)
                st = datatypes.Structures(
                    nodes=datatypes.NodesInfo(
                        positions=pos[s:e].astype(np.float32),
                        species=sp[s:e].astype(np.int32),
                    ),
                    edges=None, receivers=None, senders=None, globals=None,
                    n_node=np.asarray([e - s]), n_edge=None,
                )
                self.all_structures.append(st)
                idxs.append(offset)
                offset += 1
            self._splits[split] = np.asarray(idxs)

    def structures(self) -> Iterable[datatypes.Structures]:
        self._load()
        return self.all_structures

    def split_indices(self) -> Dict[str, np.ndarray]:
        self._load()
        return {
            "train": self._splits["train"][: self.nt],
            "val": self._splits["val"][: self.nv],
            "test": self._splits["test"][: self.nte],
        }
