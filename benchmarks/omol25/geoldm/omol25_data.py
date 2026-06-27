"""OMol25 -> EDM dataset adapter (bond-free, I/O only; no architecture change).

Yields per-molecule dicts in the exact format EDM's PreprocessQM9.collate_fn
consumes: positions (N,3), one_hot (N, n_species), charges (N,) = nuclear charge
Z (= atom-type index + 1, since the OMol25 atom map is contiguous Z 1..83).
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class OMol25EDMDataset(Dataset):
    def __init__(self, processed_dir, split, n_species=83):
        d = torch.load(f"{processed_dir}/{split}_data_processed.pt")
        self.positions = d["positions"].float()           # (total, 3)
        self.atom_types = d["atom_types"].long()           # (total,) idx 0..82
        self.node_idx = d["node_idx_array"].long()         # (n_mol, 2)
        self.n_species = int(n_species)

    def __len__(self):
        return self.node_idx.shape[0]

    def __getitem__(self, i):
        s, e = int(self.node_idx[i, 0]), int(self.node_idx[i, 1])
        pos = self.positions[s:e]
        idx = self.atom_types[s:e]
        one_hot = F.one_hot(idx, num_classes=self.n_species).float()
        charges = (idx + 1).long()                          # nuclear charge Z
        return {
            "num_atoms": torch.tensor(e - s),
            "positions": pos,
            "one_hot": one_hot,
            "charges": charges,
        }
