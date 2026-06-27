"""Convert our torch-saved OMol25 processed tensors to a torch-free .npz that
Symphony's (JAX) env can read with numpy. Run with a torch env (e.g. flowmol):
    python omol25_to_npz.py <processed_dir> <out.npz>
"""
import sys
import numpy as np
import torch

proc, out = sys.argv[1], sys.argv[2]
data = {}
for split in ["train", "val", "test"]:
    d = torch.load(f"{proc}/{split}_data_processed.pt")
    data[f"{split}_pos"] = d["positions"].numpy().astype("float32")
    data[f"{split}_species"] = d["atom_types"].numpy().astype("int32")  # idx 0..82
    data[f"{split}_nodeidx"] = d["node_idx_array"].numpy().astype("int64")
np.savez(out, **data)
print("wrote", out, "| splits:",
      {s: int(data[f"{s}_nodeidx"].shape[0]) for s in ["train", "val", "test"]})
