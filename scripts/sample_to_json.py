"""Sample molecules from a FlowMol/BGFM checkpoint and export geometries to the
common JSON ([{atomic_numbers, positions, charge}]) used by the shared
xyz2mol-validity and GFN2-xTB-relaxation evaluators. This is the FlowMol-family
analogue of the baselines' sample_omol25_to_json.py, so all 5 models feed the
SAME downstream physical evaluators (fair cross-model comparison).

Run in envs/flowmol. Reuses the sampling helpers in scripts/evaluate_validity.py.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_validity import (_load_model, _load_size_distribution,
                               _sample_n_atoms, _sample_batch)


def _mol_to_record(m, atom_map=None):
    import ase.data
    at = m.atom_types
    if hasattr(at, "argmax"):                       # one-hot / logits tensor
        idx = at.argmax(dim=-1).cpu().numpy()
        syms = [atom_map[i] for i in idx] if atom_map else None
        if syms is None:
            raise ValueError("atom_types is a tensor but no atom_map provided")
    else:                                           # already symbol strings
        syms = list(at)
    Z = [int(ase.data.atomic_numbers[s]) for s in syms]
    pos = m.positions.detach().cpu().numpy().astype(np.float64)
    return {"atomic_numbers": Z, "positions": pos.tolist(), "charge": 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--eval_data", type=Path, required=True,
                    help="dir with the n_atoms histogram (size distribution)")
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--n_timesteps", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--vanilla", action="store_true", help="skip patch_flowmol")
    ap.add_argument("--no-discrete", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from flowmol.model_utils.load import read_config_file
    cfg = read_config_file(str(a.config))
    model = _load_model(a.checkpoint, cfg, apply_patch=not a.vanilla,
                        discrete_projection=not a.no_discrete).to(a.device)
    dist = _load_size_distribution(a.eval_data)

    records, remaining = [], a.n_samples
    while remaining > 0:
        nb = min(a.batch_size, remaining)
        mols = _sample_batch(model, _sample_n_atoms(nb, dist), a.n_timesteps, a.device)
        for m in mols:
            try:
                records.append(_mol_to_record(m))
            except Exception as e:
                print(f"  [skip] {e}", flush=True)
        remaining -= nb
        print(f"  sampled {len(records)}/{a.n_samples}", flush=True)
    json.dump(records, open(a.out, "w"))
    print(f"wrote {a.out}: {len(records)} molecules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
