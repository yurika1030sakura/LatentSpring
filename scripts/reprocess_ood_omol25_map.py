"""Re-process the OOD eval sets (tmQM / kraken / hypervalent) into the OMol25
83-element atom_map index space, so an OMol25-trained model can actually be
evaluated on them.

Why this is needed: the pre-existing *_processed dirs were each built with their
OWN small atom_map (one-hot widths 29/14/10/12). An OMol25 model expects width 83
and a specific element->index ordering, so feeding it those tensors would either
crash on the dim mismatch or silently mislabel every atom.

Usage (flowmol env):
  python scripts/reprocess_ood_omol25_map.py --config configs/omol25_4m_bgfm_onpolicy_v3_energy.yaml \
      --out_root /n/holylabs/woo_lab/Lab/yulili/bgfm/processed_data/ood_omol25map
"""
from __future__ import annotations
import argparse, sys, traceback
from pathlib import Path
import yaml

REF = Path("/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/data")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True,
                    help="training config; its dataset.atom_map is the target index space")
    ap.add_argument("--out_root", type=Path, required=True)
    ap.add_argument("--only", default=None, help="comma list: tmqm_broad,kraken,hypervalent")
    ap.add_argument("--max_mols", type=int, default=20000)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    atom_map = list(cfg["dataset"]["atom_map"])
    max_atoms = int(cfg["dataset"].get("max_atoms", 200))
    print(f"target atom_map: {len(atom_map)} elements (OMol25 space). max_atoms={max_atoms}")

    jobs = {
        # name          module              raw_dir                 kwargs
        "tmqm_broad":   ("cfm_mol.data.tmqm",         REF / "tmqm",        dict(mode="broad", max_mols=a.max_mols)),
        "kraken":       ("cfm_mol.data.kraken",       REF / "kraken",      dict()),
        "hypervalent":  ("cfm_mol.data.hypervalent",  REF / "hypervalent", dict()),
    }
    if a.only:
        keep = {s.strip() for s in a.only.split(",")}
        jobs = {k: v for k, v in jobs.items() if k in keep}

    rc = 0
    for name, (mod, raw, kw) in jobs.items():
        out = a.out_root / name
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {name}: {raw} -> {out} ===", flush=True)
        if not raw.exists():
            print(f"  SKIP: raw dir missing {raw}"); rc = 1; continue
        try:
            m = __import__(mod, fromlist=["process"])
            # every adapter takes (raw_dir, processed_dir, atom_map, ...)
            try:
                m.process(raw, out, atom_map=atom_map, max_atoms=max_atoms, **kw)
            except TypeError:
                # adapters without max_atoms in their signature
                m.process(raw, out, atom_map=atom_map, **kw)
            print(f"  OK {name}")
        except Exception:
            traceback.print_exc()
            print(f"  FAILED {name}")
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
