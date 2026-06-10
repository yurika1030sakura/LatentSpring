"""Write a permissive `train_data_valencies_omol25.json` for SampleAnalyzer.

SampleAnalyzer (flowmol.analysis.metrics) requires a valencies JSON at
model init. OMol25 covers 83 elements across diverse oxidation states;
a strict valence table would need chemistry-expert curation. We instead
write a permissive table (any valence 0..8 is accepted for every element
and charge). The `frac_atoms_stable` metric SampleAnalyzer reports
during training is therefore uninformative for OMol25; post-hoc OMol25
energy evaluation is the real quality metric.

Usage:
  python scripts/write_omol25_valencies.py --out_dir <path-to-processed-dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_ATOM_MAP_OMOL25 = [
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--atom_map", default=None,
                    help="Comma-sep overriding atom map. Default = 83 elems.")
    args = ap.parse_args()

    atom_map = args.atom_map.split(",") if args.atom_map else list(DEFAULT_ATOM_MAP_OMOL25)
    permissive = {
        s: {str(c): list(range(9)) for c in range(-2, 4)}
        for s in atom_map
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.out_dir / "train_data_valencies_omol25.json"
    with open(out_file, "w") as f:
        json.dump(permissive, f)
    print(f"wrote {out_file} ({len(atom_map)} elements)")


if __name__ == "__main__":
    main()
