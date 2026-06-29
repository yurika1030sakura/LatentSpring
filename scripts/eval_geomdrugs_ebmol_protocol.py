"""Tier 2: GEOM-Drugs generation benchmark under EBMol / GEOM-Drugs Revisited protocol.

Compares BGFM against EBMol on GEOM-Drugs at matched NFE budgets:
    1080, 1960, 3720, 7240.

Metrics:
    atom stability, molecule stability, validity, uniqueness,
    novelty, Vendi diversity, bond length / angle / torsion distributions
    on the valid-and-connected subset.

References:
    EBMol GEOM-Drugs Revisited reported numbers (Diversity, mol-stab,
    validity, etc.) from Section 1.2 of notes/EBMOL_BENCHMARK_PLAN_2026-06-29_CN.md.

Target claim: at matched molecule stability, BGFM preserves higher
diversity; or at matched diversity, BGFM achieves lower xTB relaxation
energy (see scripts/eval_xtb_relaxation.py).

Status: TEMPLATE.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--n_samples", type=int, default=10000)
    ap.add_argument("--nfe", type=int, default=1960,
                    choices=[1080, 1960, 3720, 7240])
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raise NotImplementedError(
        "Tier 2 (GEOM-Drugs EBMol protocol) is scaffolded but not yet wired. "
        "Implementation plan: "
        "(1) sample n_samples molecules at args.nfe ODE steps; "
        "(2) OpenBabel bond inference; "
        "(3) RDKit sanitization with GEOM-Drugs Revisited valency tables; "
        "(4) compute Vendi diversity (https://github.com/vertaix/Vendi-Score); "
        "(5) compute bond/angle/torsion distribution differences vs. data; "
        "(6) write JSON/CSV and Pareto-plot inputs to args.out_dir."
    )


if __name__ == "__main__":
    sys.exit(main())
