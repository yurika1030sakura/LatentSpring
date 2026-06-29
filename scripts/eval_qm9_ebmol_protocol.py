"""Tier 1: QM9 generation benchmark under EBMol protocol.

Compares a BGFM checkpoint against EBMol on the QM9 standard metrics:
    atom stability, molecule stability, validity, uniqueness, valid&unique,
    novelty, NFE-matched.

Usage:
    python scripts/eval_qm9_ebmol_protocol.py \
        --checkpoint <bgfm.ckpt> \
        --config <bgfm.yaml> \
        --n_samples 10000 \
        --nfe 1810 \
        --out_dir runs/eval/qm9_ebmol

References:
    EBMol QM9 protocol: 10k samples, OpenBabel bond inference,
    RDKit sanitization, NFE budgets {930, 1370, 1810}.
    EBMol1810 reports: atom stab 99.76, mol stab 97.09, validity 98.93,
    uniqueness 82.30, novelty 42.08.

Target claim: BGFM matches stability/validity while preserving higher
uniqueness and novelty (FM backbones don't collapse like EBMs at high NFE).

Status: TEMPLATE -- the QM9 metric pipeline is fairly standard
(see EDM, MiDi, FlowMol3 code releases for reference implementations).
We will wire up the OpenBabel + RDKit + EBMol-revised valency tables
when the EBMol public release lands.
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
    ap.add_argument("--nfe", type=int, default=1810,
                    help="NFE budget. Match EBMol's reported settings: 930, 1370, 1810.")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raise NotImplementedError(
        "Tier 1 (QM9 EBMol protocol) is scaffolded but not yet wired. "
        "Implementation plan: "
        "(1) load BGFM checkpoint; "
        "(2) sample n_samples molecules using args.nfe ODE steps; "
        "(3) convert to RDKit Mol via OpenBabel bond inference; "
        "(4) compute atom stab, mol stab, validity, uniqueness, novelty, "
        "    valid&unique; "
        "(5) write JSON + CSV to args.out_dir; "
        "(6) cross-tabulate against EBMol-published numbers. "
        "Awaiting EBMol public release for exact protocol parity."
    )


if __name__ == "__main__":
    sys.exit(main())
