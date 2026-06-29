"""Tier 3: Independent physical-oracle benchmark via GFN2-xTB relaxation.

This is the most important table for the paper: GFN2-xTB was NOT used in
any BGFM training loss, so a quality gain here cannot be attributed to
training-evaluator overlap.

Inputs:
    samples.json  -- one record per generated molecule:
                       {atomic_numbers, positions, charge, spin}
                     produced by scripts/eval_qm9_ebmol_protocol.py or
                     scripts/eval_geomdrugs_ebmol_protocol.py.

Computes per molecule:
    1. xTB relaxation energy delta:  E_before - E_after
    2. max |force| before relaxation (eV/A)
    3. RMSD after relaxation (A)
    4. number of xTB optimizer steps to convergence
    5. relaxation failure rate (xTB SCF/geom fail)
    6. bond length / angle / torsion distributions (vs. held-out reference)

Comparator: EBMol's reported median xTB ΔE on GEOM-Drugs is 1.78 kcal/mol
at NFE=7240 and 4.12 kcal/mol at NFE=1960.

Status: TEMPLATE. Requires xtb python bindings (xtb-python) or subprocess
calls to the `xtb` binary. Recommend running on `envs/omol25` since that
env already has DFT/QC dependencies.

Example invocation (planned):

    python scripts/eval_xtb_relaxation.py \
        --samples runs/eval/geomdrugs_ebmol/samples.json \
        --out_csv runs/eval/geomdrugs_ebmol/xtb_relax.csv \
        --max_steps 200 \
        --n_workers 8

For the cross-model ranking table (Tier 5), run this script on samples
from EDM, FlowMol3, EBMol, and BGFM and pool the outputs.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, required=True,
                    help="JSON list of {atomic_numbers, positions, charge, spin}.")
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--max_steps", type=int, default=200,
                    help="Max xTB relaxation steps.")
    ap.add_argument("--energy_threshold_evA", type=float, default=0.05,
                    help="Convergence: ||F||_inf < this.")
    ap.add_argument("--n_workers", type=int, default=4)
    ap.add_argument("--xtb_binary", type=str, default="xtb")
    args = ap.parse_args()

    raise NotImplementedError(
        "Tier 3 (xTB relaxation) is scaffolded but not yet wired. "
        "Implementation: "
        "(1) parse samples JSON; "
        "(2) for each: write .xyz, call xtb (or xtb-python) to relax; "
        "(3) parse final energy, RMSD, step count, failure flag; "
        "(4) compute structural distributions on the valid-and-connected subset; "
        "(5) write per-molecule rows + summary to args.out_csv. "
        "Use --n_workers for parallelization; xTB is single-threaded per call."
    )


if __name__ == "__main__":
    sys.exit(main())
