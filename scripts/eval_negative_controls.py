"""Tier 6: Negative controls.

For the BGFM paper's causal claim to be defensible, training with
*shuffled* OMol25 labels must NOT lift the independent oracle (xTB)
metrics, even though it may still lift the OMol25 Tier 0 metric.

This script orchestrates the control runs. It does not train the model
itself; it generates the perturbation shards with shuffled labels and
records the shuffled-shard paths so that scripts/run_train.py
can be invoked against them.

Controls implemented:

    --shuffle_energy_within_parent      (energy values permuted among siblings)
    --shuffle_energy_across_parents     (energy values permuted across parents)
    --shuffle_force_within_atoms        (force vectors permuted among atoms)
    --shuffle_force_across_molecules    (force vectors permuted across molecules)
    --random_sign_force                 (random sign flip per atom)
    --wrong_kT_train                    (relabel kT to wrong value)

Status: TEMPLATE.

Usage (planned):

    # produce a shuffled shard:
    python scripts/eval_negative_controls.py \
        --in_shard /n/netscratch/.../perturbation_train_n30000_s0.pt \
        --out_shard /n/netscratch/.../perturbation_train_shuffled_energy_within_parent.pt \
        --shuffle_energy_within_parent

    # train BGFM against the shuffled shard:
    sbatch scripts/launch_omol25_bgfm_energy_h200.sh \
        configs/.../v8a_room_T.yaml \
        --override_bgfm_energy_perturbation_shards \
            /n/netscratch/.../perturbation_train_shuffled_energy_within_parent.pt

    # then run Tier 1-3 evals against the resulting checkpoint.

Expected result: shuffled-label BGFM may lift Tier 0 (because the loss
is still mathematically optimizable) but does NOT lift Tier 3
(independent xTB / DFT). The gap between Tier 0 and Tier 3 outcomes
is the causal evidence in the paper.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_shard", type=Path, required=True)
    ap.add_argument("--out_shard", type=Path, required=True)

    ap.add_argument("--shuffle_energy_within_parent", action="store_true")
    ap.add_argument("--shuffle_energy_across_parents", action="store_true")
    ap.add_argument("--shuffle_force_within_atoms", action="store_true")
    ap.add_argument("--shuffle_force_across_molecules", action="store_true")
    ap.add_argument("--random_sign_force", action="store_true")
    ap.add_argument("--wrong_kT_train", type=float, default=None,
                    help="If set, relabel kT to this value.")

    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raise NotImplementedError(
        "Tier 6 (negative controls) is scaffolded but not yet wired. "
        "Implementation: "
        "(1) load --in_shard (torch.load); "
        "(2) apply requested shuffle(s) to energies / forces; "
        "(3) save --out_shard with metadata describing the shuffle "
        "    so reviewers can reproduce / verify."
    )


if __name__ == "__main__":
    sys.exit(main())
