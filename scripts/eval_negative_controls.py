"""Experiment 5: Negative-control shard generator.

Creates a copy of a precomputed perturbation shard with shuffled labels,
to be used as a drop-in replacement under
`bgfm.energy_perturbation_shards` in a training config. The resulting
shuffled shards are then trained against the same BGFM config, and the
resulting checkpoints are evaluated under Experiments 2 (independent
xTB), 4 (OMol25 mechanism), etc.

The causal claim BGFM needs to defend is:

  Training with CORRECT OMol25 energies and forces lifts the
  independent xTB physical-quality metrics, while training with
  SHUFFLED labels does not (even though it can still mathematically
  optimize the OMol25-correlated training objective).

The gap between true-label and shuffled-label outcomes on the
independent xTB metric establishes the causal contribution of the
BGFM physical signal, rather than the existence of an extra
regularization term per se.

Implemented shuffles
--------------------

--shuffle_energy_within_parent
    For each parent molecule, randomly permute the K perturbation
    energies among the K perturbations of that same parent. The set
    of (energy, perturbation-geometry) pairings is broken, but the
    energy values that appear are exactly the ones OMol25 produced.
    This is the cleanest control: it isolates "energy assigned to the
    RIGHT geometry" from "energy is in the right ballpark numerically."

--shuffle_energy_across_parents
    Permute energies across the entire shard (i.e. between parents).
    Removes both within-parent ordering AND any cross-parent
    composition signal. Stronger ablation.

--shuffle_force_within_atoms
    For each molecule, permute the per-atom force vectors among the
    atoms of that molecule. Breaks "force assigned to the right atom"
    while preserving aggregate force statistics.

--shuffle_force_across_molecules
    Permute force vectors at random across the entire shard.

--random_sign_force
    Multiply each per-atom force vector by an i.i.d. random sign
    (+/-1). Tests whether the model is exploiting force *direction*
    or merely force *magnitude*.

--wrong_kT_label <value>
    Relabel the shard's kT metadata so a downstream config that reads
    kT from the shard receives a wrong value. (Only matters if the
    training pipeline reads kT from the shard rather than the config;
    use --override_bgfm_kT in run_train.py for the more direct test.)

Usage
-----

    python scripts/eval_negative_controls.py \
        --in_shard /n/netscratch/.../perturbation_train_n30000_s0.pt \
        --out_shard /n/netscratch/.../perturbation_train_n30000_s0_shuf_energy_within_parent.pt \
        --shuffle_energy_within_parent \
        --seed 0

Each shard records its shuffle provenance under the key
`negative_control_meta` so a reviewer can audit which shuffle was
applied without having to diff the files.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch


def _shuffle_within_groups(values: torch.Tensor, group_ids: torch.Tensor,
                           rng: torch.Generator) -> torch.Tensor:
    """Permute `values` within each contiguous block defined by `group_ids`.

    Returns a permuted copy of `values` such that the multiset of
    values inside each group is unchanged but the assignment to
    positions inside the group is randomized.
    """
    out = values.clone()
    unique, counts = torch.unique_consecutive(group_ids, return_counts=True)
    idx_start = 0
    for cnt in counts.tolist():
        if cnt > 1:
            perm = torch.randperm(cnt, generator=rng)
            block = out[idx_start:idx_start + cnt].clone()
            out[idx_start:idx_start + cnt] = block[perm]
        idx_start += cnt
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_shard", type=Path, required=True)
    ap.add_argument("--out_shard", type=Path, required=True)

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--shuffle_energy_within_parent", action="store_true")
    g.add_argument("--shuffle_energy_across_parents", action="store_true")
    g.add_argument("--shuffle_force_within_atoms", action="store_true")
    g.add_argument("--shuffle_force_across_molecules", action="store_true")
    g.add_argument("--random_sign_force", action="store_true")
    g.add_argument("--wrong_kT_label", type=float, default=None)

    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"[neg-ctrl] loading {args.in_shard}", flush=True)
    shard = torch.load(args.in_shard, map_location="cpu", weights_only=False)

    rng = torch.Generator()
    rng.manual_seed(args.seed)

    meta = {
        "source_shard": str(args.in_shard),
        "seed": args.seed,
    }

    # Resolve the per-row group-id key actually present in this shard.
    # Energy-perturbation shards produced by
    # scripts/precompute_energy_perturbations.py store it as 'group_id';
    # older shards may have used 'parent_idx'. Both name the same thing.
    group_key = None
    if 'group_id' in shard:
        group_key = 'group_id'
    elif 'parent_idx' in shard:
        group_key = 'parent_idx'

    if args.shuffle_energy_within_parent:
        if group_key is None or 'energies' not in shard:
            raise RuntimeError(
                "shard missing 'group_id'/'parent_idx' or 'energies' field; "
                "cannot shuffle. shard keys: "
                f"{list(shard.keys()) if isinstance(shard, dict) else type(shard).__name__}")
        original = shard['energies'].clone()
        shard['energies'] = _shuffle_within_groups(
            shard['energies'], shard[group_key], rng)
        meta["mode"] = "shuffle_energy_within_parent"
        meta["group_key"] = group_key
        meta["n_changed"] = int((shard['energies'] != original).sum().item())

    elif args.shuffle_energy_across_parents:
        if 'energies' not in shard:
            raise RuntimeError("shard missing 'energies'")
        perm = torch.randperm(shard['energies'].numel(), generator=rng)
        original = shard['energies'].clone()
        shard['energies'] = shard['energies'].view(-1)[perm].view_as(shard['energies'])
        meta["mode"] = "shuffle_energy_across_parents"
        meta["n_changed"] = int((shard['energies'] != original).sum().item())

    elif args.shuffle_force_within_atoms:
        # NOTE: force-shuffle modes operate on train_data_processed.pt
        # (which carries per-atom forces for L_force), not on the
        # perturbation shard (which only carries per-perturbation
        # energies for L_energy). See docstring.
        if 'forces' not in shard:
            raise RuntimeError("shard missing 'forces'; use train_data_processed.pt for force shuffles")
        atom_group_key = ('mol_idx' if 'mol_idx' in shard
                          else ('node_batch_idx' if 'node_batch_idx' in shard else None))
        if atom_group_key is None:
            raise RuntimeError(
                "no per-atom group id found in shard; cannot shuffle "
                "force-within-atoms. expected 'mol_idx' or 'node_batch_idx'.")
        original = shard['forces'].clone()
        # _shuffle_within_groups expects 1D contiguous groups; treat the
        # 3-vector dimension as carried along.
        flat = shard['forces']
        groups = shard[atom_group_key]
        shard['forces'] = _shuffle_within_groups(flat, groups, rng)
        meta["mode"] = "shuffle_force_within_atoms"
        meta["group_key"] = atom_group_key
        meta["n_changed"] = int((shard['forces'] != original).any(dim=-1).sum().item())

    elif args.shuffle_force_across_molecules:
        if 'forces' not in shard:
            raise RuntimeError("shard missing 'forces'")
        perm = torch.randperm(shard['forces'].shape[0], generator=rng)
        original = shard['forces'].clone()
        shard['forces'] = shard['forces'][perm]
        meta["mode"] = "shuffle_force_across_molecules"
        meta["n_changed"] = int((shard['forces'] != original).any(dim=-1).sum().item())

    elif args.random_sign_force:
        if 'forces' not in shard:
            raise RuntimeError("shard missing 'forces'")
        signs = torch.randint(0, 2, (shard['forces'].shape[0], 1),
                              generator=rng, dtype=shard['forces'].dtype) * 2 - 1
        original = shard['forces'].clone()
        shard['forces'] = shard['forces'] * signs
        meta["mode"] = "random_sign_force"
        meta["n_changed"] = int((shard['forces'] != original).any(dim=-1).sum().item())

    elif args.wrong_kT_label is not None:
        shard['kT_label'] = float(args.wrong_kT_label)
        meta["mode"] = "wrong_kT_label"
        meta["new_kT"] = float(args.wrong_kT_label)

    shard['negative_control_meta'] = meta
    args.out_shard.parent.mkdir(parents=True, exist_ok=True)
    torch.save(shard, args.out_shard)
    print(f"[neg-ctrl] wrote {args.out_shard}", flush=True)
    print(f"[neg-ctrl] meta: {meta}", flush=True)
    print("[neg-ctrl] next: train with this shard via\n"
          f"    sbatch scripts/launch_omol25_bgfm_energy_h200.sh \\\n"
          f"        <config_with_this_shard.yaml> \"\" <seed>\n"
          "then evaluate the resulting ckpt under Experiments 2 + 4.",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
