# v9 NaN fix + Experiment 5 (negative controls) launch plan

## Diagnosis: v8 NaN root cause

Across all five v8 variants the NaN appeared between steps 30k and 99k
(variant-dependent; v8b T-conditional reached the furthest, step 98,550).
Inspection of the loss trajectory shows the **forward loss values stay
finite** through the entire run, including the step immediately before
the NaN cascade. Concretely, v8b at step 98,550:

| field | value |
|---|---|
| `train_total_loss`  | 0.285 (finite) |
| `train_L_anchor`    | 2.12×10¹⁰ (finite, well below the earlier-survived 1.9×10¹² peak) |
| `train_L_energy`    | **1.30×10⁶** (≈25× the typical 5×10⁴ peak — a clear outlier) |

The next training step is fully NaN. Existing forward-side guards
(`isfinite(L_energy)` / `isfinite(L_anchor)`) did not trigger, because
the loss values themselves were finite. The most consistent
explanation is that the **backward pass** through the FFJORD divergence
integral overflowed under bf16 mixed precision and produced a non-finite
gradient component, which `gradient_clip_val: 0.5` cannot fix
(`clip_grad_norm_` blows up when any grad is non-finite).

## v9 fix (two layers)

1. **Forward-side outlier cap on `L_energy` / `L_anchor`.**
   Even when finite, treat the step as a no-op for the BGFM branch if
   `|L_energy| > 1e5` or `|L_anchor| > 1e12`. This catches the 1.3×10⁶
   energy spike directly.
   Implementation: `cfm_mol/bgfm_train_hook.py:bgfm_training_step`
   reads `l_energy_outlier_cap` and `l_anchor_outlier_cap` from the
   config (defaults `1e5`, `1e12`) and zeroes the loss + sets `l_2=0`
   / `l_3=0` when the cap is exceeded.

2. **Gradient-level NaN-skip.**
   New `on_before_optimizer_step` Lightning hook scans every parameter
   gradient; if any element is non-finite, zero that parameter's grad
   for this step. The step then becomes a no-op for the affected
   parameters but the model survives.
   Implementation: `cfm_mol/bgfm_train_hook.py:patch_flowmol_bgfm`
   monkey-patches `model.on_before_optimizer_step` when
   `bgfm.grad_nan_skip: true` (default).

3. **λ_3 step-down.**
   v8 used `λ_3 = 1e-10`. The peak anchor magnitude observed was
   1.9×10¹² at step 97,600 (multiplied by λ_3 gives 190, which is too
   big a perturbation when combined with a stochastic outlier). v9
   uses `λ_3 = 1e-12` to keep the anchor contribution well below 10
   even in the worst observed batch.

## v9 configs landed

- `configs/omol25_4m_bgfm_energy_v9a_room_T.yaml`
  Production: v8a recipe + the three fixes above. Recommended target
  for the post-fix Tier-0 (mechanism) re-measurement and downstream
  Experiments 2 / 3 once v9 reaches step ≥ 100k.

- `configs/omol25_4m_bgfm_energy_v9neg_shuffle_energy.yaml`
  Negative control for Experiment 5: same hyperparameters as v9a but
  trains against
  `perturbation_train_n30000_s0_SHUFFLE_energy_within_parent.pt`,
  produced by `scripts/eval_negative_controls.py
  --shuffle_energy_within_parent`. Within each parent the K=5
  perturbation energies are randomly permuted, so the multiset of
  energies a parent sees is unchanged but the energy-geometry pairing
  is broken.

## Experiment 5 launch plan

Once H200 capacity is available:

```bash
# 1. Produce additional shuffled shards
PSHARD=/n/netscratch/ryl_lab/Lab/yulili_cfm_mol/omol25_4m_processed
python scripts/eval_negative_controls.py \
    --in_shard "$PSHARD/perturbation_train_n30000_s0.pt" \
    --out_shard "$PSHARD/perturbation_train_n30000_s0_SHUFFLE_energy_across_parents.pt" \
    --shuffle_energy_across_parents --seed 0

# 2. Launch v9a (real labels) on three seeds + v9neg variants
sbatch scripts/launch_omol25_bgfm_energy_h200.sh \
    configs/omol25_4m_bgfm_energy_v9a_room_T.yaml "" 42
sbatch scripts/launch_omol25_bgfm_energy_h200.sh \
    configs/omol25_4m_bgfm_energy_v9neg_shuffle_energy.yaml "" 42

# 3. After ≥100k steps, evaluate both with the same Tier-0 +
#    Experiment 2 (xTB) pipeline. The headline causal claim:
#
#      Experiment 2 lift on independent xTB metrics ONLY appears for
#      real-label training, NOT for shuffled-energy training.
```

The shuffle-within-parent control is the cleanest test: it isolates
"energy assigned to the RIGHT geometry" from "energy is numerically
in the right ballpark." If BGFM still improves xTB metrics under
shuffled-within-parent training, the gain is not coming from real
local Boltzmann supervision.

## Why this layered fix is enough

Under v8 the only NaN guard was on the loss value itself. With v9:

- The outlier cap catches an extreme but finite loss before its
  gradient is computed at all.
- The grad NaN-skip catches the residual case where the loss is
  inside-cap but its backward pass still produces a non-finite grad
  (e.g. a divergence integral that overflows on a single Hutchinson
  sample).
- The λ_3 step-down provides headroom against future outliers in
  bounds we have not yet seen.

Smoke test: a 5k-step v9a smoke launch should show zero NaN events
and `train_grad_nan_skip > 0` only on rare batches (telemetry, not a
failure). If the smoke is clean, the full run proceeds.

## Open items

- DFT-subset version of the Experiment 5 negative-control evaluation
  (Tier 3 oracle, not OMol25). Needs xtb-python wiring first.
- Force-shuffle controls require operating on
  `train_data_processed.pt` (per-atom forces) rather than the
  perturbation shard; the script will detect a missing `forces` key
  and error cleanly.
