# OMol25 benchmark: 5 generators × 2 metrics

All five 3D molecular generators were trained on the **same OMol25 50k split**
(46,816 train / 2,464 val, bond-free, all 83 elements) with **I/O changes only —
no architecture changes** — and evaluated on two axes:

1. **Original metric** — each family's published de-novo-3D-generation metric,
   here unified as geometry-based **xyz2mol validity / connectivity** (RDKit
   `DetermineBonds`, works across all 83 elements, unlike each repo's QM9/GEOM
   valence tables). Computed on 100 natively-generated molecules per model.
2. **New metric — Boltzmann consistency** (this project's contribution): for 50
   held-out OMol25 molecules (≤50 atoms), make 16 Gaussian perturbations
   (σ=0.15 Å) plus the original geometry, compute each model's per-geometry
   log-density `log p_θ`, and correlate it with `−E_DFT/kT` from the OMol25 eSEN
   oracle (`esen_sm_conserving_all`). A density that equals the Boltzmann
   distribution has `log p_θ = −E/kT + const`, i.e. Pearson `r → 1`.
   **All models share the exact same molecules / perturbations / charge / spin**
   (the BGFM stage-1 set), so the per-molecule `r` is apples-to-apples.

`log p_θ` is computed in each model's native way:
- **BGFM / FM-only**: FFJORD reverse-time ODE log-density (Hutchinson divergence,
  12 ODE steps, 4 probes) — a *stochastic* estimate.
- **EDM / GeoLDM**: `−`(diffusion NLL), averaged over 16 random diffusion times.
- **Symphony**: `−`(sum of autoregressive per-fragment generation losses).

## Results (OMol25 50k, short training — illustrative scale, not the 4M headline run)

| Model | Validity | Connected | Mean Pearson r | Median r | frac r>0.5 | Mean slope | #mols |
|---|---|---|---|---|---|---|---|
| **BGFM (ours)**     | 0.03 | 1.00 | **0.305** | 0.375 | 0.24 | 0.31 | 50 |
| FM-only (Level-1)   | 0.08 | 1.00 | 0.266 | 0.310 | 0.18 | 0.15 | 50 |
| EDM                 | 0.00 | 0.00 | −0.004 | −0.036 | 0.02 | 0.46 | 50 |
| GeoLDM              | _stage-2 latent diffusion training in progress_ | | | | | | |
| Symphony            | 0.23 | 0.18 | **0.709** | 0.723 | 0.92 | 2.62 | 50 |

Primary Boltzmann-consistency number = **mean per-molecule Pearson r** (invariant
to each model's per-molecule log-p scale/offset; slope and pooled-R² are
scale-sensitive and only roughly comparable across the different `log p_θ`
definitions).

## Findings

- **EDM's diffusion NLL is uncorrelated with energy (r ≈ 0).** A vanilla
  Cartesian diffusion model does *not* learn a Boltzmann-consistent density — its
  likelihood is unrelated to the DFT energy of a perturbed geometry. This is the
  cleanest negative control and motivates the physics losses.
- **Symphony's autoregressive likelihood is strongly geometry-sensitive (r = 0.71).**
  Its per-fragment position loss rises sharply as a geometry is distorted away
  from equilibrium, which tracks the energy increase. Among these baselines it is
  the strongest "for free" Boltzmann signal — worth flagging in the paper rather
  than hiding.
- **BGFM > FM-only on r (0.305 vs 0.266)**: the force loss nudges the learned
  density toward the energy landscape, but at this scale the gain is modest.
- **Validity is low across the board** (0–23%) because these are short-training
  50k-molecule runs over the full 83-element OMol25 chemistry (transition-metal /
  radical / hypervalent complexes), far harder than QM9. Symphony's
  fragment-by-fragment construction yields the most valid geometries.

## Caveats

- **Scale.** These are 50k-molecule, short-training runs (FlowMol ~15 epochs;
  Symphony 180k steps; EDM/GeoLDM ~30 epochs). They are *illustrative* of the
  benchmark pipeline, **not** the paper's headline 4M-scale BGFM result, where the
  physics-loss advantage is expected to dominate.
- **Estimator asymmetry.** BGFM/FM-only `log p_θ` is a noisy stochastic FFJORD
  estimate (12 ODE steps); Symphony/EDM use deterministic loss sums. This
  disadvantages the flow models on the correlation metric relative to a
  variance-free estimator.
- **Connectivity column** is computed by FlowMol's own analyzer (over valid
  molecules) for BGFM/FM-only, but over all samples for EDM/Symphony — not
  perfectly comparable.

## Reproduce

Scratch drivers (UMass Unity): `~/scratch_workspace/bgfm/eval_{real,edm,symphony,symphony_validity}.sh`
and `assemble_table.py`. Boltzmann stage-2 oracle: `scripts/eval_boltzmann_stage2.py
--ckpt <esen_sm_conserving_all.pt>`. EDM/GeoLDM/Symphony Boltzmann stage-1 reuse
the BGFM stage-1 molecule set via `--ref_json`.
