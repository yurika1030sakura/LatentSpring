# BGFM newversion zip — code-grounded novelty and ICLR-readiness audit

## Executive conclusion

BGFM is a genuinely interesting idea, but the safest ICLR framing is not “a new diffusion/flow backbone” and not “an exact natural Boltzmann sampler.” The code supports a stronger and more defensible claim:

> **BGFM is an OMol25-guided, Boltzmann-regularized training framework for de novo 3D molecular flow matching. It keeps a FlowMol3-style backbone, adds training-time OMol25 force/energy supervision, and wraps the conditional path/sampler with geometric constraint hooks.**

The method’s novelty lives in the objective and physical evaluation, not in replacing the neural architecture.

## What the new zip contains

The uploaded `newversion_bgfm.zip` contains the core BGFM extension files under `cfm_mol/`, model configs under `configs/`, and several evaluation/precompute scripts under `scripts/`. It does **not** contain the full vendored `baselines/flowmol3/`, the bib file, or several Level-2/Level-3 scripts mentioned in `CODE_MAP.md`. This is enough to audit the BGFM contribution, but not enough as a fully self-contained reviewer reproduction package.

The code compiles with `python3 -m compileall -q cfm_mol scripts` after extraction.

## What the code actually implements

### 1. FlowMol3-style backbone is retained

`cfm_mol/bgfm_train_hook.py` wraps the existing model’s `training_step`: it runs the original FM step first, then computes auxiliary BGFM losses. This means the base neural vector field remains the main generator. The code does not introduce a new molecular generator architecture.

### 2. The training objective is changed

The BGFM hook adds:

- `L_force`: score-force consistency from `cfm_mol/bgfm_loss.py`.
- `L_energy`: position-only FFJORD density-energy consistency from `cfm_mol/bgfm_density.py`.
- `L_anchor`: composition-conditioned log-Z/offset anchor from `cfm_mol/log_z_predictor.py`.
- Optional kT conditioning from `cfm_mol/kt_conditioning.py`.

This is the main methodological novelty.

### 3. It is not only a loss; there are geometry/path hooks

`cfm_mol/flow_model.py` monkey-patches FlowMol’s conditional path and sampler with:

- steric-fibre retraction after conditional-path interpolation;
- tangent projection of velocity helper output;
- retraction after Euler steps;
- optional discrete valence/connectivity projection;
- optional late-time BGFM score guidance.

So the right wording is:

> BGFM does not replace the FlowMol3 neural backbone or CTMC formulation, but it does modify the training objective and wraps path/sampling execution with geometric constraint hooks.

Do **not** write “we did not change the flow/diffusion bottom layer at all.”

## Exact Boltzmann interpretation

The current code supports **local conditional coordinate-density regularization**, not global exact Boltzmann generation.

Use this target:

\[
p_B(r\mid c,T)=\frac{1}{Z_c(T)}\exp[-E_{\rm NP}(r,c)/kT],
\]

where `c` is fixed molecular identity/composition/atom types/charge and `r` is coordinates.

Do not use this as the main claim:

\[
p_B(x)=Z^{-1}\exp[-E(x)/kT]
\]

over arbitrary atom counts, compositions, charges, atom types, and coordinates.

The reason is code-level: `cfm_mol/bgfm_density.py` uses a **position-only** velocity closure. The discrete atom/charge/bond channels are held fixed while the coordinates vary. Therefore, the FFJORD density term estimates a coordinate log-density conditional on molecular identity, not a full joint molecular probability.

## Force loss: exact implementation

The new `FORCE_LOSS_AUDIT.md` is accurate. The current training pairs:

- score evaluated at `x_t` on the conditional FM path, with `t_eval_values = [0.70, 0.80, 0.90]` in the v8 configs;
- force target equal to precomputed endpoint force `F(x_1)` stored as `force_1_true`.

So the loss is best described as a **late-time force-score regularizer**:

\[
\mathcal{L}_{\rm force}
=\mathbb{E}_{t\in\mathcal{T}}\left[1-
\cos\left(s_\theta(x_t,t), F_{\rm NP}(x_1)/kT\right)\right].
\]

It is not using `F(x_t)` and it is not using perturbation forces for `L_force`. Perturbation energies are used for `L_energy` and `L_anchor`.

This is defensible, but must be stated honestly as an approximation: near late `t`, `x_t` is close to `x_1`, so `F(x_1)` is a cheap surrogate for `F(x_t)`.

## Energy-density loss: strongest method point

The strongest ICLR method contribution is the within-parent perturbation variance:

\[
\mathcal{L}_{\rm energy}
=\mathbb{E}_{m}\operatorname{Var}_{k}\left[
\log p_\theta^{\rm pos}(r_m^{(k)}\mid c_m)+\beta E_{\rm NP}(r_m^{(k)},c_m)
\right].
\]

This is a good idea because it avoids needing the true partition function `Z_c`: within the same parent molecule, the Boltzmann condition only fixes relative probabilities. This is the most novel part of the project.

## Log-Z predictor interpretation

`LogZPredictor` is a composition-conditioned auxiliary network using atom types and total charge. It should be called an **offset stabilizer** or **composition-conditioned anchor**.

Do not claim it estimates the true molecular partition function unless you have true `Z_c` labels or an independent validation of partition functions. A reviewer will not accept “log-Z predictor” as true thermodynamic partition-function estimation by name alone.

## Important code/paper mismatches and bugs

### 1. Stage-1 Boltzmann evaluation writes charge = 0

`scripts/eval_boltzmann_stage1.py` exports every record with `"charge": 0`. Stage 2 then uses this charge when computing OMol25 energy. This can corrupt evaluation on charged species and weakens the universal-chemistry claim.

I produced a patch that changes this to recover molecular charge from `val.atom_charges[ns:ne].sum()`.

### 2. T-conditional energy loss currently does not pass kT into FFJORD

In `bgfm_train_hook.py`, the force forward receives `kT`, but the energy-consistency calls do not pass `kT_tensor` to `log_density_via_flow`. The residual uses the sampled scalar kT, but the density model is evaluated without kT conditioning. This undermines v8b multi-T energy training.

I produced a patch that creates `kT_pert` for the perturbation batch and passes it into `energy_consistency_loss_per_mol(_with_anchor)`.

### 3. Stage-1 evaluation cannot evaluate T-conditional checkpoints correctly

`scripts/eval_boltzmann_stage1.py` has no `--kT` argument and removes the `bgfm` config before patching kT conditioning. T-conditional checkpoint loading/evaluation is therefore incomplete.

The patch adds `--kT`, installs the kT-conditioning module before loading the checkpoint, and passes kT into FFJORD density evaluation.

### 4. v7c is mislabeled as force-only unless runtime overrides exist

`configs/omol25_4m_bgfm_energy_v7c_from_fm.yaml` has `lambda_2: 0.001`, so the checked-in config is force + energy, not pure force-only. If the reported `R^2=0.278` came from a CLI override with `lambda_2=0`, document that. Otherwise, stop calling it force-only.

### 5. `compute_bgfm_step` is NotImplemented

`cfm_mol/bgfm_loss.py:compute_bgfm_step` is not implemented. The real training assembly is in `bgfm_train_hook.py`. Update `CODE_MAP.md` and paper references accordingly.

### 6. Some docstrings still overclaim or contradict code

Examples:

- `bgfm_train_hook.py` docstring says `lambda_2` must be 0 / energy inactive. False.
- `bgfm_loss.py` score derivation still has “DOUBLE-CHECK” text. Bad for reviewers.
- `eval_boltzmann_stage2.py` prints “learned density IS Boltzmann.” Too strong.
- Several comments describe anchor as preventing a “trivial constant” failure mode. Better: variance-only constrains relative probabilities but not absolute offset.

The patch cleans the most visible instances.

### 7. Latest zip is not a full reviewer package

The zip lacks the vendored FlowMol3 baseline, bib, notes, and some Level-2/Level-3 scripts. If handing to a reviewer, include either the full repository or an exact commit/submodule setup.

## ICLR novelty score after reading the new zip

- **Core idea novelty:** 7.5/10.
- **Backbone/architecture novelty:** 4.5/10.
- **Objective novelty:** 8/10.
- **Implementation maturity:** 5.5/10.
- **Current ICLR readiness:** not ready as a strong main-conference submission until pending v8/v8b/v8c results, charge handling, T-conditional evaluation, and Level-2/Level-3 evidence are fixed.

The idea can be made ICLR-shaped if the paper is honest:

> Existing de novo 3D generators learn data-like molecules; existing Boltzmann generators mostly sample fixed systems or conformers. BGFM bridges these by using a universal neural potential to regularize the conditional coordinate density of a de novo flow-matching generator through force-score and density-energy consistency.

## Recommended title

**Boltzmann-Regularized Flow Matching for Universal-Neural-Potential Guided 3D Molecular Generation**

## Recommended one-paragraph abstract core

We propose BGFM, a training-time physical regularization framework for de novo 3D molecular flow matching. BGFM keeps a FlowMol3-style backbone, but uses OMol25 neural-potential energies and forces to align the generator’s coordinate density, conditional on molecular identity, with local Boltzmann relative probabilities. The method combines a late-time force-score regularizer with a position-only FFJORD density-energy variance loss over precomputed perturbation clouds, plus a composition-conditioned offset anchor. This turns a universal neural potential from a post-generation relaxation tool into a train-time supervisor. We evaluate not only molecular validity, but also local Boltzmann correlation, force-score alignment, downstream relaxation cost, and ensemble overlap.

## What not to claim

Do not claim:

- exact natural Boltzmann sampling;
- full joint probability over atom types, charges, atom counts, and coordinates;
- true partition-function estimation;
- downstream-deployment guarantee;
- v8/v8b/v8c improvements before stable full evaluations are complete;
- no modification to FlowMol internals at all.

## What to claim

Claim:

- local conditional coordinate Boltzmann alignment;
- train-time OMol25 supervision rather than post-hoc relaxation;
- FlowMol3-style backbone plus BGFM objective and geometric hooks;
- force-score consistency at late FM path times;
- FFJORD density-energy variance over perturbation clouds;
- composition-conditioned offset stabilization.
