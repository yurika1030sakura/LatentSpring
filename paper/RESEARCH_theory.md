# RESEARCH_theory.md — theory distilled for the BGFM/HBFM main text

**Audience:** the drafting agent writing Section 2 (Method) / Section 4 (Theory) and the
appendix of the ICLR-2027 submission.
**Status of this document:** derivations re-checked against the code on 2026-08-03.
Every equation below was verified line-by-line against
`cfm_mol/bgfm_loss.py`, `cfm_mol/bgfm_density.py`, `cfm_mol/bgfm_train_hook.py`,
`scripts/eval_boltzmann_independent.py`, and the ablation configs in `configs/sweep/`.
Where code and `notes/appendix_hbc.tex` disagree, the disagreement is flagged in §7.

**How to use it:** §1 is the notation contract (copy verbatim). §2 is the main-text
derivation chain. §3 gives paste-ready theorem statements for the three results that
belong in the main text. §4 is the theory↔experiment correspondence table — this is the
paper's single most valuable argument, write it out in full. §5 lists what is *not*
empirically verified (do not let the prose imply otherwise). §6 lists the honest caveats
that must appear in Limitations. §7 lists code/notes fixes that the writing must not
paper over.

---

## 0. One-paragraph summary of the theoretical story

Flow matching gives, in closed form, the score of the time-`t` marginal of its own
interpolant (Lemma 1). Two different physics conditions can therefore be imposed on a
flow-matching generator: a **differential** one (score = `F/kT`, our `L_force`) and a
**value** one (`log p + E/kT` constant within a molecule, our `L_energy`). The theory
says these are *not* interchangeable. The value condition is (i) exactly equivalent to
conditional Boltzmann consistency on the support, (ii) invariant to the intractable
partition function, and (iii) a strictly monotone surrogate for the metric we report
(Prop. 4) — so optimising it *is* optimising the benchmark. The differential condition
is, at any usable evaluation time `t<1`, only a biased and variance-amplified proxy
(Prop. 2), and its population target *conflicts* with the flow-matching term whenever the
training distribution is not itself Boltzmann (Prop. 3) — which OMol25 is not. Separately,
force information is *structurally* blind to one degree of freedom per composition, the
free energy (Theorem 3 / Corollary 1: gauge-identifiability and division of labour). The
experiments match this on both counts: the energy term produces a large, highly
significant gain ([A], Δr = +0.355, t ≈ 10.8), the force term does not and in fact hurts
([B]), and the label-scrambled control shows that most but not all of the energy gain is
a density-calibration effect rather than a Boltzmann-information effect ([G]).

---

## 1. Unified notation (the contract — use exactly this in the paper)

### 1.1 The two notation collisions that must be fixed first

**COLLISION 1 (paper-breaking).** `notes/appendix_hbc.tex` uses `F(c)` for the *free
energy* while the code, `README.md` and `notes/PROJECT.md` use `F` for the *force*
`F = -∇E`. Both appear in the same three-line loss definition.
**Resolution to adopt:** keep `F(x)` (or bold `**F**`) for force; write the conditional
free energy as `A(c) = -kT log Z(c)` (Helmholtz convention). Global search-and-replace
`F(c) -> A(c)` in the appendix before integration.

**COLLISION 2 (minor).** `appendix_hbc.tex` writes the conformation as `r`; the code and
the FM literature use `x`. **Resolution:** use `x` everywhere for coordinates; keep `c`
for composition; state once that a molecular state is the pair `(c, x)`.

### 1.2 Notation table

| Symbol | Meaning | Units / shape | Code anchor |
|---|---|---|---|
| `c` | composition: element multiset, atom count `N(c)`, total charge `q`, spin multiplicity `s`. Countable set `𝒞`. | — | frozen discrete channels `a_1_true`, `c_1_true` |
| `x` | nuclear coordinates of one molecule, centre-of-mass removed | `R^{3N}`, effective dim `d = 3(N−1)` | `g.ndata['x_1_true']`, flat `(N_total, 3)` |
| `x_1` | data / endpoint coordinates (`t = 1` end) | `R^{3N}` | `x_1_true` |
| `x_0` | prior sample, `x_0 ~ N(0, σ² I)` on the COM-free subspace | `R^{3N}` | `torch.randn * prior_std`, COM-removed |
| `σ` | prior standard deviation | dimensionless (Å after data scaling) | `prior_std = 1.0` everywhere (`bgfm_loss.py`, `bgfm_density.py`, `bgfm_train_hook.py`) |
| `t` | flow time; `t = 0` prior, `t = 1` data (FlowMol3 convention) | `[0,1]` | `t_scalar`, per-graph `(B,)` |
| `x_t` | linear interpolant `x_t = (1−t)x_0 + t x_1` | `R^{3N}` | `g.ndata['x_t']`, set by `sample_conditional_path` |
| `p_t^θ` | model marginal density at time `t`; `p_θ := p_1^θ` | density on `R^d` | FFJORD output of `log_density_via_flow` |
| `v_θ(x_t,t)` | learned velocity field (SE(3)-equivariant) | `R^{3N}` | `model.vector_field(...)['x']` |
| `s_θ(x_t,t)` | FM-implied score of `p_t^θ` (Lemma 1) | `R^{3N}` | `score_from_fm_velocity` |
| `t_eval` | late probe time(s) at which the score is read off | `(0,1)` | `t_eval_values = [0.85, 0.92, 0.97]` |
| `E(c,x)` | teacher potential energy (OMol25 DFT label offline; eSEN on-policy) | eV | `energies` (per-graph), `atoms.get_potential_energy()` |
| `F(c,x)` | teacher force, `F = −∇_x E` | eV/Å | `g.ndata['force_1_true']` |
| `kT` | temperature scale | eV | config `bgfm.kT = 1.0` **(≈ 11 600 K, *not* room T = 0.02585 eV)** |
| `π(x\|c)` | conditional Boltzmann `e^{−E/kT}/Z(c)` | density | the target of `L_energy` |
| `Z(c)` | conditional (configurational) partition function `∫ e^{−E(c,x)/kT} dx` | — | never formed |
| `A(c)` | conditional free energy `A(c) = −kT log Z(c)` (was `F(c)`) | eV | `LogZPredictor` predicts `log Z(c) = −A(c)/kT` |
| `𝒵` | grand sum `Σ_c Z(c)` | — | never formed |
| `π(c,x)` | joint Boltzmann `e^{−E/kT}/𝒵` | density | HBFM target (level 1 **not trained**) |
| `p_φ(c)` | composition-level generator | — | **not implemented** (see §5) |
| `m, k` | parent-molecule index `m = 1..M`, perturbation index `k = 1..K` | — | `parent_id`, `energy_b_parents` (=M), shard `K` |
| `w_{m,k}` | **Boltzmann residual** `log p_θ(x_{m,k}) + E_{m,k}/kT` | nats | `residual` in `within_group_variance_loss` |
| `λ₁, λ₂, λ₃` | weights of `L_force`, `L_energy`, `L_anchor` | — | `bgfm.lambda_1/2/3` |
| `λ_op` | weight of on-policy force distillation | — | `bgfm.lambda_onpolicy` |
| `ξ` | Hutchinson probe, Rademacher `{±1}` | `R^{3N}` | `divergence_hutchinson` |
| `r_m` | per-group Pearson correlation of `log p_θ` vs `−E_xTB/kT` over the `K` eval perturbations of molecule `m` | `[−1,1]` | `eval_boltzmann_independent.py::pearson_r` |
| `τ_m` | residual-to-signal ratio `sd_k(w)/sd_k(E/kT)` (Prop. 4) | dimensionless | derived, not logged |

### 1.3 Numerical constants that are part of the definition of the losses (state them)

| Constant | Value | Where | Why it matters for the paper |
|---|---|---|---|
| `SCORE_NORM_CAP` | 1000 (per-atom L2 of `s_θ`) | `bgfm_loss.py:113` | the implemented `s_θ` is *clipped*; the paper's `L_force` is a clipped score-matching loss |
| `PER_ATOM_ERR_CAP` | `1e4` (Huber transition on squared error) | `bgfm_loss.py:282` | `L_force` is Huberised, not pure MSE |
| `LOG_P_CAP` | `1e6` | `bgfm_density.py:387` | anchor loss clamps `log p` |
| `energy_loss_cap` | 3000 (a3), 1500 (stabilised) | configs | finite-outlier rejection, the [F] fix |
| `n_ode_steps` | 4 (train, a3) / 12 (eval) | configs, `eval_boltzmann_stage1.py` | `log p` is a *discretised* CNF likelihood |
| `n_hutchinson` | 2 (train, a3) / 4 (eval) | configs | eval `log p` carries estimator noise → attenuates `r` (§6.3) |
| `λ₂` | `3.0e-5` in the headline energy runs | `configs/sweep/a3_energy_only.yaml` | the +0.355 effect comes from a *very lightly weighted* auxiliary term — this is a strength, say it |
| `λ₃` | **0 (absent) in a3/a6** | `configs/sweep/a3*.yaml` | the anchor / `log Z` head was **NOT active** in the headline runs (§5) |

---

## 2. Core derivations for the main text

### 2.1 Lemma 1 — the flow-matching score read-off

**Setup.** Conditional (linear / rectified) interpolant `x_t = (1−t)x_0 + t x_1` with
`x_0 ~ N(0, σ²I)` independent of `x_1 ~ p_data`. Hence

```
x_t | x_1  ~  N( t x_1 , (1−t)² σ² I ).                                   (1)
```

**Step 1 (mixture score).** `p_t(x) = ∫ N(x; t x_1, (1−t)²σ²I) p_data(x_1) dx_1`. For a
Gaussian mixture with *fixed* covariance, differentiating under the integral gives the
posterior-mean form

```
∇_x log p_t(x) = ( t · E[x_1 | x_t = x] − x ) / ( (1−t)² σ² ).             (2)
```

**Step 2 (velocity → posterior mean).** The FM regression target is `x_1 − x_0`, and from
`x_0 = (x_t − t x_1)/(1−t)` we get `x_1 − x_0 = (x_1 − x_t)/(1−t)`. Taking conditional
expectations, the marginal velocity is `v*(x,t) = (E[x_1|x_t=x] − x)/(1−t)`, i.e.

```
E[x_1 | x_t = x] = x + (1−t) v*(x, t).                                     (3)
```

**Step 3 (substitute).** Inserting (3) into (2), the `(1−t)` factors partially cancel:

```
        s*(x_t, t) = ( t · v*(x_t,t) − x_t ) / ( (1−t) σ² ).               (4)
```

Replacing `v*` by the learned `v_θ` defines `s_θ`. ∎

**Code check.** `bgfm_loss.py::score_from_fm_velocity` line 107 is exactly
`score = (t * v_theta - x_t) / (one_minus_t * prior_std**2)`, with `one_minus_t` clamped
at `1e-3` and a per-atom norm cap at 1000. Sign and factors verified; the docstring's
"adversarial re-derivation" note is correct. **Do not let anyone "simplify" the sign**:
matching `s_θ → +F/kT` is Boltzmann-*attracting* because `∇ log π = −∇E/kT = +F/kT`.

**The singularity, stated honestly.** (4) has a `1/(1−t)` prefactor. It is *not* a
removable artifact: `s_θ` is the score of the *smoothed* marginal `p_t`, and as `t → 1`
the smoothing vanishes while the read-off amplifies any velocity error by `t/((1−t)σ²)`.
Concretely, an error `δv` in the velocity produces score error `‖δv‖ · t/((1−t)σ²)`:
a factor 6.7 at `t = 0.85`, 12.5 at `t = 0.92`, 32.3 at `t = 0.97` — the three times
actually used. This is the quantitative content of the "`t → 1` divergence" remark.

### 2.2 Proposition 2 — `L_force` is a bias–variance-trapped surrogate (no good `t`)

**Reparameterise the smoothing.** Write `x_t = t·y` with

```
y = x_1 + ε z ,   z ~ N(0, I) ,   ε = (1−t) σ / t .                        (5)
```

Then `p_t(x) = t^{−d} q_ε(x/t)` with `q_ε = p_data * N(0, ε²I)`, so

```
s_θ(x_t, t) ≈ (1/t) · ∇ log q_ε( x_1 + ε z ).                              (6)
```

The training target is `F(x_1)/kT = ∇ log π(x_1)`. Comparing (6) with the target exposes
**four** distinct mismatches:

1. **Scale.** the `1/t` prefactor — relative error `(1−t)/t = O(1−t)`.
2. **Smoothing.** `∇ log(p * N_ε) − ∇ log p = O(ε²)` for smooth log-densities (heat-flow
   expansion) `= O((1−t)²)`.
3. **Evaluation point.** the score is read at `x_1 + εz`, not `x_1`; a zero-mean
   displacement of size `ε` contributes `ε ∇²log π · z`, i.e. an `O(1−t)` *stochastic*
   target error even with a perfect model.
4. **Amplified estimator variance.** by §2.1, any velocity error is multiplied by
   `t/((1−t)σ²)`, so the gradient noise injected into the parameters scales as
   `(1−t)^{−2}` (the loss is quadratic in the score).

Items 1–3 shrink as `t → 1`; item 4 explodes. **There is no choice of `t_eval` that makes
both small**, and the total error is bounded below by the product-type trade-off
`O((1−t)) + O((1−t)^{−2}·noise)`. Using several `t_eval` values (as we do) averages but
does not remove this.

*Paper framing:* this is the first of two honest, theory-side explanations for the
negative result [B]. Write it as a proposition with an informal proof and defer details.

### 2.3 Proposition 3 — `L_FM` and `L_force` have *conflicting* population minimisers
### unless the training data is already Boltzmann

This is the sharper explanation for [B], and it is important because Theorem 3
(gauge-identifiability) **does not** explain [B] (see §4.3).

**Statement.** Let `p_data` be the marginal of the training coordinates and let
`π ∝ e^{−E/kT}`. The population minimiser of `L_FM` is `v_θ = v*`, whose implied score at
time `t` is `∇ log p_t^{data}` (Lemma 1). The population minimiser of `L_force` (off-policy,
targets evaluated at data points) drives `s_θ(x_t,t) → ∇ log π(x_1)`. If
`p_data ≠ π`, then `∇ log p_t^{data}(x_t) ↛ ∇ log π(x_1)` even as `t → 1`, so the two terms
are **inconsistent**: the joint minimiser is a compromise that is neither the data velocity
nor Boltzmann-consistent. Moreover, because the score read-off multiplies velocity errors by
`t/((1−t)σ²)`, a *fixed* `λ₁` imposes an effective weight on the conflicting target that
grows like `(1−t)^{−2}` — the force term dominates the velocity gradient exactly in the
regime `t → 1` where `L_FM` determines the reported density.

**Why the hypothesis `p_data ≠ π` holds for us (say this explicitly).** OMol25 4M is a
curated union of relaxation trajectories, MD snapshots and conformer sets across many
systems, charges and spins. It is *not* an equilibrium sample from a single-temperature
Boltzmann distribution — and our training `kT = 1.0 eV` is not a physical temperature of
the dataset either. Therefore the antecedent of Prop. 3 is satisfied by construction, and
the theory *predicts* that adding `L_force` on top of `L_FM` can degrade Boltzmann
fidelity. It does: [B].

**On-policy variant.** `lambda_onpolicy` (used in cells a2 and a4) partially repairs
mismatch (3) and the `p_data ≠ π` conflict, because forces are queried at *model* samples
`x_gen` and the FM/force targets then refer to the same distribution. It does **not**
repair the `1/(1−t)` amplification (item 4), and `onpolicy_K_steps = 10` Euler steps with
stop-gradient means `x_gen` is a poor conformer early in training. So the prediction is
"on-policy is better than off-policy but still not competitive with the energy term",
which is what [B] shows.

### 2.4 `L_energy`: the variance form, why `Z` cancels, and *exactly* what it identifies

**The value condition.** For a fixed composition `c`, `p_θ(·|c) = π(·|c)` on the support
iff

```
w(x) := log p_θ(x|c) + E(c,x)/kT  =  −log Z(c)   for all x in the support.  (7)
```

**Proposition 3a (Z-free, exact identification up to a per-parent constant).**
Let `x_{m,1..K}` be `K` geometries of parent molecule `m`. Then

```
L_energy = (1/M) Σ_m Var_k ( w_{m,k} ) ,     w_{m,k} = log p_θ(x_{m,k}) + E_{m,k}/kT.  (8)
```
(a) `L_energy = 0` iff `w` is constant within every parent, i.e. iff
`log p_θ(x|c) = −E(c,x)/kT + const_c` on the sampled support.
(b) Because `p_θ(·|c)` is a *normalised* density (FFJORD is an exact change of variables
of a normalised prior), that constant is *forced* to be `−log Z(c)`; hence the variance
form is not merely "Boltzmann up to a constant" — at the population optimum it is exactly
`π(·|c)`.
(c) `Z(c)` is never formed: the within-parent centring subtracts it. The estimator is
therefore invariant to the intractable normaliser *and* to any per-parent offset of the
`log p` estimator (an important robustness property given (d)).
(d) `L_energy` is exactly invariant to adding any function that is constant within each
parent. **Its null space is precisely the per-composition free-energy gauge `{A(c)}`.**

Part (d) is the bridge to §3.3: the variance loss fixes the *shape* of `log p_θ(·|c)` and
leaves `A(c)` free; the anchor loss + `LogZPredictor` (`log_z_predictor.py`,
`energy_anchor_loss`) is what fixes `A(c)`. That is the code-level realisation of
Corollary 1 (division of labour).

> **Correction to a code comment.** `bgfm_train_hook.py:230-233` and `bgfm_density.py:361-365`
> say the anchor prevents a "trivial-constant failure mode (model satisfies Var by predicting
> constant log p per parent)". That is **not** a failure mode of (8): if `log p` were constant
> within a parent, `w = const + E/kT` and `Var_k(w) = Var_k(E)/kT² > 0`, which the loss
> penalises maximally. The anchor's real role is (d) — pinning the per-parent additive
> constant, i.e. the free energy. Say the correct thing in the paper; it is a *better* story.

**Why cross-batch variance is broken (keep this remark, it is a good one).** Across
molecules, (7) holds with a *molecule-dependent* constant `−log Z(c)` whose spread is
dominated by size and composition; the cross-batch variance therefore measures `log Z`
spread (~1e9 observed), not Boltzmann deviation. This is the same Simpson-type trap as
pooling the evaluation correlation across molecules — one paragraph can cover both, and it
motivates both the per-parent loss and the per-group metric. (`bgfm_density.py::
energy_consistency_loss` is retained only as a deprecated reference; the live path is
`within_group_variance_loss`.)

### 2.5 Proposition 4 — the training loss is a monotone surrogate for the reported metric

This is the tightest theory↔experiment link in the paper. **Write it out.**

Fix a parent `m` with `K` eval perturbations. Let `u_k = −E_k/kT` (the independent xTB
energy at eval time), `L_k = log p_θ(x_k)`, and `w_k = L_k − u_k` (the Boltzmann residual
of (7)). Write `τ = sd_k(w)/sd_k(u)` and `ρ = corr_k(w,u)`. Then, from
`Cov(L,u) = Var(u)(1+ρτ)` and `Var(L) = Var(u)(1+2ρτ+τ²)`,

```
r_m = (1 + ρτ) / sqrt(1 + 2ρτ + τ²) ,   and if ρ = 0:   r_m = 1/sqrt(1 + τ²).   (9)
```

**Consequences to state.**
* `r_m` is a strictly decreasing function of `τ` (at `ρ = 0`), and `L_energy` for parent `m`
  is exactly `τ² · Var_k(u)`. **Minimising `L_energy` is minimising the metric's residual.**
  No other term in `L_total` has this property.
* Inverting (9), `τ = sqrt(1/r² − 1)`, converts the reported `r` into an interpretable
  residual-to-signal ratio (derived quantities, label them as such):

  | run | reported `r` | implied `τ = sd(w)/sd(u)` |
  |---|---|---|
  | energy, true labels [A] | +0.430 | 2.10 |
  | no-physics baseline [A] | +0.075 | 13.30 |
  | energy, Y-scrambled [C] | +0.339 | 2.78 |
  | force-only [B] | +0.109 | 9.12 |
  | tmQM energy [D] | +0.365 | 2.55 |

  i.e. the energy term cuts the Boltzmann residual from **13.3× to 2.1×** the energy
  spread — a **6.3× reduction** in residual. This is a faithful restatement of the same
  measurement and reads far stronger than "r went from 0.08 to 0.43". Recommended for the
  abstract/intro, *with* the `ρ = 0` idealisation stated in a footnote.
* Caveats to state alongside: (9) is per-molecule while we report the mean over molecules
  (Jensen); `ρ = 0` is an idealisation (a systematic `log p` bias that correlates with `E`
  makes `ρ ≠ 0`); a negative `r` (as in force-only tmQM, −0.158) means `ρ ≠ 0` and (9) does
  not apply.
* **`r` is invariant to `kT`.** Pearson `r` (and `R²`) are invariant under positive affine
  maps of either variable, so our headline metric certifies "`log p_θ` is affine in `E`" but
  **cannot certify the temperature**. The slope of the regression is the temperature test;
  `eval_boltzmann_independent.py` already writes a `slope` column but no slope numbers are
  reported. → Limitations (§6.2), and a cheap high-value experiment.

### 2.6 FFJORD log-density and the Hutchinson estimator

**Instantaneous change of variables.** For the flow `dx/dt = v_θ(x,t)` transporting
`p_0 = N(0,σ²I)` to `p_1^θ`,

```
d/dt log p_t^θ(x_t) = − ∇·v_θ(x_t, t)   ⟹   log p_θ(x_1) = log p_0(x_0) − ∫_0^1 ∇·v_θ(x_t,t) dt,   (10)
```
where `x_0` is obtained by integrating the ODE *backwards* from `x_1`.

**Code check (sign).** `bgfm_density.py::log_density_via_flow` integrates `t: 1 → 0` with
midpoint times `t = 1 − (step+½)dt`, accumulates `logp_integral += dt·div`, steps
`x ← x − dt·v`, and returns `logp0 − logp_integral`. This is exactly (10). ✔

**Prior term.** `gaussian_prior_log_density` uses effective dimension `d = 3(N−1)`
(COM removed) with `Σ‖x_i‖²` in ambient coordinates — correct, because COM removal is an
orthogonal projection and hence an isometry onto the `3(N−1)`-dim subspace.

**Hutchinson.** `tr J = E_ξ[ξᵀJξ]` for any `ξ` with `E[ξ]=0`, `E[ξξᵀ]=I`; Rademacher
halves the variance versus Gaussian. One backward pass per probe via
`ξᵀJξ = ∇_x(v·ξ)·ξ`. `divergence_exact_atomwise` (3N backward passes) is the unit-test
reference. Two assumptions worth one sentence each:

* **(A-trans)** The ambient `3N`-dim Hutchinson trace equals the intrinsic `3(N−1)`-dim
  trace provided `v_θ` is invariant to global translation (then the directional derivative
  along the three COM directions vanishes and contributes nothing to the trace). FlowMol3's
  COM-removed equivariant field satisfies this.
* **(A-disc)** `log p` is a *first-order-discretised* CNF likelihood (4 steps in training,
  12 at eval) with a stochastic trace. It is a biased, noisy estimate of the true model
  log-density. §6.3 spells out the consequences.

**Truncated gradient (must be disclosed).** In `log_density_via_flow`, the trajectory is
advanced under `torch.no_grad()` and each `x` is `detach()`ed before the divergence is
taken, and `logp0` is computed from a detached `x_0`. Therefore the gradient actually
back-propagated is

```
∇_θ L_energy  ≈  Σ_steps dt · ∇_θ ∇·v_θ(x_step, t_step)      (trajectory held fixed)
```

i.e. the terms `∂logp_0/∂x_0 · ∂x_0/∂θ` and the trajectory's `θ`-dependence are dropped.
This is a *frozen-trajectory / truncated-adjoint* gradient, not the exact
`∇_θ log p_θ(x_1)`. It is a defensible surrogate (it is the dominant term and it avoids a
full second-order adjoint), but the paper must say so — do not write "we backpropagate
through the exact likelihood".

---

## 3. What goes in the main text (paste-ready statements)

Recommended main-text budget: **one lemma + two theorem-level results + one corollary**,
≈ ¾ page, all proofs one-liners or deferred. Everything else stays in the appendix.

### 3.1 Main text, Lemma 1 (Score read-off) — §2.1 above

> **Lemma 1 (Flow-matching score read-off).** Let `x_t = (1−t)x_0 + t x_1` with
> `x_0 ~ N(0,σ²I) ⫫ x_1 ~ p_data`, and let `v*(x,t) = E[x_1 − x_0 | x_t = x]` be the
> marginal velocity. Then for every `t ∈ [0,1)` the score of the time-`t` marginal is
> `∇_x log p_t(x) = (t v*(x,t) − x)/((1−t)σ²)`. The identity degenerates at `t = 1`: a
> velocity perturbation `δv` moves the implied score by `t‖δv‖/((1−t)σ²)`.

Follow immediately with **Proposition 2** (§2.2) in two sentences and a footnote — the
bias–variance trap is what earns the negative result its place in the main text.

### 3.2 Main text, Theorem 1 (Boltzmann certificate / zero-variance identity)

This is `cor:readout` of `appendix_hbc.tex`, promoted and split so that only the
empirically exercised half is asserted in the main text.

> **Assumptions.** (A1) `E(c,·)` is the teacher potential at fixed `(N,q,s)`, smooth in `x`;
> (A2) `p_θ(·|c) > 0` on the sampled support and is a normalised density on the COM-removed
> subspace; (A3) the FFJORD log-density is evaluated on that same subspace with a
> translation-invariant velocity field (A-trans).
>
> **Theorem 1.** Fix a composition `c` and define the Boltzmann residual
> `w(x) = log p_θ(x|c) + E(c,x)/kT`. Then
> (i) `p_θ(·|c) = π(·|c) = e^{−E/kT}/Z(c)` on the support **iff** `w` is constant on the
> support, in which case `w ≡ −log Z(c) = A(c)/kT`;
> (ii) consequently, for any sampling distribution `ρ` supported there,
> `Var_{x∼ρ}[w(x)] = 0` and `E_{x∼ρ}[w(x)] = A(c)/kT`;
> (iii) `Var_{x∼ρ}[w]` is an *unnormalised-teacher-free and `Z`-free* discrepancy: it is
> invariant to `E → E + g(c)` and requires no evaluation of `Z(c)` or `𝒵`.
>
> **Corollary (what the loss identifies).** The empirical version
> `L_energy = M^{-1}Σ_m Var_k(w_{m,k})` has null space exactly `{functions constant within
> each parent}` — i.e. it determines `log p_θ(·|c)` up to the per-composition free-energy
> gauge `A(c)`, and no further.

**Main-text sentence to attach:** "Part (ii) also says the *mean* that the variance form
discards is the conditional free energy — the quantity the hierarchical extension needs.
We use the variance half in this paper; the free-energy readout is evaluated in
Appendix X and left to future work." (See §5 — the mean half is **not** measured.)

### 3.3 Main text, Theorem 2 (Gauge-identifiability) + Corollary (Division of labour)

`lem:gauge` + `thm:gauge` + `cor:labor`, compressed. Keep the assumptions visible.

> **Assumptions.** (B1) For each `c`, the coordinate domain is connected and `E(c,·)` is
> `C¹`; (B2) the estimator has access only to `{∇_x E(c,x)}` (forces / scores), not to
> absolute energies; (B3) compositions are compared through the joint density (i.e. the
> quantity of interest includes the composition marginal).
>
> **Lemma 2 (force gauge).** `∇_x E ≡ ∇_x E'` for all `(c,x)` iff `E'(c,x) = E(c,x) + g(c)`
> for some `g : 𝒞 → R`.
>
> **Theorem 2 (gauge-identifiability).** Under (B1)–(B3), force information identifies the
> conditional Boltzmann `π(·|c)` **exactly, for every `c`**, but identifies the joint
> `π(c,x)` **only up to the transformation `E → E + g(c)`**, which leaves every force and
> every conditional invariant while reweighting the joint by `e^{−g(c)/kT}` and shifting
> `A(c) → A(c) + g(c)`. Hence no estimator consuming only forces/scores — flat force
> matching, denoising energy matching (iDEM/iEFM), or the gradient signal of
> adjoint/stochastic-optimal-control samplers — determines `π(c,x)`; the unidentifiable
> degrees of freedom are exactly the per-composition free energies `{A(c)}`.
>
> **Corollary 1 (division of labour).** The two-level factorisation
> `p_{θ,φ}(c,x) = p_φ(c) p_θ(x|c)` is the minimal parameterisation that confines the entire
> unidentifiable gauge to a single object. Forces constrain `p_θ(·|c)`; absolute energies,
> through the readout of Theorem 1(ii), constrain the gauge `{A(c)}` and hence `p_φ`.
> Neither supervision term is redundant.

**Critical framing instruction.** Theorem 2 is a statement about *identifiability*, i.e.
about what forces are **insufficient** for. It is **not** a statement that forces are
*harmful*, and it is **not** the explanation of experiment [B] — see §4.3. Write the two
mechanisms as two separate claims. Conflating them would be a real logical error that a
reviewer will catch.

### 3.4 Stays in the appendix (with the labels from `appendix_hbc.tex`)

| Label | Result | Why appendix |
|---|---|---|
| `thm:hbc` | Hierarchical Boltzmann Consistency (`p=π` iff (i)+(ii)) | elementary identity; condition (ii) is **never trained** in this paper |
| `cor:reward` | composition reward `R(c) = log Z(c)` closes the loop | level 1 not implemented; purely prospective |
| `prop:physics` | the factorisation is the statistical-mechanical one | interpretive, no test |
| `thm:additive` | `KL(π‖p) = KL_c + E_c[KL_{x|c}]`, no cross term | exact identity, but `ε_c`, `ε_r` unmeasured |
| `prop:selfconsistent` | plug-in `Â(c)` bias `≤ kTδ(c) + Δ_E` | `δ(c)`, `Δ_E` unmeasured |
| `prop:variance` | variance collapse / scalar reweighting, ESS argument | **ESS never measured** — see §5, must be reworded |
| `ass:rigid`, other assumptions | rigid-motion / symmetry factors, estimator regularity, `(q,s)` well-posedness | keep, and update per §7 |

---

## 4. Theory ↔ experiment correspondence (write this table out in the paper)

### 4.1 The table

| # | Theoretical claim (where) | Direction of prediction | Experiment | Numbers | Verdict |
|---|---|---|---|---|---|
| 1 | `L_energy = 0` ⟺ conditional Boltzmann on the support; and `L_energy` is a monotone surrogate for per-group `r` (Thm 1, Prop 4) | training on the energy term should raise per-group `r` a lot | [A] energy vs no-physics, n=120, 4 seeds each, Welch | `r = 0.430 ± 0.033` vs `0.075 ± 0.004`; `Δ = +0.355`, `t ≈ 10.8`; implied residual ratio `τ`: 13.3 → 2.1 | **CONFIRMED, strongly.** The headline. |
| 2 | The `Z`-free variance form works only *within* a molecule; cross-molecule pooling measures `log Z` spread (§2.4) | pooled correlation must be computed per group, else Simpson | metric design + the observed `~1e9` cross-batch `L_energy` | live loss = `within_group_variance_loss`; metric = mean per-group `r`; `eval_boltzmann_independent.py` pools only after per-group mean-centring | **CONFIRMED** (design-level; the 1e9 observation is the evidence) |
| 3 | Force matching at any usable `t<1` is bias–variance trapped (Prop 2), **and** its population target conflicts with `L_FM` when `p_data ≠ π` (Prop 3) | adding `L_force` need not help and can hurt | [B] force-only vs no-physics (same architecture, n=30 round) | `r = 0.109` vs `0.223` — force is **worse** than no physics | **CONFIRMED** (this is the honest reading; see §4.3 for what does *not* explain it) |
| 4 | Same, at the composition-shift extreme where the smoothed-marginal proxy is worst | force should fail hardest OOD | [D] tmQM (Cr/Fe/Mn/Mo/Ru/Ti/V, 3000 complexes) | force-only `r = −0.158`; energy `r = +0.365` (median 0.429, 37 % of molecules `r>0.5`); 270/270 xTB single points converged (no survivorship bias) | **CONFIRMED**, and the sign flip is a strong figure |
| 5 | Forces alone identify `π(x\|c)` but not the joint; the free gauge is exactly `{A(c)}` (Thm 2) | a force-only model is "Boltzmann up to a per-molecule constant" | **not directly testable with the current metric**: per-group Pearson `r` is shift-invariant, so it is *blind* to the gauge | — | **NOT TESTED.** State it as theory; do not cite [B] as its evidence. |
| 6 | Division of labour: variance form fixes shape, absolute-energy anchor fixes `A(c)` (Cor 1, Prop 3a(d)) | with the anchor off, `log p` levels are free per molecule | the headline runs a3/a6 have **λ₃ absent (=0)** | anchor + `LogZPredictor` were **not active** in [A]/[C] | **NOT TESTED.** Present Cor 1 as motivation for the anchor, flag the anchor as future/appendix work. |
| 7 | Y-scrambling: within-parent energy *set* and *variance* are unchanged; only the geometry↔energy pairing is destroyed, so the loss retains a density-*calibration* signal but loses the *ranking* signal (§4.2) | scrambled control should retain part, not all, of the gain | [C] n=120: scrambled 3 seeds `[0.284, 0.315, 0.417]`, mean `+0.339`; real `+0.430`; `Δ = +0.092`, `t ≈ 1.77` | within-parent energy std identical (77.6 kcal/mol real = scrambled) | **PARTIALLY CONFIRMED — currently NOT significant at n=120.** Placeholder rules apply (§4.4). |
| 8 | Effect decomposition into a calibration component and a Boltzmann-information component (§4.2) | most of the gain survives scrambling; a residual needs true pairing | [G] | +0.224 of +0.355 (63 %) survives; +0.131 (37 %) needs true pairing | **CONFIRMED as arithmetic, but see the seed-set inconsistency in §4.4 — 63/37 uses the 2-seed scrambled mean; the 3-seed mean gives 74/26.** |
| 9 | The variance estimator with `M = energy_b_parents` parents per step has variance `∝ 1/M`; a single-parent estimate is uncontrolled (Thm 1 is a population statement) | small `M` ⟹ gradient spikes ⟹ divergence | [F] `b_parents=1`: `L_energy` observed jumping 48 → 974; 3/9 seeds diverged; `b_parents=8` + `energy_loss_cap`: 2/2 stable | code: `energy_b_parents`, `energy_loss_cap`, `FiniteWeightGuard` | **CONFIRMED.** Also the engineering contribution: the pre-existing NaN guard zeroed the loss and *kept training*, hiding the failure for ~1 GPU-day. |
| 10 | Bond-free is required because the teacher has no bond labels; theory only ever needs `E(c,x)` and `∇_x E` | a bond-free generator can still be Boltzmann-scored | [E] OMol25 de-novo table | ours 43.0 / 32.0 / 23.0 / 100 vs Zatom-1 30.4 / 17.0 / 15.1 / 93.5 (Zatom-1 **non-converged at 80 epochs — must be labelled**); xTB relaxation ΔE/atom median 6.90 kcal, 3 % failures | **CONSISTENT** (capability, not a theory test) |
| 11 | Variance collapse / no ESS collapse (`prop:variance`) | hierarchical reweighting has `3N`-independent variance | `ess_frac` **is implemented** in `eval_boltzmann_independent.py` but **no numbers reported** | — | **NOT TESTED.** Must not claim "explains the absence of ESS collapse". |
| 12 | Free-energy readout `A(c) = E + kT log p_θ` (Thm 1(ii), mean half) | `Â(c)` should be low-variance and predictive of relative populations | [P5] no ΔF / ESS script results | — | **NOT TESTED.** |

### 4.2 The two flagship correspondences, spelled out

**(a) "Forces fix gradients, energies fix the density" ↔ [B] + [A].**
Do **not** write the naive version — *"`L_force` only constrains `∇ log p`, so it cannot
determine the density"*. **That sentence is mathematically false**: on a connected domain
at fixed composition, the gradient field determines the density completely (up to a
normalisation that normalisation itself fixes). That is precisely Theorem 2's first half.
The correct three-part statement is:

1. **Across compositions** the gradient is genuinely insufficient — the free-energy gauge
   `{A(c)}` is invisible to forces (Theorem 2). *(But note our metric cannot see this
   either — row 5.)*
2. **At any usable `t`** the FM read-off gives the score of the *smoothed* marginal `p_t`,
   not of `p_1`, and the read-off amplifies velocity error by `1/(1−t)`: bias `O(1−t)`,
   injected gradient noise `O((1−t)^{−2})` — no `t` is safe (Prop. 2).
3. **The population targets conflict.** `L_FM` drives `s_θ → ∇log p_t^{data}`, `L_force`
   drives `s_θ → ∇log π`; these differ unless the data is already Boltzmann, which OMol25
   is not (Prop. 3). A fixed `λ₁` moreover weights the conflicting target by `(1−t)^{−2}`.

Against this, `L_energy` is a *value* constraint on exactly the quantity the metric reads
(`log p`), at exactly the geometries the metric probes (within-parent perturbations), with
`Z` cancelled and — by Prop. 4 — with population optimum equal to the metric's optimum.
So the theory predicts a large energy effect and an unreliable force effect. Measured:
`+0.355` (t≈10.8) for energy; `−0.114` for force-only vs. no-physics. **Write this as the
paper's central consistency argument.**

**(b) Division of labour ↔ [G]'s 63 % / 37 % decomposition.**
The honest mechanism for why the Y-scrambled control still helps:

* Scrambling permutes the geometry↔energy pairing **within** a parent. The multiset
  `{E_{m,k}}` and hence `Var_k(E)` are untouched (measured: within-parent std 77.6 kcal/mol
  in both arms). What changes is only the *assignment*.
* Under a scrambled pairing, the minimiser of `Var_k(log p(x_k) + E_{σ(k)}/kT)` still
  requires `log p` to have the *same spread* over the `K` geometries as `−E/kT` does —
  it is a **dynamic-range / calibration constraint** on the model's log-density. What it
  loses is the *ranking*: which geometry gets which value.
* `p_θ` is a smooth function of geometry, so it cannot realise a random permutation of a
  smooth target. The projection of the scrambled target onto the smooth hypothesis class
  retains the leading smooth component, which correlates with the true energy surface.
  Hence a substantial positive `r` from the scrambled arm is *expected*, not anomalous.
* Therefore: **calibration component** (log-density spread; survives scrambling) +
  **Boltzmann-information component** (correct ranking; requires true pairing). This is a
  refinement of Corollary 1's shape/gauge split at the level of a single molecule.

This is a genuinely valuable ablation because it rules out the trivial explanation "any
extra auxiliary loss with roughly this scale would have helped": the scrambled arm has the
*same* loss scale, *same* energy statistics, *same* λ₂, *same* seeds count — only the
pairing differs.

### 4.3 What the theory does **NOT** explain (guard rail)

* **Theorem 2 does not predict [B].** Per-group Pearson `r` measures the *conditional*
  Boltzmann shape (condition (i)), which Theorem 2 says forces *do* identify in the
  population limit, and `r` is shift-invariant so the gauge is invisible to it. If forces
  were the exact score constraint, `L_force` should have *helped* row 5's target. It hurt.
  The explanation must be Prop. 2 + Prop. 3 (finite-`t` estimator + objective conflict),
  not identifiability. Any sentence of the form "as predicted by gauge-identifiability,
  the force term underperforms" is wrong and must be cut.
* **`force+energy (+0.294) < energy-only (+0.420)` is CONFOUNDED.** `configs/sweep/a4_force_energy.yaml`
  uses `energy_b_parents: 1` and **no** `energy_loss_cap`, while
  `configs/sweep/a3_energy_only.yaml` uses `energy_b_parents: 4` and `energy_loss_cap: 3000`.
  That is exactly the pathological configuration that [F] identifies as the cause of
  divergence. a4 also runs `lambda_onpolicy: 0.1` (a2 as well), so "force" in the ablation
  means off-policy force MSE **plus** on-policy teacher distillation. **Report the a3-vs-a4
  comparison only with these two confounds stated in the caption**, or drop it and rest the
  force-is-harmful claim on the clean a2-vs-a1 comparison (`0.109` vs `0.223`, same
  `λ₂ = 0`, differing only in `λ₁`/`λ_op`). Recommended: rest on a2-vs-a1, mention a4 as
  suggestive-but-confounded.
* **Nothing in the theory predicts the *magnitude* 0.43.** `r` is bounded above by the
  agreement between the training potential (OMol25/eSEN) and the evaluation potential
  (GFN2-xTB) on these perturbations, and further attenuated by `log p` estimator noise
  (§6.3). Both ceilings are unmeasured. Do not present 0.43 as approaching a theoretical
  limit; present it as `6.3×` residual reduction over baseline (Prop. 4) and note the
  ceilings.

### 4.4 Placeholder handling for rows 7 and 8 (follow the user's rule exactly)

* Row 7 (Y-scrambling): write the analysis in the **positive** direction as instructed —
  `Δ ≈ \pending{+0.13}`, `t ≈ \pending{3.9}`, `p < \pending{0.05}` at n=240 — **and** state
  in the same subsection and in the PLACEHOLDER INVENTORY that at the current n=120 the
  observed values are `Δ = +0.092`, `t = 1.77`, **not significant**, and that the earlier
  2-seed result (`t = 3.63`) weakened when a third scrambled seed was added, i.e. this
  second-order effect is sample-size sensitive. n=240 is running ([P3]).
* Row 8 (63 %/37 %): **arithmetic inconsistency to resolve.** `+0.224/+0.355 = 63 %`
  corresponds to a scrambled mean of `0.2995`, which is the **2-seed** mean of
  `[0.284, 0.315]`. The current **3-seed** mean from [C] is `0.339`, which gives
  `0.264/0.355 = 74 %` calibration and `0.091/0.355 = 26 %` Boltzmann-information.
  Options, in order of preference: (a) report 74 %/26 % from the 3-seed mean and footnote
  that the 2-seed mean gave 63 %/37 %; (b) report 63 %/37 % but state explicitly that it
  uses the 2-seed scrambled mean. **Do not report 63 %/37 % next to the 3-seed
  `mean = +0.339` without an explanation** — the numbers do not add up and a reviewer will
  check. Wrap the split in `\pending{}` if it will be recomputed at n=240.

---

## 5. Honest labelling: appendix results with **no** experimental support

State this explicitly, ideally in one short "Scope of the theory" paragraph plus the
PLACEHOLDER INVENTORY. Nothing in the main text may imply these were tested.

| `appendix_hbc.tex` result | Empirical status | Required wording |
|---|---|---|
| `thm:hbc` (i)+(ii) | Condition (i) is the object of all our experiments. **Condition (ii) is never trained**: `p_φ(c)` does not exist in the codebase (`PROJECT.md §12`: "Level 1 … next phase"). | "This paper trains and evaluates the conformation level (condition (i)); the composition level is a prospective consequence of the same identity." Never write "HBFM generates compositions in proportion to `e^{−A(c)/kT}`" in the present tense. |
| `cor:readout` **mean** half (`A(c) = E + kT log p_θ`) | **Not measured.** No ΔF numbers, no `Â(c)` variance numbers ([P5]). The `LogZPredictor` / `energy_anchor_loss` path exists but `λ₃ = 0` in the headline runs a3/a6. | "The mean is the conditional free energy; we do not evaluate free-energy accuracy in this work." |
| `cor:readout` **variance** half | Exercised: it *is* the training loss, and by Prop. 4 it is monotonically tied to the reported metric. | fine to assert |
| `cor:reward` (composition reward closes the loop) | **Not implemented, not measured.** | "future work" / "outlines how the loop closes"; strictly future tense |
| `thm:additive` (additive KL, no cross term) | Exact identity, correct. But `ε_c` and `ε_r` are **never estimated**, so the bound is not instantiated. | "we do not report a numerical instantiation of this bound" |
| `prop:selfconsistent` | `δ(c)` and `Δ_E` **not measured**. | future work |
| `prop:variance` (variance collapse / ESS) | **`ess_frac` is implemented** in `eval_boltzmann_independent.py` (weights `w = e^{−E/kT}/p_θ`, shift-invariant) but **no ESS numbers exist**. The appendix currently asserts the hierarchy "explain[s] the absence of ESS collapse" — that phrase must be deleted or made conditional. | "we do not report ESS"; and the cited 0.04–0.5 ESS figures are *other papers'* numbers, so attribute them |
| `prop:physics` | Interpretive; no test needed. | fine, but present as interpretation not result |
| `lem:gauge` / `thm:gauge` | Mathematically sound. Consistent with force insufficiency; **not** evidence-backed by [B] (§4.3) and **not** probed by our shift-invariant metric (row 5). | present as theory only |
| The "oral-level" paragraph in `appendix_hbc.tex` ("Why this is oral-level", the four mutually-reinforcing claims) | Self-assessment, not a result. | **delete before submission** |

**Additional claims that must not appear** (from the project's wording discipline):
zero-shot to unseen elements (the transition metals *are* in OMol25 training, as a sparse
tail: 0.34 % of atoms, 0.7 M/205 M, 15 k–51 k atoms per metal — the correct framing is
"sparse tail + distribution shift to a different TM dataset"); beating all baselines (we
beat a non-converged Zatom-1 and our own properly-converged FlowMol3-bond-free control
a1_fm_only); Table 3's N/A entries are a *capability* argument (other methods have no exact
`log p`), not a performance win.

---

## 6. Caveats the theory forces into Limitations

### 6.1 The training temperature is not physical
`bgfm.kT = 1.0 eV` in every headline run (`a1`–`a6`), i.e. ≈ 11 600 K, chosen to keep
`E/kT` and `F/kT` numerically tame. The variance-form loss is scale-covariant, not
scale-invariant (`Var(log p + E/kT)` depends on `kT`), so the trained object is the
Boltzmann distribution *at that `kT`*, not at 298 K. The `v8a_room_T` / `v8b_T_conditional`
configs exist for this; the headline result does not use them. Say so.

### 6.2 The metric certifies shape, not temperature
Pearson `r` / `R²` are invariant to positive affine maps, so `r` cannot distinguish
`p ∝ e^{−E/kT}` from `p ∝ e^{−E/kT'}` for any `T'>0`. The per-group regression **slope**
would test the temperature and is already written to the CSV by
`eval_boltzmann_independent.py`; no slope numbers are reported. → Limitation + cheap
experiment.

### 6.3 The measured `r` is attenuated by `log p` estimator error (this favours us — say it carefully)
Eval `log p` uses 12 first-order reverse-Euler steps and 4 Hutchinson probes, so it carries
both discretisation bias and stochastic trace noise. Classical attenuation gives
`r_meas ≈ r_true / sqrt(1 + Var(noise)/Var(log p))`, so the reported `r` is a **lower
bound** on the true correlation. Two consequences: (i) the honest statement is "our metric
under-reports"; (ii) an inexpensive, high-value check is to recompute a subset with exact
atom-wise divergence (`n_hutchinson = 0` triggers `divergence_exact_atomwise`) and/or more
ODE steps, and report the `r` change. Additionally, geometry-dependent discretisation bias
enters `Var_k(w)` as an estimator floor, which bounds how small `L_energy` — and hence
`τ` — can get; part of the gap between `r = 0.43` and `r = 1` is estimator, not model.

### 6.4 Independent-potential evaluation has an irreducible ceiling
Training energies are OMol25/eSEN; evaluation energies are GFN2-xTB. This removes oracle
circularity (a real strength — say it), but a *perfect* OMol25-Boltzmann model would still
score `r < 1` against xTB. The OMol25–xTB agreement on these perturbation ensembles is
unmeasured. Present it as a ceiling, not as a defect.

### 6.5 Rotations are not quotiented
`gaussian_prior_log_density` uses effective dimension `3(N−1)` — COM removed only. The
appendix's `ass:rigid` claims the density lives in "the COM-removed, **rotation-quotiented**
space that the FFJORD density actually models"; the code does **not** quotient rotations.
Fix the assumption text. Mitigating fact worth stating: the within-parent design protects
the metric — all `K` perturbations of a parent are small (`σ = 0.15 Å`) displacements of one
reference geometry in one frame, so any rotational volume factor is approximately constant
within a group and cancels from the per-group Pearson `r` (and is exactly annihilated by
the within-parent variance loss). This is another reason the per-group metric is the right
one.

### 6.6 Charge is threaded, spin is not
`bgfm_train_hook.py::_mol_total_charge` decodes the total charge from the frozen charge
channel and passes it to the teacher, so `appendix_hbc.tex`'s third assumption ("the client
hardcodes `q=0, s=1`") is **half stale**: charge is now correct, but `mol_spin` is never
passed (`physics_drift.py` supports it; the hook omits it) so the teacher defaults to
`s = 1` (singlet). For radicals and open-shell TM complexes this is the wrong electronic
state. Update the assumption and list it as a limitation.

### 6.7 The implemented losses are clipped/Huberised versions of the stated ones
`s_θ` is norm-capped at 1000 per atom; `L_force` is Huberised above per-atom squared error
`1e4`; the anchor clamps `log p` at `±1e6`; `L_energy` above `energy_loss_cap` is rejected
outright. One honest sentence in the method: "for numerical stability the implemented
objectives are clipped/Huberised variants of (eqs.); the clip thresholds are given in
Appendix Y and are inactive for a converged model."

---

## 7. Fix-list before the appendix is integrated (do not silently paper over these)

1. `F(c) → A(c)` throughout `appendix_hbc.tex` (force/free-energy collision, §1.1).
2. `r → x` for coordinates throughout `appendix_hbc.tex` (§1.1).
3. Delete the "**Why this is oral-level**" paragraph (self-assessment, and it asserts the
   untested ESS claim).
4. `prop:variance`: remove "explaining the absence of ESS collapse" — we have not measured
   ESS. Attribute the 0.04–0.5 ESS range to the cited papers.
5. `ass:rigid`: delete "rotation-quotiented" (§6.5) or state the code does COM removal only.
6. Third assumption: update — charge is now decoded and passed; **spin** is still hardcoded
   to 1 (§6.6).
7. Stale file reference: the appendix points at `notes/appendix_A_v4.tex`-era names and at
   `scripts/level3_bgfm_sample.py`; verify those paths exist in the submitted artifact list.
8. Correct the "trivial-constant failure mode" comment in `bgfm_train_hook.py:230-233` and
   `bgfm_density.py:361-365`; the anchor's role is gauge-fixing (§2.4).
9. Disclose the frozen-trajectory (truncated-adjoint) `L_energy` gradient (§2.6). Do **not**
   write "we differentiate the exact likelihood".
10. `kT_conditioning` is patched **only inside** the `if energy_enabled:` block of
    `patch_flowmol_bgfm` (lines 634–682), so a config with `kT_conditioning: true` and
    `λ₂ = 0` silently trains **without** kT conditioning while `bgfm_training_step` still
    samples `kT_step` per step. If any reported run has that combination, its "T-conditional"
    label is wrong. Check `v8b` before citing it.
11. `bgfm_loss.py::energy_loss_variance` is the **cross-batch** (deprecated) form and
    `compute_bgfm_step` raises `NotImplementedError`; the live path is
    `bgfm_density.py::within_group_variance_loss`. Cite the right function in the paper.
12. The [G] 63 %/37 % vs [C] 3-seed arithmetic (§4.4) must be reconciled in the text.

---

## 8. Suggested main-text theory subsection skeleton (≈¾ page)

```
2.x  From flow matching to physics
     Lemma 1 (score read-off, eq. 4) + one-line proof sketch.
     Remark: 1/(1-t) amplification; the three t_eval values and their factors 6.7/12.5/32.3.
     Proposition 2 (bias-variance trap) — 3 sentences, proof in appendix.
     Proposition 3 (objective conflict when p_data != pi) — 3 sentences. State that OMol25
       is not an equilibrium sample, so the antecedent holds. Forward-reference the
       negative result in Sec. 5.
2.y  The value condition: Boltzmann certificate
     Theorem 1 (zero-variance identity) with assumptions A1-A3.
     Corollary: null space = per-composition gauge {A(c)}; hence L_energy fixes shape,
       an absolute-energy anchor fixes A(c).
     Proposition 4 (r = 1/sqrt(1+tau^2)): the loss is a monotone surrogate for the metric.
       This is the sentence that justifies the whole experimental design.
     Remark: within-parent grouping cancels Z and is required (cross-batch reads 1e9).
2.z  What forces cannot do
     Lemma 2 + Theorem 2 (gauge-identifiability) with assumptions B1-B3.
     Corollary 1 (division of labour).
     Explicit guard sentence: Theorem 2 is about insufficiency across compositions; the
       empirical underperformance of L_force is explained by Props 2-3, not by Theorem 2;
       our shift-invariant metric cannot probe the gauge.
2.w  Scope of the theory  (3 sentences)
     Trained/evaluated: condition (i) at kT = 1 eV. Not evaluated in this paper:
     free-energy readout, composition level, ESS, additive-KL instantiation.
```

---

## 9. One-line summary for the drafting agent

Write the theory as *two* claims, kept strictly separate: **(1) the value/energy condition
is the one that is `Z`-free, equivalent to conditional Boltzmann, and provably monotone in
the metric we report — hence the large, significant `+0.355`; (2) forces are structurally
insufficient across compositions (Theorem 2) *and*, independently, a bias–variance-trapped
and objective-conflicting surrogate at any finite `t` (Props 2–3) — hence the negative
result.** Everything hierarchical (`p_φ`, `A(c)` readout, ESS, additive KL) is theory-only
in this paper and must be in the future tense.
