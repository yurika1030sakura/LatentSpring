# SPINE.md — the single source of truth and writing contract for the BGFM/HBFM manuscript

**Status:** binding. Written 2026-08-03 by the editor-in-chief agent after reading
`RESEARCH_story.md`, `RESEARCH_theory.md`, `RESEARCH_figures.md`, `TEMPLATE_NOTES.md`
and the project fact sheet.

**Rule for every section writer:**

1. You may not introduce a number that is not in §7 of this file.
2. You may not introduce a symbol that is not in §5 of this file.
3. You may not make a claim that is not in the §4 claim→evidence table.
4. You may not use a sentence that §9 forbids.
5. If you believe this file is wrong, you *stop and report it*; you do not silently deviate.
   (This file already overrides three things the upstream research docs got inconsistent —
   see the RULINGS boxes. Those rulings are final.)

Target venue: **ICLR 2026**. Venue hard limit **9 pages main text**; this draft is written to
**≤12 pages** by explicit user instruction, with the compression path pre-designed in §6.3.

---

## 1. Title candidates

| # | Title | Notes |
|---|---|---|
| T-A | *Boltzmann-Guided Flow Matching: Calibrating the Density of a Bond-Free 3D Molecular Generator with a Universal Neural Potential* | Safe, descriptive, colourless. Long (17 words). |
| **T-B (RECOMMENDED)** | **Energies Calibrate, Forces Do Not: Boltzmann-Guided Flow Matching for Bond-Free 3D Molecular Generation** | Carries the headline *and* the negative result in the title. This is the paper's differentiator (see §2); putting it in the title is what makes reviewers remember it. 15 words. |
| T-C | *From Structure Emulators to Boltzmann-Consistent Generators: Flow Matching Supervised by a Universal Neural Potential* | Leads with the Klein–Noé terminology move (T4 in §10). Good if we want the framing to do the work, but hides the empirical contribution. |

**Decision: T-B.** Method name in the abstract's first method sentence
("we introduce **Boltzmann-Guided Flow Matching (BGFM)**"), not in the title's first clause.

Do **not** use "HBFM" / "hierarchical" anywhere in the title or abstract. The hierarchical
(composition-level) layer is untrained and untested in this paper (§9 ban B-7).

---

## 2. One-sentence story and three-part pitch

### 2.1 The "so what" (one sentence)

> A universal neural potential can be turned into a *training target* rather than a post-hoc
> filter, which makes a bond-free, 83-element, de-novo 3D molecular generator's own learned
> density measurably track the Boltzmann distribution — and, surprisingly, the *global*
> (energy-value) form of that supervision is what works, while the *local* (force/score) form,
> the one everybody reaches for first, makes calibration worse.

### 2.2 Three-part pitch (use this shape in the abstract, the intro, and the conclusion)

**(1) The regime nobody occupies.** Chemistry cares about Boltzmann *ensembles*, not single
structures. Two literatures each have half of what that requires. 3D molecular generators
(EDM, GeoLDM, FlowMol3, Zatom-1) generate composition and geometry de novo but carry **no
calibrated density** — they are *structure emulators*. Boltzmann generators and diffusion
samplers (TBG, SBG, ArBG, Adjoint Sampling, EWFM) have tractable densities but are per-system
or peptide-scale and **do not generate composition**. No generator is simultaneously de-novo,
periodic-table-scale, and Boltzmann-calibrated.

**(2) What makes it possible now, and what we do.** Universal MLIPs (OMol25/eSEN) supply a
transferable `E` and `∇E` over 83 elements, converting energy supervision from a per-system
luxury into a data-scale training signal. We add two physics terms to flow matching:
a *local* one (the closed-form score read-out of the FM velocity matched to `F/kT`) and a
*global* one (`Var(log p_θ + E/kT)` within a molecule, in which the partition function cancels
identically). At the joint optimum the model density equals the Boltzmann density.

**(3) What we found, including the part that hurts.** The global term works and is large:
per-group Boltzmann `r` rises 0.075 → 0.430 (Welch `t = 10.8`), scored against an
*independent* potential (GFN2-xTB). The local term does not work — it is *worse than no
physics at all* — and we report it, with a bias–variance and objective-conflict explanation.
A within-molecule label-scrambling control shows ~3/4 of the gain is generic density
calibration and ~1/4 is Boltzmann-specific pairing information.

---

## 3. Contribution bullets (5, verbatim-draftable)

Style: bolded 2–4-word type label; ≤2 sentences; a number in every bullet that has one; the
negative result gets its own bullet so it cannot be called buried. Evidence tags in `[..]`
are for *our* traceability — **strip them before compiling**.

> **Formulation and guarantee.** We introduce Boltzmann-Guided Flow Matching (BGFM), a
> bond-free 3D generator trained against a universal neural potential, in which the
> flow-matching velocity is read out in closed form as a score to impose a *local* gradient
> condition against DFT forces, and a variance-form energy objective imposes a *global* shape
> condition in which the partition function cancels exactly. We prove that the variance form
> is zero if and only if the model equals the conditional Boltzmann density on the support,
> and that it is a strictly monotone surrogate for the correlation metric we report
> (Theorem 1, Proposition 4). *[theory; Thm 1, Prop 4]*

> **Energy supervision measurably calibrates the density.** Across 120 held-out molecules and
> four independently trained seeds per arm, energy supervision raises the per-group Boltzmann
> correlation from `r = 0.075 ± 0.004` to `r = 0.430 ± 0.033` (`Δ = +0.355`, Welch `t = 10.8`),
> scored against GFN2-xTB — a reference *independent* of the potential used for training —
> which is a 6.3× reduction in the Boltzmann residual relative to the energy spread. *[A]*

> **A negative result that should change practice.** Force/score matching — the local
> condition the score read-out most naturally suggests — *degrades* Boltzmann calibration
> rather than improving it, both alone (`r = 0.109` versus `0.223` for the matched
> physics-free control) and on transition-metal complexes (`r = −0.158` versus `+0.365` for
> energy supervision). We report and explain it rather than presenting only the winning
> configuration. *[B], [D]*

> **Mechanism isolated by a negative control.** Permuting the geometry↔energy pairing *within*
> each parent molecule leaves the energy multiset and the within-parent energy spread exactly
> unchanged (77.6 kcal/mol in both arms) and alters only the correspondence; it retains ≈74%
> of the gain, attributing that share to generic density calibration and the remaining ≈26%
> (`Δ ≈ \pending{+0.13}`, `t ≈ \pending{3.9}`, `p < \pending{0.05}` at `n = \pending{240}`) to
> Boltzmann-specific information. *[C], [G]; the n=240 statistics are placeholders — see §8*

> **A reproducible failure mode, diagnosed and fixed.** Single-parent estimation of the energy
> objective has such high variance (observed jumps 48 → 974 between steps) that gradient
> spikes drive weights to NaN in 3 of 9 seeds, while a conventional NaN guard silently zeroes
> the loss and *keeps training*, masking the failure for about a GPU-day. Increasing the parent
> batch to 8, capping the energy loss at 1500, and aborting on non-finite weights gives 2/2
> stable seeds. *[F]*

*Optional 6th bullet — include only if §6 shows slack, otherwise fold into §9 of the paper:*

> **Generation quality and a capability frontier.** On bond-free OMol25 de-novo generation BGFM
> exceeds the only published generator trained on the same corpus on every reported column
> (RDKit-valid 43.0 vs 30.4, connected 32.0 vs 17.0, PoseBusters 23.0 vs 15.1, uniqueness 100
> vs 93.5, at the training budget those authors report), and is the only model in that
> comparison for which an exact `log p_θ` — and therefore a Boltzmann-calibration measurement
> at all — is defined. *[E]*

---

## 4. Claim → evidence map

**Every claim the paper makes must appear in this table.** If you want to write something not
listed here, it must be added here first (report to the editor), or deleted.
Status column: **REAL** = measured, use in black; **\pending** = placeholder, must be wrapped;
**THEORY-ONLY** = mathematical statement, no experimental support, must be presented as such;
**BANNED** = do not write (kept in the table so nobody re-invents it).

| # | Claim (as it will appear) | Where | Evidence / statistic | Status |
|---|---|---|---|---|
| C1 | The FM velocity yields the score of its own time-`t` marginal in closed form, `s = (t v − x)/((1−t)σ²)`. | §5 Lemma 1 | derivation; verified against `bgfm_loss.py::score_from_fm_velocity` | THEORY-ONLY (exact) |
| C2 | The read-out amplifies velocity error by `t/((1−t)σ²)`: 6.7×, 12.5×, 32.3× at the three probe times used. | §5 Remark | `t_eval = 0.85 / 0.92 / 0.97`, `σ = 1` | REAL (arithmetic) |
| C3 | `L_force` is bias–variance trapped: no `t` makes both the `O(1−t)` bias and the `O((1−t)^{-2})` injected gradient noise small. | §5 Prop 2 | derivation (§2.2 of RESEARCH_theory) | THEORY-ONLY |
| C4 | `L_FM` and `L_force` have conflicting population minimisers unless `p_data = π`; OMol25 is not a single-temperature equilibrium sample, so the antecedent holds. | §5 Prop 3 | derivation; dataset construction | THEORY-ONLY (antecedent is a factual statement about OMol25) |
| C5 | `L_energy = 0` ⟺ `p_θ(·\|c) = π(·\|c)` on the support; `Z(c)` never appears; the null space is exactly the per-composition free-energy gauge `{A(c)}`. | §5 Thm 1 + Cor | derivation | THEORY-ONLY (exact) |
| C6 | The training loss is a monotone surrogate for the reported metric: at `ρ = 0`, `r = 1/√(1+τ²)` with `τ² Var(u) = L_energy` for that parent. | §5 Prop 4 | derivation | THEORY-ONLY (exact) |
| C7 | Energy supervision raises per-group Boltzmann `r` from 0.075±0.004 to 0.430±0.033, `Δ = +0.355`, Welch `t = 10.8`, `p ≈ 0.002`. | Abstract, §1, §7, Fig. 3(a) | [A]: n=120 held-out parents, 4 seeds/arm; seeds in §7.1 | **REAL** |
| C8 | Equivalently, the Boltzmann residual falls from 13.3× to 2.1× the energy spread — a 6.3× reduction. | Abstract, §1, §7 | Prop 4 inversion of C7; state the `ρ = 0` idealisation in a footnote | **REAL (derived)** |
| C9 | The metric must be computed per parent molecule; pooling mixes per-molecule `log Z` (Simpson's paradox). Pooled `r²` is 0.11 (energy) / 0.01 (no physics) against mean per-parent `r` of +0.43 / +0.07. | §6, Fig. 2 right panel | [A] + `runs/eval_ours/wide_*/boltz_independent.csv`; the cross-batch loss reads ~1e9 | **REAL** |
| C10 | The evaluation potential (GFN2-xTB) is independent of the training potential (OMol25/eSEN). | Abstract, §1, §6, Fig. 2 caption | protocol fact | **REAL** |
| C11 | Force supervision alone is worse than no physics: `r = 0.109` vs `0.223`, same architecture, same round. | Abstract, §1, §8.1, Fig. 3(a) | [B], the *clean* a2-vs-a1 contrast, 1 seed, n=30 parents | **REAL** (must carry the n=30 / 1-seed qualifier — see RULING R2) |
| C12 | Adding force to energy lowers the energy-only result (0.294 vs 0.420). | §8.1 only, with both confounds stated | [B]; a4 uses `energy_b_parents=1` and no `energy_loss_cap` (the pathological config of [F]) and `lambda_onpolicy=0.1` | **REAL but CONFOUNDED** — see RULING R2 |
| C13 | On tmQM transition-metal complexes the ordering reproduces and the sign flips: force-only `r = −0.158`, energy `r = +0.365` (median +0.429, 37% of complexes `r > 0.5`). | §1 P7, §9.2, Fig. 5 | [D], 3000 complexes (Cr/Fe/Mn/Mo/Ru/Ti/V) | **REAL** |
| C14 | No survivorship bias in the tmQM result: 270/270 xTB single points converged. | §9.2 | [D] | **REAL** |
| C15 | These transition metals are in the OMol25 training data, in its sparse tail (0.34% of atoms; 0.7M of 205M; 1.5×10⁴–5.1×10⁴ atoms per metal); the experiment measures transfer under distribution shift. | §1 P7, §9.2, §11 | [D] + dataset statistics | **REAL** (and mandatory whenever C13 appears — §9 ban B-1) |
| C16 | Within-parent label scrambling leaves the energy multiset and the within-parent energy std (77.6 kcal/mol) exactly unchanged; only the pairing is permuted. The config diff to the real-label arm is three lines (output dir, run name, shard path). | §8.2 | [C] | **REAL** |
| C17 | The scrambled arm still reaches `r = 0.339` (3 seeds); real-vs-scrambled `Δ = +0.092`, `t = 1.77`, `p = 0.147` — **not significant at n = 120**. | §8.2, Fig. 3(b), §11 | [C] | **REAL** |
| C18 | At `n = \pending{240}` the real-vs-scrambled contrast reaches `Δ ≈ \pending{+0.13}`, `t ≈ \pending{3.9}`, `p < \pending{0.05}`. | §8.2, Fig. 3(b), §11, PLACEHOLDER INVENTORY | [P3] running | **\pending** |
| C19 | Effect decomposition: ≈74% of the +0.355 gain (+0.264) survives scrambling and is generic density calibration; ≈26% (+0.092) requires the true pairing and is Boltzmann-specific information. | §8.3, Fig. 4 | [C]/[G] arithmetic — see RULING R1 | **REAL at n=120** (footnote the 2-seed 63/37) |
| C20 | Scrambling cannot be explained by "any auxiliary loss of this scale would have helped": same λ₂, same loss scale, same energy statistics, same seed count; only the pairing differs. | §8.2 | [C] + config diff | **REAL** |
| C21 | The variance estimator has variance ∝ 1/M in the number of parents per step; at M=1 the loss jumped 48 → 974 between steps and 3 of 9 seeds diverged to NaN weights. | §10, Fig. 6 | [F] | **REAL** |
| C22 | The pre-existing NaN guard zeroed the loss and continued training, masking the failure for about a GPU-day. | §10 | [F] | **REAL** |
| C23 | The fix (`energy_b_parents` 1→8, `energy_loss_cap = 1500`, `FiniteWeightGuard` abort) gives 2/2 stable seeds. | §10 | [F] | **REAL** (say "2 of 2", do not say "robust" — §9 ban B-9) |
| C24 | On bond-free OMol25 de-novo generation we exceed Zatom-1 on all four reported columns: 43.0/32.0/23.0/100 vs 30.4/17.0/15.1/93.5. | §9.1, Table 1 | [E], n=100 samples, one scoring script | **REAL** (Zatom-1 side needs source verification V1) |
| C25 | xTB relaxation of our samples moves them by a median 6.90 kcal/mol per atom, with a 3% relaxation failure rate. | §9.1 | [E] | **REAL** |
| C26 | Symphony / EDM / GeoLDM retrained on OMol25 score below BGFM. | Table 1 | [P1], [P2] not started/queued | **\pending** (whole rows) |
| C27 | BGFM is the only model in the comparison for which an exact `log p_θ`, and therefore a Boltzmann-calibration measurement, is defined. | §9.3, Table 2 | capability audit of cited papers | **REAL** (each cell must be checkable — V2) |
| C28 | GEOM COV/AMR numbers are context only: those baselines solve a conditional problem (graph given); measured AMR-R 3.03 Å at 50k vs 0.073 Å for conditional baselines. | §9.4 (or appendix), Table 3 | [P4] | **REAL number, non-comparable by design** |
| C29 | The energy effect comes from a *very lightly weighted* auxiliary term, `λ₂ = 3×10⁻⁵`. | §4, §7 | `configs/sweep/a3_energy_only.yaml` | **REAL** (a strength — say it) |
| C30 | The implemented objectives are clipped/Huberised variants of the stated ones, and the `L_energy` gradient is a frozen-trajectory (truncated-adjoint) gradient, not `∇_θ` of the exact likelihood. | §4 (implementation paragraph), §11 | code audit (RESEARCH_theory §2.6, §6.7) | **REAL disclosure — MANDATORY** |
| C31 | Anchor loss / `LogZPredictor` were **not active** (`λ₃ = 0`) in the headline runs. | §4 footnote, §11 | `configs/sweep/a3*.yaml`, `a6*.yaml` | **REAL disclosure — MANDATORY** |
| C32 | Training used `kT = 1.0 eV` (≈11 600 K), not a physical temperature. | §4, §11 | configs | **REAL disclosure — MANDATORY** |
| C33 | Pearson `r` is invariant to positive affine maps, so it certifies shape, not temperature. | §6, §11 | Prop 4 remark | **REAL (mathematical)** |
| C34 | The reported `r` is attenuated by `log p` estimator noise (12 Euler steps, 4 Hutchinson probes at eval) and is therefore a lower bound. | §6, §11 | RESEARCH_theory §6.3 | **THEORY-ONLY reasoning about REAL protocol** — phrase as "under classical attenuation, our `r` under-reports" |
| C35 | Forces identify `π(x\|c)` for each `c` but identify the joint only up to `E → E + g(c)`; the unidentifiable degrees of freedom are exactly `{A(c)}`. | §5 Thm 2 + Cor 1 | derivation | **THEORY-ONLY — and NOT the explanation of C11/C12** (RULING R3) |
| — | "Zero-shot to unseen elements" | — | — | **BANNED** (B-1) |
| — | "State of the art" / "beats all baselines" | — | — | **BANNED** (B-2) |
| — | Free-energy readout accuracy, ΔF, ESS numbers | — | [P5] no script, no numbers | **BANNED** (B-4) |
| — | Composition-level generation `p_φ(c) ∝ e^{−A(c)/kT}` in the present tense | — | not implemented | **BANNED** (B-7) |
| — | "As gauge-identifiability predicts, the force term underperforms" | — | logically invalid | **BANNED** (B-6) |
| — | "L_force only constrains ∇log p, so it cannot determine the density" | — | mathematically false | **BANNED** (B-5) |
| — | Any GEOM COV/MAT superiority or competitiveness claim | — | AMR-R 3.03 vs 0.073 | **BANNED** (B-3) |

---

### RULING R1 — the 63/37 vs 74/26 inconsistency (FINAL)

The fact sheet's `[G]` split (63% / 37%, from a scrambled mean of 0.2995) uses the **2-seed**
scrambled mean. `[C]` now has **3** scrambled seeds, mean **0.339**. These cannot both be
written.

**The paper reports 74% / 26%.** Reasons, in order:
- It is the only split arithmetically consistent with the `Δ = +0.092` that `[C]` also reports:
  `0.339 − 0.075 = 0.264` (74%) and `0.430 − 0.339 = 0.092` (26%), and `0.264 + 0.092 = 0.356`.
  Writing 63/37 next to `Δ = +0.092` is self-contradictory and a reviewer will catch it.
- It uses all data currently on disk.

Required footnote, exact content: *"Computed from the three-seed scrambled mean (0.339). An
earlier two-seed mean (0.300) gave a 63%/37% split; the decomposition is sensitive to the seed
set, which is the same sensitivity documented in Section [Limitations]."*

`figures/make_experiment_figures.py --split seeds` must be re-run so Figure 4 shows 74/26, and
placeholder **P-F3** is then retired from the figure inventory.

### RULING R2 — how the force result may be stated (FINAL)

- The **load-bearing** force claim is the clean contrast **C11**: `0.109` (force-only) vs
  `0.223` (matched no-physics control), *same evaluation round, 1 seed, n = 30 parents*.
  Every occurrence must carry "in a smaller earlier round (1 seed, n = 30 parents)" or an
  equivalent qualifier, and must **never** be compared to the n=120 baseline value 0.075.
- **C12** (`0.294` vs `0.420`) is *supporting and confounded*. It may appear once, in §8.1 or a
  caption, and only with **both** confounds stated in the same sentence: the force+energy cell
  used `energy_b_parents = 1` with no `energy_loss_cap` — exactly the pathological
  configuration diagnosed in §10 — and both force cells additionally carry
  `lambda_onpolicy = 0.1`, so "force" there means off-policy force MSE *plus* on-policy teacher
  distillation.
- **C13** (tmQM, sign flip) is the independent replication of the ordering and should be given
  equal weight to C11 in the intro.

### RULING R3 — theory guard rail (FINAL)

Theorem 2 (gauge-identifiability) is a statement about what forces are **insufficient for
across compositions**. Our metric is shift-invariant and therefore *blind* to that gauge.
Theorem 2 **does not** explain the empirical force failure; Propositions 2 and 3 do. Any
sentence linking them causally is a logical error and must be cut. The main text must contain
the explicit guard sentence given in §6.2 of the section plan below.

### RULING R4 — headline statistics rounding (FINAL)

The manuscript uses `Δ = +0.355`, `t = 10.8`, `p ≈ 0.002`.
`make_experiment_figures.py` currently annotates Figure 3(a) with `Δ = +0.356`, `t = 10.9`,
`p ≈ 0.0015` (unrounded seed values). **Action item A1:** regenerate Figure 3 with the
manuscript's rounding, or the two will disagree in print.

---

## 5. Notation contract (final; adapted from RESEARCH_theory §1.2)

Reuse `math_commands.tex` macros — `\E \R \Ls \Var \Cov \KL \1 \argmin \Tr` — and **never
`\newcommand` those names** (they are already defined; see TEMPLATE_NOTES §6.3).
Add exactly two new macros in the preamble:

```latex
\usepackage{tikz}                                  % required by figures/fig1_method
\providecommand{\pending}[1]{\textcolor{red}{#1}}  % every placeholder number goes in this
```

| Symbol | Meaning | Notes for writers |
|---|---|---|
| `c` | composition: element multiset, atom count `N(c)`, total charge `q`, spin multiplicity `s`; countable set `𝒞` | a molecular state is the pair `(c, x)` — say this once |
| `x` | nuclear coordinates of one molecule, COM removed; `x ∈ ℝ^{3N}`, effective dim `d = 3(N−1)` | **never `r`** for coordinates |
| `x_0, x_1` | prior sample `x_0 ~ N(0, σ²I)` on the COM-free subspace; data endpoint `x_1` | |
| `σ` | prior standard deviation, `σ = 1` throughout | |
| `t` | flow time, `t = 0` prior → `t = 1` data (FlowMol3 convention) | state the convention explicitly; half the literature is reversed |
| `x_t` | linear interpolant `x_t = (1−t)x_0 + t x_1` | |
| `v_θ(x_t,t)` | learned SE(3)-equivariant velocity field | |
| `s_θ(x_t,t)` | FM-implied score of the time-`t` marginal (Lemma 1) | |
| `p_t^θ`, `p_θ := p_1^θ` | model marginal density; final model density | |
| `t*` | late probe time at which the score is read out; `t_eval ∈ {0.85, 0.92, 0.97}` in training, `t* = 0.95` in the Figure 1 illustration | |
| `E(c,x)` | teacher potential energy (OMol25 DFT label; eSEN on-policy), eV | |
| `F(c,x)` | teacher force, `F = −∇_x E`, eV/Å | **`F` is force, always** |
| `A(c)` | conditional free energy `A(c) = −kT log Z(c)`, eV | **`A`, never `F`, for free energy** |
| `kT` | temperature scale, eV; `kT = 1.0` in all reported runs | |
| `π(x\|c)` | conditional Boltzmann `e^{−E/kT}/Z(c)` | the target |
| `Z(c)`, `𝒵` | conditional partition function; grand sum | never formed — say so |
| `p_φ(c)` | composition-level generator | **future tense only** |
| `m, k` | parent index `m = 1..M`; perturbation index `k = 1..K` | |
| `w_{m,k}` | Boltzmann residual `log p_θ(x_{m,k}) + E_{m,k}/kT` (nats) | |
| `λ₁, λ₂, λ₃` | weights of `L_force`, `L_energy`, `L_anchor` | `λ₂ = 3×10⁻⁵`, `λ₃ = 0` in reported runs |
| `ξ` | Hutchinson probe, Rademacher | |
| `r_m` | per-group Pearson correlation of `log p_θ` vs `−E_xTB/kT` over the `K` eval perturbations of parent `m` | the **primary metric** is `mean_m r_m` |
| `τ_m` | residual-to-signal ratio `sd_k(w)/sd_k(E/kT)` (Prop 4) | label as derived |

Loss names in text and equations, fixed: `\Ls_{\mathrm{FM}}`, `\Ls_{\mathrm{force}}`,
`\Ls_{\mathrm{energy}}`, `\Ls_{\mathrm{anchor}}`, total `\Ls`.
Master equation, always written this way:

```
L = L_FM + λ₁ L_force + λ₂ L_energy      (+ λ₃ L_anchor, inactive in this paper)
```

Cross-references: **`\Cref{}` only** (cleveref). Do not mix in `math_commands`' `\secref`,
`\figref`, `\algref`. Equation references: `\eqref{}` (the skeleton restores amsmath's "(1)").

---

## 6. Section structure, page budget, word targets

### 6.1 The 12-page draft budget

`W` = target words of prose (excludes displayed equations, captions, table bodies).
Rendered density on this template is ≈650 words per full text page.

| § | Title | Pages | Words | Floats placed here | Writer must include |
|---|---|---|---|---|---|
| — | Title block + abstract | 0.45 | 190 (abstract) | — | abstract shape §6.4 |
| 1 | Introduction | 1.30 | 850 | Figure 1 (+0.40 p) | 8 paragraphs (§6.2), then the 5 bullets of §3 |
| 2 | Background | 0.45 | 300 | — | flow matching, the FM marginal, universal MLIPs. No new results. |
| 3 | Related work | 0.70 | 450 | — | 5 method-family buckets, each ending in one "how BGFM differs" sentence |
| 4 | Method | 1.60 | 900 | — | the three losses; the FFJORD cost; the implementation-disclosure paragraph (C30–C32) |
| 5 | Theory | 1.30 | 700 | — | Lemma 1, Prop 2, Prop 3, Thm 1 + Cor, Prop 4, Thm 2 + Cor 1, Scope paragraph |
| 6 | Evaluation protocol | 0.70 | 450 | — | per-group Boltzmann `r`; independence of xTB; why grouping (C9); T1/T3 disclosures |
| 7 | Main result: energy supervision calibrates | 0.95 | 350 | Figure 2 (+0.40 p incl.) | C7, C8, C29; the first line flags §7–8 as the core |
| 8 | Negative results and mechanism | 1.55 | 600 | Figure 3, Figure 4 | 8.1 force is harmful (C11/C12 under R2); 8.2 Y-scrambling (C16–C18, C20); 8.3 decomposition (C19 under R1) |
| 9 | Generation quality, transfer, and capability | 1.35 | 500 | Table 1, Table 2, Table 3 | 9.1 Table 1 + C24/C25; 9.2 tmQM C13–C15; 9.3 Table 2 + T2 caption; 9.4 GEOM context C28 |
| 10 | Numerical stability as an engineering contribution | 0.50 | 320 | (Figure 6 optional) | C21–C23 |
| 11 | Limitations | 0.55 | 360 | — | the 7 items of §6.5 |
| 12 | Conclusion | 0.20 | 130 | — | restate the three-part pitch in 3 sentences |
| **Total** | | **≈11.9** | **≈5100** | | |

Appendix (after `\bibliography`, uncounted): proofs of Lemma 1 / Props 2–4 / Thms 1–2;
numerical-constant table; FFJORD algorithm listing; full ablation-cell table; dataset
construction; per-metal tmQM breakdown; the placeholder inventory in machine-readable form.

### 6.2 Introduction: paragraph functions (do not renumber)

P1 stakes as an ensemble problem (**not** "generative models are transforming chemistry") ·
P2 the two literatures and the gap, using the emulator/Boltzmann-consistent split ·
P3 why now — universal MLIPs — **with the one-sentence demarcation from Adjoint Sampling**
(they sample conformers of a given molecule; we generate composition and geometry
unconditionally, bond-free) ·
P4 the formulation, one inline equation, the score read-out, the `Z`-cancellation clause, one
sentence on the theorem, **one sentence volunteering the FFJORD second-order memory cost** ·
P5 the three findings with numbers, including "we report (ii) rather than dropping it" ·
P6 evaluation validity: the word "**independent**"; why grouping is mandatory; the
incomparability disclosure (T1/T3) and the honest baseline scope (T5) ·
P7 tmQM, and in the *same paragraph* the 0.34% sparse-tail / distribution-shift clause (T6) ·
P8 the five bullets of §3.

Theory-section guard sentence (must appear verbatim in §5, per RULING R3):

> Theorem 2 concerns insufficiency *across* compositions. The empirical underperformance of
> `L_force` reported in Section [8] is explained by Propositions 2 and 3 — a finite-`t`
> estimator trap and an objective conflict — and not by Theorem 2; moreover, our
> shift-invariant per-group metric cannot probe the free-energy gauge at all.

### 6.3 Compression path 12 → 9 pages (pre-designed; execute in this order)

Mark each block with `% APPENDIX-MOVABLE` as you write it so the cut is a paste, not a rewrite.
Figures and tables must live in `\input{figures/xxx.tex}` files for the same reason.

| Order | Move | Saves |
|---|---|---|
| 1 | §5: keep Lemma 1, Theorem 1 + Corollary, Proposition 4 in the main text; move the full statements of Prop 2, Prop 3, Thm 2, Cor 1 to the appendix, leaving two-sentence prose summaries. | 0.55 |
| 2 | §4: move the numerical-constants paragraph, the FFJORD algorithm listing and the perturbation-shard construction to the appendix. | 0.45 |
| 3 | Table 3 (GEOM context) → appendix; keep the two-sentence T1 statement in the main text. | 0.35 |
| 4 | Figure 4 → merge into Figure 3 as a third panel, or demote to appendix. | 0.28 |
| 5 | §3 Related work → 0.45 p (bucket sentences only, one citation cluster each). | 0.25 |
| 6 | §6 → 0.45 p (move the estimator-attenuation argument to the appendix). | 0.25 |
| 7 | §10 → 0.25 p (keep C21–C23 as three sentences; trace figure to appendix). | 0.25 |
| 8 | §1 → 1.05 p (merge P2 and P3; tighten bullets). | 0.25 |
| 9 | §8 and §2 line-level trims. | 0.37 |
| | **Total available** | **3.00** |

Illegal levers (grounds for desk rejection): changing `\textwidth`, `\textheight`, font size,
`\parskip`, or anything in `iclr2026_conference.sty`. Legal: `microtype` (already loaded),
tighter float sizing, sparing `\vspace{-...}`, appendix demotion, deleting words.

Page check after every compile:

```bash
sed -n 's/.*newlabel{endofmaintext}{{[^}]*}{\([0-9]*\)}.*/main text pages: \1/p' main.aux
```

### 6.4 Abstract shape (190 words, 7 sentences)

1. Stakes as the ensemble problem. 2. The gap (emulators have no calibrated density; Boltzmann
generators do not assemble molecules). 3–4. Method: the two conditions, the `Z`-cancellation,
the guarantee. 5. Headline number: `0.075 → 0.430`, `t = 10.8`, against an *independent*
GFN2-xTB reference; optionally the 6.3× residual restatement. 6. **The negative result** —
unusual in an abstract and precisely why it is remembered. 7. Capability-frontier close.
Any "first" in sentence 7 must be hedged ("to our knowledge") and is subject to verification
item **V3**.

### 6.5 Limitations: the seven mandatory items

Write them in ArBG's register — flat, technical, unsoftened. No "we leave this to exciting
future work" padding.

1. The Y-scrambling decomposition is a second-order effect: at `n = 120` it is **not
   significant** (`Δ = +0.092`, `t = 1.77`, `p = 0.147`); an earlier two-seed contrast gave
   `t = 3.63` and weakened when a third seed was added. The `n = \pending{240}` figures we
   report are `\pending{provisional}`.
2. The force/score term is harmful in our setting; we characterise it empirically and offer two
   candidate mechanisms (Props 2–3) but no complete explanation.
3. Energy supervision is expensive and was numerically fragile; we report a fix (2 of 2 stable
   seeds), not a proof of robustness.
4. Baseline coverage is deliberately narrow (T5): one published generator on the same corpus
   and one matched converged bond-free control we trained ourselves; Symphony/EDM/GeoLDM
   retrains are `\pending{in progress}`.
5. Element coverage is 83 elements **within the training distribution**; heavy-element
   behaviour is measured only under distribution shift (T6).
6. Training used `kT = 1.0` eV (≈11 600 K), not a physical temperature; and Pearson `r` is
   affine-invariant, so the metric certifies the *shape* of `log p_θ(E)` and not the
   temperature. The per-group regression slope would test the temperature; we do not report it.
7. Implementation caveats: the reported `log p` is a discretised (12-step) CNF likelihood with
   a stochastic trace, so the reported `r` is attenuated and is a lower bound; the `L_energy`
   gradient is a frozen-trajectory truncated-adjoint gradient; the objectives are
   clipped/Huberised; rotations are not quotiented (mitigated by the within-parent design);
   spin is not threaded to the teacher (charge is), so open-shell species are treated as
   singlets.

---

## 7. The numbers table — writers may take numbers from here and nowhere else

### 7.1 Primary Boltzmann-calibration results (REAL, black)

Metric everywhere: **mean per-parent Pearson `r` between `log p_θ` and `−E_GFN2-xTB/kT`**,
`K` perturbations per parent at `σ = 0.15 Å`.

| Arm | `n` parents | seeds | per-seed `r` | mean | SEM |
|---|---|---|---|---|---|
| Energy supervision, true labels (a3) | 120 | 4 | 0.377, 0.402, 0.525, 0.418 | **+0.430** | 0.033 |
| No-physics control (a1, FM only) | 120 | 4 | 0.075, 0.083, 0.065, 0.077 | **+0.075** | 0.004 |
| Energy supervision, Y-scrambled (a6) | 120 | 3 | 0.284, 0.315, 0.417 | **+0.339** | — |

| Contrast | Δ | Welch `t` | `p` | status |
|---|---|---|---|---|
| energy vs no-physics | **+0.355** | **10.8** | ≈0.002 | REAL, primary |
| real vs Y-scrambled (n=120) | **+0.092** | **1.77** | 0.147 | REAL, **not significant** |
| real vs Y-scrambled (n=\pending{240}) | `\pending{+0.13}` | `\pending{3.9}` | `<\pending{0.05}` | placeholder |

Pooled (non-grouped) `r²`: **0.11** for the energy model, **0.01** for the no-physics control —
used only to justify the grouped metric (C9). Cross-batch energy loss magnitude: **~1e9**.

### 7.2 Derived quantities (REAL, label as derived; Prop 4 with `ρ = 0`)

| Run | reported `r` | implied `τ = sd(w)/sd(u)` |
|---|---|---|
| energy, true labels | +0.430 | 2.10 |
| no-physics baseline | +0.075 | 13.30 |
| energy, Y-scrambled | +0.339 | 2.78 |
| force-only | +0.109 | 9.12 |
| tmQM, energy | +0.365 | 2.55 |

Headline restatement: **13.3× → 2.1×, a 6.3× reduction in the Boltzmann residual.**
Footnote required: this uses the `ρ = 0` idealisation of Prop 4 and is computed per molecule
while we report the mean over molecules (Jensen); it does not apply to negative `r`.

### 7.3 Force ablation (REAL; qualifiers mandatory — RULING R2)

| Cell | Config | `n` | seeds | `r` | how it may be used |
|---|---|---|---|---|---|
| force-only (a2) | λ₁>0, λ₂=0, `lambda_onpolicy=0.1` | 30 | 1 | **+0.109** | primary force claim, vs the row below only |
| no-physics (a1), same round | — | 30 | 1 | **+0.223** | the *only* legitimate comparator for +0.109 |
| force + energy (a4) | `energy_b_parents=1`, no `energy_loss_cap`, `lambda_onpolicy=0.1` | 30 | 1 | **+0.294** | once, with both confounds stated |
| energy-only, same round (a3) | `energy_b_parents=4`, `energy_loss_cap=3000` | 30 | 1 | **+0.420** | comparator for the row above |

### 7.4 Effect decomposition (REAL; RULING R1)

| Component | Value | Share | Operational definition |
|---|---|---|---|
| Total energy-supervision gain | +0.355 | 100% | energy − no-physics |
| Density-calibration component | **+0.264** | **≈74%** | survives within-parent label scrambling |
| Boltzmann-information component | **+0.092** | **≈26%** | destroyed by scrambling |

Footnote (mandatory): the two-seed scrambled mean (0.300) gave +0.224 / +0.131 = 63% / 37%;
the split is seed-set sensitive.

### 7.5 tmQM transfer (REAL)

3000 transition-metal complexes; metals Cr, Fe, Mn, Mo, Ru, Ti, V.

| Quantity | Value |
|---|---|
| force-only `r` | **−0.158** |
| energy `r`, mean | **+0.365** |
| energy `r`, median | **+0.429** |
| fraction of complexes with `r > 0.5` | **37%** |
| xTB single-point convergence | **270/270** |
| TM share of OMol25 training atoms | **0.34%** (0.7M of 205M) |
| atoms per metal in training data | **1.5×10⁴ – 5.1×10⁴** |

### 7.6 Generation quality (REAL, except the \pending rows)

n = 100 samples per model, one scoring script, recomputed by us.

| Model | RDKit-valid | connected | PoseBusters | uniqueness |
|---|---|---|---|---|
| **BGFM (ours)** | **43.0** | **32.0** | **23.0** | **100** |
| Zatom-1 (author-reported budget; see V1) | 30.4 | 17.0 | 15.1 | 93.5 |
| FlowMol3 bond-free control (a1, ours, converged) | *[fill from `runs/eval_ours/` — writer must query, do not invent]* | | | |
| Symphony (OMol25 retrain) | `\pending{…}` | `\pending{…}` | `\pending{…}` | `\pending{…}` |
| EDM (OMol25 retrain) | `\pending{…}` | `\pending{…}` | `\pending{…}` | `\pending{…}` |
| GeoLDM (OMol25 retrain) | `\pending{…}` | `\pending{…}` | `\pending{…}` | `\pending{…}` |

Also real: xTB relaxation **ΔE/atom median 6.90 kcal/mol**, relaxation **failure rate 3%**.

> **Note to the §9 writer:** the FlowMol3-bond-free control row is required by T5 (it is half of
> our honest baseline set) but its four generation-quality numbers are not in the fact sheet.
> Either read them off disk and add them to this table (then tell the editor), or state in
> prose that the bond-free control is our calibration baseline and omit the row. **Do not
> invent it, and do not mark it `\pending` — `\pending` is reserved for numbers that do not
> exist yet, not for numbers we have not looked up.**

### 7.7 Configuration and protocol constants (REAL)

| Constant | Value | Where it must be stated |
|---|---|---|
| training data | OMol25 4M: 3.9M molecules, 205M atoms, 83 elements, DFT forces + energies, no bond labels | §1, §4, Fig. 1(d) |
| `max_atoms` | 200 | §4 or appendix |
| `kT` (training) | 1.0 eV (≈11 600 K) | §4, §11 |
| `λ₂` | 3×10⁻⁵ | §4, §7 |
| `λ₃` | 0 (anchor inactive) | §4 footnote, §11 |
| `t_eval` | {0.85, 0.92, 0.97}; amplification 6.7× / 12.5× / 32.3× | §5 remark |
| `σ` (prior) | 1.0 | §4 |
| eval perturbation `σ` | 0.15 Å | §6 |
| `n_ode_steps` | 4 (train) / 12 (eval) | §4/§6 |
| `n_hutchinson` | 2 (train) / 4 (eval) | §4/§6 |
| `energy_b_parents` | 1 (pathological) → 8 (fixed) | §10 |
| `energy_loss_cap` | 3000 (a3) → 1500 (stabilised) | §10 |
| `SCORE_NORM_CAP` | 1000 per-atom L2 | §4 disclosure / appendix |
| `PER_ATOM_ERR_CAP` | 1e4 (Huber transition) | appendix |
| `LOG_P_CAP` | 1e6 | appendix |
| within-parent energy std, both arms | 77.6 kcal/mol | §8.2 |
| single-parent loss excursion | 48 → 974 | §10 |
| divergent seeds | 3 of 9 | §10 |
| stable seeds after fix | 2 of 2 | §10 |
| masked-failure duration | ≈1 GPU-day | §10 |
| GEOM AMR-R, ours at 50k | 3.03 Å (vs 0.073 Å for conditional baselines) | §9.4 / appendix |

---

## 8. PLACEHOLDER INVENTORY

This table goes **verbatim into a comment block at the top of `main.tex`**, and is mirrored in
the appendix. Every `\pending{}` in the manuscript must have a row here. `make_experiment_figures.py`
prints its own inventory at the end of each run; the two must agree line for line.

| ID | Section / float | Semantics | Placeholder value | True current status | Source when it lands |
|---|---|---|---|---|---|
| P1 | §8.2, Fig. 3(b), §11, bullet 4 | Y-scrambling contrast at larger `n` | `Δ ≈ \pending{+0.13}` | measured at n=120: `Δ = +0.092` | `runs/eval_ours/wide_a6_*` + `wide_a3_*` at n=240 |
| P2 | §8.2, Fig. 3(b), §11 | Welch `t` for that contrast | `t ≈ \pending{3.9}` | measured at n=120: `t = 1.77` | same |
| P3 | §8.2, Fig. 3(b), §11 | significance for that contrast | `p < \pending{0.05}` | measured at n=120: `p = 0.147`, **n.s.** | same |
| P4 | §8.2, §11, bullet 4 | evaluation group count | `n = \pending{240}` | current record is n=120 | eval job [P3] running |
| P5 | §11 item 1, bullet 4 | status word | `\pending{provisional}` | — | drop the macro when P1–P4 land |
| P6 | Table 1, Symphony row | four generation-quality numbers | `\pending{…}` ×4 | retrain **queued**, no numbers | [P1] Symphony OMol25 retrain |
| P7 | Table 1, EDM row | four generation-quality numbers | `\pending{…}` ×4 | env + data **not ready**, no numbers | [P2] |
| P8 | Table 1, GeoLDM row | four generation-quality numbers | `\pending{…}` ×4 | env + data **not ready**, no numbers | [P2] |
| P9 | §11 item 4 | baseline status phrase | `\pending{in progress}` | accurate today | drop when P6–P8 land |
| P-F1 | Fig. 2, three left panels | individual `(log p, −E/kT)` point positions | synthesised points reproducing the real per-parent `r` and OLS slope | per-parent `r` and slope are **real**; only the scatter positions are drawn | `paper/figures/dump_boltz_records.py` then `make_experiment_figures.py --only fig2` |
| P-F3 | Fig. 4 | the 74/26 split | *retired by RULING R1* — regenerate with `--split seeds` | the figure currently ships 63/37 | `make_experiment_figures.py --split seeds` |

**Camera-ready gate:** `make_experiment_figures.py --strict-real` must exit 0, and
`grep -c 'pending' main.tex` must be 0, before submission of the final version. For the
*initial* submission, placeholders may remain but every one must appear in this table.

**Not placeholders — do not wrap these in `\pending`:** anything in §7.1–7.7 that is listed as
REAL, and the FlowMol3-bond-free generation row of Table 1 (that is a *lookup*, not a missing
experiment — see the note in §7.6).

---

## 9. Wording bans — 不可写 / 应写

Each ban is a hard filter. If a draft sentence matches the left column, it is rejected.

| # | 不可写 (never write) | 应写 (write instead) |
|---|---|---|
| **B-1** | "zero-shot generalisation to unseen elements"; "generalises to elements never seen in training"; "unseen transition metals" | "transfer under distribution shift to a different transition-metal corpus. These metals are present in the OMol25 training data, in its sparse tail: 0.34% of all training atoms, between 1.5×10⁴ and 5.1×10⁴ atoms per metal." The sparse-tail clause must appear **in the same paragraph** as any tmQM number. |
| **B-2** | "state of the art"; "outperforms existing 3D generators"; "beats all baselines" | "Our external comparisons are deliberately narrow: one published generator trained on the same corpus and one matched, fully converged bond-free control we trained ourselves. We make no claim of superiority over the broader 3D-generation literature, which is trained on different corpora with bond supervision and evaluated under different protocols." |
| **B-3** | "competitive on GEOM COV/AMR"; any bar chart or ranking against the conditional conformer literature | "COV and AMR are defined per reference molecule and therefore score a strictly conditional problem; a direct ranking would be uninformative in either direction and we make no such claim." Context table only, never a chart. |
| **B-4** | "our model estimates free energies"; any ΔF number; any ESS number; "explains the absence of ESS collapse" | "The mean that the variance form discards is the conditional free energy; we do not evaluate free-energy accuracy or effective sample size in this work." Attribute any cited 0.04–0.5 ESS range to the papers it comes from. |
| **B-5** | "`L_force` only constrains `∇ log p`, so it cannot determine the density" (**mathematically false**) | The three-part statement: (i) across compositions the gradient is genuinely insufficient — the gauge `{A(c)}` is invisible to forces; (ii) at any usable `t` the read-out gives the score of the *smoothed* marginal with `O(1−t)` bias and `O((1−t)^{-2})` injected noise; (iii) the population targets of `L_FM` and `L_force` conflict unless the data is already Boltzmann, and OMol25 is not. |
| **B-6** | "as gauge-identifiability predicts, the force term underperforms" | See RULING R3 and the mandatory guard sentence in §6.2. |
| **B-7** | any present-tense statement that BGFM generates compositions in proportion to `e^{−A(c)/kT}`; "hierarchical Boltzmann consistency is achieved" | "This paper trains and evaluates the conformation level; the composition level is a prospective consequence of the same identity and is left to future work." Future tense throughout. |
| **B-8** | "we backpropagate through the exact likelihood" | "The `L_energy` gradient is taken with the trajectory held fixed (a truncated-adjoint / frozen-trajectory surrogate); we do not differentiate through the ODE solve." |
| **B-9** | "the training is robust/stable"; "we solve the instability" | "Two of two seeds trained to completion after the fix; we report the fix, not a proof of robustness." |
| **B-10** | "0.43 approaches the theoretical ceiling"; "near-perfect Boltzmann agreement" | "Nothing in the theory predicts the magnitude. `r` is bounded above by the agreement between the training potential and the evaluation potential on these ensembles and further attenuated by `log p` estimator noise; both ceilings are unmeasured." |
| **B-11** | "Zatom-1 is non-converged" (unless V1 produces a locatable author statement) | "at the training budget reported by the authors (≈400 GPU-hours for the jointly trained 80M model)". |
| **B-12** | comparing `+0.109` to the n=120 baseline `+0.075`; any cross-round comparison | Always attach "in a smaller earlier round (1 seed, `n = 30` parents)" and compare only to `+0.223`. |
| **B-13** | "the anchor prevents a trivial-constant failure mode" | "The anchor's role is to pin the per-parent additive constant, i.e. the free-energy gauge. It was inactive (`λ₃ = 0`) in all reported runs." |
| **B-14** | present-tense claims about `\pending` numbers, or `\pending` numbers written in black | Every not-yet-real number is inside `\pending{}` and appears in the §8 inventory. |
| **B-15** | real names, institutions, identifiable repository URLs, funder names in the body or a title footnote | Anonymous throughout; funding only in `\subsubsection*{Acknowledgments}` at the very end, added at camera-ready. |

---

## 10. Reusable sentence templates (renumbered for **SPINE table numbering** — §11)

**T1 — GEOM context (goes in §9.4 and Table 3's caption).**
> Table 3 places our numbers alongside the conditional conformer-generation literature for
> context only. Those methods receive the molecular graph and are asked to produce its
> low-energy conformers; BGFM receives nothing and must produce composition, geometry, and
> (post hoc) bonds jointly. COV and AMR are defined per reference molecule and therefore score
> a strictly conditional problem; a direct ranking would be uninformative in either direction
> and we make no such claim.

**T2 — capability matrix disclaimer (must appear in Table 2's caption).**
> Entries marked N/A in Table 2 are not failures. Those methods do not define a tractable exact
> `log p_θ`, so the quantity in that column cannot be computed for them at all. Table 2
> therefore states which questions each model class can be asked, not which model is better.

**T3 — why we re-scored everything.**
> We do not transcribe numbers across papers. Published results in this area differ in
> train/test split, hydrogen handling, and sample count, so every entry in Table 1 was
> recomputed by us with a single scoring script on `n = 100` samples per model. Where a number
> is author-reported we mark it as such and state the training budget the authors report.

**T4 — the terminology move (Related Work opener and/or intro P2).**
> Following the distinction drawn for Boltzmann generators, we call a model a *structure
> emulator* if it produces chemically plausible geometries without a calibrated density, and
> *Boltzmann-consistent* if its density provably tracks `exp(−E/kT)`. Essentially all 3D
> molecular generators are emulators by construction. The contribution of this paper is not to
> out-emulate them, but to make a periodic-table-scale de-novo generator Boltzmann-consistent.

**T5 — honest baseline scope.** As in ban B-2's right column; use it verbatim in §1 P6 and §11.

**T6 — distribution shift, not unseen elements.** As in ban B-1's right column; use it verbatim
in §1 P7 and §9.2, and in Figure 5's caption.

**T7 — the negative result as a contribution.**
> We report the force term's failure in full because the negative result is the informative
> one: the local gradient condition is the term a reader would adopt first, and our evidence is
> that at 83-element scale it trades away precisely the global calibration the energy term buys.

**T8 — the two-tier claim strength (§8.2 and §11).**
> We separate the two claims by strength. The primary effect — energy supervision versus no
> physics — is large and unambiguous (`Δ = +0.355`, `t = 10.8`). The decomposition into
> calibration and Boltzmann-specific components rests on a second-order contrast that is
> sensitive to the number of evaluation groups: at `n = 120` it is not statistically
> significant (`Δ = +0.092`, `t = 1.77`), and the `n = \pending{240}` figures reported above
> are `\pending{provisional}`. We therefore state the decomposition as a hypothesis supported
> by a consistent point estimate, not as an established result.

---

## 11. Figure and table register

**Numbering in this register is authoritative and supersedes the numbering used in
`RESEARCH_story.md` / `RESEARCH_figures.md`.** The mapping is given in the last column so
nobody mis-resolves a cross-reference.

### Figures (caption **below** the figure; sentence case)

| # | Label | Title / content | Section | Source | Status |
|---|---|---|---|---|---|
| 1 | `fig:method` | BGFM method overview: FM strip, global-energy panel, local-force panel, oracle bus, total objective | §1 (page 2) | `\input{figures/fig1_method}` — pure TikZ, natural width 5.49 in, **never** `\resizebox` or a width option | ready; caption drafted in RESEARCH_figures §5 |
| 2 | `fig:boltzmann` | per-parent `log p_θ` vs `−E_xTB/kT`: three parents at the 20th/55th/92nd `r` percentile + the distribution of per-parent `r` for energy vs no-physics | §7 | `figures/out/fig2_boltzmann_scatter.pdf`, `width=\linewidth` | left-panel points are **P-F1**; fix with `dump_boltz_records.py` |
| 3 | `fig:ablation` | (a) ablation bars with per-seed dots; (b) Y-scrambling contrast | §8 | `figures/out/fig3_ablation_bars.pdf`, `width=\linewidth` | **action A1** (rounding, RULING R4); n=240 projection is P1–P4 |
| 4 | `fig:mechanism` | waterfall decomposition of the +0.355 gain | §8.3 | `figures/out/fig4_mechanism_decomposition.pdf`, `width=0.62\linewidth` | **must be regenerated at 74/26** (RULING R1); first candidate for demotion |
| 5 | `fig:tmqm` | per-metal strip/violin of per-parent `r`, force-only vs energy, with each metal's training-atom share printed | §9.2 | **not yet built**; data in `runs/eval_ours/OOD_tmqm_*` | optional; caption must carry T6 |
| 6 | `fig:stability` | loss / weight-norm traces, divergent seeds in vermillion, before and after the fix | §10 or appendix | **not yet built**; data in the [F] runs | optional; high reviewer value |

Figure 1 must be declared with `[t]` **after** the text that first cites it, or with `[h]`,
to avoid floating above the title on page 1 (TEMPLATE_NOTES §6.4).

### Tables (caption **above** the table; `booktabs` rules only)

| # | Label | Title / content | Section | Filled from | Status |
|---|---|---|---|---|---|
| 1 | `tab:generation` | OMol25 bond-free de-novo generation quality: RDKit-valid / connected / PoseBusters / uniqueness | §9.1 | §7.6 | ours + Zatom-1 REAL (V1); Symphony/EDM/GeoLDM rows are P6–P8; FlowMol3-control row is a lookup |
| 2 | `tab:capability` | capability matrix: elements · needs bond labels · unconditional de-novo composition · tractable exact `log p_θ` · trained against a transferable energy · amortised across molecules · Boltzmann calibration reported | §9.3 | audit of the cited papers (rows: EDM/GeoLDM, FlowMol3 + our control, Zatom-1, Torsional Diffusion, TBG, SBG/ArBG, Adjoint Sampling, **BGFM**) | caption **must** contain T2; every cell is verification item **V2** |
| 3 | `tab:geomcontext` | GEOM COV/AMR context (ours AMR-R 3.03 Å at 50k vs 0.073 Å conditional) | §9.4 | §7.7 | caption **must** contain T1; first table to demote to the appendix |
| A1 | `tab:cells` | full ablation-cell table: config, `n`, seeds, `r`, and the confound flags of RULING R2 | appendix | §7.1, §7.3 | required, so the main text can stay terse |
| A2 | `tab:constants` | numerical constants that are part of the loss definitions | appendix | §7.7 | required by C30 |

`RESEARCH_*.md` calls Table 1 "Table 2", Table 2 "Table 3", and Table 3 "Table 1". **Use this
register's numbers.**

---

## 12. Blocking verification items (assign before the relevant section is written)

| ID | Item | Blocks | Fallback if unresolved |
|---|---|---|---|
| **V1** | Locate the Zatom-1 OMol25 *generation* results table (version + table number) and the exact "80 epochs / non-converged" author sentence. The arXiv version appears to report OMol25 only for energy/force prediction. | Table 1, C24, bullet 6 | Cite the numbers with the version they come from and use ban B-11's wording. **Never** write "non-converged" without a locatable author statement. |
| **V2** | Verify every cell of the capability matrix against the cited paper. | Table 2 | Drop any row whose cells cannot all be verified; a single wrong checkmark costs more credibility than the table earns. |
| **V3** | Verify the "first" in the abstract's closing sentence against TBG / SBG / ArBG / Adjoint Sampling / Torsional Diffusion. | abstract sentence 7 | Replace with "to our knowledge, the first … " or drop the superlative and state the capability. |
| **V4** | Confirm authors + year of the unconditional-3D-generation evaluation paper (arXiv:2505.00518) before using it in T3. | §6, §9 | Cite by arXiv id only. |
| **A1** | Regenerate Figure 3 so its annotations read `Δ = +0.355`, `t = 10.8` (RULING R4). | Fig. 3 | — |
| **A2** | Regenerate Figure 4 with `--split seeds` so it reads 74/26 (RULING R1). | Fig. 4, §8.3 | — |
| **A3** | Look up the four generation-quality numbers for the FlowMol3-bond-free control, or drop the row (§7.6 note). | Table 1 | prose statement, no row |

---

## 13. Canonical citation keys (use exactly these in `\citep{}` / `\citet{}`)

One writer owns `references.bib`; everyone else uses these keys and never invents new ones
without adding them there first. `\bibliographystyle{iclr2026_conference}`,
`\bibliography{references}`, author–year style (numeric is impossible — TEMPLATE_NOTES §6.1).

`flowmol3` · `omol25` · `esen` · `edm` · `geoldm` · `zatom1` · `torsionaldiff` ·
`tbg` (Klein & Noé, transferable BG) · `sbg` · `arbg` · `adjointsampling` · `ewfm` ·
`eval3dgen` (arXiv:2505.00518, see V4) · `posebusters` · `gfn2xtb` · `xyz2mol` · `ffjord` ·
`hutchinson` · `lipman2023flowmatching` · `liu2023rectifiedflow` · `albergo2023interpolants` ·
`noe2019boltzmann` · `tmqm` · `geom` · `ruecker2007yrandom` · `midi` · `semlaflow` · `symphony`

---

## 14. Mechanical checklist for every section writer

- [ ] Preamble is `skeleton_min.tex` **verbatim** plus exactly the two lines in §5. Nothing else
      added to the preamble without editor approval.
- [ ] No `\newcommand` on any `math_commands.tex` name (`\E \R \Ls \Var \KL \1 \argmin`, the
      `\va…\vz`, `\mA…\mZ` families, …).
- [ ] Cross-references via `\Cref{}`; equations via `\eqref{}`.
- [ ] Figure captions **below**, table captions **above**, sentence case.
- [ ] Every float in its own `\input{figures/…}` file, and every appendix-movable block tagged
      `% APPENDIX-MOVABLE`.
- [ ] No packages from the forbidden list: `siunitx`, `makecell`, `adjustbox`, `algorithm2e`,
      `pgfplots`, `bbm`, `dsfont`. Pseudocode = `algorithm` + `algpseudocode`.
- [ ] No `\tableofcontents` (disabled by the `.sty`).
- [ ] Every number traced to §7; every claim traced to §4; every `\pending` traced to §8.
- [ ] `grep -n 'zero-shot\|state of the art\|SOTA\|outperform' ` on your section returns nothing
      that survives §9.
- [ ] Build: `pdflatex → bibtex → pdflatex → pdflatex`, then the page check of §6.3.
