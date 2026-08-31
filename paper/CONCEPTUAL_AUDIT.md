# CONCEPTUAL AUDIT

Paper: **What Energy Values Teach De Novo Molecular Flows: Ordering Without Calibration**
Date: 2026-08-14
Status: read-only audit record. Nothing here edits the manuscript. Corrections derived from
this record are applied and logged in `REVISION_LOG.md`; compression is planned separately in
`PAGE_CUT_PLAN.md`.

This document holds the things a reader of the manuscript cannot check for themselves: what the
composition variable actually is in the code, what the supervision design actually looks like at
the level of the training pool, which theorem hypotheses hold for which measure class, which
training run produced which number, and what remains unknown because the artifacts to settle it
do not exist.

---

## 1. The one message, and the three properties it separates

Energy grounding is not one property. The manuscript's entire claim structure rests on keeping
three things apart:

| | Property | Question | Evidence in this paper |
|---|---|---|---|
| (i) | **Local ordering** | are nearby geometries ranked consistently with energy? | improvement, measured |
| (ii) | **Local calibration** | does the *magnitude* of the log-density response match `−E/kT`? | failure, measured |
| (iii) | **Global mass allocation** | do distinct basins receive correct relative probability? | **not measured at all** |

Every finding below is filed against one of these three. The most common failure mode in the
pre-revision manuscript was a sentence that silently promoted evidence for (i) into a claim about
(ii) or (iii): "the anchor is the remedy for the calibration failure" moves an offset argument
(a (iii)-shaped object) onto a (ii)-shaped metric; "is the model Boltzmann?" collapses all three.

---

## 2. The composition variable `c`, exactly as implemented

Three different definitions of `c` were in circulation. They are not interchangeable.

| Where | Definition | Status |
|---|---|---|
| `A1_proofs.tex` `app:setup` (pre-revision) | element multiset, atom count `N(c)`, total charge `q`, **spin multiplicity `s`** | wrong: `s` is never threaded |
| Composition-multiplicity audit (identifiability analysis) | element multiset **+ total charge** | this is the operative key |
| Pipeline as implemented | element multiset + total charge; **spin fixed at singlet** | verified first-hand, below |

**Verified first-hand for this audit:**

- `scripts/precompute_energy_perturbations.py:154` sets `atoms.info["spin"] = 1` unconditionally,
  alongside the comment at `:144` that atom 0 carries the total charge. Spin is a hard-coded
  constant, not a queried property of the parent.
- `scripts/eval_boltzmann_independent.py:86-87,143,154` threads `charge` into the xTB single-point
  call and into the per-record CSV; there is no spin/`uhf` argument anywhere in the evaluator.
- `cfm_mol/physics.py:67,78,116` and `cfm_mol/physics_drift.py:104,115,127,147` do accept a spin
  argument, but both default to 1 and neither is on the training or evaluation path used for any
  reported number.

**Consequence for the theory.** `E(c, ·)` in every theorem is not the function the code queries
for any open-shell species: for a doublet, the teacher is asked for the singlet energy of the same
nuclear configuration. The evaluated population is main-group organic, where the singlet
assignment is the standard one, but multiplicity was never verified per parent. The correct
statement is a *standing restriction*, not a component of `c`: no radical, open-shell or
transition-metal result is claimed anywhere in the paper. `refs` to "83 elements" describe the
**training corpus and the teacher**, never a demonstrated evaluation scope.

**Consequence for the anchor.** `cfm_mol/log_z_predictor.py` (read for this audit) builds its
prediction from `elem_e` (a per-element scalar embedding summed over atoms) plus a 3-feature
correction `(n_atoms, total_charge, |total_charge|)`. Its output is therefore a **function of `c`
alone**. `cor:gauge` identifies one free constant **per group**. The implemented anchor can fix at
most one constant **per composition**. Where a composition supplies several groups — 31% of
training parents, §3 — the implemented anchor is structurally incapable of fixing the constants
the corollary names. This is a fact about the code, not a judgement about the design.

---

## 3. Composition multiplicity: the settled audit

`c` = element multiset + total charge, taken from the stored per-atom types and charges.

**Training perturbation shard** (`perturbation_train_n30000_s0.pt`):

| Quantity | Value |
|---|---|
| parent groups | 30,000 |
| distinct `c` | 22,822 |
| compositions carrying more than one parent | 2,204 |
| parents in those compositions | 9,382 (**31%**) |
| maximum multiplicity | 80 |

**Evaluation population** (`val_data_processed.pt`; the 120 matched-grid parents and the 93
primary-endpoint parents): every parent has a **unique** `c`. Multiplicity is 1 for all.

### What the manuscript may and may not say

**MAY NOT** (false as a property of the supervision design):
> "each composition contributes exactly one group, so `J = C = 1`"

It was false for training in four places (`03_theory.tex` `cor:basins`, its proof in
`A1_proofs.tex`, the introduction, and the `app:scope` inventory) and it was the load-bearing step
of the corollary, which derived its conclusion from `J = 1` rather than from disjointness.

**MAY** (the empirical corollary, which is what is actually true):
> Each parent supplies one empirical perturbation cloud. Clouds sharing a composition almost surely
> share no sampled geometry, so the empirical overlap graph consists of isolated vertices; the
> objective constrains the local relation inside each cloud and leaves an independent offset for
> every disconnected component.

**MAY also** (and should, because it sharpens rather than weakens the argument):
> Repeated compositions do occur in training — 2,204 compositions covering 31% of parents, maximum
> multiplicity 80 — and they are still disconnected.

**MAY, scoped** (about the evaluated set only):
> Every evaluated parent has a distinct composition, per this audit.

The repaired corollary is derived from **disjointness of sampled supports**, which is robust to
multiplicity and does not require any statement about `J`. That repair also removes an internal
contradiction: the old proof concluded `J = 1` and then, five lines later, reasoned about "every
pair of groups of a common composition", which presupposes `J > 1`.

---

## 4. The empirical overlap graph, correctly stated

Two vertices (parent clouds) are joined when their reference measures both assign positive measure
to a common set on which the residual-constancy statements must agree.

- **Continuous reading** (each `ρ_j` has a positive Lebesgue density on `U_j`): two groups of a
  common composition drawn as full-support Gaussians would overlap, and `G_c` would be *complete*.
- **Empirical reading** (each `ρ_j` is the atomic measure on the `K` drawn geometries): distinct
  parents draw distinct geometries, so with probability one no two clouds share a sampled point,
  and `G_c` is **edgeless** — whatever `J` is.

The empirical reading is the operative one, because it is what the loss sees. The consequence is an
**identifiability** statement, not a normalisation statement: the objective places no constraint
relating the offsets of distinct clouds, so their relative mass is unconstrained by the loss. That
is property (iii) in §1, and it is the sense in which this paper does not measure basin mass.

Connectivity is therefore **sufficient, and necessary in the identifiability sense**: a disconnected
graph leaves zero-loss solutions with differing offsets; it does not force every zero-loss solution
to have them. Four sites asserted unqualified necessity ("unless group supports connect", "needs the
overlap graph to be connected"), including the abstract, and all four are now identifiability-scoped.

---

## 5. Theorem / measure consistency

The pre-revision theory declared the reference measures empirical (K sampled points) and then
required each of them to have a strictly positive Lebesgue density thirteen lines later. A K-point
atomic measure has no Lebesgue density. Six sites were affected; the table records what each one
needed and what it now carries.

| Site | Defect | Repair |
|---|---|---|
| `03_theory.tex` `def:overlap` | required a strictly positive Lebesgue density of every group measure | restated for measures **dominated by a common σ-finite measure**; an edge is a shared set of positive measure under both |
| `03_theory.tex` (reference-measure sentence) | sat between the certificate and the definition, swapping measure class mid-argument | moved to follow the definition and phrased as the choice of corollary |
| `A1_proofs.tex` `rem:refmeasure` | "`U_j` is a finite point set" contradicted `def:overlap` outright | promoted to statement level as a new **empirical corollary** |
| `A1_proofs.tex` `thm:certificate`(i) | asserted a density proportionality from `K` pointwise equations | split: proportionality on `U` in the continuous case; "one common value at the `K` sampled points" (`K−1` equations, nothing about unsampled points) in the atomic case |
| `A1_proofs.tex` `thm:certificate`(ii) | assumed `p_θ(U|c) = 1` — vacuous when `U` is Lebesgue-null; `rem:epsilon` repairs only the bounded-`U_c` case | restricted to the continuous corollary, with an explicit statement that **no normalisation hypothesis of this form is available for atomic reference measures at any ε** |
| `A1_proofs.tex` `thm:l2g`(iii) | used the positive-density clause to get Lebesgue-nullity of inter-component intersections and hence `Σ_i e^{−b_i} Z^{(i)} = 1` | proved only in the continuous corollary; the empirical corollary carries the weaker, correct conclusion (offsets unrelated ⇒ relative mass unidentified) |

**Not a defect, and kept:** the identity `NRV_m = τ_m²` (`eq:app-nrvtau`) is exact algebra within a
single group under a single potential. What was wrong was the *prose* around it (§7, C2), not the
derivation.

---

## 6. Theory versus code: seven mismatches

1. **The reported density is not the generator's conditional.** Every theorem is written for
   `log p_θ(x | c)`. `cfm_mol/bgfm_density.py::log_density_via_flow` (with
   `make_position_velocity_fn`, read at `:90-115`) computes the log-density of a **position-only
   flow with the discrete channels `a_t, c_t, e_t` clamped at data labels for the whole reverse
   trajectory**. The joint generator evolves types and charges through a CTMC, so the clamped-channel
   flow density is a different, separately normalised object. It coincides with the generator's
   conditional only if the discrete trajectory is determined by its endpoint. Pre-revision this
   appeared once, as a subordinate clause in Section 2; the theory appendix asserted normalisation
   "by construction" without disclosing the clamping.
2. **The implemented value estimator is not the displayed one.** `within_group_variance_loss`
   (`bgfm_density.py:252`, read for this audit) uses the **biased** variance, drops non-finite
   residuals, excludes groups left with fewer than two valid members, and averages over **surviving**
   groups rather than over `M`. Separately, a whole step's value loss is rejected above the loss cap,
   so the optimised objective is a conditionally-sampled version of the displayed `(1/M) Σ_m Var_k`.
   The null-space results are unaffected by the biased normalisation; they are stated for the
   full-group objective, and that is now said.
3. **Groups contain the data geometry.** Both the training shards and the evaluation groups contain
   the parent geometry itself, not only displacements. The primary endpoint drops it **at evaluation
   only**; the training objective always includes it.
4. **The anchor is composition-keyed, the gauge is group-keyed.** See §2.
5. **`c` includes spin in the theory and never in the code.** See §2.
6. **The estimated density is `log p_{1−ε}`, not `log p_θ`.** `eq:epsgrid` establishes that the
   integration grid makes the estimated object a *different* density (the smoothed marginal), not a
   noisy version of the same one. No theorem was restated at that smoothing scale, so the certificate
   the objective can enforce is about `p_{1−ε}` restricted to the sampled points. The ε-dependence is
   a property of the method, not an accuracy limit of the estimator — which is exactly why the
   resolution ablation (§8, and D1 in `REVISION_LOG.md`) is a statement about *the scale at which
   local ordering improves*.
7. **Prior log-density normaliser.** The ambient-trace argument in Assumption A3 is sound, but the
   prior log-density mixes a `3(N−1)` normaliser with a full `3N` quadratic form. This is valid only
   if `x_0` is exactly centre-of-mass-free. Flagged; not load-bearing for any reported number.

---

## 7. Conceptual contradictions found (register)

Severity order. "Sites" counts live (compiled) occurrences.

| # | Contradiction | Sites | Status |
|---|---|---|---|
| 1 | **Anchor vs NRV.** `04_experiments.tex:68-69` states `NRV_m` is invariant to the per-group constant; `:237-239` names the anchor (which pins exactly that constant) as the repair for NRV. Both in one file, 170 lines apart. | 6 wrong, 2 correct-and-kept | corrected; the two statements assigning the anchor only its *offset* role were kept, with the per-composition caveat added |
| 2 | **`J = C = 1`.** Asserted as a property of the supervision design; false for training (§3); internally contradicted twice within one proof. | 4 + 2 | corrected via the disjointness derivation |
| 3 | **Measure class.** Empirical reference measures declared, Lebesgue densities required. | 6 | corrected (§5) |
| 4 | **τ.** `tab:tau`'s caption claimed value supervision "cuts the Boltzmann residual from 4.85 to 2.52", while `rem:nrvtau` thirty lines earlier proves `NRV = τ²` and reports measured NRV **rising** 1.405 → 2.144, i.e. measured τ rising 1.19 → 1.46. Two mutually inconsistent values of one symbol for one arm, supporting opposite conclusions. | 2 | the residual-reduction sentence deleted; the table now prints implied τ **beside** measured √NRV, and the discrepancy is labelled as the evidence that ρ < 0 |
| 5 | **Round provenance.** The earlier round was called "uninterpretable" and "superseded in full" while supplying the headline table's checkpoints. | 4 vs 1 | split into which contrasts are superseded (force, scramble) and which survive as a corrected held-out evaluation of that checkpoint family |
| 6 | **Ceiling.** 26 live uses of "ceiling" / "achievable maximum" / "attainable ordering" / "oracle noise". `r = 0.927` is what exact reproduction of the eSEN teacher scores against GFN2-xTB on this ensemble; a model could in principle align with GFN2-xTB *better*. It is not an upper bound. | 26 | renamed **teacher-matching reference** throughout; the 39.8% figure survives only as "absolute correlation is ≈40% of the reference", never as the gain; gap closure is **23.0%** |
| 7 | **Figure 4 aggregation.** A **per-parent** identity `1 − r²` was overlaid on a point built from **mean** `r` and **median** NRV. Neither aggregate commutes with `1 − r²`. | 2 | curve removed; `NRV_min` reported as its own seed-aggregated statistic (FM 0.830, Value 0.747) in a new panel |
| 8 | **"Slope".** Figure 2(c) printed `slope 0.202 / 0.369` for a standardised association equal to the mean within-parent correlation, while "slope" elsewhere means the unstandardised calibration slope `0.127 / 0.448` — a factor of ~2.4 apart in one arm. | 2 | relabelled "mean within-parent r", with an in-panel disclaimer |
| 9 | **`prop:surrogate` status.** Listed under "Exercised empirically" while its stated condition ρ = 0 is contradicted by the paper's own measurement (the value arm has both the larger `r` *and* the larger NRV, impossible under ρ = 0). | 2 | renamed to a diagnostic identity; moved to a fourth scope category |
| 10 | **Group identity.** `cor:basins` claimed the objective and the metric average over the same group. They differ in potential (eSEN vs GFN2-xTB), population (training shards vs held-out parents), `K` and σ (5 at 0.03–0.40 Å vs 9 at 0.15 Å), and smoothing scale (`n = 4` vs `n = 12`). | 2 | corollary scoped to the objective's groups; the difference stated |
| 11 | **Abstract vs Figure 3.** The abstract attributed the gain unconditionally; Figure 3's caption said the same evidence "is consistent with either a real effect or with survivorship, and we cannot separate the two". | 2 | both replaced with the completion-conditioned formulation: means conditional on completion, unconditional magnitudes not identified, direction supported |
| 12 | **Scrambled arm flagging.** Table 1 printed the earlier round's scrambled arm at `r = 0.226` with no defect flag at the point of use, although only one of its two shards was scrambled, so ~¼ of its parent pool carried true pairings — the negative control was partly a positive control. | 3 | flagged in the row, the caption and the prose; the clean scramble is grid cell C |
| 13 | **"All arms differ only in the physics block and the seed."** True of the matched grid; false of the earlier checkpoint family, which differs in batch size, parents-per-value-step, loss cap and shard list. | 2 | scoped to the grid |
| 14 | **Figure 1(d).** Labelled the two clouds `group c_A` / `group c_B`, i.e. two *compositions* — illustrating a claim the theory does not make and reinforcing the `J = 1` error. | 1 | redrawn as two disconnected groups of the **same** composition inside one conditional density, with free offsets `b_A`, `b_B` |

---

## 8. Full run-status audit

64 training runs under `runs/{abl,p0}`. 141 checkpoint files across 47 run directories were loaded
(`torch.load`, `isfinite` over every floating-point tensor of the state dict). **Zero non-finite
parameters in every file.** That is not luck: `FiniteWeightGuard` halts training the moment weights
go non-finite, so a diverged run leaves either a clean earlier checkpoint or none at all. No
corrupted checkpoint exists on disk and nothing in the manuscript rests on one.

### Three failure modes, separated

| Mode | Meaning | Count | Runs |
|---|---|---|---|
| (i) | diverged **before** the first checkpoint; nothing to score; unrecoverable without training | **11** | `a6_..._s4`, `a6_..._s5`, `D_s1`, `D_s4`, `F_s1`, `F_s2`, `F_s4`, `F_s6`, `F_s7`, `F_s8`, `F_s9` |
| (ii) | diverged **after** the 15,000-step checkpoint; a finite half-budget model, correctly not scored | **3** | `a3_energy_only_s4` (@19600), `a6_energy_only_shuffled` (@17200), `p0_B_flat_s1` (@25800) |
| (iii) | **finished and finite but never scored** under the primary protocol | **4** | `a1_fm_only`, `a6_..._stab_s4` (recovered), `p0_D_energy_s8`, `p0_F_both_s10` |

Nothing was "never launched"; every prepared configuration ran. Category (iii) is the one the audit
exists to find.

> **Arithmetic note.** The audit's summary sentence says "9 runs" for mode (i) but then enumerates
> eleven run names. Counting the enumeration: 7 in the earlier round and grid wave 1, plus 4 in wave
> 2, = **11**. The manuscript uses the enumerated count.

### Earlier checkpoint family (`runs/abl`, "R1") — supplies `tab:primary`, `tab:scaleseeds`, `tab:respowered`, `tab:cells`

| Run | Condition | Training | Last step | Ckpt finite | Scored (primary) | Missingness informative |
|---|---|---|---|---|---|---|
| `a1_fm_only` | FM control, seed 42 | completed | 30000 | yes | **no** | **cannot be excluded, and in the paper's favour** — see below |
| `a1_fm_only_s2` | FM control, 1234 | completed | 30000 | yes | yes | no |
| `a1_fm_only_s3` | FM control, 3 | completed | 30000 | yes | yes | no |
| `a1_fm_only_s4` | FM control, 4 | completed | 30000 | yes | yes | no (absent from the fixed-RMSD / normal-mode rounds, which use 42/1234/3/5) |
| `a1_fm_only_s5` | FM control, 5 | completed | 30000 | yes | yes | no |
| `a2_force_only` | gradient only, 42 | completed | 30000 | yes | no (n=30 round only, r = 0.109) | no — superseded by cell E; `tab:cells` declares the different population |
| `a3_energy_only` | value, 42 | completed | 30000 | yes | yes | no |
| `a3_energy_only_s2` | value, 1234 | completed | 30000 | yes | yes | no |
| `a3_energy_only_s3` | value, 3 | completed | 30000 | yes | yes | no |
| `a3_energy_only_s4` | value, 4 | **diverged @19600** | 15000 | yes (half-budget) | no | **YES** — divergence occurs only in value-carrying arms, so completion is not independent of the arm |
| `a3_energy_only_s5` | value, 5 | completed | 30000 | yes | yes | no |
| `a4_force_energy` | gradient + value, 42 | completed | 30000 | yes | no (n=30 round only, r = 0.294) | no — superseded by cell F |
| `a5_shuffle_energy` | scramble, 42; **only one of two shards scrambled** | completed | 30000 | yes | no (n=30 only, r = 0.262) | no — excluded on a disclosed design defect, not on its value |
| `a6_energy_only_shuffled` | scramble, 42 | **diverged @17200** | 15000 | yes | no | **YES** |
| `a6_energy_only_shuffled_s2` | scramble, 1234 | completed | 30000 | yes | yes | no |
| `a6_energy_only_shuffled_s3` | scramble, 3 | completed | 30000 | yes | yes | no |
| `a6_energy_only_shuffled_s4` | scramble, 4 | **diverged @1600** | — (no ckpt) | — | no | **YES** |
| `a6_energy_only_shuffled_s5` | scramble, 5 | **diverged @1400** | — (no ckpt) | — | no | **YES** |
| `a6_..._stab_s4` | scramble, 4, **re-run** (batch 8→4, M 4→8, cap 3000→1500) | completed | 30000 | yes | **no — launcher path fault**; recovered | **YES, but conservative** — see below |
| `a6_..._stab_s5` | scramble, 5, **re-run**, same stabilised config | completed | 30000 | yes | yes | no for its own value; the manuscript did not disclose that it is a *replacement* |

**`a1_fm_only` — the one omission whose missingness cannot be certified ignorable.** The 120-parent
round was launched 2026-07-19 against the set named in the then-current plan ("the seed-2 triple
plus a3 seed 1"); when seeds 3–5 were appended this seed was never added. No `wide_a1_fm_only`
directory and no such SLURM job exists, so no primary-protocol result for it was ever produced and
the omission cannot be shown to be result-driven. Equally it cannot be certified ignorable: the only
score that ever existed for it (`r = 0.223` in the superseded n = 30 round) was the **higher** of
that round's two FM values, and dropping a strong control seed lowers the control mean and widens
the reported gap. One evaluation job would settle it. It is disclosed in `app:eval` in exactly these
terms.

**`a6_..._stab_s4` — recovered at zero compute cost.** Two independent faults: the watchdog saw the
training job as TIMEOUT on 2026-07-30 and marked the arm done, but the good step-30000 checkpoint
landed 2026-08-01; and the manual wide-eval submitted 2026-08-04 pointed at the *preempted* attempt's
version directory and died with `FileNotFoundError`, leaving an empty output directory nobody
audited. The score was recovered from records already on disk (draw-identity established to 1.2e-4
in `log p`; route validated by reproducing two published NRVs to six decimals): **r = 0.161**, the
**lowest** of the four scrambled seeds, slope 0.104, kT_eff 9.59 eV. The arm at four seeds is 0.210
against the three-seed 0.226. Its absence flattered the control; including it moves the scrambled
arm *further below* the value arm's 0.369.

**The honest replacement for "five, five and seven prepared, of which four, four and three are
reported":** the two unreported value and scrambled runs plus two further scrambled runs diverged to
non-finite weights before the training budget and were not replaced; one scrambled run completed with
finite weights but lost its evaluation to a launcher path error and has since been scored (r = 0.161,
the arm's lowest); and one flow-matching run completed with finite weights and was never submitted to
this round's evaluation at all.

### Matched six-cell grid (`runs/p0`, "R2" wave 1) — supplies `tab:p0cells`, `tab:p0seeds`, Figure 3

| Cell | Condition | Launched / completed / diverged | Diverged at | Scored |
|---|---|---|---|---|
| A | FM-only (`λ₁ = 0, λ_E = 0`) | 5 / 5 / 0 | — | s1–s5 |
| B | zero energies (`λ_E = 3e−5` on an identically-zero shard) | 5 / 4 / 1 | s1 @25800 (15k ckpt survives) | s2–s5 |
| C | energies permuted within parent | 5 / 5 / 0 | — | s1–s5 |
| D | true teacher energies | 5 / 3 / 2 | s1 @9600, s4 @1800 (no ckpt) | s2, s3, s5 |
| E | gradient only (`λ₁ = 0.1`) | 5 / 5 / 0 | — | s1–s5 |
| F | gradient + value | 5 / 2 / 3 | s1 @3000, s2 @4600, s4 @8800 (no ckpt) | s3, s5 |

### Later relaunch block (wave 2 and the gradient-weight variant) — **not in the pre-revision manuscript**

| Run | Condition | Training | Scored | Value |
|---|---|---|---|---|
| `p0_D_energy_s6` | cell D, fresh seed; config byte-identical to `s5` apart from name and `output_dir` | completed | yes, **not reported** | **0.353251** |
| `p0_D_energy_s7` | cell D, fresh seed | completed | yes, **not reported** | **0.373749** |
| `p0_D_energy_s8` | cell D, fresh seed | completed | **no eval job ever submitted** | — |
| `p0_D_energy_s9` | cell D, fresh seed | completed | yes, **not reported** | **0.470672** (highest cell-D value on record) |
| `p0_F_both_s6..s9` | cell F, fresh seeds | **all diverged** (@5600, @8800, @8800, @13400; no ckpt) | — | — |
| `p0_F_both_s10` | cell F, fresh seed | completed | **no eval job ever submitted** | — |
| `p0_E2_force_s1..s5` | cell E variant, `λ₁ = 0.01` (one line different from cell E) | all 5 completed | yes, **not reported** | 0.117697, 0.076397, 0.089002, 0.078024, 0.038388 → **0.080 ± 0.013** |

`paper/figures/fig3_grid_data.json` excludes the D and F relaunches by name as "post-hoc relaunch"
and the E2 family as "a different force variant".

**Why this mattered.** `tab:p0cells`'s caption claimed the status columns were the complete launch
record and that **no seed was re-rolled**. Fourteen further runs of the same design were launched
2026-08-11. Over the whole campaign cell D is **9 / 7 / 2** and cell F is **10 / 3 / 7**. The
earlier round's scrambled arm was re-rolled too (`stab_s4`, `stab_s5`, under a changed
configuration that the manuscript disclosed without saying they were replacements).

**Why the relaunch evidence does not cut against the paper.** The three scored relaunched D seeds
are 0.353, 0.374 and 0.471 — every one **above every control seed in the grid** (highest control
0.243). The worry that D's survivors are a lucky subset is now bounded by data instead of by
argument. Similarly E2 shows that "gradient supervision hurts local ordering" is not an artefact of
one `λ₁`: at a tenfold smaller gradient weight the force-only arm still scores 0.080 ± 0.013, below
FM-only's 0.217. The correct repair is to report the complete launch record and add the relaunch
seeds separately, not to keep the smaller honest-sounding one.

### Stability

| Population | Diverged / total |
|---|---|
| grid wave 1, value-density estimator present (B, C, D, F) | **6 / 20** |
| grid wave 1, estimator absent (A, E) | **0 / 10** |
| whole campaign, estimator present (B 5, C 5, D 9, F 10) | **10 / 29** |
| whole campaign, estimator absent (A 5, E 5, E2 5) | **0 / 15** |

The value-density estimator is a necessary condition for divergence on this evidence. Cell B, whose
energies are **identically zero**, still lost a seed — so instability lives in the reverse-time
log-density estimate, not in anomalous teacher energies. The per-cell dose-response (B 1/5, C 0/5,
D 2/5, F 3/5 in wave 1) is suggestive only at these counts, and the grid does not separate the
estimator from its gradient.

---

## 9. Claim-to-round map

Two evidence families. They are cleanly separable and must never be pooled.

### Family 1 — the earlier checkpoint family, re-evaluated under the corrected primary endpoint (93 shard-disjoint parents, data geometry dropped, GFN2-xTB, `n_ode = 12`)

| Claim | Defects carried |
|---|---|
| **Primary ordering**: mean within-parent `r` 0.202 ± 0.023 → 0.369 ± 0.023; Welch t = 5.16; exact permutation p = 2/70 = 0.029; value higher on 69/93 | Round-level: value arms read one shard built from the validation split the evaluation parents come from. **Repaired for this endpoint** by restricting to the 93 disjoint parents; high-leverage data geometry removed by dropping it. Both corrections *reduce* the contrast. Residual: 4 of 5 prepared seeds per arm reported; adverse imputation removes significance; value-arm means completion-conditional; effect size resolution-conditional. |
| **Retrieval**: top-1 19.6% → 29.6%; top-3 49.2% → 62.4% (chance 12.5% / 37.5%) | Same shard-overlap repair. Added **after** the evaluation was designed; not a pre-registered endpoint. Secondary, within-cloud read-out only. |
| **Calibration**: NRV 1.405 → 2.144; NRV_min 0.830 → 0.747; slope 0.127 → 0.448; scale ratio 0.834 → 1.394; kT_eff 8.23 → 2.34 eV (target 1.0 eV) | Same round defects. **NRV is a median over parents per seed then a mean over seeds; `r` is a mean over parents then over seeds.** They are differently aggregated, which is why the `1 − r²` overlay in Figure 4 was invalid. |
| **Teacher-matching reference** `r = 0.927` (retrieval 87.1% / 98.9%) | Not a model and **not an upper bound**: what exact reproduction of the eSEN teacher scores against GFN2-xTB on this ensemble. Ensemble-dependent and unusually high because within-group energy spread averages 14.3 eV against 4.15 eV mean absolute teacher–evaluator disagreement. Absolute ratio 0.369/0.927 = 39.8%; **gap closure (0.369−0.202)/(0.927−0.202) = 23.0%**. |
| **Fixed-RMSD robustness**: value 0.305 vs FM 0.182, +0.124, t = 4.94, 4 seeds/arm | Only 2 of 4 planned confound-breaking families run. No single artifact reproduces the verified pair (the two available rounds give 0.303/0.189 over 40 parents and 0.350/0.248 over 120), so Figure 2 draws no panel for it. Reference for this narrower ensemble never measured. |
| **Normal-mode family**: +0.185, t = 1.58, 5 seeds | **Not significant.** Supporting context only; never a confirmation. |
| **Resolution dependence** (re-scoring 9 existing checkpoints at a pinned evaluation seed, 120 parents) | Not an independent replication — same checkpoint family. See §10 for the corrected values. |
| **Generation quality** (validity 43.0 vs 35.5, etc.) | One value run against a two-seed control; n = 100 samples per model, binomial s.e. ≈5 points, not seed-controlled. Supports only *absence of evidence of degradation*. |

### Family 2 — the matched six-cell grid (120 parents, reference dropped, `n_ode = 12`, training-split shards only)

| Claim | Value | Defects |
|---|---|---|
| A FM-only | 0.217 ± 0.008 (5/5/0) | none material: no shard overlap by construction, common warm start |
| B zero energies; B − A | 0.187 ± 0.030 (5/4/1); −0.030, t = −0.96 | one diverged seed; among the contrasts least exposed to selection. Supplies the stability finding that instability does not require anomalous energy labels |
| C clean scramble; D − C | 0.185 ± 0.020 (5/5/0); +0.236, t = 9.28 | none on C itself. **Supersedes** the earlier round's scrambled arm, whose validation shard was never scrambled and whose third seed was not configuration-matched |
| D true pairing; D − A, D − B | 0.421 ± 0.016 (5/3/2); +0.204 t = 11.36, +0.234 t = 6.89 | two diverged seeds; mean conditional on completion, **unconditional magnitude not identified**. This cell establishes **direction, not magnitude** |
| E gradient only; E − A | 0.085 ± 0.016 (5/5/0); −0.132, t = −7.60 | clean. Supports only the narrow claim that naive off-policy endpoint force matching through the flow-matching score read-out is harmful *in this non-equilibrium-data configuration*. The earlier round's force cells (a2/a4) carried an on-policy distillation term and are confounded; that older comparison must not be reused |
| F gradient + value; F − D | 0.194 ± 0.016 (5/2/3); −0.227, t = −10.05 | three diverged seeds in wave 1, seven of ten over the campaign; **the most completion-conditional number in the paper** |
| Stability census | 6/20 with estimator, 0/10 without | none as a census; per-cell dose-response suggestive only |

### Superseded and used for nothing

The old four-cell table `a1`–`a4` (30 parents, reference **included**) carries four defects: on-policy
distillation in `a2`/`a4` only, a validation-split shard in `a3`, an unscrambled validation shard in
the scrambled control, and unmatched batch / M / cap. Genuinely superseded in full by the matched
grid. It is used for no headline number and should stay that way.

---

## 10. One place where the manuscript's numbers were stale, not wrong

The revision brief fixed the resolution ablation at "+0.382 at n = 4, +0.177 at n = 12, and the
n = 48 cell has one value seed and is not a result". That was true when `tab:respowered` was
assembled. It is not true of the artifacts on disk: `runs/eval_res` now holds **four** completed
value-arm evaluations at `n_ode = 48` (`a3_energy_only`, `_s2`, `_s3`, `_s5`) and five FM-only, all
120 parents, all on the pinned evaluation seed 12345. The three previously preempted evaluations
were relaunched and completed 2026-08-11 at 20:39, 20:50 and 20:58, after the table was written.

Running the paper's own `scripts/analyze_resolution_ablation.py` unmodified (an allowed re-analysis
of existing records; no training, no new GPU job):

| `n` | ε | value | FM-only | gap | t |
|---|---|---|---|---|---|
| 4 | 0.1250 | +0.571 (4 seeds) | +0.189 (5) | **+0.382** | 9.40 |
| 12 | 0.0417 | +0.418 (4) | +0.204 (5) | **+0.214** | 6.35 |
| 48 | 0.0104 | +0.306 (4) | +0.269 (5) | **+0.037** | 1.94 |

An independent second family confirms the shape: the matched-grid checkpoints scored at the same two
extremes give gap +0.453 at ε = 0.125 (6 vs 5 seeds, t = 27.85) falling to +0.053 at ε = 0.0104
(t = 3.06).

The defensible statement is therefore not "we have no result at the finest grid" but **the gap falls
by roughly an order of magnitude between the training smoothing scale and the finest grid we
measured, and at that grid it is small and not separated from zero at these seed counts.** That is a
statement about the *scale* at which local ordering is improved — property (i) in §1 — and it says
nothing about calibration or mass allocation. It is a stronger and more defensible version of the
manuscript's own limitation on resolution. Logged as REVIEWER_DISAGREE D1 in `REVISION_LOG.md`.

---

## 11. Cannot be resolved from existing artifacts

Each item below needs work this pass forbids. All are disclosed in the manuscript.

1. **Three finite checkpoints have no primary-protocol score.** `abl/a1_fm_only`,
   `p0/p0_D_energy_s8`, `p0/p0_F_both_s10`. Each needs one fresh multi-hour GPU evaluation. All
   three were finiteness-verified, so the jobs would pass their own guard. Exact commands:
   - `sbatch -J w_a1_fm_only scripts/fasrc/eval_boltzmann_wide.slurm <ckpt> configs/sweep/a1_fm_only.yaml wide_a1_fm_only 120`
   - `sbatch -J p0_D_energy_s8 scripts/fasrc/eval_clean_arm.slurm p0_D_energy_s8 120`
   - `sbatch -J p0_F_both_s10 scripts/fasrc/eval_clean_arm.slurm p0_F_both_s10 120`

   Of these, `a1_fm_only` is the one that touches the headline (§8). `p0_F_both_s10` is the one that
   would most change a reported contrast: cell F has the worst completion rate and the smallest
   reported `n`, so a third F value would materially move the F − D contrast's credibility in either
   direction.
2. **The unconditional magnitudes of cells D and F are not identified.** Divergence is not
   independent of the arm, so no reweighting of the completed runs recovers them. Only a stabilised
   training configuration would, and that is a new experiment.
3. **The fixed-RMSD panel cannot be drawn** without a cross-round comparison (§9), so that result
   stays prose.
4. **No teacher-matching reference exists for the narrower perturbation families** (fixed-RMSD,
   normal-mode), so no correlation on those families may be expressed as a fraction of a reference.
   A narrower ensemble has a materially lower reference, and it must be measured, never inherited.
5. **Property (iii), global mass allocation, is not measured anywhere.** Measuring it requires a
   supervision design whose overlap graph is connected — groups tiled along a path between basins —
   which is a different experiment and is named as such rather than claimed.
6. **The multiplicity of every evaluated parent was not verified**, so the singlet default is
   asserted as the standard assignment for closed-shell main-group organics rather than as a
   checked fact.
7. **`refs.bib` carries 17 `% TODO-verify` markers** on bibliographic fields never confirmed against
   a primary source. They render nowhere (0 occurrences in the PDF and in `main.bbl`), so this is not
   a submission gate, but they remain unverified facts inside the shipped bibliography.
8. **The AI-use statement's factual content** is unverifiable by any tool and is flagged for author
   sign-off by a non-rendering comment in `main.tex`.

---

## 12. What was checked, and how

Verifications performed for this audit (all read-only; no training, no new GPU job):

- **Checkpoint finiteness**: all 141 `.ckpt` files under `runs/{abl,p0}` loaded and scanned.
- **Divergence ledger**: `NON-FINITE WEIGHTS at global_step=` grepped from all 63 `sweep-*.out`
  training logs; RNG seeds recovered from `Seed set to N` in the matching `.err` logs (seeds are a
  positional argument to `sweep.slurm`, not stored in the YAML).
- **Grid protocol replication**: the six-cell statistic recomputed from
  `runs/eval_ours/clean_p0_*/boltz_independent_records.csv` (120 parents, `pert_id == 0` dropped,
  `xtb_ok` only, per-parent Pearson `r`, mean over parents), reproducing every value in
  `fig3_grid_data.json` to six decimals before the same computation was applied to the eight
  evaluated-but-unreported runs.
- **Draw identity**: `n240_*` and the primary wide round evaluate the same parents and the same
  Gaussian displacements for `group_id < 120` (key-matched on `(group_id, pert_id)` across four
  arms; `log p` agrees to 1.22e-4, float32 CSV rounding; 1074 of 1078 records shared, the remainder
  xTB convergence differences). The wide round and the `clean_p0` grid round share all 1074 record
  keys, so the primary endpoint and the matched grid sit on **one evaluation draw**.
- **Configuration diffs**: `p0_D_energy_s6..s9` and `p0_F_both_s6..s10` are byte-identical to their
  wave-1 counterparts apart from name and `output_dir`; `p0_E2_force_*` differ from `p0_E_force_*`
  in exactly one line (`λ₁` 0.1 → 0.01); `a6_..._stab_s4/s5` differ from `a6_..._s4/s5` in three
  lines (batch 8→4, `energy_b_parents` 4→8, cap 3000→1500).
- **Code read for this document**: `cfm_mol/log_z_predictor.py` (anchor feature set),
  `cfm_mol/bgfm_density.py:90-115` (position-only velocity function) and `:252ff`
  (`within_group_variance_loss`), `scripts/precompute_energy_perturbations.py:144-154` (spin
  hard-coded to 1), `scripts/eval_boltzmann_independent.py` (charge threaded, no spin).
- **Deliberately not run**: any training; any new GPU evaluation.
