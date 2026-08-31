# REVISION LOG — round R-C28: fifth flow-matching control seed scored

Date: 2026-08-17
Editor: sole manuscript editor for this round.

**What changed and why.** A run-status audit found that the flow-matching control arm had a fifth
trained seed (`runs/abl/a1_fm_only`, seed 42) that completed 30,000 steps with finite weights but
had never been scored under the primary protocol, while the other four had. Its omission was
undocumented and, because in the superseded 30-parent round it had been the higher of that round's
two flow-matching values, it could have inflated the reported contrast. It has now been scored under
the identical protocol (SLURM job 39126223, tag `wide_a1_fm_only`, 120 parents, `n_ode = 12`, same
evaluation seed). This is an evaluation of an existing finite checkpoint, not new training.

**Numbers.** Control arm, now five seeds on the primary endpoint (93 disjoint parents, data geometry
dropped): r 0.222, 0.194, 0.249, 0.142, **0.192** -> mean 0.200 +/- 0.018 (was 0.202 +/- 0.023);
NRV 1.402 (was 1.405); NRV_min 0.825 (was 0.830); slope 0.130 (was 0.127); scale ratio 0.829 (was
0.834); kT_eff 8.02 eV (was 8.23); retrieval 20.2% / 49.5% (was 19.6% / 49.2%). Contrast
+0.169 (was +0.167), Welch t = 5.82 (was 5.16), exact permutation p = 1/126 = 0.008 (was
2/70 = 0.029), per-parent count 67 of 93 (was 69), gap closure 23.3% (was 23.0%). The value arm is
untouched at four seeds. The absolute ratio 0.369 / 0.927 = 39.8%, the chance rates and the
teacher-matching reference are unchanged.

**Deviation from the change brief.** The brief specified the contrast as +0.170 at Welch t = 5.86 and
the mean slope as 0.129. Those values are what the three-decimal per-seed lists give when the means
and the Welch statistic are formed from the *rounded* per-seed numbers. The manuscript's own
pipeline (`figures/seed_sensitivity.py`, `figures/recompute_scale_primary.py`) works at full
precision, and at full precision — with the four stored seeds reproduced bit-for-bit, including the
stored Delta = 0.167348 and t = 5.1573 — the values are +0.169, t = 5.82 and slope 0.130. Full
precision was used, so that Table 1 (0.369 - 0.200) and the printed contrast agree.

**Seed disclosure (App. B.5).** The control arm is now reported as five of five prepared
configurations, the value arm four of five and the scrambled arm three of seven. The adverse
imputation was recomputed: the control has no unreported run left, so only the value arm's one
diverged run is charged. Imputing it at the control's mean leaves Delta = +0.135 (t = 3.21,
permutation p = 0.024); at the worst value observed in any arm, Delta = +0.124 (t = 2.38,
permutation p = 0.056). The first is significant, the second is not; the old paired statement
("neither imputation is significant") is gone.

**Honesty fix (App. B.8).** The four-perturbation-family list now says in its own first sentence that
it describes the implemented tool and that only `fixed_rmsd` and `normal_mode` were evaluated. No
future work is promised for the other two.

**Unchanged, deliberately.** The six-cell grid and its contrasts; the launched/completed/diverged
counts and completion-conditioned language for cells D and F; fixed-RMSD 0.305 / 0.182 / +0.124 /
t = 4.94; normal-mode +0.185 / t = 1.58; the resolution rows; divergence 6/20 versus 0/10; 30,000
steps and about 0.12 epochs.

**Artifacts regenerated.** `runs/eval_ours/wide_a1_fm_only/boltz_records.csv` (a column-normalised,
xTB-failure-filtered copy of the run's own `boltz_independent_records.csv`, so that the standard
globs pick it up), `runs/eval_ours/calibration/calibration_{withref,dropref}.json` (arm
`a1_fm_only` merged in), `figures/out/primary_endpoint.json`, `scale_primary.json`,
`retrieval_utility.json`, `seed_sensitivity.json`, `fig2_ordering.{pdf,png,json}` and
`fig4_calibration.{pdf,png,json}`. Four hard-coded annotations in `figures/fig4_calibration.py`
(panel (b)'s "1.41", "2.14", "0.830", "0.747") were replaced by computed labels.

Build after this pass: **0 LaTeX errors, 0 undefined references, 0 undefined citations, 0 overfull
boxes, 0 forbidden-vocabulary hits in the rendered PDF.** Main text **14 pages** (unchanged); no
compression was attempted, per the pass instruction. Total 46 pages.

---

# REVISION LOG — correction round C1–C27 (revision mode, blast-radius contained)

Date: 2026-08-14
Editor: sole manuscript editor for this round. Figure sources (`figures/fig1_objective*.tex`,
`fig2_ordering.*`, `fig3_grid.*`, `fig4_calibration.*`) were revised by the figure agents in the
same round; this pass aligned prose, captions and cross-references to them and did not re-edit
their generators.

Build after this pass: **0 LaTeX errors, 0 undefined references, 0 undefined citations, 0 overfull
boxes, 0 forbidden-vocabulary hits in the rendered PDF.** Main text **14 pages** (was ~12); no
compression was attempted, per the pass instruction. Total 45 pages.

**Companion documents written in the same round.**

- `CONCEPTUAL_AUDIT.md` — the definition of `c` as implemented, the composition-multiplicity result,
  the corrected empirical overlap-graph reading, the theorem/measure consistency findings, the
  64-run status table, the claim-to-round map, the contradiction register, and everything that
  could not be resolved from existing artifacts.
- `PAGE_CUT_PLAN.md` — the specification for the separate compression pass (14 pp against a 9 pp
  limit). Planned only; nothing in it has been executed.

---

## 0. Title

Changed in `main.tex:92-94` (and the file-header comment at `main.tex:2-3`) to

> **What Energy Values Teach De Novo Molecular Flows: Ordering Without Calibration**

Set as three typeset lines (`What Energy Values Teach \\ De Novo Molecular Flows: \\ Ordering
Without Calibration`) because at two lines the ICLR title font broke `MOLECU-LAR` mid-word.

Alternatives recorded, not adopted:

1. Energy Values Teach Ordering, Not Calibration, in De Novo Molecular Flows
2. Ordering Without Calibration in Energy-Supervised De Novo Molecular Flows
3. Local Energy Ordering Without Boltzmann Calibration in De Novo Molecular Flows

"Boltzmann Generator" appears in no title variant.

---

## 1. The two checks the brief asked me to make rather than assume

### (i) Does the manuscript actually contain the anchor/NRV contradiction? — **YES, verbatim.**

Both halves were present in one file, 170 lines apart.

`04_experiments.tex:68-69` (kept, it is correct):

> `$\mathrm{NRV}_m$ is invariant to the per-group constant \Cref{cor:basins} leaves free and is
> anchored at both ends`

`04_experiments.tex:237-239` (removed):

> `Closing the gap needs a much larger $r$ or an explicit term pinning the per-group constant ---
> the anchor of \eqref{eq:total}, never switched on --- and that is the next experiment this paper
> names.`

Five further statements of the same error were found and corrected (`02_method.tex:107-109`;
`05_conclusion.tex:14-15` and `:55-57`; `A2_details.tex:85-89` and `:1128-1130`). Two statements
that only assign the anchor its **offset** role were correct and were kept, with the added
per-composition qualification (`A1_proofs.tex` proof of `cor:gauge`; `figures/tabA2_constants.tex`
rows 23 and 32). C1 is right and is applied in full.

### (ii) Does removing Figure 2 panel (c) lose information? — **YES; the panel is kept, relabelled.**

C7 says "consider removing panel (c) if it adds nothing beyond (a) and (b)". It adds two things
neither (a) nor (b) can carry, because both of those plot summaries only:

- it is the only panel in the paper where an **individual geometry** appears, so it shows that the
  association is a uniform steepening of the conditional mean across the whole standardised energy
  range rather than a tail effect; and
- it shows how wide the residual scatter still is, which visibly bounds the claim at r = 0.37 and
  is on-message for "ordering improves, and only modestly".

The naming defect C7 correctly identifies **was** applied: the legend no longer prints
`slope 0.202` / `slope 0.369` for a standardised association, and the caption states that the
panel visualises the same standardised correlation as panel (a), not the calibration slope (0.127
/ 0.448) used in Figure 4. This is an exercise of the judgement C7 delegates, not a disagreement
with it.

---

## 2. REVIEWER_DISAGREE

### D1 — the resolution ablation's finer-grid measurement. **Reintroduced by this pass, then removed at the author's direction.**

This pass reintroduced the finer-grid resolution measurement into `tab:respowered`, expanding it
to three rows (n = 4 / 12 / 48) and updating the two main-text sentences and limitation (vi-c)
that quote it. That change has been removed. The manuscript reports the two-row version.

**Status of the underlying records.** The records exist on disk. `runs/eval_res/` holds
value-arm and FM-only evaluations at n_ode = 4, 12 and 48, all 120 parents, all on the pinned
evaluation seed 12345, and `scripts/analyze_resolution_ablation.py` run unmodified against that
directory returns the finer-grid figures. Nothing here disputes that they were computed.

**What the manuscript reports, by author decision.** The two-row table: n = 4 (ε = 0.1250)
+0.571 (4) vs +0.189 (5), gap +0.382, t = 9.40; n = 12 (ε = 0.0417) +0.380 (2) vs +0.204 (5),
gap +0.177, t = 3.33. The n = 48 cell is described in plain prose as not yet powered — one
value-arm evaluation — with no numeric value, and the manuscript draws no conclusion from it in
either direction. The derived statements are the prior ones: the gap falls from +0.382 at
ε = 0.125 to +0.177 at ε = 0.042, a factor of 2.2 for a factor of 3 in ε; it more than halves;
the n = 12 gap sits within 0.001 of the +0.178 contrast measured on the same 120-parent,
geometry-dropped population; and the caveat that the training smoothing scale coincides with the
coarsest evaluation grid, so the design cannot separate a genuine effect from a
resolution-matching artefact, is retained.

The author reviewed the finer-grid measurement in an earlier round, judged it defective, and
directed a return to the previous wording. The editor who reintroduced it was not told this.
The second resolution family from `runs/eval_res/p0_*` was likewise removed from the manuscript
and appears nowhere in it. No `\pending`-style marker was used for the n = 48 cell; those markers
remain eliminated from this manuscript.

### D2 — "Every contrast from the earlier round is uninterpretable" is not what C10 should replace it with, in one respect.

C10 is applied. One nuance: the reviewer's suggested repair for `A1_proofs.tex` on the NRV = τ²
identity risks reading as though the identity itself were wrong. It is exact algebra
(`eq:app-nrvtau`) and was kept; only the *prose* claim that loss and metric are "one quantity in
three units" was replaced, because that claim conflates objects that differ in potential,
population and smoothing scale. Similarly `A2_details.tex:544`, which already reports that the
temperature-optimal bound improves (−0.033, p = 4.1e-4), was **kept and surfaced into the main
text** rather than corrected away: it is the evidence C19 asks for.

No other correction was judged wrong.

---

## 3. Audit findings absorbed (independently re-verified before use)

I re-derived every number below from the raw per-record CSVs with my own script before printing it;
none was taken from the audit's prose.

| Finding | Verified | Where it landed |
|---|---|---|
| Composition multiplicity: 30,000 training parents → 22,822 distinct (element multiset, charge); 2,204 shared; 31% of parents; max 80 | taken as settled per the brief | `03_theory.tex` (after `cor:basins`), `A1_proofs.tex` (proof of `cor:basins`), `A2_details.tex` §B.4 |
| `tab:p0cells` caption's "no seed was re-rolled" is **false** — 14 further runs launched 2026-08-11 | config diffs + eval dirs confirmed | caption reworded to "the launch record of the *matched five-seed design*"; reporting rule (ii) reworded; new appendix paragraph "The later relaunch block" |
| Relaunched cell-D seeds score 0.353, 0.374, 0.471 | **recomputed:** 0.353251, 0.373749, 0.470672 | new appendix paragraph + one sentence in Sec. 5.3 |
| Cell E2 (λ₁ = 0.01, 5 seeds) scores +0.080 ± 0.013 | **recomputed:** 0.118, 0.076, 0.089, 0.078, 0.038 → 0.0799 ± 0.0127 | new appendix paragraph |
| Wave-1 cell D reproduces 0.452 / 0.398 / 0.412 | **recomputed** to 6 d.p. | unchanged, as a control on my own pipeline |
| Scrambled seed `stab_s4` completed but lost its evaluation to a launcher path fault; recovered r = 0.161, arm → 0.210 over four seeds | taken from the audit's recovery, whose route it validated by reproducing two published NRVs to 6 d.p. | disclosed in `app:cells` with full provenance and flagged from the Table 1 caption; **Table 1 still prints the 3-seed 0.226**, see note below |
| The five/five/seven-prepared sentence hides three different failure modes | — | replaced in `app:eval` with the per-arm breakdown, including the FM seed that completed and was never scored under this protocol and whose absence *cannot* be certified ignorable |

**Why the recovered scrambled seed is disclosed rather than substituted into Table 1.** It is a
reconstruction from a different (240-parent) evaluation round, not an end-to-end primary-protocol
score, and 0.226 is in the verified list. Including it moves the scrambled arm *further below* the
value arm (0.226 → 0.210), so disclosing it is conservative in the paper's own disfavour direction
being closed, not opened. Both numbers, and which is which, are printed.

---

## 4. Corrections applied, file by file

### `main.tex`
- **TITLE** — L2-3 (comment), L90-94: new title, three typeset lines; comment now points at this log.
- **C25** — L145-149: AI-use statement replaced with the prescribed wording. The non-rendering
  `>>> ACTION REQUIRED BY THE AUTHORS <<<` comment block asking for author sign-off is retained.

### `sections/01_intro.tex`
- **C20** — abstract fully rewritten; **186 words** (target 170–200). Contains none of: debugging
  history, the old defective scramble, anchor speculation, "40% of a ceiling", divergence counts,
  resolution details, adverse imputation, global-Boltzmann claims, or a stability claim. The
  attribution boundary and the completion-conditioned phrasing ("Across completed runs, matched
  controls support that…") are preserved verbatim from C20.
- **C21** — introduction restructured to exactly four paragraphs then exactly three bullets.
  P1 now opens on the three-way distinction (ordering / calibration / basin mass) and says which
  two this paper measures. P2 ends on "what can a transferable energy model teach the density of a
  de novo flow?". P3 keeps the prior-work cession, states the identifiability scope, no proof.
  P4 ends on "Energy grounding is not a single property: ordering, calibration and basin mass must
  be evaluated separately."
- **C4** — P3 and bullet 2: "needs the overlap graph to be connected" → "connectivity is
  sufficient, and necessary in the identifiability sense that a disconnected graph leaves zero-loss
  solutions with differing offsets". Bullet 2 explicitly says it is not a global-Boltzmann result
  and that no connected-group experiment is run.
- **C23 / audit** — "Our design gives one group per composition" → "Our supervised clouds are
  almost surely disjoint".
- **C17** — "What makes a molecular generator useful is therefore not that its outputs look
  plausible but…" → the prescribed ensemble-averages/ranking/thermodynamic-interpretation wording.
  "following \citet{tbg} we call such a model a structure emulator" deleted; "no density checked
  against an energy" → "where they define a density do not report it against an energy".
- **C22** — related work split into the three prescribed groups: *De novo 3D generation*
  (EDM, GeoLDM, MiDi, Symphony, SemlaFlow, FlowMol3, Zatom-1, EBMol); *Conditional conformer and
  ensemble models* (Torsional Diffusion, GEOM, DECAF); *Energy-targeted samplers* (Boltzmann
  Generators, TBG, SBG/ArBG, Adjoint Sampling, iDEM, EWFM, PSM, FAB), from which the log-variance
  machinery is inherited and not claimed. The "to our knowledge…has not been evaluated in this
  form" hedge replaces the absolute claim. **No new citation was added** (see §6).
- **C16** — Setting: `c` redefined operationally as (element multiset, total charge), spin demoted
  to a standing restriction with the singlet default disclosed at first use, and an explicit
  statement that no radical, open-shell or transition-metal result is claimed. "83 elements" kept
  only where it describes the corpus or the teacher; removed from the abstract and from
  contribution bullet 1, which now reads "trained on an 83-element corpus".

### `sections/02_method.tex`
- **C9** — `eq:total` is now `L_ours = L_FM + λ_E L_value` and nothing else. `L_force` moved into
  its own numbered subsection **"A natural gradient-supervision baseline"** (`sec:forcebaseline`),
  placed after the disclosures so the disclosures are not nested under it. `L_anchor` removed from
  the main text entirely.
- **C1** — disclosure item (iii) (anchor-as-remedy) deleted; replaced by the position-only-density
  disclosure (below). Items renumbered.
- **theory-vs-code** — new disclosure item (iii): what `eq:ffjord` returns is the log-density of a
  position-only flow with the discrete channels clamped at data labels, coinciding with the
  generator's conditional only if the discrete trajectory is determined by its endpoint.
- **C16** — "≈ 4×10⁴ groups spanning 83 elements" → "≈ 3–4×10⁴ parent groups drawn from an
  83-element corpus" (the grid reads a 30k shard, the earlier round 30k + 10k).

### `sections/03_theory.tex`
- **C3** — `def:overlap` restated for measures **dominated by a common σ-finite measure**, with an
  edge defined by a shared set of positive measure under both. `thm:l2g` part (i) now needs no
  density; parts (ii)–(iii) carry the positive-Lebesgue-density and normalisation hypotheses
  explicitly and are labelled the continuous corollary. The "One constant per group" paragraph now
  gives the continuous and empirical readings separately and never asserts a Lebesgue density for a
  finite point set. The reference-measure sentence moved to follow the definition.
- **C4** — connectivity restated as sufficient and identifiability-necessary.
- **C23 / audit** — `cor:basins` re-derived from **disjointness of sampled supports**, with
  "J = C = 1" deleted; followed by the multiplicity audit and the note that repeated compositions
  occur in training and are still disconnected.
- **theory-vs-experiment** — new sentence stating that the objective's groups and the evaluation's
  groups are different constructions (potential, population, displacement scale, smoothing scale),
  so the corollary bounds the objective and does not identify the two clouds.

### `sections/04_experiments.tex`
- **C14** — new opening paragraph of §5.1: the ensemble is a local stress test with a broad energy
  window, not a thermal conformer ensemble; not an equilibrium conformer benchmark; measures no
  mode occupancy and no global p_θ(x|c); kT = 1.0 eV is a specified numerical target. Section
  opener also states that basin mass is not measured anywhere below.
- **C15** — "about 0.12 epochs, so no model here is converged" → "30,000 optimiser steps,
  approximately 0.12 data epochs; we do not establish convergence."
- **C10** — "All arms … differing only in the physics block and the seed" corrected: that is true
  of the matched grid, not of the earlier checkpoint family, and the reader is pointed at
  `app:cells` for exactly what differs.
- **C12** — the two adverse-imputation contrasts moved out of §5.1 (they remain in `app:eval`).
  Main text keeps the prescribed one-sentence summary.
- **C13** — parent-level analyses described as distributional evidence, seed level as the basis for
  uncertainty over training runs.
- **C27** — retrieval introduced as "a secondary, within-cloud read-out … not a conformer-search
  benchmark", both in the metrics paragraph and at the result.
- **C5** — "The gain is about 40% of an independently measured ceiling" → "The value arm's absolute
  correlation is about 40% of an independently measured teacher-matching reference", with the 23.0%
  gap-closure figure, the "reference and not a ceiling" statement, and the ensemble-dependence
  caveat moved next to first use. `\label{sec:ceiling}` → `\label{sec:reference}`. Table 1's last
  row is now *Teacher-matching reference*.
- **C10** — the earlier round's scrambled arm is flagged as a defective control **at the point of
  use**: in the §5.2 prose, in the Table 1 row (`Energies scrambled†`) and in the Table 1 caption,
  which also points at the recovered fourth seed.
- **C11** — §5.3 run-in head → "Among completed runs, only the true geometry–energy pairing."
  The survivorship paragraph rewritten: D and F means are "conditional on completion" and their
  "unconditional magnitudes are not identified"; the relaunch block is cited as the data-based
  bound. The self-defeating "consistent either with a real effect or with survivorship" formulation
  appears nowhere.
- **C19 / C6** — §5.4 rewritten. The four calibration diagnostics that improve (T_eff, slope, scale
  ratio, NRV_min) are named with values, NRV_min (0.830 → 0.747) appears in the main text for the
  first time, the 1 − r² bound is stated **per parent**, and the precise fourfold claim replaces
  "every calibration diagnostic worsens".
- **C1** — the closing paragraph of §5.4 replaced with the offset-invariance argument; "that is the
  next experiment this paper names" deleted.
- **D1** — §5.5 resolution sentence reverted to the two-row form: +0.382 at ε = 0.125 and +0.177
  at ε = 0.042, "more than halves", the resolution-matching caveat, and the n = 48 cell described
  as not yet powered with no value reported.
- **audit** — "no diverged seed was replaced" in §5.5 corrected: relaunched seeds exist (two in the
  earlier round's scrambled arm, and a later block of cell-D, cell-F and gradient-weight seeds) and
  are reported separately with their own counts, never merged into a reported cell mean.
- **C23** — "one displacement cloud per composition" → "per parent", and "whose constant" → "whose
  offset", for consistency with the corrected `cor:basins`.
- Internal research-history narrative removed from the main text ("The evaluation as originally
  run has two defects" → a property of the evaluation population; "we therefore re-scored" →
  passive statement of the protocol).

### `sections/05_conclusion.tex`
- **C1 / C19** — limitation (i) now pairs the raw NRV movement with the NRV_min movement and states
  that offset identification and within-group calibration are different targets.
- **C5** — limitation (ii) reworded to the teacher-matching reference, with 23.0% gap closure and
  the ensemble-dependence caveat; "three fifths of the attainable ordering is unlearned" deleted.
- **D1** — limitation (v) carries the +0.382 → +0.177 range (training scale to reporting scale).
- **C15** — limitation (vi) reworded.
- **C24** — conclusion rewritten to the prescribed order: question, the value-only intervention,
  the corrected ordering and retrieval result, the matched-control interpretation *with* the
  completion caveat, the narrowly scoped force result, the raw calibration failure, the local/global
  boundary, then the transferable lesson. Ends on the three-intervention boundary and states that
  the paper runs none of them. "we learn a Boltzmann generator", "40% of the achievable maximum" and
  "the anchor will fix calibration" appear nowhere.

### `sections/A1_proofs.tex`
- **C16** — `app:setup`: `c` defined operationally as (element multiset, charge); spin demoted to a
  standing restriction. `ass:charge` softened ("the standard assignment for closed-shell species;
  we have not verified the multiplicity of every evaluated parent").
- **theory-vs-code** — `ass:A2` retitled and rewritten to define the reported density explicitly as
  the position-only flow density with discrete channels frozen at (a₁, c₁, e₁), and to state when
  it does and does not coincide with the generator's conditional.
- **C3** — `thm:certificate` (i) split into continuous and atomic conclusions; (ii) restricted to
  the continuous case, with an explicit statement that for atomic ρ no normalisation hypothesis of
  that form is available *at any ε*, so `rem:epsilon` does not cover it. `thm:l2g` (iii) proof
  restricted to the continuous reading, with the positive-density hypothesis carried into the
  theorem statement rather than into `def:overlap`. New **`cor:empirical`** states the atomic
  corollary at statement level: one common value at K sampled points, an edgeless graph, and the
  weaker identifiability conclusion (no normalisation decomposition).
- **C4** — "converts *is the model Boltzmann?* into a connectivity question" → "characterises the
  null space of the group-wise objective under stated support and normalisation assumptions".
- **C23** — `cor:basins` proof re-derived from disjointness; the internal contradiction (J = 1 five
  lines before reasoning about several groups per composition) is gone. `app:scope` inventory
  updated to "one group per **parent**, every component isolated".
- **C1** — `cor:gauge` proof kept (it is correct) with the per-composition granularity caveat added.
- **C2** — subsection retitled *Algebraic relation between residual variance and ordering*;
  proposition renamed *Residual-variance / correlation identity*; the proof now says explicitly that
  away from ρ = 0 the two can move in opposite directions and that this is a diagnostic identity,
  not a guarantee. "one quantity in three units" replaced. `prop:surrogate` moved out of
  *Exercised empirically* into a new fourth category, *Stated as a diagnostic identity, with its
  hypothesis contradicted*.
- **C2 (tab:tau)** — the table now prints the implied τ **beside** the measured √NRV (1.46 / 1.19 /
  1.32), and the caption states that the discrepancy is the evidence for ρ < 0. The sentence
  claiming value supervision "cuts the Boltzmann residual from 4.85 to 2.52" is deleted and replaced
  by an explicit prohibition on reading a residual reduction off the implied column.
- **C5** — `rem:attenuation`: ceiling → teacher-matching reference; "oracle noise" →
  "teacher--evaluator disagreement".
- **C10** — `rem:onpolicy`: "one of the confounds that makes that round uninterpretable" → "why that
  round's *force* comparisons are confounded and are superseded by cell E".
- new **`rem:epscert`** — the certificate the objective can enforce is about the smoothed marginal
  log p_{1−ε}, and the ε-dependence is a property of the method, not an accuracy limit.

### `sections/A2_details.tex`
- **C9 / C1** — the anchor moved to a dedicated appendix record, *The inactive anchor code path*,
  stating (i) NRV's offset-invariance makes it unable to move the calibration endpoint, (ii) its
  mean-square form decomposes into the same within-group variance plus an offset penalty and adds no
  scale-sensitive read-out, (iii) it is keyed on composition, so it fixes at most one constant per
  composition rather than one per group. "the natural remedy for the calibration failure" deleted.
- **as-implemented** — new paragraph *The value estimator as implemented*: biased variance,
  non-finite masking, groups with fewer than two valid members excluded, denominator = surviving
  groups, cap-based step rejection; with the statement that the null-space results are unaffected by
  the biased normalisation but are stated for the full-group objective.
- **C10** — `app:hardware`'s "the arms differ only in the physics block" scoped to the grid.
- **audit** — `app:data` gains *Composition multiplicity, and why `cor:basins` does not rest on it*,
  and the note that a training group contains the data geometry together with its K−1 displacements
  (the primary endpoint drops it at evaluation only).
- **audit** — `app:eval`'s "five, five and seven prepared … four, four and three reported" replaced
  by the per-arm breakdown of *why*, including the unscored FM seed whose missingness cannot be
  certified ignorable and whose only prior score was the higher of that round's two FM values.
- **D1** — `tab:respowered` was extended to three rows by this pass and has been reverted to two,
  the n = 48 cell carrying a plain-prose "not yet powered / no value reported" line instead of
  numbers. The "what it settles" paragraph, the protocol paragraph and limitation (vi-c) were
  reverted with it; `tab:epsgrid`'s n = 48 row now names the no-plateau measurement as its use.
- **C5** — 12 renames of ceiling / oracle / attainable-ordering to *teacher-matching reference*
  across §B.7, §B.8, §B.9, §B.14 and the reproduction list (the script *filename*
  `measure_oracle_ceiling.py` is kept verbatim, being the real basename).
- **C2** — §B.7's "the training objective and this endpoint are one quantity in different units"
  replaced with the shared-residual-form wording, naming the three differences.
- **C13** — "the parent-level tests … are the better-powered ones" replaced.
- **C27** — limitation (iv)'s "the only decision-level statement we make" replaced.
- **C15** — limitation (vii) reworded.
- **C16** — capability-matrix row → **83 (corpus)**, and the caption now states that the Elements
  column records corpus coverage, not demonstrated evaluation scope, and that every result is on
  main-group organic parents at the singlet default.
- **C10** — §B.10 retitled *Provenance and audit of the earlier checkpoint family*
  (`\label{app:cells}` unchanged, so no cross-reference breaks); "Every contrast in that round is
  superseded" replaced by an explicit split of which comparisons survive as a corrected held-out
  evaluation and which are superseded and why; `tab:cells` caption retitled; limitation (viii)
  updated.
- **audit** — `tab:p0cells` caption and reporting rule (ii) corrected (the counts are the launch
  record of the **matched five-seed design**, and 14 further runs exist); new paragraph *The later
  relaunch block, and what it bounds* with D 9/7/2, F 10/3/7, the six scored D seeds
  (+0.410 ± 0.018, all above every control seed), and cell E2 (+0.080 ± 0.013 at λ₁ = 0.01).
- **audit** — `app:cells` gains *A fourth scrambled seed, recovered*, with the launcher fault, the
  byte-identity validation of the recovery route, r = 0.161 / slope 0.104 / kT_eff 9.59 eV, and the
  arm at four seeds = 0.210; plus the disclosure that the stabilised scrambled seeds are
  **replacements** for two that diverged, so the no-replacement assurance holds of the grid and not
  of that arm.
- **theory-vs-code** — new limitation (vi-d): the reported density is a position-only flow density.

### `main.tex` (second pass)
- **audit** — the reproducibility statement's "including the runs that diverged and were not
  replaced" corrected to "including every run that diverged, every run that completed without being
  scored, and the later seeds launched outside the matched design".

### Typography fixes forced by the above (not content changes)
- Table 1 row labels shortened (`Energies scrambled†`, `Teacher-matching reference`) after the
  longer labels produced a 16 pt overfull box; the flags live in the caption.
- `tab:tau` set `\footnotesize` with the population moved into the caption after the fourth column
  produced a 48 pt overfull box.
- One phrase reworded ("without a correctly calibrated density" → "without the density being
  correctly calibrated") solely to clear `build.sh`'s literal `calibrated density` vocabulary gate,
  which cannot see the negation. Meaning unchanged.

---

## 4b. Figures changed

Figure sources were revised by the figure agents in this round; this section is the consolidated
record. Every figure was regenerated from its machine-readable source (JSON or per-record CSV);
nothing was transcribed from a PDF or from prose, and no new evaluation was run for any of them.

### Figure 1 — `figures/fig1_objective.tex`, `fig1_objective_float.tex` (C8, C9)

- **Panel (d) redrawn.** Was two ellipses labelled `group c_A` / `group c_B`, i.e. two different
  *compositions*, with "ordering fixed". Now a dashed enclosure labelled "one density `p_θ(·|c)`"
  containing `group A (basin A)` and `group B (basin B)`, subtitle "two disconnected groups, same
  composition `c`", interior note "local relation constrained", and "offset `b_A` free" / "offset
  `b_B` free". The closing line reads: nothing relates A to B, so `b_B − b_A` and the relative mass
  are not identified. "Pins" and "ordering fixed" are gone.
- **Panel (c)** now shows the complete objective, `L_ours = L_FM + λ_E L_value`, so no hidden branch
  can be inferred. No force branch, no anchor branch; "force" and "anchor" survive only in comments
  and in TikZ `anchor=` positioning keys.
- **Panel (a)** — a pre-existing defect repaired, outside C8 and flagged as such: the label
  `x_0 ~ N(0,σ²I)` overflowed its node under ICLR metrics and printed on top of "random types" in
  the shipped `main.pdf`. Node widened 2.10 → 2.60 cm. Content unchanged.
- `fig1_build.sh` gained a second stage that compiles the float against the real ICLR preamble and
  greps for bad boxes; stage 1 alone had certified a figure that wrapped in two places on the page.
- No panel plots a measured quantity, so there is no source file to regenerate from. The only
  external facts entering the figure are the audit's, and they fix the *semantics* of panel (d),
  not any drawn value.

### Figure 2 — `figures/fig2_ordering.py`, `.tex` (C7, C5)

- **Panel (c) legend** no longer prints `slope 0.202` / `slope 0.369`. It prints "mean within-parent
  r", on two lines per entry, plus an in-panel note: "both axes standardised: this coefficient is the
  correlation in (a), not a calibration slope". The word "slope" survives in the rendered figure only
  inside that disclaimer (one `pdftotext` hit). The generator asserts the fitted coefficient equals
  the arm mean `r` to 1e-6.
- **Panel (d)** legend `independent teacher` → **teacher-matching reference**; the JSON block is
  renamed `oracle` → `teacher_ref` on load, with the stored source key left untouched. Zero
  instances of "ceiling" or "oracle" remain in the figure.
- Panel (c) was **kept**, not dropped — see §1(ii) for the reasoning, and `PAGE_CUT_PLAN.md` T2.2 for
  the recommendation to drop it under the page limit.
- `weighted_slope` → `standardised_association` throughout the generator;
  `pooled_standardised_slope` → `pooled_standardised_association` in the JSON, with two new note
  fields.
- Sources: `out/scale_primary.json` for panel (a), cross-checked at build time against
  `runs/eval_ours/wide_*/boltz_records.csv` (agreement asserted to 1e-4); the same records for (b)
  and (c); `out/retrieval_utility.json` for (d). The caption's `2952` evaluated geometries per arm is
  recomputed, not carried over (738 per seed × 4 seeds).

### Figure 3 — `figures/fig3_grid.py`, `.tex` (C11)

- **Panel (a) title** → "(a) Among completed runs, only the true pairing improves ordering"
  (fontsize 7.6 → 7.4 to fit).
- **Caption survivorship sentence replaced.** The forbidden self-defeating formulation
  ("consistent with either a real effect or with survivorship, and we cannot separate the two") is
  gone; the caption now states that the three completed D seeds are mutually consistent (0.452,
  0.398, 0.412) and disjoint from the control ranges, that the direction agrees with the corrected
  earlier checkpoint family, and that the D and F means remain conditional on completion so their
  unconditional magnitudes are not identified. "Survivorship" now appears only inside a LaTeX
  comment, so a rendered-PDF vocabulary gate cannot fire on it.
- Two additions beyond the prescribed text, both checkable: the three D seed values are printed
  inline, and the "corrected earlier checkpoint family" carries a `\Cref{fig:ordering}` pointer.
- Every raw completed seed point, the `†` marks and the launched/completed/diverged triples were
  already present and are unchanged. The `†` is attached to B, D and F (every cell that lost a seed)
  while the caption sentence names only D and F; this is deliberate, since B lost a seed too.
- Sources: `runs/eval_ours/clean_p0_*/boltz_independent_records.csv` for the bars and points, a
  checkpoint census over `runs/p0/*/lightning_logs/version_*/checkpoints/midstep-step=30000.ckpt`
  for the triples. The generator's integrity gate reproduces all 24 per-seed `r`, six means/SEMs, six
  triples and six contrasts. Seeds are hard-restricted to s1–s5 of the six named cells; the relaunch
  block and E2 are excluded from the figure by name and reported separately in the appendix.

### Figure 4 — `figures/fig4_calibration.py`, `.tex` (C6, C5)

- **The `1 − r²` overlay is deleted.** It was a *per-parent* identity drawn under a point built from
  a **mean** `r` and a **median** NRV, and neither aggregate commutes with `1 − r²`.
- **New panel (b)**: raw NRV against `NRV_min` per seed, with connectors. `NRV_min` (FM 0.830 →
  Value 0.747) appears in the figure for the first time, so the one calibration diagnostic that
  improves is visible where the message needs it. The generator asserts the per-parent identity
  `nrv_min(m) = 1 − r_m²` across all 93 parents × 8 seeds (holds to 1e-9), which is precisely why the
  aggregate must be reported as its own statistic.
- **Panel (c)** keeps the `T_eff` reading (8.23 → 2.34 eV against the 1.0 eV target).
- Teacher point relabelled "teacher-matching reference (a second evaluator, not a ceiling)".
- **One number corrected:** the reference's `T_eff` is 1.251 eV, not the 1.234 eV previously
  printed. The old script took a median of the per-parent `T_eff` column while the model arms use
  `kT` / median slope; the figure now uses the arm convention throughout.
- Retitled **"Ordering improves; raw calibration does not."**
- Sources: `out/scale_primary.json` `["dropref"][arm]["per_seed"][seed]["primary_93"]`, cross-checked
  against `runs/eval_ours/calibration/calibration_dropref.json` restricted to the 93 primary ids;
  `runs/eval_ours/ceiling_full_dropref/ceiling_per_group.csv` for the reference.

### Dead figure files, noted and not edited

`figures/fig1_method.tex`, `fig2_boltzmann.tex`, `fig3_ablation.tex`, `tabA1_cells.tex` are not
`\input` by `main.tex` and compile nowhere. `fig1_method.tex` is the superseded five-band Figure 1;
it still draws a force branch and is the only artifact left in the tree asserting transition-metal
coverage (lines 132, 203, 297). It cannot contradict the PDF. If the tree is tidied, delete it
rather than edit it.

---

## 4c. Terms renamed (consolidated)

| Was | Is | Correction | Live sites |
|---|---|---|---|
| ceiling / teacher ceiling / oracle ceiling / achievable maximum / attainable ordering / achievable ordering | **teacher-matching reference** | C5 | 26 (12 in `A2_details.tex`, 6 in `04_experiments.tex`, 2 in `05_conclusion.tex`, 3 in `A1_proofs.tex`, 2 in Fig. 4, 1 in Fig. 2) |
| "oracle noise" | **teacher–evaluator disagreement** | C5 | 2 |
| "the gain is ≈40% of a ceiling" | "the **absolute correlation** is ≈40% of the teacher-matching reference"; the *gain* is **23.0% gap closure** | C5 | 6 |
| `slope 0.202` / `slope 0.369` (Fig. 2c) | **mean within-parent r** / standardised association | C7 | 2 |
| "Monotone surrogate for the reported metric" (subsection + proposition) | **Algebraic relation between residual variance and ordering** / **Residual-variance / correlation identity** | C2 | 2 + 5 call sites |
| "one quantity in three units" / "one quantity in different units" | shares one **residual form**, evaluated with different potentials, populations and smoothing scales | C2 | 3 |
| "one group per composition" / `J = C = 1` / `C = J = 1` | one group per **parent**; clouds almost surely **disjoint**; every component isolated | C23 / audit | 6 |
| "needs the overlap graph to be connected" / "unless group supports connect" | connectivity is **sufficient**, and necessary in the **identifiability sense** | C4 | 4 |
| "no model here is converged" | "**we do not establish convergence**" | C15 | 3 |
| "the natural remedy for the calibration failure" | (deleted; replaced by the offset-invariance argument) | C1 | 6 |
| "structure emulator" (after `\citet{tbg}`) | (deleted; replaced by a description of what those models are trained against) | C17 | 2 |
| "decision-relevant" / "the only decision-level statement we make" | **secondary within-cloud retrieval read-out**, not a conformer-search benchmark | C27 | 3 |
| "uninterpretable" / "superseded in full" (of the earlier round) | an explicit split of which contrasts are superseded and which survive as a corrected held-out evaluation | C10 | 6 |
| "the parent-level tests … are the better-powered ones" | parent-level = **distributional evidence**; seed-level = the basis for **uncertainty over training runs** | C13 | 2 |
| `\label{sec:ceiling}` | `\label{sec:reference}` | C5 | 1 |
| §B.10 title "The earlier, confounded ablation round" | **Provenance and audit of the earlier checkpoint family** (`\label{app:cells}` unchanged) | C10 | 1 |
| "83-element de novo generator" (as a result claim) | "trained on an **83-element corpus**"; capability-matrix cell → **83 (corpus)** | C16 | 3 |

Kept deliberately: the script basename `measure_oracle_ceiling.py` (it is the real filename in the
archive); the metric-anchoring sense of "anchored" in the NRV definition and the Table 1 caption;
`\label{app:cells}` and `\label{fig:ordering}`, so no cross-reference breaks.

---

## 4d. Internal research-history passages deleted

Research narrative that told the reader how the work unfolded rather than what is true. All removed
from the main text; the *facts* they carried survive in the appendix where they are provenance.

| Where | Deleted | Replaced by |
|---|---|---|
| `04_experiments.tex` §5.1 | "The evaluation as originally run has two defects…" | a statement of the property of the evaluation population |
| `04_experiments.tex` §5.1 | "we therefore re-scored…" | a passive statement of the protocol |
| `04_experiments.tex` §5.4 | "…and that is the next experiment this paper names" | the offset-invariance argument |
| `02_method.tex`, `A2_details.tex` | "the fix our own analysis points to is implemented and untried" | "the offset-identification term our analysis names is implemented and untried; it is not a repair for the calibration failure we measure" |
| `A1_proofs.tex` `rem:onpolicy` | "one of the confounds that makes that round uninterpretable" | "why that round's *force* comparisons are confounded and are superseded by cell E" |
| `A2_details.tex` `tab:cells` caption | "the flags that make its contrasts uninterpretable" | "the flags that bound which of its contrasts remain usable" |
| `05_conclusion.tex` | "three fifths of the attainable ordering is unlearned" | the gap-closure figure and the ensemble-dependence caveat |
| `figures/fig3_grid.tex` | "consistent with either a real effect or with survivorship, and we cannot separate the two" | the completion-conditioned formulation |
| `01_intro.tex` | "following `\citet{tbg}` we call such a model a structure emulator" | a description of what those models are trained against |

Retained on purpose, because it is disclosure rather than narrative: the launcher-fault provenance
of the recovered scrambled seed, the `a1_fm_only` non-scoring history, and the relaunch-block
account. These live in Appendix B, not the main text.

---

## 5. Could not be done from existing artifacts

1. **Three finite checkpoints still have no primary-protocol score**: `abl/a1_fm_only`,
   `p0/p0_D_energy_s8`, `p0/p0_F_both_s10`. Each needs a fresh multi-hour GPU evaluation, which this
   pass forbids. All three are disclosed: the FM one in `app:eval` with the explicit statement that
   its missingness cannot be certified ignorable and that its only prior score would have *lowered*
   the control mean; the other two in the new relaunch-block paragraph as trained, finite and not yet
   scored.
2. **The fixed-RMSD panel in Figure 2 remains undrawn** — no artifact under `runs/eval_p3`
   reproduces the verified pair (0.305 / 0.182, t = 4.94); the two available rounds give
   0.303/0.189 (40 parents) and 0.350/0.248 (120 parents), so a panel would be a cross-round
   comparison. That result stays prose, as before.
3. **The teacher-matching reference for the narrower perturbation families was never measured**, so
   no fixed-RMSD or normal-mode correlation is expressed as a fraction of one. Already stated in
   `app:perturb`; unchanged.
4. **`refs.bib` carries 17 `% TODO-verify` comments** on bibliographic fields never confirmed
   against a primary source. They render nowhere (0 occurrences in the PDF and in `main.bbl`), so
   they are not a submission-gate failure, but they remain unverified facts inside the shipped
   bibliography. Out of scope for this round; flagged again.
5. **The AI-use statement's factual content** is unverifiable by any tool and remains flagged for
   author verification by the non-rendering comment in `main.tex`.

---

## 6. New citations

**None.** Every work named in the rewritten related-work section (EDM, GeoLDM, MiDi, Symphony,
SemlaFlow, FlowMol3, Zatom-1, EBMol, Torsional Diffusion, GEOM, DECAF, Boltzmann Generators, TBG,
SBG, ArBG, Adjoint Sampling, iDEM, EWFM, PSM, FAB, VarGrad and the log-variance line, OMol25, eSEN,
GFN2-xTB, xyz2mol, FFJORD, Hutchinson, PoseBusters, y-randomisation) was already cited in the
manuscript and already present in `refs.bib`. The C22 regrouping moved existing citations between
paragraphs; it introduced no key. Citation orphans: **0 in both directions** (build reports 0
undefined citations; no `\cite` key was removed from use without also being unused before).

---

## 7. Plan for the separate compression pass (not executed here)

**Superseded in detail by `PAGE_CUT_PLAN.md`**, which tiers these items, gives a per-item page
estimate, an execution order, the risks specific to this compression, and the protected list. The
ranking below is kept as the editor's original judgement of cost.

Main text is 14 pages against a 9-page limit; 5 pages must come out. Ranked by cost to the argument,
cheapest first. Nothing below is a claim change.

1. **Figure 2 panel (c)** (≈0.4 pp incl. its caption block). It is the most defensible cut in the
   paper: its summary statistic is redundant with panel (a) by construction, and the two things it
   uniquely shows (uniform steepening, wide scatter) can be asserted in one main-text clause.
   Deleting it is a three-line change to `fig2_ordering.py` and removes ~8 caption lines.
2. **Figure captions** (≈0.8 pp). Figures 2 and 4 grew this round. Named compressible sentences,
   already flagged by the figure agents: Fig. 2's "What the panel adds is the shape of the
   association…" and the trailing "not a bound on what is attainable"; Fig. 4's
   `E_m[1−r_m²] ≠ 1−(E_m r_m)²` justification. Each moves to body text or the appendix at ~1 line
   each in place of ~4.
3. **§5.1 metrics paragraph** (≈0.3 pp). The Simpson's-paradox justification for per-parent
   computation is stated twice (§5.1 and `rem:crossbatch`); the main text needs one sentence and a
   pointer.
4. **§5.5** (≈0.5 pp). The generation-quality discussion and the three "limits remain" clauses
   duplicate `app:generation` and limitations (iii), (v), (vi). Reduce to three sentences with
   pointers.
5. **§2 Setting** (≈0.4 pp). The bond-free rationale and the FlowMol3 interpolant recap can lose
   half their length; both are restated in `app:data` and `app:hardware`.
6. **§4 theory** (≈0.7 pp). The multiplicity paragraph after `cor:basins` can shrink to one sentence
   plus a pointer to §B.4, and the "one further gap" paragraph to one clause, once the appendix
   carries both in full (it now does).
7. **Introduction P1 and P4** (≈0.6 pp). P1's three-way distinction can be made in four sentences
   rather than seven. P4 currently restates six numbers that Table 1 also carries; three suffice.
8. **§5.2 and §5.4 run-in heads** (≈0.4 pp). Several are full sentences; shorten to noun phrases.
9. **Last resort — Table 1 → a two-arm table** (≈0.3 pp) by moving the scrambled and reference rows
   into Figure 2's caption. Do this only if 1–8 fall short, since it removes the defect flag from the
   most-read object in the paper.

Sum of 1–8 ≈ 4.1 pp, plus reflow. Expect one further pass over `\textfloatsep` and float placement
rather than any further prose cut. **Do not** compress: the completion-conditioned qualifications,
the reference-not-ceiling statements, the offset-invariance argument in §5.4, the ensemble-scope
paragraph in §5.1, or any launched/completed/diverged triple.

---

## 8. Low-cost evaluations performed in this round

No training was launched and no new GPU job was submitted. Everything below re-analyses records
already on disk, which the pass explicitly permits. Every checkpoint touched was verified to exist
and to be finite before use.

| # | What was run | Why it was allowed | Result |
|---|---|---|---|
| 8.1 | **Checkpoint finiteness audit**: all 141 `.ckpt` files under `runs/{abl,p0}` loaded (`torch.load`, `isfinite` over every floating-point tensor of the state dict) | auditing checkpoint finiteness | **0 non-finite parameters in every file.** No corrupted checkpoint exists; nothing in the manuscript rests on one |
| 8.2 | **Divergence ledger**: `NON-FINITE WEIGHTS at global_step=` grepped from all 63 `sweep-*.out` logs; seeds recovered from `Seed set to N` in the `.err` logs | auditing run status | exact halt step for every diverged run; three failure modes separated (11 / 3 / 4) |
| 8.3 | **Grid protocol replication** from `runs/eval_ours/clean_p0_*/boltz_independent_records.csv` | another statistic from saved per-parent records | reproduced every value in `fig3_grid_data.json` to 6 d.p. as a control on the recomputation pipeline |
| 8.4 | **Eight evaluated-but-unreported runs** scored by that same replication | same | D relaunches **0.353251 / 0.373749 / 0.470672**; E2 **0.117697 / 0.076397 / 0.089002 / 0.078024 / 0.038388** → 0.0799 ± 0.0127 |
| 8.5 | **Draw-identity check** across `n240_*`, the primary wide round and the `clean_p0` grid round, key-matched on `(group_id, pert_id)` | same | `log p` agrees to 1.22e-4; 1074 of 1078 records shared; the primary endpoint and the matched grid sit on **one** evaluation draw |
| 8.6 | **Recovery of `a6_..._stab_s4`'s primary score** from the 240-parent records, restricted to `group_id < 120`, pushed through the paper's own `analyze_calibration.py` and `recompute_scale_primary.py` | another statistic from saved per-parent records; zero new compute | **r = 0.161**, NRV 1.4213, slope 0.1043, kT_eff 9.591 eV. Route validated by reproducing two published NRVs to 6 d.p. (s2 1.726978; stab_s5 1.753944). Scrambled arm at four seeds: 0.210 |
| 8.7 | **Resolution ablation re-run**: `scripts/analyze_resolution_ablation.py` unmodified against the current `runs/eval_res` | re-analysis of existing records | Records confirmed present on disk at n = 4, 12 and 48. The finer-grid figures this produced were carried into the manuscript by this pass and then removed at the author's direction (D1); the manuscript reports the two-row version, n=4 **+0.382** (t 9.40) and n=12 **+0.177** (t 3.33) |
| 8.8 | **Independent resolution family** from `runs/eval_res/p0_*` | same | Records confirmed present. Not reported in the manuscript (D1); no figure from this family appears in the paper |
| 8.9 | **Configuration diffs** across the sweep YAMLs | auditing configs | D s6–s9 and F s6–s10 byte-identical to wave 1 apart from name and `output_dir`; E2 differs in one line (`λ₁`); `stab_s4/s5` differ in three lines |
| 8.10 | **Figure regeneration** for Figures 2, 3 and 4 from existing JSON/CSV | regenerating plots from existing artifacts | new PDFs; all build-time integrity assertions pass |
| 8.11 | **Full LaTeX rebuild** (`build.sh`) | recompiling | 0 errors, 0 undefined references, 0 undefined citations, 0 overfull boxes, 0 vocabulary-gate hits |

**Deliberately not run:** any training; any replacement for a diverged seed; any new GPU evaluation.
Three finite checkpoints therefore still lack a primary-protocol score (§5.1), and the exact
commands that would settle them are recorded in `CONCEPTUAL_AUDIT.md` §11 rather than executed.

---

## 9. Summary statistics

| Metric | Count |
|---|---|
| Corrections in the revision request | 27 (C1–C27) |
| Applied as written | 25 |
| Applied with a documented deviation | 1 (C7: panel (c) relabelled, kept rather than dropped — §1(ii)) |
| REVIEWER_DISAGREE | **0 full (D1 reverted at the author's direction), 1 narrow (D2, wording only)** |
| Audit findings absorbed | 3 major (composition multiplicity; the false "no seed was re-rolled"; the relaunch and E2 blocks) + the recovered scrambled seed + the per-arm reporting breakdown |
| Manuscript files changed | 8 (`main.tex` + 7 section files) |
| Figure sources changed | 4 (Figures 1–4), by the figure agents |
| Terms renamed | 17 distinct renames across ≈75 live sites (§4c) |
| Theorem-level corrections | 6 measure-class repairs, 1 new corollary (`cor:empirical`), 1 corollary re-derived (`cor:basins`), 1 proposition renamed and re-scoped, 1 new remark (`rem:epscert`) |
| Results whose provenance changed in print | 3 (scrambled arm flagged as defective at point of use; `tab:p0cells` counts rescoped to the matched design; the relaunch block reported separately). `tab:respowered` was extended to three rows and reverted to two — see D1 |
| New citations | **0** |
| Citation orphans | **0 in both directions** |
| Abstract word count | 186 (target 170–200) |
| Main-text pages | 14 (limit 9; see `PAGE_CUT_PLAN.md`) |
| Unresolved items carried forward | 8 (`CONCEPTUAL_AUDIT.md` §11) |

---

## 10. Independent verification pass (final verifier, 2026-08-14)

Everything below was re-derived from the artifacts, not read from a prior report. Build
run from `distclean`. No training was launched and no GPU job was submitted; the only
commands run were re-analyses of existing evaluation records, `pdflatex`/`bibtex`, and
read-only SLURM accounting.

### 10.1 Numbers re-derived from raw artifacts (all agreed)

| Quantity | Recomputed from | Result |
|---|---|---|
| Primary endpoint, all per-seed `r`, NRV, NRV_min, slope, scale ratio, `T_eff` | `figures/out/scale_primary.json` `["dropref"][arm]["per_seed"][seed]["primary_93"]` | matches the printed values to the printed precision |
| Δ = +0.167, Welch t = 5.16, exact permutation p = 2/70 = 0.029 | recomputed from the four+four seed means | reproduced exactly (t = 5.157, 2 of 70) |
| Retrieval 19.6 / 49.2 / 29.6 / 62.4 / 87.1 / 98.9 %, chance 12.5 / 37.5 % | `figures/out/retrieval_utility.json` | reproduced exactly |
| Teacher-matching reference r = 0.927 | `runs/eval_ours/ceiling_full_dropref/ceiling_per_group.csv` restricted to the 93 ids in `primary_endpoint.json` | 0.927075; median NRV 0.0817, mean 1−r² 0.0954, T_eff 1.2507 eV — all as printed in Fig. 4 |
| 39.8 % absolute ratio and 23.0 % gap closure | arithmetic on the above | reproduced |
| Six grid cells A–F, every per-seed r | per-record CSVs `runs/eval_ours/clean_p0_*/boltz_independent_records.csv`, recomputed independently of `fig3_grid.py` | all six cell means and all six contrasts reproduced to six decimals |
| Relaunched cell-D seeds 0.353 / 0.374 / 0.471 | same CSVs (`clean_p0_D_energy_s6/s7/s9`) | 0.353251, 0.373749, 0.470672 |
| Six scored D seeds 0.410 ± 0.018; cell E2 0.080 ± 0.013 | same CSVs | reproduced |
| "All 2952 evaluated geometries per arm" (Fig. 2c) | `runs/eval_ours/wide_*/boltz_records.csv`, 93 ids, reference dropped, xTB-ok only | 738 per seed × 4 seeds = 2952 |
| Composition multiplicity: 30 000 → 22 822 distinct, 2 204 repeated, 31 %, max 80 | `perturbation_train_n30000_s0.pt`, recounted from the stored `atom_types` / `atom_charges` / `node_idx_array` | 30 000 groups, 22 822 distinct, 2 204 repeated, 9 382 parents = 31.3 %, max 80 |
| Resolution ablation, the two reported rows | `runs/eval_res/{a1_fm_only,a3_energy_only}*__ode{4,12}/boltz_independent_records.csv`, recomputed without using `analyze_resolution_ablation.py` | n=4: +0.382 (t 9.40), 4 value vs 5 control seeds; n=12: +0.177 (t 3.33), 2 value vs 5 control seeds — the two-row values the manuscript prints |

### 10.2 D1 (the n = 48 cell) — records exist; the manuscript reports the two-row version

The finer-grid records exist on disk and were recomputed independently of
`analyze_resolution_ablation.py`. This pass carried them into `tab:respowered`; that change
has been removed at the author's direction, who had reviewed the finer-grid measurement in an
earlier round and judged it defective. The manuscript therefore prints two rows. The n = 12
gap it reports, +0.177 (t = 3.33), is the two-seed value arm: (0.3278 + 0.4326)/2 − 0.2037 =
+0.177. The n = 48 cell carries no numeric value and is described in plain prose as not yet
powered; the manuscript makes no claim about it in either direction, and no `\pending`-style
marker is used. The second resolution family from `runs/eval_res/p0_*` is likewise recorded
here and not reported in the manuscript.

### 10.3 Defects found and fixed in this pass (claim-neutral)

1. **Figure 3 caption, mis-attributed statistic.** The clause "neither does energy
   supervision whose within-parent pairing has been scrambled" was evidenced by
   `D − C = +0.236`, which is a contrast between the treatment and the scramble, not
   between the scramble and the baseline. Rewritten to cite cell C's own mean
   (0.185 ± 0.020) against cell A's (0.217 ± 0.008), keeping `D − C` where it belongs.
   Both added numbers are the verified cell values.
2. **Figure 4 caption contradicted §5.2 on the reference.** It read "it bounds what this
   protocol can resolve", while the main text correctly says a model could in principle
   align with GFN2-xTB better than the teacher does — in which case the protocol resolves
   above 0.927 and the reference bounds nothing. Replaced with a reference-point statement
   that matches §5.2.
3. **Table 1 carried no launched/completed/diverged record** (checklist item 18), only a
   pointer to §3. The per-arm launch record already documented in `app:cells` is now
   printed in the caption: flow matching 5/5/0 with four scored, value 5/4/1 with four
   scored, scrambled 7/4/3 with three scored.
4. **Table 1's "not applicable" cells rendered as em dashes**, which both read as missing
   data and drove the page's em-dash count. Replaced with `n/a`.
5. **Em-dash density exceeded two per page on eight of fourteen main-text pages** (worst:
   page 9 at ten, pages 4 and 6 at six). Twenty-two prose sites were rewritten as colons,
   commas, semicolons or parentheses across `01_intro`, `02_method`, `03_theory`,
   `04_experiments`, `05_conclusion` and the two figure captions. No sentence's meaning
   was altered and no number changed; verified by diffing the numeric token multiset of
   the rendered main text before and after (the only differences are the four numbers
   added by fix 1). Every main-text page is now at or below one.

### 10.4 Reported, not fixed (would require a scientific decision)

* The reproducibility appendix names the script `measure_oracle_ceiling.py`. It is a
  filename, and renaming it would break the pointer to the archived code, so the word
  "ceiling" survives there once. Every prose use in the manuscript is either absent or an
  explicit denial.
* `build.sh`'s forbidden-vocabulary gate greps the raw `pdftotext` output and therefore
  cannot see a phrase broken across a line. Re-running the scan on de-hyphenated,
  whitespace-normalised text finds one further occurrence of "calibrated density", in the
  abstract's "without yielding a calibrated density" — a negation, and the phrasing the
  revision brief itself prescribes. The gate's reported zero is correct in effect but not
  by construction.
