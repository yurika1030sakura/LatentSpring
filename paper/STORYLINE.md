# STORYLINE.md — binding spine for the rewrite

Status: BINDING. The manuscript writer and the figure builders follow this document.
Where this document and any older file in `paper/` (SPINE.md, SPINE_v2.md, RESEARCH_story.md,
DRAFT_STATUS.md, the comment block at the top of `main.tex`) disagree, **this document wins**.
Every number below is drawn from the VERIFIED NUMBERS list; nothing here may be re-derived from
the old PDF or from prose in the current `sections/*.tex`.

---

## 1. One-sentence pitch

Supervising a bond-free de novo 3D molecular flow with *energy values* from a universal neural
potential teaches it which nearby geometries of a molecule should be more probable — but not how
much more probable — and a matched six-cell control grid shows that this gain requires the true
geometry–energy correspondence, not a generic density regulariser, not scrambled energies, and
not naive off-policy force supervision.

---

## 2. Central scientific question (verbatim)

> How can an external universal energy model ground the density of a data-trained de novo
> molecular flow?

The paper is a controlled study of what energy supervision does and does not teach a de novo
generative density. It is **not** "we invented a new divergence and learned a Boltzmann generator".

**Naming ruling.** The acronym *BGFM* is retired from the manuscript. The proposed configuration is
called **energy-value supervision** and its experimental arm is labelled **Value (ours)**. The
comparator is **gradient-supervision baseline** / **Force**. This removes the "Boltzmann generator"
adjacency the rules forbid and makes the value-only identity of the method unambiguous. If a legacy
label is unavoidable in a config name or an appendix path, it is defined once in App. B as the
value-only configuration (`lambda_1 = 0`, `lambda_2 = 3e-5`).

---

## 3. Contribution bullets (exactly three — no fourth may be added)

1. **A value-only energy-grounding objective deployed at chemical scale.** We add a single term to
   flow matching, `L = L_FM + lambda_E * Var_k[ log p_theta(x_k | c) + E(c, x_k)/kT ]`, evaluated
   per composition/charge group over geometries scored by a universal neural potential across an
   83-element corpus, inside a mixed discrete–continuous bond-free de novo generator. *The
   log-variance divergence and its partition-function cancellation are prior work
   (`vargrad`, `nusken2021pathspace`, `richter2024improvedsampling`); we claim neither.* What is
   ours is the composition- and group-conditioned deployment, and a local-to-global identifiability
   analysis (Def. 1, Thm. 1, Cor. 1) that states exactly what group-conditioned supervision can and
   cannot determine.
2. **The ordering result, measured against an independent evaluator and an oracle ceiling.**
   On 93 held-out parents disjoint from the energy-shard pool, with the reference geometry dropped,
   local density–energy correlation rises from `r = 0.202 ± 0.023` (FM-only) to `0.369 ± 0.023`
   (value), `+0.167`, Welch `t = 5.16`, exact permutation `p = 0.029`, 4 seeds per arm; lowest-energy
   retrieval rises from 19.6% to 29.6% (top-1) and 49.2% to 62.4% (top-3) against 12.5% / 37.5%
   chance. The independent GFN2-xTB evaluator scores the eSEN teacher itself at `r = 0.927` on the
   same population, so the model reaches about 40% of the measured ceiling.
3. **A matched six-cell grid separating what is learned from what is not, and a calibration
   boundary.** Only the true geometry–energy pairing helps (`D = 0.421` against `A = 0.217`,
   `B = 0.187`, `C = 0.185`); a zero-energy pure `Var[log p]` regulariser is indistinguishable from
   no supervision (`B - A = -0.030`, not significant); naive off-policy force supervision is worse
   than no physics (`E = 0.085`, `E - A = -0.132`). Meanwhile the learned density stays
   thermodynamically miscalibrated: normalised residual variance rises from 1.405 to 2.144 where a
   constant log-density scores exactly 1, and the implied `T_eff` is 2.34 eV against a training
   target of `kT = 1.0` eV. The model learns the right ordering, but not the right calibration.

---

## 4. Claim-to-evidence matrix

Four rows. Every number in this table is verbatim from VERIFIED NUMBERS. Nothing else may be
reported as a headline.

| Row | Claim (as stated in the paper) | Exact numbers | Where it lands |
|---|---|---|---|
| **PRIMARY** | Energy-value supervision improves local density–energy ordering. | 93 disjoint parents, reference dropped, 8 displaced geometries/parent, 4 seeds/arm. FM-only per-seed `r` = 0.222, 0.194, 0.249, 0.142 → mean **0.202**, sem 0.023. Value per-seed `r` = 0.316, 0.378, 0.426, 0.358 → mean **0.369**, sem 0.023. Difference **+0.167**, Welch `t = 5.16`, exact permutation `p = 0.029`. Scrambled (3 seeds): 0.226, 0.192, 0.260 → **0.226**. Retrieval: top-1 FM **0.196** vs value **0.296** (chance 0.125, oracle 0.871); top-3 FM **0.492** vs value **0.624** (chance 0.375, oracle 0.989). Teacher ceiling on the same 93 parents, reference dropped: **0.927**; model reaches **39.8%** of it. | **Sec. 5.2** (result + ceiling repositioning) with **Figure 2** and **Table 1**. Per-seed retrieval values and the all-120 permissive population go to App. B. |
| **MECHANISM** | Only the true geometry–energy correspondence produces the gain; a generic density regulariser and scrambled energies do not, and off-policy force supervision hurts. | Matched six-cell grid, 120 parents, reference dropped, evaluation `n_ode = 12`. `A` FM-only 5/5/0 **0.217 ± 0.008**; `B` zero energies 5/4/1 **0.187 ± 0.030**; `C` scrambled-within-parent 5/5/0 **0.185 ± 0.020**; `D` true pairing 5/3/2 **0.421 ± 0.016**; `E` force-only 5/5/0 **0.085 ± 0.016**; `F` force + true energy 5/2/3 **0.194 ± 0.016**. Contrasts: `D−A +0.204` (t = 11.36), `D−C +0.236` (t = 9.28), `D−B +0.234` (t = 6.89), `B−A −0.030` (t = −0.96, not significant), `E−A −0.132` (t = −7.60), `F−D −0.227` (t = −10.05). | **Sec. 5.3** with **Figure 3**. Per-seed `r` values, the matched-configuration table and the six-contrast table with the Welch-df caution go to App. B. |
| **BOUNDARY** | The ordering improves; the calibration does not. | Same 93 parents, same seeds. NRV per seed is a **median over the 93 parents**; the reported arm value is the **mean over seeds**. FM-only 1.393, 1.342, 1.337, 1.548 → **1.405**. Value 2.955, 2.738, 1.547, 1.337 → **2.144**. A constant log-density scores NRV exactly **1**. `T_eff` (eV) FM-only 7.04, 10.63, 6.17, 9.06 → **8.23**; value 2.34, 1.86, 1.95, 3.19 → **2.34**; training target `kT = 1.0` eV. | **Sec. 5.4** with **Figure 4** and the NRV / `T_eff` columns of **Table 1**. |
| **ROBUSTNESS** | The ordering effect survives a perturbation family that removes the radial-distance confound; its size is resolution-conditional. | Fixed-RMSD perturbations (within-group distance constant to numerical precision), 4 seeds/arm: value **0.305**, FM-only **0.182**, difference **+0.124**, `t = 4.94`. Normal-mode perturbations: difference **+0.185**, `t = 1.58`, 5 seeds — **not significant, supporting context only**. Resolution: FFJORD grid ties smoothing scale to step count, `eps = 1/(2n)`; gap **+0.382** at `n = 4` (`eps = 0.125`, the training resolution) and **+0.177** at `n = 12` (`eps = 0.042`, the resolution at which every reported number is computed). | **Sec. 5.5**, prose only, two short paragraphs, no float. Full derivation, the preliminary ten-parent table and the powered paired ablation go to App. B (`app:estimator`, `app:perturb`). |

### Mandatory attachments to these rows

- **Completion counts.** Every main-text mention of cell `D` or cell `F` carries
  launched/completed/diverged in the same sentence or in the figure caption
  (`D` 5/3/2, `F` 5/2/3), and states that the mean is conditional on completion.
- **Statistics presentation.** Welch statistics are reported as `t` values with the seed counts,
  never as precise p-values (seed-level dof are small). The one exception is the exact permutation
  `p = 0.029` on the primary contrast, which is exact and may be printed. Figures 2 and 3 show raw
  seed points; the argument rests on non-overlapping cell patterns and effect sizes.
- **The `n = 48` resolution cell has one energy seed. It is underpowered and is not a result.**
  It must not appear anywhere in the submission — no numeric value, and never described as zero,
  negative, vanished, attenuated or non-significant. The only permitted statement is that the
  effect size is resolution-conditional.
- **Stability.** 6 of 20 runs carrying the value-density estimator diverged; 0 of 10 without it.
  The zero-energy cell `B` also diverged (5/4/1), so the instability does not require anomalous
  energy labels. The paper never claims the proposed configuration is numerically robust.
- **Training budget.** Every arm trains approximately 0.12 epochs. No model here is converged.
  This sentence appears once in Sec. 5.1 and once in Sec. 6.

---

## 5. Explicit non-claims (the paper's own scope statement)

Written as one short paragraph in **Sec. 6**, and echoed in one clause in the abstract.

> We do not claim correct relative mass across basins: every measurement here is within a single
> parent composition and a local displacement neighbourhood, and Corollary 1 states that
> group-conditioned supervision leaves a per-group constant free. We do not claim a globally
> calibrated `p(x | c)`; the normalised residual variance moves in the wrong direction. We do not
> claim a Boltzmann generator, correct basin occupancy, or correct physical-temperature sampling —
> the implied `T_eff` disagrees with the training target by a factor of two. We do not claim
> superiority over force supervision in general: our negative force result is scoped to naive
> off-policy endpoint force matching read out through the flow-matching score identity on this
> non-equilibrium corpus. We do not claim superiority over the broad de novo generation literature;
> generation quality is reported only to show it is not degraded.

Forbidden vocabulary in the submission PDF (enforced by a build check): `\pending`, any red macro,
"in progress", "placeholder", "TODO", "TBD", "retract"/"retraction", "withdraw", "earlier version",
"our earlier draft", "surprisingly", "we did not expect", "Boltzmann generator" as a description of
our model, "global Boltzmann density", "correct basin occupancy", "zero-shot", "outperform all",
"calibrated density" as a description of our result.

Prefer **calibration** over **scale** for the density conclusion ("scale" survives only inside
mathematical definitions such as *scale ratio* and *smoothing scale*). Use **local density–energy
ordering**, **value supervision**, **gradient-supervision baseline** consistently. "We find" for
measured results; "may be explained by" for mechanisms not experimentally isolated.

---

## 6. Main-paper section outline and page budget

Target: **9.0 pages** of main text (`\label{endofmaintext}` on page 9), ICLR 2027 style.
Front-matter statements (AI use, reproducibility, ethics) sit after the main text and are off budget.

| § | Title | Budget | Contents | Floats charged here |
|---|---|---|---|---|
| — | Abstract | included in §1 | ~200 words: question, method in one clause, primary number, mechanism grid in one clause, calibration failure, one scope clause. No per-cell SEMs, no resolution clause, no history. | — |
| 1 | Introduction | **1.2 pp** | Stakes (ensembles from a de novo generator); the question; the value-only objective in one sentence with the attribution cession verbatim; the four findings paragraph; two sentences on what makes the metric admissible (independent scorer, per-parent, both corrections at once); the three contribution bullets. | **Figure 1** (0.45 pp) |
| 2 | Setting and related work | **0.8 pp** | State `(c, x)`, the eSEN teacher, `kT = 1.0` eV as a numerical scale, why bond-free is forced by the teacher, the FlowMol3 interpolant and `L_FM`. Then three related-work buckets: de novo 3D generation and conformer models; divergences for training samplers (**the attribution cession, unambiguous**); universal neural potentials and force supervision. | — |
| 3 | Energy-value supervision | **1.2 pp** | The FFJORD log-density in one displayed line; `L_value` as the group-wise variance; `L = L_FM + lambda_E L_value`; the relation to the log-variance divergence and why grouping is mandatory (cross-parent variance measures `log Z` spread); the score read-out `s = (t v − x)/((1−t) sigma^2)` and `L_force` in two sentences as the **later** comparator; two disclosures (how `lambda_E` was fixed; `lambda_3 = 0` in every run); one clause that what is reported is a smoothed marginal at `eps = 0.042`; one sentence that 6 of 20 estimator-carrying runs diverged against 0 of 10 without, so every energy-cell mean is conditional on completion. | — |
| 4 | What does group-wise supervision identify? | **0.8 pp** | Lemma 1 (score read-out) as one displayed identity; the zero-variance certificate in two sentences; the two-basin counterexample; **Definition 1** (overlap graph); **Theorem 1** (local-to-global identifiability); **Corollary 1** (our overlap graph has no edges, so only part (i) applies). Two sentences of proof idea. All proofs in App. A. | — |
| 5 | Experiments | **4.1 pp** | 5.1 Setup and endpoints (0.6) · 5.2 Value supervision improves local ordering, with retrieval and the ceiling (1.1) · 5.3 The matched six-cell grid (1.1) · 5.4 Ordering improves, calibration does not (0.85) · 5.5 Robustness and resolution, prose only (0.3) · 5.6 One sentence on generation quality (0.05) | **Table 1** (0.35), **Figure 2** (0.55), **Figure 3** (0.5), **Figure 4** (0.45) |
| 6 | Limitations and conclusion | **0.9 pp** | Limitations, one paragraph, six items: calibration, 40% of ceiling, cross-basin unmeasured, seed imputation, radial confound + estimator resolution in one clause, 0.12-epoch budget. Then the non-claims paragraph from §5 above. Then the conclusion: ordering without calibration, the mechanism finding, the scoped force result, and the named next experiment (the anchor term, implemented and untried). | — |
| — | *(after `endofmaintext`)* AI-use statement; Reproducibility statement; Ethics statement | off budget | New text, authored from scratch. Placed as unnumbered `\subsubsection*{}` blocks immediately before `\label{endofmaintext}`. | — |

**Float ruling.** The main text carries **four figures and exactly one table**. `tab:diverge`,
`tab:boltzmann`, `tab:scale` and `tab:p0grid` from the current draft are all superseded: their
content is absorbed by Table 1 (primary endpoint), Figure 3 (the grid, with L/C/D annotations
replacing `tab:diverge`) and Figure 4 (calibration). No `\resizebox` on Table 1; if it does not
fit at 9 pt, drop a column, not the font size.

**Table 1 specification.** Rows: *FM-only*, *Value (ours)*, *Scrambled energies*, *Teacher ceiling*.
Columns: `r` (mean ± sem over seeds), NRV (median over parents, mean over seeds), `T_eff` (eV),
retrieval top-1, retrieval top-3, seeds. Population: the 93 disjoint parents, reference dropped.
The ceiling row leaves NRV and `T_eff` blank unless those values are recomputed and the derivation
is stated in the caption (see the Figure 4 ruling below). Caption states the population, the
reference-drop, the aggregation rule, and that a constant log-density scores NRV exactly 1.

**Template ruling.** Switch `main.tex` to `iclr2027_conference.sty` and
`\bibliographystyle{iclr2027_conference}`; the running head currently prints ICLR 2026. Keep
`\author{Anonymous}` and keep `\iclrfinalcopy` commented out. Verify the 2027 `.sty` page macros
before trusting `build.sh`'s page check.

---

## 7. Figure plan

All four figures are drawn at final size (5.5 in single column, 7–8 pt in-figure serif text) and
included with `width=\linewidth`. Each gets a wrapper `figures/figN_<name>.tex` holding the
`figure` environment, `\input` from the section that argues it; generators emit both `.pdf` and
`.png` into `figures/out/`. Reuse `set_style()`, `PALETTE`, `sem()`, `welch()`, `permutation_p()`,
`read_csv_cols()`, `save()` from `paper/figures/make_experiment_figures.py`. Palette semantics are
fixed: **green = value/energy**, **vermilion = force**, **grey = FM-only baseline**,
**purple = scrambled control**; **red is retired** (it was the placeholder colour). Plotting
interpreter: `/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/omol25/bin/python` (matplotlib + scipy,
no pandas).

### Figure 1 — The objective and its scope (conceptual, TikZ, Sec. 1)

| Panel | Content | Source |
|---|---|---|
| (a) | Bond-free flow matching: interpolant, reverse-time ODE, `log p_theta` read out by FFJORD at `eps = 1/(2n)`. | none (conceptual) |
| (b) | **The value term, given full visual weight**: `K` displaced geometries of one parent, teacher energies, the group-wise variance of `log p + E/kT`, the `log Z` cancellation. | none |
| (c) | **Subordinate strip**, visually smaller and set apart: the gradient-supervision baseline via the score read-out, labelled "evaluated separately in Sec. 5.3". | none |
| (d) | What is certified and what is not: within-group ordering identified up to a per-group constant (Cor. 1); cross-group mass not identified. | none |

**Ruling: the force path must not have equal visual status.** Panel (c) occupies at most one third
of the width of panel (b), sits below it, and carries no arrow into the training loop that panel (b)
does not dominate. No panel may assert a result; the old caption's "significantly worse than no
physics" is deleted. Existing `figures/fig1_method.tex` is the starting point; strip its
`\providecommand{\pending}`, delete the five-panel structure's panel (e) result claim, and demote
the force panel.

**One sentence it must make obvious:** *the proposed method adds a single value-only term that
ties the model's own log-density to teacher energies within a composition group, and the
partition function drops out.*

### Figure 2 — Value supervision improves local ordering (Sec. 5.2)

Three panels, no fourth. **Panel 2d (fixed-RMSD) is cancelled**: no artifact on disk reproduces the
verified pair (0.305 / 0.182), so that result is reported as prose in Sec. 5.5 only. Verified
scalars must never be paired with per-parent clouds recomputed from a round that disagrees with them.

| Panel | Content | Source file |
|---|---|---|
| 2a | Per-seed `r` for FM-only (4), Value (4), Scrambled (3) on the 93-parent primary endpoint; raw seed dots, arm mean, sem bar; the `t = 5.16` / permutation `p = 0.029` annotation on the FM↔Value contrast only. | `/n/home04/yulili/bgfm/paper/figures/out/scale_primary.json` → `["dropref"][arm]["per_seed"][seed]["primary_93"]["pearson_r"]`. Cross-check: `/n/holylabs/woo_lab/Lab/yulili/bgfm/runs/eval_ours/calibration/calibration_dropref.json` → `["arms"][arm_seed]["per_group"][gid]`, filtered to the 93 ids in `figures/out/primary_endpoint.json["disjoint_group_ids"]`. |
| 2b | Distribution of per-parent `r_m` over the 93 parents, FM vs Value, overlaid, plus one representative per-parent scatter of `log p_theta` against `−E/kT` inset. | Distribution: `calibration_dropref.json` `per_group` filtered to the 93 ids (`n_pert = 8`, reference already dropped). Scatter: `/n/holylabs/woo_lab/Lab/yulili/bgfm/runs/eval_ours/wide_<tag>/boltz_records.csv`, dropping `pert_id == 0` and keeping the 93 ids. **Do not lift the existing `fig2_boltzmann_scatter.pdf`** — it is the permissive all-120, reference-included population. |
| 2c | Lowest-energy retrieval: top-1 and top-3 for FM and Value, with the chance line (0.125 / 0.375) and the oracle line (0.871 / 0.989); per-seed dots. | `/n/home04/yulili/bgfm/paper/figures/out/retrieval_utility.json` → `["arms"][arm]["per_seed"][tag]["primary_93"]` and `["oracle"]["primary_93"]`, `["chance_top1"]`, `["chance_top3"]`. |

**One sentence it must make obvious:** *on held-out parents the model never trained on, energy-value
supervision moves per-parent density–energy correlation up by a margin larger than the seed spread,
and that margin converts into a usable improvement in picking the lowest-energy geometry.*

Caption must state: 93 parents disjoint from the energy-shard pool, reference geometry dropped,
8 displaced geometries per parent, 4/4/3 seeds, scored by an independent GFN2-xTB evaluator that is
not the eSEN training teacher.

### Figure 3 — The matched six-cell grid (Sec. 5.3)

Single-panel dot-and-bar chart, six cells `A_fmonly`, `B_flat`, `C_scram`, `D_energy`, `E_force`,
`F_both`, ordered as launched. This figure **replaces** the per-seed grid table in the main text.

| Element | Content | Source |
|---|---|---|
| Bars + seed dots | Per-seed `r` per cell (raw dots), cell mean, sem bar. A 0.234/0.200/0.232/0.221/0.197; B 0.233/0.125/0.147/0.243; C 0.232/0.224/0.168/0.125/0.174; D 0.452/0.398/0.412; E 0.105/0.130/0.044/0.086/0.058; F 0.179/0.210. | `/n/holylabs/woo_lab/Lab/yulili/bgfm/runs/eval_ours/clean_p0_<CELL>_s<N>/boltz_independent_records.csv`. Recipe: drop `pert_id == 0`, keep `xtb_ok`, per-parent Pearson `r(log_p_theta, negE_kT)`, mean over the 120 parents. |
| L/C/D annotation | Under each cell: `5/5/0`, `5/4/1`, `5/5/0`, `5/3/2`, `5/5/0`, `5/2/3`. | Census over `/n/holylabs/woo_lab/Lab/yulili/bgfm/runs/p0/p0_<CELL>_s<N>/lightning_logs/version_*/checkpoints/` — a completed run has `midstep-step=30000.ckpt`. Hard-restrict to seeds s1–s5. |
| Contrast brackets | `D−A +0.204` (t = 11.36), `D−C +0.236` (t = 9.28), `B−A −0.030` (n.s.), `E−A −0.132` (t = −7.60). At most four brackets; `D−B` and `F−D` go in the text. | derived from the per-seed values above |

**Three traps the builder must avoid, restated:** (i) the verified force cell is
`clean_p0_E_force_*`, **not** `clean_p0_E2_force_*`; (ii) `clean_p0_D_energy_s6/s7/s9` and
`p0_F_both_s6..s10` are post-hoc relaunches outside the matched grid and must be excluded — no seed
is ever replaced; (iii) `boltz_independent.json`'s `mean_pearson_r` is reference-**included** and
must never be used.

**One sentence it must make obvious:** *only the cell with the true geometry–energy pairing rises
above the no-physics baseline; a zero-energy density regulariser and scrambled energies do not, and
force supervision falls below it — and the two energy-carrying cells are also the two that lost
seeds to divergence.*

Caption must carry the launched/completed/diverged triples again and state that `D` and `F` means
are conditional on completion.

### Figure 4 — Ordering versus calibration (Sec. 5.4)

Two-panel figure, both panels on the 93-parent primary endpoint.

| Panel | Content | Source |
|---|---|---|
| 4a | Scatter with ordering `r` on the x-axis and NRV on the y-axis; one point per seed (4 FM grey, 4 Value green), arm means with sem crosses; a horizontal reference line at **NRV = 1** annotated "a constant log-density scores 1"; the `NRV_min = 1 − r^2` curve as a light lower-bound guide; a vertical marker at `r = 0.927` labelled "independent-teacher ceiling". | `/n/home04/yulili/bgfm/paper/figures/out/scale_primary.json` → `["dropref"][arm]["per_seed"][seed]["primary_93"]` fields `pearson_r`, `nrv`, `nrv_min`. Ceiling x-value from `/n/holylabs/woo_lab/Lab/yulili/bgfm/runs/eval_ours/ceiling_full_dropref/ceiling_per_group.csv` restricted to the 93 ids in `figures/out/primary_endpoint.json` (mean `pearson_r` = 0.9271). |
| 4b | Implied `T_eff` per seed for both arms as a strip plot, with a horizontal line at the training target `kT = 1.0` eV. FM 7.04/10.63/6.17/9.06 (mean 8.23); Value 2.34/1.86/1.95/3.19 (mean 2.34). | same `scale_primary.json` records, field `T_eff_eV` |

**Oracle-NRV ruling.** The teacher's NRV is **not** in the verified list. Default: the ceiling
appears in panel 4a only as a vertical marker on the ordering axis, not as a 2-D point. If the
builder wants the 2-D oracle point, it must be recomputed from `ceiling_per_group.csv` restricted
to the 93 ids using the same median-over-parents convention, and the caption must say the number
was derived for this figure from that file. Nothing is taken from the old PDF.

**One sentence it must make obvious:** *the arm that orders geometries better sits further from the
calibrated line, not closer — improved ordering and improved calibration are decoupled here.*

---

## 8. Main-versus-appendix content map

Derived from the recon survey. "Delete" means the material does not appear anywhere in the
submission, source comments included.

### Delete outright (does not survive anywhere)

- `main.tex` lines 35–545: the float register, the Q1–Q8 placeholder inventory, all ROUND LOGs,
  the BAN amendment, the AUDIT PASS list, the open verification items, the camera-ready gate.
- The `\pending` macro and all 20 call sites; the three `\providecommand` duplicates.
- `app:placeholders` and `tab:placeholders` (App. B.21) — this also removes the only consumer of
  `longtable`.
- Every retelling of the unscrambled-validation-shard history, the withdrawn "2 of 15 = 13%"
  mitigation, the retracted 63/37 mechanism decomposition, and the withdrawn energy-label
  attribution of the divergences. Specifically: intro bullet 4; the abstract's two retraction
  sentences; `sec:ablation` "An earlier decomposition of the gain is retracted, and why";
  `app:p0cells` "The decomposition this grid overturns" and "What was wrong with the previous
  scrambled control"; `app:stability`'s "The attribution was wrong" heading and the
  withdrawn-mitigation paragraph; `app:limitations` item (viii)'s narration; the editorial
  sentences in `app:data`.
- `sec:ablation` "A negative control on the labels" (the old n = 120 scramble arm narrative) —
  superseded by cell `C`.
- Every cross-round comparison: the n = 30 one-seed force round used as a contrast; the sentence
  aligning grid `A` with the headline round's physics-free arm; `tab:tau`'s "Force-only (earlier
  round, n = 30)" row; `app:movedresults` "Why the force result is scoped the way it is";
  `app:eval`'s cross-round guard sentence; `tab:cells`' "cf. / compare with" entries.
- `app:tmqm` (B.13), the tmQM rows of `tab:cells`, and the sentence in `ass:charge` naming it.
  Preliminary transition-metal results with incorrect spin handling.
- `app:methoddetails` (B.18) — duplication end to end. Its one unique sentence (the force term is
  retained in the formulation because Sec. 5 measures it) folds into Sec. 3.
- `fig:ablation` / `app:ablationfig` (App. B.12) — the per-seed picture of the superseded round;
  Figure 3 now carries every per-seed value a reader needs.
- The three `\pending` baseline-retrain rows (Symphony, EDM, GeoLDM) in `tab:generation` and the
  caption sentence describing them.
- The `n = 48` row of `tab:respowered`.
- `app:scope`'s "Deleted rather than deferred" paragraph.
- `refs.bib` lines 320–336, the dead prose stub block, and all resolved `% TODO-verify` comments.

### Main text (see §6 for placement)

Abstract · Sec. 1 P1/P2/P4 and a two-sentence version of P5 · the three bullets · `sec:related`
three buckets, compressed · `sec:method:setting` · `eq:score` + one sentence of `L_force` ·
`eq:lenergy` + one line of `eq:ffjord` · "Relation to the log-variance divergence" (the single
surviving copy) · `eq:total` · disclosures (i) and (iv) · one stability sentence · Lemma 1 in one
line · the certificate in two sentences · the two-basin counterexample · Definition 1 · Theorem 1 ·
Corollary 1 · `sec:experiments` preamble · the ensemble/setup paragraph · `eq:rm` and `eq:nrvmin` ·
the primary-endpoint construction, with the shard leak compressed to one clause · the adversarial
seed-imputation result (`+0.100, p = 0.111`; `+0.077, p = 0.294`; neither significant) · the ceiling
paragraph · the primary contrast · retrieval · the calibration subsection · two robustness
paragraphs · one generation-quality sentence · limitations (one paragraph) · non-claims ·
conclusion · the three front-matter statements.

### Appendix A (proofs) — keep as-is except where noted

`app:setup` (Assumptions 1–7; rewrite `ass:charge`'s justification as a plain scope limitation once
`app:tmqm` is gone) · `app:score` (score read-out proof, sign remark, amplification remark — this is
also where the Huber threshold and per-atom norm cap discussion from Sec. 3 lands) · `app:trap`
(**required correction**: `rem:onpolicy` currently says the reported force cells carry an on-policy
distillation term; cell `E` carries none — rewrite or the appendix contradicts Figure 3) ·
`app:certificate` · `app:l2g` (full proof of Theorem 1, Corollary 1, `rem:epsilon`, `rem:l2g-design`,
`rem:crossbatch`) · `app:movedtheory` merged with the first four paragraphs of `app:scope` under a
real title (not "moved by the page pass") · `app:surrogate` + `tab:tau` (minus the cross-round row;
`rem:attenuation` cites `app:estimator` instead of restating it) · `app:gauge` including `rem:guard` ·
`rem:refmeasure` moves here from Sec. 4.

### Appendix B (details) — keep, with edits

`app:asimplemented` (clipping, Huberisation, truncated-adjoint gradient — destination for Sec. 3
disclosure (ii)) · `app:hyper` + `tab:constants` (drop the old force cells' distillation settings;
**strip the `scripts/fasrc/` path**) · `app:hardware` · `app:data` — **the one and only home for the
shard-construction facts**, editorial narration removed · `app:eval` · `app:estimator` +
`tab:epsgrid` + `tab:resolution` + `tab:respowered` — absorbs Sec. 5.5 and the `eq:epsgrid`
paragraph; four `\pending` markers and the `n = 48` row deleted; the three-readings / what-it-settles
pair trimmed · `app:scalediag` + `tab:scaleperm` (definitions, identities, permissive populations,
median-aggregation rationale, retrieval read-out, ceiling projections; the "Q1, closed" paragraph
deleted) · `app:perturb` (confound statement, fixed-RMSD construction and its `1.3e-16` verification,
**both** measured families including the non-significant normal-mode result; say plainly that two of
four families were evaluated) · `app:cells` + `tab:cells` as a provenance record only, no contrasts
drawn · `app:p0cells` + `tab:p0cells` + `tab:p0seeds` (configuration table, per-seed table, the six
contrasts with the Welch-df caution, "what cell B is for", the selection-risk analysis; the launch-
record paragraph trimmed against `app:stability`) · `app:boltzfig` + `fig:boltzmann` (**required fix**:
legend `+0.430 / +0.075` vs caption `+0.434 / +0.074` — pick one aggregation and make panel, caption
and script agree) · `app:generation` + `tab:generation` (three real rows, the ~5-point binomial SE,
the explicit no-superiority conclusion) · `app:capability` + `tab:capability` (**verify every cell
against the cited papers or delete the unverifiable rows**) · `app:stability` (mechanism, the
three-part fix, the guard rationale, current rates stated as facts) · `app:repro` (cut to ~4
sentences; keep the anonymised-archive promise) · `app:limitations` items (i)–(vii), item (viii)
reduced to ~300 words of facts · `app:movedresults` — keep only the paired Wilcoxon paragraph and
"The force result, unconfounded", under a real title.

### Anonymity fixes (all in Appendix B, all rendered text)

Remove `scripts/fasrc/...` from `A2_details.tex:461` and `figures/tabA2_constants.tex:4` — "FASRC"
names the authors' institution's cluster. Strip every internal repository path printed in App. B
(`paper/figures/*.py`, `scripts/*.py`, `runs/eval_ours/...`, `cfm_mol/bgfm_density.py`,
`tests/test_geom_perturb.py`). Strip the two main-text filenames (`seed_sensitivity.py` at
`04_experiments.tex:199`, `recompute_scale_primary.py` in the `tab:scale` caption). Keep the single
anonymised-archive sentence at `A2_details.tex:1343-1344`. Keep `\author{Anonymous}`; keep
`\iclrfinalcopy` commented out; keep Acknowledgments and Author Contributions commented out.

### Bibliography fixes (do in the same pass)

`sbg` — wrong title, no authors, no arXiv id, no venue; replace with *Scalable Equilibrium Sampling
with Sequential Boltzmann Generators*, ICML 2025, PMLR 267:58467–58498, arXiv:2502.18462.
`tbg` — NeurIPS 2024, not ICML 2025. `semlaflow` — AISTATS 2025, not ICML 2025.
`adjointsampling` — expand the 13-author list. `flowmol3` — now Digital Discovery 5(5):2052–2066,
DOI 10.1039/D5DD00363F. `omol25`, `esen` — expand author lists. `nusken2021pathspace` — replace
`number={4}` with the article number 48 and add the DOI. Add `tabg` (Schopmans & Friederich,
*Temperature-Annealed Boltzmann Generators*, ICML 2025) to fill the annealed-BG slot. Cite or delete
`etflow` and `mcf`. Update the stale ICLR-2026 header comment.

---

## 9. Title

**Default (use this):**

> **Grounding De Novo Molecular Flows with Energy Values: Ordering without Calibration**

Recorded alternatives, in the order they were proposed. Any switch is a single edit to `\title{}`
and requires no change to the body text:

1. Ordering without Calibration: Energy-Value Supervision for De Novo 3D Molecular Flows
2. Learning Energy-Ordered Densities for De Novo 3D Molecular Generation
3. Energy-Value Supervision Grounds De Novo Molecular Flows in Local Boltzmann Ordering

Alternative 3 uses "Boltzmann" as a property of the *ordering*, not as a description of our model,
which is permitted; it is nonetheless the weakest option because a skimming reader may read it as a
Boltzmann-generator claim. Alternative 2 loses the calibration boundary, which is half the thesis.

---

## 10. The five anticipated reviewer attacks, and where each is answered

### (i) "The log-variance divergence is prior work; what is new here?"

**Answered in three places, and the concession is never hedged.** Sec. 1, in the objective
paragraph: one sentence ceding the divergence and its partition-function cancellation to
`vargrad`, `nusken2021pathspace`, `richter2024improvedsampling`, with the phrase *neither the
divergence nor that cancellation is ours* preserved verbatim. Sec. 2, related-work bucket 2: the
full attribution, with `sendera2024offpolicy` and `berner2024optimalcontrol`. Sec. 3, "Relation to
the log-variance divergence": what is specific here is the composition- and group-conditioned
deployment in a bond-free mixed discrete–continuous de novo generator over an 83-element corpus
scored by a universal neural potential. The claimed novelty is enumerated in contribution bullet 1
and consists of exactly five items: group-conditioned deployment; use with a universal neural
potential at 83-element scale; the local-to-global identifiability analysis; the controlled
value-versus-gradient experiments under one backbone and budget; the empirical separation of
ordering from calibration. Three redundant copies of this concession (Sec. 4 preamble,
`rem:logvar`, `app:scope`) are deleted so the surviving statements read as attribution rather than
as a defensive refrain.

### (ii) "The evaluation is local, not cross-basin; you have not shown a Boltzmann density."

**Answered structurally, not rhetorically — this is why Sec. 4 exists.** Definition 1 introduces the
overlap graph; Theorem 1 states that group-conditioned supervision identifies the density globally
only up to a `(C−1)`-parameter family determined by the graph's connectivity; **Corollary 1 states
that our overlap graph is a single vertex with no edges, so only part (i) applies and a per-group
constant is genuinely free**. Sec. 5.1 therefore defines both endpoints per parent. Sec. 6's
non-claims paragraph states outright that we do not claim correct relative mass across basins, a
globally calibrated `p(x | c)`, a Boltzmann generator, correct basin occupancy, or correct
physical-temperature sampling. Figure 1 panel (d) shows the same boundary graphically. The title
itself concedes the limit. A reviewer who raises this attack should find that the paper raised it
first and built a theorem around it.

### (iii) "Gaussian displacement confounds energy with distance from the reference geometry."

**Answered in Sec. 5.5 with a measured control, and bounded honestly.** The primary perturbation
family is isotropic (`sigma = 0.15 Å`) and its within-group radial spread has a coefficient of
variation of 8.1%, which is stated in Sec. 5.1 as a known confound rather than discovered under
questioning. Sec. 5.5 then reports the fixed-RMSD family, in which within-group distance is constant
to numerical precision: value **0.305** against FM-only **0.182**, difference **+0.124**, `t = 4.94`,
4 seeds per arm — the effect survives with the confound removed. The normal-mode family gives
**+0.185**, `t = 1.58`, 5 seeds, **not significant**, and is reported as supporting context only and
never described as significant. Sec. 6 states that two of four planned confound-breaking families
were evaluated. `app:perturb` carries the fixed-RMSD shell construction and its `1.3e-16`
verification. Note for the writer: the fixed-RMSD and normal-mode numbers are quoted as verified
scalars and are **not** plotted against per-parent data, because no on-disk round reproduces them
exactly; the sentence must not name a file it cannot back.

### (iv) "The density estimator is resolution-dependent, so the effect may be an artefact of the grid."

**Answered in Sec. 3 as a disclosure and in Sec. 5.5 as a measurement.** Sec. 3 states that the
FFJORD grid ties the smoothing scale to the step count, `eps = 1/(2n)`, so what is reported is a
smoothed marginal at `eps = 0.042` (`n = 12`), the resolution at which **every** reported number in
the paper is computed — this is said before any result appears. Sec. 5.5 then reports the powered,
strictly-paired resolution ablation: the gap is **+0.382** at `n = 4` (`eps = 0.125`, the training
resolution) and **+0.177** at `n = 12`. The paper states that the effect size is
**resolution-conditional** and does not claim a converged density. `app:estimator` carries the full
derivation, the non-convergence of the log-density with `n`, the train/eval `eps` mismatch, the
preliminary ten-parent check and the named repair (decoupling the smoothing scale from the step
count) as future work. **Enforcement:** the `n = 48` cell has one energy seed; it must not appear in
the submission in any form — no value, and never described as zero, negative, vanished, attenuated
or non-significant.

### (v) "The energy cells lost seeds to divergence, so their means are selected."

**Answered as a first-class result rather than a caveat.** Sec. 3 states, before any experiment, that
6 of 20 runs carrying the value-density estimator diverged against 0 of 10 without it, and that every
energy-cell mean is therefore conditional on completion. Figure 3 prints launched/completed/diverged
under every cell (`A` 5/5/0, `B` 5/4/1, `C` 5/5/0, `D` 5/3/2, `E` 5/5/0, `F` 5/2/3), and its caption
repeats that `D` and `F` are conditional on completion. Sec. 5.3 names the selection risk explicitly:
the three surviving `D` seeds are tightly clustered (0.452, 0.398, 0.412), which is consistent with
either a real effect or with survivorship, and we cannot separate the two. Sec. 5.2's seed-imputation
analysis bounds the primary endpoint adversarially (`+0.100`, `p = 0.111`; `+0.077`, `p = 0.294`;
neither significant) and that weakening of our own headline is printed in the main text. The
zero-energy cell `B` also diverged (5/4/1), so the instability does not require anomalous energy
labels; this is stated as evidence about where the instability lives, and **the paper never claims
the proposed configuration is numerically robust**. No diverged seed is ever replaced: the relaunched
runs `D_energy_s6/s7/s9` and `F_both_s6..s10` exist on disk and are excluded from every reported
number.

---

## Enforcement checklist for `build.sh` (add to the existing gate)

1. Page count: `\label{endofmaintext}` on page ≤ 9.
2. Zero occurrences of `\pending` and of the forbidden-vocabulary list from §5.
3. Zero occurrences of `n = 48`, `n=48`, `48 steps` in any rendered file.
4. Zero occurrences of `fasrc`, `FASRC`, `holylabs`, `netscratch`, `home04` in rendered text.
5. Running head reads ICLR 2027.
6. Required figure inputs present: `figures/out/fig2_ordering.pdf`, `fig3_grid.pdf`,
   `fig4_calibration.pdf` (Figure 1 is inline TikZ).
7. Front-matter statements present: AI use, reproducibility, ethics.
8. Exactly three items in the contributions list; exactly one table in the main text.
