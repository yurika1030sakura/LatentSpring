# SPINE_v2.md — the writing contract for the BGFM manuscript, revision 2

**Status:** BINDING. Supersedes `SPINE.md` §§1, 2, 3, 4, 6, 7, 8, 9 and every ruling
in `SPINE.md` §4 (R1–R4). `SPINE.md` remains authoritative *only* for
§5 (notation contract), §10 (sentence templates, modulo the bans below),
§11 (float register, as amended in §5.3 here), §13 (citation keys) and §14
(mechanical checklist).

**Date:** 2026-08-04 · **Round:** N1/N2 repositioning ·
**Editor's ruling:** this is a *repositioning*, not a patch. Two measurements
landed that change what the paper is allowed to claim:

* **N1 — the oracle ceiling.** Pushing the eSEN teacher's own energies through the
  identical metric gives mean per-group `r = 0.958`. Our energy arm's `r = 0.434`
  is therefore **45% of the achievable maximum**, not an approach to a limit.
  The old ban B-10 ("do not say 0.43 approaches the ceiling") is now enforced by a
  *number*, and the number is a contribution in its own right.
* **N2 — normalised residual variance (NRV) and effective temperature.** Every arm
  we have, including the energy arm, scores **NRV ≥ 1** — the value attained by a
  model that emits a *constant* log-density — and the energy arm's effective
  temperature is **≈2.0 eV against a 1.0 eV training target**. The learned density
  has the right *direction* and roughly the right *dynamic range* but is **not on
  the right scale**. Consequently the word **"calibrated" is retired from the
  title, the abstract, the claim vocabulary and the section headings.**

The tension — *ordering improves, scale does not* — is not damage control. It is
the paper's most transferable scientific statement and it must be written as a
result, in the abstract, with its next experiment named.

---

## 0. Rulings carried into this revision

| # | Ruling | Enforcement |
|---|---|---|
| **R1** | The energy term is the **log-variance divergence**; the `Z`-cancellation and the off-policy freedom are *prior work* (`vargrad`, `nusken2021pathspace`, `richter2024improvedsampling`, `sendera2024offpolicy`). Novelty = composition/group-conditioned off-policy deployment inside a mixed discrete–continuous de-novo molecular flow at broad chemical scale + the value-vs-gradient study + the identifiability theorem. | **ALREADY IMPLEMENTED** in `01_intro.tex` (abstract, P4, bullet 1, Related Work ¶4) and `03_theory.tex` (¶"Relation to prior work"). Do not weaken. |
| **R2** | Parent ≠ composition. Per-parent centring certifies *local* shape consistency only; disconnected basins may carry different constants at zero loss. `\Cref{thm:l2g}` + `\Cref{cor:basins}` state exactly when local certificates fuse. | **ALREADY IMPLEMENTED** in `03_theory.tex` (Def. overlap graph, Thm. l2g, Cor. basins) and echoed in `04_experiments.tex` §5.6. Keep the theorem; **change its prose label** from "local density–energy *calibration*" to "local density–energy *ordering*" (see B-16). |
| **R3** | Title must not assert "calibrate" (N2 kills it) and must not assert a general force claim (PSM / drifting-to-Boltzmann work; our force arm is confounded). | **§2 below.** New title mandatory this round. |
| **R4** | Delete: joint composition-level Boltzmann extension; free-energy/ESS read-outs; tmQM from the main text; the 74/26–63/37 mechanism decomposition; the two-round juxtaposition in Fig. 3. | **§9 below**, file-by-file. Items (a), (b, partial), (c), (d) are already done; verify and finish. |
| **R5** | Limitations must state: scale un-calibrated (NRV > 1, `T_eff ≈ 2` eV); 45% of ceiling; local-only evaluation, no cross-basin test; `r` measures shape only; 12% of one epoch; radial-distance confound open; estimator not validated; force result confounded and hypothesis-generating only. | **§5.2 (Sec. 5.6) + §4 claim map.** Eight mandatory items, all with a number or an explicit "not measured". |
| **R6** *(new)* | **Ordering vs. scale must be reported as two co-primary metrics**, never one. Any sentence quoting `r` without the NRV/`T_eff` companion within the same paragraph is rejected. | B-17, §10. |
| **R7** *(new)* | **Every `r` in the manuscript must be readable against the ceiling.** Table 1 carries a ceiling row; §5.3 carries the ratio. | B-18, §10. |
| **R8** *(new)* | **Estimator-convergence gate.** `ESTIMATOR_VALIDATION.md` (on disk) reports that `log p_θ` does not converge in `n_ode_steps` and that the energy-vs-control gap falls from +0.45 (4 steps) to ≈0 (48 steps). Until the editor-in-chief re-verifies this, **no draft may claim estimator-independence**, and the limitations item "the likelihood estimator has not been validated" must stay and be strengthened. See §11 (open conflicts). | B-19, §11. |

---

## 1. The honest one-sentence story

> **A de-novo 3D molecular flow trained on data is a structure emulator, not a
> thermodynamic ensemble model. Aligning its conditional log-density to a universal
> neural potential with a group-wise log-variance objective substantially improves
> the *local ordering* of density against energy (mean per-group `r`: 0.07 → 0.43
> with the reference geometry, 0.22 → 0.40 without it, four seeds per arm), yet the
> learned density remains *un-scaled*: its normalised residual variance stays at or
> above 1 — the value a constant log-density attains — its effective temperature is
> ≈2× the 1.0 eV training target, and it reaches only ~45% of the `r ≈ 0.96` ceiling
> that an exact copy of the teacher attains on the same metric. The judgement that
> buys the ordering gain is density-**value** based and works off-policy, whereas
> naive force-to-score matching was worse than no physics in our
> non-equilibrium-data setting. Ordering-without-scale is the result, and it names
> the next experiment: explicit scale/temperature calibration.**

### 1.1 Three-part pitch (use this shape in abstract, intro P1–P5, conclusion)

1. **Stakes.** Chemistry is an ensemble problem; de-novo generators have no density
   grounded in a physical energy, and Boltzmann samplers do not assemble molecules.
2. **Move.** Turn a universal neural potential into a *training target* through a
   group-conditioned log-variance objective (prior-art divergence, new deployment),
   and be exact about what group-wise centring identifies (`\Cref{thm:l2g}`).
3. **Finding, with its own limit.** It measurably moves the density's *ordering*
   toward the Boltzmann weight — and, measured properly, leaves the *scale* wrong;
   both halves are quantified against an oracle ceiling, and the gradient-form
   alternative fails in our construction.

### 1.2 Abstract shape (mandatory, 8 sentences, ≤ 200 words)

1. Stakes (ensemble problem, structure emulator vs. Boltzmann model).
2. The move: group-conditioned log-variance objective over a bond-free 83-element
   flow; **attribution sentence** (divergence is prior work, deployment is ours).
3. Identifiability: per-group constants fuse iff the overlap graph is connected;
   ours has no edges → **local ordering**, never a conditional Boltzmann density.
4. **Ordering result** on the primary endpoint (disjoint 93 parents, reference
   excluded): `0.202 ± 0.023 → 0.369 ± 0.023`, Welch `t = 5.16`, permutation
   `p = 0.029`; scrambled control indistinguishable from no physics (`p = 0.486`).
5. **Ceiling sentence (N1):** the same metric applied to the teacher's own energies
   gives `r = 0.958`, so we reach ~45% of what is achievable.
6. **Scale sentence (N2):** NRV stays ≥ 1 and `T_eff ≈ 2.09` eV against a 1.0 eV
   target — the density's ordering improves while its scale does not, which is the
   result we most want carried forward.
7. Force: gradient-form alternative was worse than no physics **in our
   construction**, scoped, not general.
8. What is not tested: cross-basin occupancy, temperature calibration, the
   radial-distance confound, estimator convergence; plus the numerical fragility.

**Forbidden in the abstract:** "calibrated density", "Boltzmann-calibrated",
"state of the art", "zero-shot", any ΔF or ESS number, any `\pending` number.

---

## 2. Title

Requirements: no "calibrate/calibrated"; no unscoped force claim; states the
positive finding *and* its limit; ≤ 2 lines; no "HBFM"/"hierarchical" (ban B-7).

| # | Candidate | Notes |
|---|---|---|
| **T-A** ⭐ | **Ordering Without Scale: Energy-Value Supervision for Bond-Free De Novo 3D Molecular Generation** | **RECOMMENDED.** Names the tension in three words, so the honest limit is the memorable part rather than the buried part. "Energy-value" foreshadows the value-vs-gradient study without asserting anything about forces. No banned word. |
| T-B | Density–Energy Ordering, and Its Limits, in a Bond-Free De Novo 3D Molecular Flow | Safest, least quotable. Use if a reviewer objects that "Ordering Without Scale" over-promises a scale analysis. |
| T-C | A Log-Variance Objective Improves Density–Energy Ordering but Not Density Scale in De Novo 3D Molecular Generation | Most literal; also the most awkward and the longest. Fallback only. |

**Decision: T-A.** Optional subtitle line if the venue permits two lines:
*"…: what a log-variance objective against a universal neural potential does and does not buy."*

**Retired titles (do not resurrect):** "Energies Calibrate, Forces Do Not: …" (both
halves over-claim); "From Structure Emulation to Energy-Calibrated Densities …"
(current `main.tex` line 193 — contains the banned word and asserts the endpoint
N2 refutes).

---

## 3. Contribution bullets (5; each carries its evidence tag)

Tags: **[REAL]** = in §6 numbers table, verified · **[DERIVED]** = arithmetic on
REAL numbers, must be labelled as such · **[PENDING]** = `\pending{}` + inventory row
· **[DISK-UNVERIFIED]** = exists on disk, not yet re-verified by the editor; **may
not be cited in the main text this round**.

1. **A log-variance objective deployed at chemical scale, with its attribution
   stated.** A bond-free 83-element de-novo flow is trained against a universal
   neural potential through a group-conditioned log-variance objective; the
   divergence, its partition-function cancellation and its off-policy estimability
   are prior work, and the contribution is the composition-conditioned deployment
   over `3×10⁴` chemically distinct groups plus the matched value-vs-gradient
   comparison. *Evidence:* protocol constants **[REAL]** (§6.6); citations
   `vargrad`, `nusken2021pathspace`, `richter2024improvedsampling`,
   `sendera2024offpolicy`.

2. **An identifiability theorem that fixes the scope of the claim.** Per-group
   residual constants are synchronised into one composition-level constant **iff**
   the overlap graph of the perturbation groups is connected; with `C` components
   the zero-loss set is a `(C−1)`-parameter family in which every assignment of
   relative basin mass is attained. Our design has no edges, so we claim and
   measure *local ordering* only — and the theorem specifies the experiment that
   would upgrade the claim. *Evidence:* `\Cref{thm:l2g}`, `\Cref{cor:basins}`,
   proofs in `A1_proofs.tex` **[REAL — proof]**.

3. **Energy-value supervision measurably improves local density–energy ordering, on
   a doubly-corrected endpoint.** 93 parents disjoint from the energy term's shard
   pool, reference geometry excluded, both corrections applied simultaneously,
   4/4/3 seeds: `+0.202 ± 0.023 → +0.369 ± 0.023` (`Δ = +0.167`, Welch `t = 5.16`,
   exact permutation `p = 0.029`); a within-parent label-scramble reaches `+0.226`,
   indistinguishable from no physics (`p = 0.486`). *Evidence:* **[REAL]** §6.1,
   `paper/figures/out/primary_endpoint.json` (per-seed values on disk).

4. **The gain is in ordering, not in scale — quantified against an oracle ceiling.**
   The same metric applied to the teacher's own energies gives `r = 0.958`
   (median 0.986), so the energy arm sits at ~45% of achievable; and on a
   scale-sensitive read-out the same model is **not** calibrated: NRV `1.211`
   (reference-included) / `2.024` (reference-excluded) against `1.0` for a constant
   log-density, `T_eff = 2.09 ± 0.41` eV against a 1.0 eV training target, while the
   log-density *dynamic range* does become approximately right (scale-ratio
   `0.56 → 1.09`). This is the paper's most transferable statement: a value-based
   objective buys direction and magnitude of response, not a normalisable density.
   *Evidence:* **[REAL]** §6.2, §6.3 (`CALIBRATION_RESULTS.md`, artefacts under
   `runs/eval_ours/calibration/` and `runs/eval_ours/ceiling_full/`).

5. **A negative result about gradients, and a reproducible failure mode.**
   Off-policy endpoint force matching read through the flow-matching score identity
   was worse than no physics in every configuration we ran (`r = 0.109` against
   `0.223`/`0.126` for the matched control of the *same* smaller round), which we
   scope to that construction — one seed per cell, `n = 30`, a distillation
   confound, and a literature that shows force information can be made to work. And
   single-parent estimation of the energy objective is so high-variance
   (`48 → 974` between consecutive steps) that 3 of 9 seeds diverged to NaN while a
   conventional guard silently zeroed the loss and kept training; parent-batch 8 +
   loss cap + a finite-weight guard gave 2 of 2 stable seeds. *Evidence:* **[REAL]**
   §6.4, §6.5; qualifiers mandatory (B-12, B-20).

*(A sixth bullet on generation quality is **forbidden this round** — see B-21.)*

---

## 4. Claim → evidence map

Every row must map to a number in §6. A claim with no row is deleted, not softened.
Column *Status*: REAL / DERIVED / PENDING / DISK-UNVERIFIED (= not citable now).

| # | Claim (as it may be written) | Evidence | Status | Where |
|---|---|---|---|---|
| C1 | Data-trained de-novo 3D generators are structure emulators, not ensemble models | FM-only arm: `r = 0.074` (all-120, +ref), NRV 1.303, `T_eff` 19.3 eV | REAL | Abs, §1, Tab. 1–2 |
| C2 | The energy term improves local density–energy **ordering** | primary endpoint `0.202 → 0.369`, `Δ=+0.167`, `t=5.16`, perm `p=0.029` | REAL | Abs, §5.2, Tab. 1 |
| C3 | The gain survives both known inflations applied at once | all four cells from one re-scored source: `+0.167` / `+0.178` / `+0.376` / `+0.360` | REAL | §5.1–5.2, Tab. 1 |
| C4 | The gain requires the geometry–energy pairing | scrambled `+0.226` vs control `+0.202`, `t=0.80`, `p=0.486`; true-vs-scrambled `Δ=+0.143`, `p=0.029` | REAL | §5.4 |
| C5 | The scrambling control has two defects that bias **against** us | one of two shards unscrambled; 3rd seed not config-matched (`+0.260` vs `+0.226`/`+0.192`) | REAL | §5.4 |
| C6 | On the permissive metric the scramble ordering inverts, which indicts the permissive metric | scrambled `+0.341` vs control `+0.074`; true-vs-scrambled `p=0.171` | REAL | §5.4 |
| **C7** | **`r = 0.43` is 45% of achievable, not near a limit** | ceiling `r = 0.958` (median 0.986), `0.434/0.958 = 45%`; dropref `0.397/0.932 = 43%` | REAL (ratio DERIVED) | Abs, §5.3, Tab. 1 row |
| **C8** | **The learned density is not on the Boltzmann scale** | NRV `1.211` (+ref) / `2.024` (no-ref) ≥ 1 = constant-density value; ceiling NRV `0.033` | REAL | Abs, §5.3, Tab. 2 |
| **C9** | **The effective temperature is ≈2× the training target** | `T_eff = 2.09 ± 0.41` eV (+ref) / `1.88 ± 0.38` (no-ref); ceiling `1.07`; FM-only `19.3` | REAL | Abs, §5.3, Tab. 2 |
| **C10** | **Even an optimal temperature rescale cannot reach NRV < 1 here — because `r` itself is too low** | `NRV_min = 1−r² = 0.708` (+ref) / `0.736` (no-ref); improves significantly vs control (Wilcoxon `p = 6.7e−10` / `4.7e−7`) | REAL | §5.3 |
| **C11** | **The energy term does fix the log-density *dynamic range*** | scale-ratio `0.559 → 1.093` (+ref), `0.838 → 1.420` (no-ref); slope `0.053 → 0.491`; Wilcoxon `p = 3.4e−19` / `4.7e−19` | REAL | §5.3, Tab. 2 |
| C12 | Direction and scale are different axes, and `r` is blind to the second | identity `slope = r × scale-ratio`; `NRV(kT)` parabola with minimum `1−r²` | DERIVED (identity) | §5.3 (mandated paragraph, §10.3) |
| C13 | The metric is scored against a potential independent of the teacher | GFN2-xTB vs OMol25/eSEN: no shared parameters, data or functional form | REAL (protocol) | §5.1 |
| C14 | Grouping is mandatory; pooling gives a Simpson artefact | pooled `r²` 0.11 vs 0.01; cross-molecule objective `~1e9` | REAL | §5.1 |
| C15 | The objective certifies local shape only; cross-basin mass is unconstrained | `\Cref{thm:l2g}(iii)`, `\Cref{cor:basins}` | REAL (proof) | §4, §5.6 |
| C16 | Off-policy endpoint force matching underperformed **in our construction** | `+0.109` vs `+0.223`/`+0.126` (same round, `n=30`, 1 seed/cell, distillation confound) | REAL + mandatory qualifiers | §5.4 |
| C17 | Two mechanisms are available for that failure; a third is ruled out | amplification `6.7/12.5/32.3`; `\Cref{prop:trap}`, `\Cref{prop:conflict}`; `\Cref{thm:gauge}` explicitly *not* the explanation | REAL (theory) | §4, §5.4 |
| C18 | The energy objective is numerically fragile, and we diagnosed and fixed it | `48 → 974`; 3/9 NaN; guard masked it ~1 GPU-day; 2/2 stable after fix | REAL | §5.6 |
| C19 | Generation quality is not visibly destroyed by the energy term | 43.0/31.0/20.0/100 vs 35.5/23.0/16.0/100, binomial SE ≈ 5 pts at `n=100` | REAL — **no superiority claim** | App. |
| C20 | The evaluation ensemble is not thermally relevant | per-group `std(E_xTB) = 20.7` eV ≈ 21 `kT` at the training `kT` | REAL | §5.6 |
| C21 | The ceiling is high **only because** the ensemble spread is huge | eSEN−xTB relative-energy RMSE 3.0 eV median vs `σ_E ≈ 20.7` eV; projected ceiling collapses to 0.03–0.32 at 0.1–1 eV spreads | REAL (projection = DERIVED, model-based) | §5.6 + App. |
| C22 | NRV on the **primary endpoint parent subset** | *not computed* — current NRV is over all 120 parents | **PENDING (N-1)** | §5.3 footnote |
| C23 | Fixed-RMSD / torsional / normal-mode perturbations remove the radial confound | scripts in place, eval queued | **PENDING (N-2)** | §5.6 |
| C24 | Cross-basin (global-ensemble) behaviour | `ENSEMBLE_RESULTS.md` exists on disk | **DISK-UNVERIFIED — not citable** | §11 |
| C25 | Estimator convergence in ODE steps / probes | `ESTIMATOR_VALIDATION.md` exists on disk | **DISK-UNVERIFIED — not citable, but blocks C2's robustness wording** | §11 |
| C26 | Clean-force arm (no distillation), 5 seeds × 6 cells | P0 training queued | **PENDING (N-3)** | §5.4, §5.6 |
| C27 | External baselines (Symphony / EDM / GeoLDM) | retrains queued | **PENDING (N-4)** | App. |

**Deleted claims (no evidence row exists — remove wherever they still appear):**
composition-level / joint Boltzmann consistency; free-energy read-out; ESS;
tmQM ordering as a main-text result; the 74/26 or 63/37 mechanism split; any
statement that the density is "calibrated"; any statement that generation quality
is better than the control.

---

## 5. Section structure and page budget

### 5.1 Budget (main text ≤ 12 pages; ≈650 words / full page)

| § | Title | Pages | Words | Floats | Must contain |
|---|---|---|---|---|---|
| — | Title + abstract | 0.45 | 200 | — | shape §1.2, 8 sentences |
| 1 | Introduction | 1.30 | 850 | Fig. 1 (+0.40) | P1 stakes · P2 two literatures + emulator framing + why-now + Adjoint-Sampling demarcation · P3 formulation + attribution + cost · **P4 three findings: ordering / ceiling+scale / force** · P5 metric validity + disclosures · P6 the 5 bullets |
| 2 | Related work | 0.70 | 450 | — | 5 buckets; the log-variance-divergence bucket is mandatory and must cede the cancellation |
| 3 | Method | 1.40 | 850 | — | three losses; FFJORD cost; implementation disclosures (clipping, frozen trajectory, no rotation quotient) |
| 4 | Theory | 1.25 | 700 | — | prior-work ¶ · Lem. score + amplification remark · prose summary of trap/conflict · Thm. certificate + Cor. gauge · **Def. overlap graph + Thm. l2g + Cor. basins** · Prop. surrogate · scope ¶ |
| 5.1 | Setup and protocol | 0.75 | 480 | — | primary endpoint definition; both corrections; grouping; independence; seeds/statistics |
| 5.2 | **Ordering result** | 1.20 | 400 | Tab. 1, Fig. 2 (+0.35) | C2, C3; `τ` restatement; **no "calibrated"** |
| 5.3 | **Scale and the oracle ceiling** *(NEW)* | 0.95 | 420 | Tab. 2 | C7–C12 + the mandated direction-vs-scale paragraph (§10.3) |
| 5.4 | Ablation: labels and gradients | 1.10 | 620 | Fig. 3 | C4–C6, C16, C17 with all qualifiers |
| 5.5 | Generation quality (pointer only) | 0.25 | 160 | — | C19, one paragraph, table in appendix |
| 5.6 | Limitations | 1.05 | 700 | — | the **eight** mandatory items of §5.2 below |
| 6 | Conclusion | 0.25 | 160 | — | three-part pitch + the named next experiment (scale/temperature calibration) |
| | **Total** | **≈10.7** | **≈6100** | | leaves ≈1.3 pages of slack for float overflow |

### 5.2 Limitations — the eight mandatory items, in this order

1. **Scale is not calibrated.** NRV ≥ 1 in every arm; `T_eff ≈ 2×` target;
   `NRV_min = 1−r²` shows no temperature rescale fixes it at the present `r`.
2. **45% of the ceiling.** With a caveat that the ceiling is high only because the
   ensemble spread is ~20.7 eV (C21), so the ceiling itself is
   ensemble-dependent and must be re-measured on any new ensemble.
3. **Local only — no cross-basin test.** `\Cref{thm:l2g}` + the named experiment.
4. **`r` certifies shape, not temperature or scale** (affine invariance) — now
   partially remedied by §5.3, and the remaining gap stated.
5. **Radial-distance confound not excluded.** Fixed-RMSD/torsional/normal-mode
   perturbations are the fix; `\pending` status.
6. **Estimator not validated.** ODE-step and probe convergence, repeat-to-repeat
   variance, agreement with exact divergence — none reported (see R8/§11).
7. **Training budget: 30k steps ≈ 0.12 epochs.** Under-training is a live
   alternative explanation for the distance to the ceiling and must be named as
   such, not as an excuse.
8. **Force result is confounded and hypothesis-generating.** Plus the remaining
   known defects (shard leak, unscrambled validation shard, unmatched seed,
   spin multiplicity, narrow baselines) and the numerical fragility.

### 5.3 Float register (amended)

| Float | Content | Change this round |
|---|---|---|
| Fig. 1 `fig:method` | method schematic | panel (e) caption: "certifies local *ordering*", not "calibration" |
| **Tab. 1** `tab:boltzmann` | per-group `r`, 4 endpoint cells, contrasts | **ADD a ceiling row** (teacher-vs-xTB `0.958` / `0.932`); relabel caption "ordering" |
| Fig. 2 `fig:boltzmann` | per-parent scatter + `r_m` distribution | unchanged; caption must not say "calibrated" |
| **Tab. 2** `tab:scale` *(NEW)* | NRV / slope / scale-ratio / `T_eff` / `NRV_min`, arms × {+ref, no-ref}, with a ceiling column | new file `figures/tab2_scale.tex` |
| Fig. 3 `fig:ablation` | ablation bars | **remove the two-round juxtaposition** (R4e); force cells go to their own panel or to the appendix |
| Fig. A1 `fig:mechanism` | 74/26 decomposition | **DELETE file and float** (R4d) |
| Tab. A1–A4 | τ, constants, cells, placeholders | tab:cells gains NRV/`T_eff` per arm; placeholders per §7 |

---

## 6. The numbers table — writers may take numbers from here and nowhere else

Convention: **bold = quotable headline**; every number carries `±` = SEM over seeds
where seeds exist. `kT_train = 1.0` eV throughout. `+ref` = all 9 geometries;
`no-ref` = the 8 displaced geometries only (the honest setting).

### 6.1 Ordering — mean per-group Pearson `r` vs `−E_GFN2-xTB/kT` **[REAL]**

Single source for all four cells: `paper/figures/recompute_disjoint_noreference.py`
→ `paper/figures/out/primary_endpoint.json` (per-seed values on disk; editor
re-derived every mean below from that file).

| Arm | seeds | **disjoint 93, no-ref (PRIMARY)** | disjoint 93, +ref | all 120, no-ref | all 120, +ref |
|---|---|---|---|---|---|
| No physics (FM only) | 4 | **+0.202 ± 0.023** | +0.045 ± 0.006 | +0.218 ± 0.019 | +0.074 ± 0.005 |
| Energy, scrambled labels | 3 | +0.226 ± 0.020 | +0.325 ± 0.037 | +0.254 ± 0.022 | +0.341 ± 0.039 |
| Energy, true labels | 4 | **+0.369 ± 0.023** | +0.420 ± 0.034 | +0.397 ± 0.022 | +0.434 ± 0.032 |
| **Oracle ceiling (teacher energies through the same metric)** | — | **+0.932** [0.891, 0.959] | — | — | **+0.958** [0.921, 0.978] |

| Contrast | PRIMARY | +ref (disjoint) | all 120, no-ref | all 120, +ref |
|---|---|---|---|---|
| energy − no physics | **Δ +0.167, t 5.16, perm 0.029** | Δ +0.376, t 11.0, 0.029 | Δ +0.178, t 6.21, 0.029 | Δ +0.360, t 11.3, 0.029 |
| energy − scrambled | Δ +0.143, t 4.73, perm 0.029 | Δ +0.095, t 1.91, 0.143 | Δ +0.143, t 4.56, 0.029 | Δ +0.093, t 1.85, 0.171 |
| scrambled − no physics | Δ +0.024, t 0.80, **0.486 (n.s.)** | Δ +0.280, t 7.47, 0.029 | Δ +0.036, t 1.23, 0.343 | Δ +0.267, t 6.76, 0.029 |

Ceiling medians: **0.986** (+ref) / 0.978 (no-ref); Spearman 0.941 / 0.923;
98% / 92% of groups have ceiling `r > 0.8`. Ceiling on 1080 geometries / 120 groups,
`esen_sm_conserving_all.pt`, 0 failures.
**Ratios [DERIVED]:** `0.434 / 0.958 = 45%`; `0.397 / 0.932 = 43%`.

### 6.2 Scale — calibration diagnostics **[REAL]**

`NRV = Var_k[log p_θ + βE] / Var_k[βE]` (perfect Boltzmann 0; constant log-density
exactly 1). Seed-aggregated medians, all 120 parents. Source
`CALIBRATION_RESULTS.md` + `runs/eval_ours/calibration/`; A6 rows are 3-seed means
over the two matched seeds and the stabilised seed.

**With reference (all 9 geometries):**

| Arm | seeds | NRV | `NRV_min = 1−r²` | slope (ideal 1) | scale-ratio (ideal 1) | `T_eff` eV (ideal 1.0) | mean `r` |
|---|---|---|---|---|---|---|---|
| A1 FM-only | 4 | **1.303 ± 0.045** | 0.844 ± 0.011 | 0.053 ± 0.009 | 0.559 | **19.3 ± 3.7** | 0.074 ± 0.010 |
| A3 energy | 4 | **1.211 ± 0.361** | 0.708 ± 0.060 | 0.491 ± 0.089 | 1.093 | **2.09 ± 0.41** | 0.434 ± 0.063 |
| A6 scrambled | 3 | 1.062 | 0.797 / 0.738 | 0.227 / 0.325 | 0.707 / 0.794 | 4.42 / 3.08 | 0.341 |
| **ceiling** | — | **0.033** | 0.052 | 0.884 (median 0.934) | — | **1.07** | 0.958 |

**Reference dropped (8 displaced geometries):**

| Arm | seeds | NRV | `NRV_min` | slope | scale-ratio | `T_eff` eV | mean `r` |
|---|---|---|---|---|---|---|---|
| A1 FM-only | 4 | **1.390 ± 0.073** | 0.830 ± 0.021 | 0.151 ± 0.023 | 0.838 | 6.75 ± 1.13 | 0.218 ± 0.037 |
| A3 energy | 4 | **2.024 ± 0.754** | 0.736 ± 0.030 | 0.548 ± 0.107 | 1.420 | **1.88 ± 0.38** | 0.397 ± 0.044 |
| A6 scrambled | 3 | 1.691 | 0.803 / 0.785 | 0.212 / 0.284 | 1.051 / 1.152 | 4.77 / 3.53 | 0.254 |
| **ceiling** | — | **0.089** | 0.094 | 0.767 (median 0.793) | — | 1.26 | 0.932 |

**Paired per-molecule Wilcoxon (energy − FM-only, same 120 molecules):**

| metric | with reference | reference dropped |
|---|---|---|
| NRV | Δ −0.083, **p = 0.87 (n.s.)** | Δ **+0.337, p = 9.2e−7 (WORSE)** |
| `NRV_min` | Δ −0.132, p = 6.7e−10 (better) | Δ −0.065, p = 4.7e−7 (better) |
| slope | Δ +0.394, p = 1.3e−18 | Δ +0.366, p = 7.9e−17 |
| scale-ratio | Δ +0.383, p = 3.4e−19 | Δ +0.471, p = 4.7e−19 |
| Pearson `r` | Δ +0.340, p = 3.5e−19 | Δ +0.210, p = 2.1e−12 |

Fraction of molecules beating the constant-density bar (NRV < 1), energy arm:
**≈42%** (+ref) / **≈27%** (no-ref). NRV is heavy-tailed — **quote medians only**
(one FM-only seed has mean NRV 7349 from a single group).

**Identities (safe to state, [DERIVED]):** `slope = r × scale-ratio`;
`NRV(kT) = a·kT² + 2b·kT + 1` with minimum `1 − r²` at `kT* = −b/a`.

### 6.3 Ceiling context **[REAL]**

| quantity | value |
|---|---|
| eSEN−xTB relative-energy RMSE | 5.11 eV mean / **3.01 eV median** per group |
| per-group `std(E_xTB)` | **20.7 eV** mean, 15.0 median, 75 max (≈21 `kT` at `kT=1` eV) |
| projected ceiling at `σ_E` = 0.1 / 0.5 / 1 / 5 eV | 0.03 / 0.16 / 0.32 / 0.86 **[DERIVED, model-based; label as a projection]** |
| runtime | 1098 s CPU, 0 failures; all 11 wide-eval arms share byte-identical geometries |

### 6.4 Force ablation **[REAL — qualifiers mandatory, B-12/B-20]**

Smaller earlier round: `n = 30` parents, **one seed per physics cell**, force arms
additionally carry an on-policy distillation term (weight 0.1).

| Cell | `r` | may be compared only with |
|---|---|---|
| force only | +0.109 | that round's control seeds +0.223 and +0.126 (mean +0.175) |
| energy only (same round) | +0.420, +0.424 | each other |
| force + energy | +0.294 | same round only; ran at `b_parents=1`, no loss cap (pathological config) |

### 6.5 Generation quality **[REAL — no superiority claim, B-21]**

`n = 100` samples per model, binomial SE ≈ 5 points per cell.

| Model | valid | connected | PoseBusters | uniqueness |
|---|---|---|---|---|
| Energy arm (`abl_a3_energy_only`) | 43.0 | 31.0 | 20.0 | 100 |
| Matched bond-free FM control (mean of 2 seeds 41.0/30.0 on validity) | 35.5 | 23.0 | 16.0 | 100 |
| Zatom-1 (author-reported, 80M, ≈400 GPU-h) | 30.4 | 17.0 | 15.1 | 93.5 |

Permitted sentence: *"we do not observe energy supervision visibly degrading
generation quality"*. Forbidden: any claim of improvement over our own control.

### 6.6 Protocol and configuration constants **[REAL]**

| item | value |
|---|---|
| corpus / elements | OMol25 4M, 83 elements, bond-free, no bond supervision |
| `λ₁, λ₂, λ₃` | 0, `3×10⁻⁵`, 0 (the reported model uses **no force term at all**) |
| `kT` (training) | 1.0 eV |
| probe times `t*` | 0.85 / 0.92 / 0.97; score amplification 6.7 / 12.5 / 32.3 |
| FFJORD | 4 ODE steps + 2 Hutchinson probes (training); **12 steps + 2 probes (eval)** |
| training budget | 30 000 optimiser steps, batch 8 × accum 2 ≈ **0.12 epochs** |
| conditioning groups | `3×10⁴` |
| training shards | `K = 5` per parent (reference + 4 displacements) at σ ∈ {0.03,0.06,0.10,0.20,0.40} Å |
| evaluation groups | 1 reference + **8** displacements at σ = 0.15 Å ⇒ RMSD ≈ 0.25 Å (`σ√3`), within-group RMSD CV **8.1%** |
| energy-term parent coverage | parents ≤ 50 atoms; evaluation parents up to 200 atoms |
| within-parent energy spread | 77.6 kcal/mol in both true and scrambled arms |
| stability | single-parent loss jump `48 → 974`; 3/9 seeds NaN; fixed config (b_parents 8, cap, FiniteWeightGuard) → 2/2 stable |
| pooled (ungrouped) `r²` | 0.11 energy / 0.01 control; cross-molecule objective magnitude `~1e9` |

### 6.7 Not citable this round — DISK-UNVERIFIED

These exist as documents on disk but were **not** in the editor-verified set for this
round, and the round brief lists them as "queued, no results". They may not appear
as numbers in the manuscript until §11 is resolved.

| source | headline it would carry | why blocked |
|---|---|---|
| `ENSEMBLE_RESULTS.md` (P1) | between-basin free-energy error 32–92× the true spread; energy term gives **no** cross-basin transfer | status conflict with the round brief; `n = 12` systems; the doc itself says the estimator is not converged |
| `ESTIMATOR_VALIDATION.md` (P6) | `log p_θ` does not converge in `n_ode_steps`; the arm gap falls +0.45 → +0.18 → ≈0 from 4 → 12 → 48 steps | **potentially headline-invalidating**; status conflict; `n = 10` molecules, 1 seed |

### 6.8 Pending — every one needs `\pending{}` + an inventory row

| ID | quantity | true current status |
|---|---|---|
| N-1 | NRV / `T_eff` restricted to the 93 disjoint parents | not computed; current NRV is over all 120 |
| N-2 | fixed-RMSD / torsional / bond-angle / normal-mode perturbation results | scripts in place, eval queued |
| N-3 | P0 grid: 6 cells × 5 seeds (incl. clean force arm, `Var[log p]` flat-energy cell) | training queued |
| N-4 | Symphony / EDM / GeoLDM retrains | queued / env not ready |
| N-5 | `n = 240` scrambling contrast | measured at `n = 120`: Δ +0.092, `t` 1.77, `p` 0.147 (**n.s.**) |

---

## 7. PLACEHOLDER INVENTORY (goes verbatim into `main.tex` and Appendix B)

| ID | Section / float | Semantics | Placeholder | True current status | Source when it lands |
|---|---|---|---|---|---|
| **Q1** | §5.3 footnote, Tab. 2 | NRV / `T_eff` on the primary endpoint subset | `NRV \pending{—}`, `T_eff \pending{—}` | **not computed**; §6.2 values are over all 120 parents | `scripts/analyze_calibration.py` restricted to `primary_endpoint.json:disjoint_group_ids`, `--drop_reference` |
| **Q2** | §5.6 (radial confound) | fixed-RMSD shell result | `\pending{in progress}` | perturbation generator verified (`std = 1.3e−16`); no eval result | P3 eval jobs |
| **Q3** | §5.4, §5.6 | clean force arm (no distillation), ≥3 seeds | `\pending{in progress}` | P0 grid queued; the only force evidence is 1 seed with a distillation confound | P0 cells E/F |
| **Q4** | §5.6, App. generation table | Symphony / EDM / GeoLDM rows | `\pending{…}` ×4 each | retrains queued / env not ready | P1–P2 baseline jobs |
| **Q5** | §5.4, Tab. 1 last row, Fig. 3(b) | `n = \pending{240}` scrambling contrast, `Δ \pending{+0.13}`, `t \pending{3.9}`, `p < \pending{0.05}` | as printed | measured at `n = 120`: Δ +0.092, `t` 1.77, `p` 0.147 — **not significant** | `runs/eval_ours/wide_a6_*` + `wide_a3_*` at `n = 240` |
| **Q6** | §5.6 (estimator) | estimator-convergence statement | `\pending{not yet measured}` **or** delete the sentence | a study exists on disk but is DISK-UNVERIFIED (§11); until resolved the manuscript must say the check has not been reported | editor re-verification of `ESTIMATOR_VALIDATION.md` |
| **Q7** | §5.6 (cross-basin) | cross-basin occupancy | prose only, **no number** | a study exists on disk but is DISK-UNVERIFIED | editor re-verification of `ENSEMBLE_RESULTS.md` |

**Retired from the v1 inventory:** P-F1 (Fig. 2 points are now re-scored real
records) and P-F3 (the 74/26 figure — the whole float is deleted, R4d).

**Camera-ready gate:** `grep -c pending main.tex sections/*.tex figures/*.tex` = 0,
and `figures/make_experiment_figures.py --strict-real` exits 0.

---

## 8. Wording bans — 不可写 / 应写

v1 bans **B-1 … B-15 remain in force** (see `SPINE.md` §9). B-10 is *superseded* by
B-18 below, which is stronger. New bans this round:

| # | 不可写 (never write) | 应写 (write instead) |
|---|---|---|
| **B-16** | "calibrated density"; "Boltzmann-calibrated"; "energy-calibrated"; "local density–energy **calibration**"; "the model is calibrated to the Boltzmann weight" — anywhere, including titles, captions, theorem names and file headers | "local density–energy **ordering**"; "local energy-**ordering** consistency"; "the density's ordering tracks the Boltzmann weight". Reserve the word *calibration* for the §5.3 metric block, where it appears **only** in the negative: "on a scale-sensitive read-out the density is *not* calibrated (NRV ≥ 1, `T_eff ≈ 2.09` eV vs 1.0 eV)". |
| **B-17** | quoting `r = 0.434` (or any `r`) without the scale companion in the same paragraph | every `r` statement is followed, in the same paragraph, by NRV and/or `T_eff`: "`r` rises from 0.074 to 0.434, while NRV stays at 1.211 — at or above the value a constant log-density attains — and `T_eff` is 2.09 eV against a 1.0 eV target." |
| **B-18** *(replaces B-10)* | "approaches the ceiling"; "near-perfect"; "the ceiling is unmeasured"; "bounded above by an unknown teacher–evaluator agreement" | "The ceiling **is** measured: the teacher's own energies score `r = 0.958` on this metric, so our 0.434 is ~45% of achievable and the residual is unlearned model capacity, not oracle noise." Always add the ensemble caveat: "the ceiling is this high only because the perturbation ensemble spans ≈20.7 eV." |
| **B-19** | "the result is robust to the likelihood estimator"; "estimator noise only attenuates `r`, so our number is a lower bound" *stated as a settled matter* | "The reported `log p_θ` is a 12-step first-order discretisation with a 2-probe stochastic trace. We have not reported convergence in ODE steps or probes, repeat-to-repeat variance, or agreement with an exact divergence; the attenuation argument is an argument, not a measurement." (R8) |
| **B-20** | "forces do not work"; "force supervision fails"; "energies calibrate, forces do not" | "*Naive off-policy endpoint* force matching, read through the flow-matching score identity, underperformed density-value supervision **in our non-equilibrium-data setting**, on one seed per cell at `n = 30` with a distillation confound. Force information demonstrably can be made to work (`psm`, `driftingboltzmann`); we treat our observation as hypothesis-generating." |
| **B-21** | "the energy term improves generation quality"; bolding either of our two rows in the generation table | "At `n = 100` the binomial standard error is ≈5 points, larger than the gap between our arms; we observe no evidence that energy supervision degrades generation quality, and claim no improvement." |
| **B-22** | any composition-level / joint `p(c,x)` Boltzmann claim; any ΔF or free-energy read-out; any ESS number | "Comparing absolute electronic energies across different elements and atom counts requires a chemical-potential or grand-canonical convention that we neither adopt nor test; this paper is at fixed composition." |
| **B-23** | "we measure Boltzmann fidelity"; "thermodynamically accurate samples" | "We measure the *within-cloud ordering* of `log p_θ` against an independent potential, on an ensemble whose per-group energy spread is ≈21 `kT`, i.e. not a thermally accessible ensemble." |
| **B-24** | attributing the residual gap solely to under-training, or solely to model capacity | "Two explanations remain open for the 55% of ceiling we do not reach: under-training (0.12 epochs) and the objective's own limits. Nothing here separates them." |
| **B-25** | mean NRV; any NRV number without saying whether the reference geometry is in | "median NRV (heavy-tailed; the plain mean is dominated by groups with `Var(E) ≈ 0`)", and always tag `+ref` / `no-ref`. |
| **B-26** | citing any number from `ENSEMBLE_RESULTS.md` or `ESTIMATOR_VALIDATION.md` this round | prose statement that the check is not reported here (Q6/Q7), pending §11 resolution. |

---

## 9. Deletions required this round (R4), file by file

| # | What | File / location | Status |
|---|---|---|---|
| **R4a** | Joint / composition-level Boltzmann extension and `\Cref{cor:labor}` ("division of labour") | `sections/03_theory.tex` (header note lines 36–41 record it), `sections/A1_proofs.tex` (`app:gauge` region) | **verify**: `cor:labor` is deleted from the theory section; confirm no surviving statement or proof in `A1_proofs.tex`, and that `\Cref{thm:gauge}` is retained *only* as "what gradients cannot see", with the chemical-potential disclaimer |
| **R4b** | Free-energy read-out and ESS claims | `03_theory.tex` (`thm:certificate`(ii) reading), `A1_proofs.tex`, `A2_details.tex` | **verify**: `grep -niE "free energ|ESS|effective sample"` across `sections/` must return only limitation-style or cited-literature sentences |
| **R4c** | tmQM comparison out of the main text | already moved to `app:tmqm` (`A2_details.tex`); intro P7 already deleted (`01_intro.tex` line 202) | **done — verify** no tmQM number remains in `01_intro.tex`, `04_experiments.tex`, `05_conclusion.tex`, and that `app:tmqm` carries: singlet-scoring defect, single seed, different run family |
| **R4d** | 74/26 and 63/37 mechanism decomposition + Figure A1 | subsubsection deleted in `04_experiments.tex` (line 303 comment); **`figures/fig4_mechanism.tex`, `figures/out/fig4_mechanism_decomposition.{pdf,png}` and the `fig:mechanism` register entry in `main.tex` lines 45 and 82 must also go** | **NOT DONE — do this round** |
| **R4e** | Figure 3's juxtaposition of two non-comparable rounds | `figures/fig3_ablation.tex` + its generator in `figures/make_experiment_figures.py` | **NOT DONE — do this round.** Either drop the force panel from Fig. 3 entirely (preferred) or place it in a separately captioned appendix figure whose caption states `n = 30`, 1 seed/cell, distillation confound, and "not comparable with Table 1" |
| **R4f** *(new)* | The title on `main.tex` line 193 and every "calibrat*" string | `main.tex`, all `sections/*.tex`, all `figures/*.tex` | **do this round**; `grep -nic calibrat` should fall to the §5.3 block + the explicit negative statements only |

---

## 10. Where N1 and N2 go in the manuscript

### 10.1 N1 — the oracle ceiling

* **Table 1: a fourth data row**, `Oracle ceiling (eSEN teacher energies, same
  metric)` = `+0.958` (+ref) / `+0.932` (no-ref), typeset in a separate block below
  a `\midrule` so no reader mistakes it for an arm. Caption must say it is *not* a
  model: it is the teacher's own energies pushed through the identical pipeline, and
  it is what a student that reproduced the teacher exactly would score.
* **§5.3 opening**, three sentences: the question ("is 0.43 close to the maximum?"),
  the answer (0.958 mean / 0.986 median, 98% of groups above 0.8), the ratio (45%).
* **Limitations item 2** and **App.**: the σ_E projection table (C21) and the
  instruction that any new ensemble needs its own directly-measured ceiling.
* **Abstract sentence 5.** **Intro P4 finding (ii).**
* *Not* a separate top-level section — it is one column-width fact and belongs
  wherever `r` is reported (B-18).

### 10.2 N2 — NRV as a co-primary metric

* **Table 2 (NEW)** `tab:scale`, immediately after Table 1, arms × {+ref, no-ref},
  columns NRV / `NRV_min` / slope / scale-ratio / `T_eff`, with the ceiling row.
  Caption defines NRV and states the two anchors: **0 = perfect Boltzmann,
  1 = constant log-density**.
* **§5.1** must introduce NRV alongside `r` as a *co-primary* endpoint, not as a
  post-hoc diagnostic — otherwise it reads as a metric found after the fact.
* **Abstract sentence 6**, **Intro P4 finding (ii)**, **Conclusion**.
* Every quotation of `r` carries the NRV companion (B-17).

### 10.3 The mandated "direction vs. scale" paragraph (§5.3, drafted; edit for style only)

> Correlation and calibration disagree here, and the disagreement is informative
> rather than contradictory. Pearson `r` is invariant to positive affine maps of
> either variable, so it scores only the *direction* of the density's response to
> energy; the normalised residual variance charges for the *magnitude* as well,
> and it is anchored: a model emitting a constant log-density scores exactly 1.
> The two decompose cleanly, since the regression slope factorises as
> `slope = r × (sd log p_θ / sd βE)`. On that decomposition the energy term does two
> things and fails at a third. It improves direction (`r`: 0.074 → 0.434) and it
> fixes the response magnitude (scale-ratio 0.56 → 1.09; slope 0.053 → 0.491;
> effective temperature 19.3 eV → 2.09 eV against a 1.0 eV target). But it also
> injects log-density variation of which only `r² ≈ 19%` is energy-aligned, and NRV
> charges for the remaining 81%: median NRV stays at 1.211 with the reference
> geometry and rises to 2.024 without it, so on this scale-sensitive read-out the
> density is not better than a constant, and the paired per-molecule comparison
> against the physics-free control is not significant (`p = 0.87`) with the
> reference and significantly worse (`p = 9.2e−7`) without it. The best NRV any
> temperature rescaling could buy is `1 − r² = 0.708`, which *does* improve
> significantly over the control's 0.844 (`p = 6.7e−10`) — the aligned signal is
> real — but it is still short of 1 only because `r` is short of 1. The honest
> summary is therefore that a value-based objective buys ordering and response
> magnitude, not a normalisable density, and that closing the gap needs either a
> much larger `r` or an explicit scale/temperature term. That is the next experiment
> this paper names.

### 10.4 One-line summary to reuse verbatim (intro, conclusion)

> *Ordering improves, scale does not:* `r` moves 0.074 → 0.434 (45% of the 0.958
> achievable), while NRV stays at 1.211 — the value a constant log-density scores —
> and the effective temperature is 2.09 eV against a 1.0 eV target.

---

## 11. Open conflicts the editor-in-chief must resolve before drafting §5.6

1. **P1 / P6 status conflict (blocking for the limitations section).** The round
   brief lists P1 (global ensemble) and P6 (estimator validation) as "queued, no
   results". Both documents exist on disk, dated 2026-08-03, with results:
   * `ENSEMBLE_RESULTS.md` — 12 systems × 4 arms; between-basin free-energy error
     32–92× the true spread; the energy term's local win **does not transfer**
     between basins (`p = 0.22–0.35`) and is *worse* on one calibration metric
     (`p = 0.005`).
   * `ESTIMATOR_VALIDATION.md` — the Hutchinson trace is clean (unbiased,
     geometry-uncorrelated, `r` flat in probe count, shuffled null ≈ 0), **but**
     `log p_θ` does not converge in `n_ode_steps`, and the energy-vs-control gap
     falls +0.45 (4 steps) → +0.18 (12, our setting) → ≈0 (48).

   The second is **potentially headline-invalidating** and cannot be left
   unaddressed: if it survives re-verification, C2's wording must become
   "at the evaluation resolution we use (12 ODE steps)", the limitation must be
   promoted to the top of §5.6, and the ODE integration scheme must be fixed
   (integrate to a fixed `t = 1 − ε` independent of `n`) before any resubmission.
   **Until the editor re-verifies both, ban B-26 applies and no number from either
   document enters the manuscript.** Recommended action: re-verify P6 first — it is
   cheap (`--analyze_only`, no GPU) and it gates everything else.

2. **NRV is computed on the wrong parent set (Q1).** All §6.2 numbers cover the 120
   parents; the primary endpoint is the 93 disjoint ones. Either recompute
   (`analyze_calibration.py` restricted to `primary_endpoint.json:disjoint_group_ids`
   with `--drop_reference`) or footnote the mismatch explicitly in Table 2. Do not
   let Table 1 and Table 2 silently describe different populations.

3. **Resolved this round — no `\pending` needed.** The brief flags the joint
   "disjoint + reference-excluded" primary endpoint as possibly uncomputed. It **is**
   computed: `paper/figures/out/primary_endpoint.json` carries per-seed values for
   all 11 arms in all four cells, and the editor re-derived every mean in §6.1 from
   it (FM-only 0.2020, energy 0.3693, scrambled 0.2260). Write these in black.

4. **A6's seed heterogeneity.** The three scrambled seeds are not
   configuration-matched (two at `M=4`/cap 3000, one stabilised at `M=8`/cap 1500,
   and the odd one is the highest). Any A6 aggregate in §6.1/§6.2 must carry that
   note wherever it is quoted (already in `04_experiments.tex` §5.4; add it to the
   Table 2 caption too).

---

## 12. Mechanical checklist for every section writer

1. Every number came from §6, verbatim, with its `±` and its `+ref` / `no-ref` tag.
2. Every `r` has an NRV or `T_eff` companion in the same paragraph (B-17).
3. Every `r` is readable against 0.958 (B-18).
4. `grep -nic calibrat` on your file returns 0, except inside the §5.3 block where
   it appears only in the negative (B-16).
5. No number from `ENSEMBLE_RESULTS.md` / `ESTIMATOR_VALIDATION.md` (B-26).
6. Every not-yet-real number is inside `\pending{}` and has a §7 row.
7. No superlative, no "first", no cross-round comparison (B-2, B-12).
8. Force sentences carry all four qualifiers: off-policy, endpoint, `n = 30`,
   distillation confound (B-20).
9. Theorem references by `\Cref{label}`, never by number.
10. Page budget checked against §5.1 after every compile
    (`build.sh` prints the main-text page count).
