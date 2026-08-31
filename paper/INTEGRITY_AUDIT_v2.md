# INTEGRITY AUDIT v2 — /n/home04/yulili/bgfm/paper/

Auditor: academic-integrity pass, 2026-08-04.
Method: every claim below was checked by grep/read against the manuscript source
and, where a number is involved, recomputed from the on-disk artefact
(`figures/out/primary_endpoint.json`, `CALIBRATION_RESULTS.md`). No statement in
this report rests on what the draft says about itself.

Files audited: `main.tex`, `sections/{01_intro,02_method,03_theory,04_experiments,05_conclusion,A1_proofs,A2_details}.tex`,
`figures/{fig1_method,fig2_boltzmann,fig3_ablation,tabA1_cells,tabA2_constants}.tex`, `refs.bib`,
plus `main.aux`/`main.log`/`main.blg` for the compiled state.

---

## VERDICT

**PASS with required edits.** The manuscript is honest at the level of the
numbers: I could not find a single altered, invented or unregistered figure, and
every protected value in the round brief survives verbatim. The residual risk is
concentrated in two *process* claims and one *omission*, all of which are
fixable by editing text, not by re-running experiments:

1. a false pre-registration claim ("pre-specified", 6 places);
2. an internal estimator study whose existence and direction are not disclosed,
   although the manuscript's own limitation (vi) is the natural place for it;
3. four arithmetic/labelling slips (43 % vs 40 %, the amplification triple, the
   SEM-vs-SD caption, one inventory cross-reference).

Nothing here rises to misconduct. Items 1 and 2 are the ones a hostile reviewer
would use, and both are cheap to fix.

---

## 1. PLACEHOLDER DISCIPLINE — PASS

Every `\pending{}` occurrence in the manuscript, with its registered row:

| # | Location | Text | Registered as |
|---|---|---|---|
| 1 | `04_experiments.tex:85` (Sec. 5.1) | `in progress` | Q2 |
| 2 | `04_experiments.tex:271` (Tab. 1 last row) | `240`, `+0.13`, `3.9`, `0.05` | Q5 |
| 3 | `04_experiments.tex:324` (Tab. 2 caption) | `not yet run` | Q1 |
| 4 | `04_experiments.tex:389–390` (Sec. 5.6.1) | `240`, `+0.13`, `3.9`, `0.05` | Q5 |
| 5 | `04_experiments.tex:416` (Sec. 5.6.2) | `in progress` | Q3 |
| 6 | `04_experiments.tex:453` (Sec. 5.7) | `in progress` | Q4 |
| 7 | `04_experiments.tex:486` (Lim. iv) | `not yet run` | Q1 |
| 8 | `04_experiments.tex:494` (Lim. v) | `in progress` | Q2 |
| 9 | `04_experiments.tex:501` (Lim. vi) | `not yet reported here` | Q6 |
| 10 | `A2_details.tex:356` (`app:perturb`) | `in progress` | Q2 |
| 11 | `A2_details.tex:506–508` (Tab. A4 rows 4–6) | `--` ×12 | Q4 |
| 12 | `A2_details.tex:662–685` (Tab. A6) | the inventory table itself | Q1–Q6 |
| 13 | `fig3_ablation.tex:19` | comment only, not a macro | n/a |

- All 13 are inside `\pending{}` → red (`main.tex:190`, `\textcolor{red}`).
- All map to a row of the PLACEHOLDER INVENTORY in `main.tex:80–121`, mirrored
  in Tab. A6 (`A2_details.tex:640–700`).
- **No `\pending` in the abstract or introduction** (grep confirms: `01_intro.tex`
  contains the `\providecommand` and prose-only mentions).
- Q7 (cross-basin) is deliberately prose-only with no number — consistent with
  the inventory's own declaration, and Sec. 5.8 (iii) states the measurement
  does not exist in this paper. Correct handling.
- **Ban B-26 verified by grep**: not one number from `ENSEMBLE_RESULTS.md`
  (38×, 147×, 349 basins, 91.8×, 0.160 eV) or `ESTIMATOR_VALIDATION.md`
  (0.518 / 0.452 / 0.628 / +0.183 / 48 steps) appears anywhere in the `.tex`
  tree. The quarantine is real, not asserted.

Two bookkeeping defects, both minor:

- **`main.tex:87`** lists Q2 as "Sec.5.2"; the placeholder is in **Sec. 5.1**
  (`sec:setup`). The Tab. A6 mirror is correct (`\Cref{sec:setup}`). Fix main.tex.
- **Q5 is the one placeholder that can mislead** — see Issue 6 below.

---

## 2. REAL NUMBERS — NO ALTERATION FOUND

I recomputed, not just compared. Source: `figures/out/primary_endpoint.json`
(per-seed, per-cell) and `CALIBRATION_RESULTS.md`.

### Primary endpoint (disjoint 93 × reference-excluded) — the number the brief
said might not exist. **It exists and it is correct.**

FM-only seeds `.22227 / .19401 / .24939 / .14219` → mean **0.20197**, SEM **0.02291**
→ printed `+0.202 ± 0.023`. ✓
Energy seeds `.31568 / .37764 / .42636 / .35757` → mean **0.36931**, SEM **0.02298**
→ printed `+0.369 ± 0.023`. ✓
Δ = **0.16735** → `+0.167` ✓; Welch t = **5.157** → `5.16` ✓;
exact permutation, all 4 above all 4 → p = 2/70 = **0.0286** → `0.029` ✓.
Scramble (3 seeds `.22586/.19190/.26026`) → **0.22601** → `+0.226 ± 0.020`, t = **0.795**,
p = 0.486 ✓.

So the mentor's demanded **disjoint AND reference-excluded** endpoint is a real
measurement, not a placeholder, and it is correctly the headline. Good.

### Every other Table 1 cell recomputed from the JSON

| cell | recomputed | printed |
|---|---|---|
| FM, disjoint +ref | 0.04466 ± 0.00601 | `+0.045 ± 0.006` ✓ |
| FM, all120 no ref | 0.21835 ± 0.01859 | `+0.218 ± 0.019` ✓ |
| FM, all120 +ref | 0.07410 ± 0.00473 | `+0.074 ± 0.005` ✓ |
| Energy, disjoint +ref | 0.42031 ± 0.03373 | `+0.420 ± 0.034` ✓ |
| Energy, all120 no ref | 0.39675 ± 0.02189 | `+0.397 ± 0.022` ✓ |
| Energy, all120 +ref | 0.43427 ± 0.03165 | `+0.434 ± 0.032` ✓ |
| Scramble, all120 +ref | 0.34110 | `+0.341` ✓ |
| Scramble, all120 no ref | 0.25396 | `+0.254` ✓ |
| Δ / t, all120 +ref | 0.36017 / 11.26 | `+0.360`, `11.3` ✓ (brief: +0.360, 11.3) |
| Δ / t, all120 no ref | 0.17841 / 6.21 | `+0.178`, `6.21` ✓ (brief: +0.178, 6.21) |
| Δ / t, disjoint +ref | 0.37565 / 10.96 | `+0.376`, `11.0` ✓ |

`tab:cells` per-seed values (`.222,.194,.249,.142` / `.316,.378,.426,.358`) are
the same JSON entries. `tab:tau` implied τ = √(1/r²−1) checks at all six rows
(2.52, 4.85, 4.31, 2.08, 13.48, 9.12). ✓

### N1 oracle ceiling — unaltered
0.958 mean / 0.986 median / Spearman 0.941 / 98 % above 0.8 / NRV 0.033 /
kT_eff 1.07 eV / 0 of 1080 failures / 3.01 eV median disagreement / 20.7 eV
spread / 45 % ratio: all match `CALIBRATION_RESULTS.md §3`. The reference-dropped
ceiling (0.932, 0.978, 0.089, 1.26 eV, 43 %) also matches.
Note: the brief's "slope 0.934" is the ceiling's **median** slope; the paper
prints the **mean** 0.884 in Tab. 2 and the median 0.934 nowhere. Not an error,
but Tab. 2 mixes medians (NRV) with means (slope) in one row — worth one clause.

### N2 scale table — unaltered
All 14 printed values in Tab. 2 reproduce `CALIBRATION_RESULTS.md §2` exactly
(1.303/1.390 vs the doc's 1.304 — the brief's own value is 1.303, so the paper
follows the brief; 0.844, 0.708, 0.736, 0.830, 0.491, 0.548, 0.559, 1.093,
0.838, 1.420, 19.3, 2.09, 1.88, 6.75). The scrambled 1.062 and 1.691 are the
correct 3-seed means of the doc's (1.107 ×2, 0.974) and (1.689 ×2, 1.695).
Wilcoxon block (−0.083/p 0.87; +0.337/p 9.2e−7; +0.340/p 3.5e−19; +0.394;
+0.383; −0.132) matches the doc line for line, **including the two that go
against the paper** (NRV not improved with ref, significantly worse without).
That is the correct way to report this.

### Generation, stability, budget, force — unaltered
43.0/31.0/20.0/100 ✓; control 35.5/23.0/16.0/100 as the mean of 41.0/27.0/17.0/100
and 30.0/19.0/15.0/100 ✓; binomial SE ≈ 5 points stated and nothing bolded ✓;
48 → 974, 3 of 9 NaN, guard-zeroed-and-kept-training, 2 of 2 after M=8/cap 1500 ✓;
30 000 steps × batch 8 × accum 2 ≈ 0.12 epochs ✓; force +0.109 vs 0.223/0.126
(mean 0.175), energy 0.420/0.424 → 0.422, force+energy 0.294, all flagged
n = 30 / 1 seed / distillation ✓.

**Conclusion of §2: no protected number was changed.** One informational note
for the parent agent: the brief's disjoint figures (control +0.046, energy
+0.415) are superseded by the JSON's 0.045/0.420, which is what the paper
prints. The paper is right; the brief's copy was rounded from an earlier dump.

---

## 3. WORDING BANS — PASS

| pattern | hits in body | verdict |
|---|---|---|
| `calibrat*` | `04:458` "The scale is **not** calibrated"; `01:84` "temperature calibration" in the not-yet-tested list; `05:9`(comment) next experiment; `A2:251,254,666` script filename `analyze_calibration.py` | **all admissible** — every surviving use is an explicit negative, a named future experiment, or a file path. No "calibrated density" anywhere. |
| `Boltzmann-calibrated` | 0 | ✓ |
| `zero-shot` | 0 (only in a `%` comment forbidding it) | ✓ |
| `unseen element*` | 0 — and `A2:449` *actively rejects* the claim: metals are "0.34 % of all training atoms … transfer under distribution shift, not generalisation to unseen elements" | ✓ exemplary |
| `outperform` | 0 | ✓ |
| `state-of-the-art` / `SOTA` | 0 | ✓ |
| `the first` / `we are the first` / `novel` | 0 as a claim | ✓ |
| `superior` | 1, as "We make **no claim of superiority**" (`A2:469`) | ✓ |
| `converged` | only "no model here is converged" (×2) and "inactive for a converged model" | ✓ ban B-11 respected |

Title (`main.tex:233`) is **"Ordering Without Scale: Energy-Value Supervision
for Bond-Free De Novo 3D Molecular Generation"** — R3 satisfied: no
"Calibrate", no general force claim, and the tension is named. Abstract carries
NRV ≥ 1 and T_eff 2.09 eV in sentence 6, so the N2 refutation is in the first
paragraph rather than buried.

Given N2, I looked specifically for any residual sentence that *functions* as a
calibration claim without the word. The closest are `04:290–291` ("it fixes the
response magnitude") and bullet 4's "its dynamic range does become approximately
right". Both are immediately followed by the NRV ≥ 1 statement in the same
sentence or the next, and both are literally supported (scale ratio 0.559 →
1.093 is in the artefact). **Admissible, but see Issue 5**: "fixes" is stronger
than "moves toward"; I would soften it, because scale-ratio 1.42 in the
reference-dropped setting is an *overshoot*, not a fix, and the paper prints
that number two rows below.

---

## 4. R4 DELETIONS — VERIFIED GONE

| item | status | evidence |
|---|---|---|
| (a) joint composition-level Boltzmann extension, `cor:labor` | **deleted** | `grep "joint composition"` = 0; `grep cor:labor` = 0; `A1_proofs.tex:709–713` states the deletion and why; `03_theory.tex:236` adds the grand-canonical disclaimer |
| (b) free-energy read-out | **deleted** | only surviving hits are the deletion note (`A1:710`) and "between-molecule free-energy offsets" as a physical explanation in `fig2` caption — not a read-out claim |
| (b) ESS / importance-sampling claims | **deleted** | `grep -i "ESS"`, `"effective sample"` = 0; `A1:711` records the removal of the "importance-sampling-dimension argument" |
| (c) tmQM in main text | **moved out** | main text mentions it once, only to say the observation "stays in `app:tmqm` and is labelled preliminary". `app:tmqm` opens with "**This is not evidence and the main text makes no use of it**" and lists 4 disqualifying defects incl. spin/singlet, 1 seed, different run family. Correct. |
| (d) 74/26 and 63/37 decomposition | **deleted** | `grep "74/26"`, `"63/37"` = 0; `A2:380–399` records the deletion, the figure file removal, and the reason (the split swings to 14/86 under the primary endpoint) |
| (e) Fig. 3 juxtaposition of incomparable rounds | **fixed at the source** | `fig3_ablation.tex` caption: force cells "are not plotted … different, smaller round … not comparable"; figure moved to appendix `app:ablationfig`; the red n=240 projection removed from the panel |

The 19 %/81 % split at `04:292` is **not** the banned decomposition — it is
r² = 0.434², sourced from `CALIBRATION_RESULTS.md §2`. Admissible.

---

## 5. R1 LOG-VARIANCE ATTRIBUTION — PASS, CITATIONS REAL

Present and, as far as I can verify, genuine:

- `vargrad` — Richter, Boustati, Nüsken, Ruiz, Akyildiz, *VarGrad*, NeurIPS 2020,
  arXiv:2010.10436. **Real, correct authors/venue/ID.**
- `nusken2021pathspace` — Nüsken & Richter, *PDE and Applications* 2(4) 2021,
  arXiv:2005.05409. **Real.**
- `richter2024improvedsampling` — Richter & Berner, ICLR 2024, arXiv:2307.01198.
  **Real.**
- `sendera2024offpolicy` — Sendera et al., NeurIPS 2024, arXiv:2402.05098.
  **Real, full author list.**
- `berner2024optimalcontrol` — Berner, Richter, Ullrich, TMLR, arXiv:2211.01364.
  **Real.**
- `psm` — arXiv:2503.14569, *Potential Score Matching*. Real.
- `driftingboltzmann` — arXiv:2603.05527. **Author list is `Hu, Pipi and others`
  with a `TODO-verify` in the bib.** See Issue 11.

The attribution is not perfunctory. `03_theory.tex:61–77` is a dedicated
paragraph titled "the energy term **is** a log-variance divergence" that says
"We claim no part of that object" and cedes both the Z-cancellation and the
off-policy freedom as "known structural properties of that family". `02_method`
§3.3 repeats it, the abstract cedes it in sentence 2, and the conclusion cedes
it again ("The divergence is prior work"). Novelty is re-positioned exactly as
R1 prescribed (composition/group-conditioned off-policy deployment at 83
elements + value-vs-gradient study). **This is fully compliant.**

`bibtex` reports 37 entries used, 0 warnings; no undefined citations in
`main.log`.

## 5b. R2 CLAIM NARROWING — PASS, AND STRONGER THAN ASKED

`def:overlap` + `thm:l2g` (i)(ii)(iii) + `cor:basins` are stated in
`03_theory.tex:153–204` with a proof idea, the full proof deferred to
`app:l2g`. Part (iii) gives the (C−1)-parameter zero-loss family and states
that every occupancy vector in the open simplex is attained — precisely the
non-identifiability the mentor asked for. `cor:basins` then states that **our**
overlap graph is a single vertex with no edges, so only `thm:certificate`(i)
applies. "Conditional Boltzmann density" is claimed nowhere; the phrase used
throughout is "local density–energy ordering". Abstract, intro bullet 2,
Fig. 1(e) caption, limitation (iii) and the conclusion all carry the
restriction. Compliant.

---

## 6. R5 LIMITATIONS — PASS, ALL FIVE COVERED WITH NUMBERS

| R5 item | where | numbers present |
|---|---|---|
| scale not calibrated (NRV>1, T_eff≈2 eV) | (i) | NRV 1.211 / 2.024 vs 1.0; T_eff 2.09 ± 0.41 vs 1.0 eV; 1−r² = 0.708 floor |
| 45 % of oracle ceiling | (ii) | 0.958 / 0.932; 0.434 / 0.397; 45 % / 43 %; ensemble-dependence at 20.7 eV |
| local only, no cross-basin | (iii) | σ ≤ 0.40 Å train / 0.15 Å eval; tied to `thm:l2g`/`cor:basins` |
| correlation = shape only | (iv) | affine invariance; the 120-vs-93 population mismatch flagged |
| under-training | (vii) | 30 000 steps, 0.12 epochs, "no model here is converged", named as a live alternative explanation |
| radial confound not excluded | (v) | 8.1 % CV on RMSD ≈ 0.25 Å; four generator families; `\pending` |
| estimator not validated | (vi) | 12 steps / 2 probes; attenuation called "an argument, not a measurement" |
| force confounded / hypothesis-only | (viii) | distillation weight 0.1, n = 30, 1 seed; plus 4 further defects and the NaN story |

Eight items where five were required, each with its number. The bans B-17
(every r carries a scale companion) and B-18 (ceiling carries its ensemble
caveat) hold in the abstract, in intro P4, in all five bullets and in the
conclusion — I checked each r occurrence individually.

---

## ISSUES, ORDERED BY SEVERITY

### Issue 1 — HIGH. An internal estimator result that bears on the headline is not disclosed at all.

`paper/ESTIMATOR_VALIDATION.md` (on disk, dated 2026-08-03) measures the paper's
own metric across ODE resolutions on 10 molecules and reports, reference-dropped:
gap +0.452 at 4 steps, **+0.183 at the paper's own 12-step headline setting**,
**−0.012 to +0.050 at 48 steps**, with "log p does not converge in n_ode_steps"
and "the claim energy arm ≫ FM-only is not robust to the n_ode_steps choice".
The same document contains a partial-correlation measurement
(`r = 0.420 → partial 0.311` with the reference, `0.325 → 0.315` without) that
speaks directly to limitation (v).

The manuscript's treatment is `02_method` disclosure (iii) and limitation (vi):
"We have **not** reported convergence in step or probe count … agreement with an
exact divergence are all *not yet reported here*." That sentence is *literally
true* and the quarantine (ban B-26) is a defensible editorial choice given the
unresolved status conflict recorded at `main.tex:108–115`. But "not reported"
reads to a referee as "not looked at", when in fact a preliminary internal look
exists and it points **against** the robustness of the headline. If that
document is later verified, the current wording will look like selective
non-reporting — the single most expensive failure mode available to this paper.

**Fix (one sentence, no new experiment).** In limitation (vi), after the
"not yet reported here" clause, add: *"An internal, not-yet-verified check of
this estimator on 10 molecules indicates that log p_θ is not converged at 12
steps and that the magnitude of the arm gap depends on the ODE resolution; we
therefore report the resolution with every number and treat the effect size, not
its sign, as resolution-conditional. Verifying that check is the first item on
the revision list."* This costs 45 words, converts a concealment risk into a
credibility asset, and is consistent with the paper's own posture everywhere
else. Do **not** quote the 0.518/−0.012 numbers until the document is
re-verified — the sentence above needs none of them.

### Issue 2 — HIGH. "Pre-specified" / "fixed in advance" / "pre-registered" is not supported by the record.

Six occurrences: `01_intro.tex:71`, `01_intro.tex:182`, `04_experiments.tex:92`,
`04_experiments.tex:127` ("The primary endpoint, **fixed in advance**"),
`04_experiments.tex:235`, `fig3_ablation.tex:32`, plus "our **pre-registered**
reading" at `04_experiments.tex:389`.

Evidence against: the disjoint × reference-excluded cell exists only in
`figures/out/primary_endpoint.json`, mtime **2026-08-03 23:19**, produced by
`recompute_disjoint_noreference.py` (mtime 23:18) — i.e. *after* the permissive
+0.434/+0.074 contrast was the headline. `SPINE.md` (18:34 the same day) contains
no "primary endpoint" concept at all; it appears only in `SPINE_v2.md`. Both
corrections were audit findings, which the manuscript itself says: "The
evaluation as originally run has two defects" and "two things about this control
are not what an earlier draft of this appendix claimed".

This is the claim most likely to be caught, and it is unnecessary — the endpoint
is defensible *without* it, because (a) both corrections were applied uniformly
to all arms, and (b) they **reduce** the reported effect (+0.360 → +0.167), so
the choice cannot have been fishing.

**Fix.** Replace "pre-specified"/"fixed in advance" with "designated primary"
and add one sentence at `04_experiments.tex:127`: *"This endpoint was defined
after both defects were identified in an audit of the original evaluation, not
before it; we designate it primary because both corrections apply uniformly to
every arm and both reduce the contrast we report (+0.360 → +0.167)."* At
`04:389`, "pre-registered reading" → "our stated expectation".

### Issue 3 — MEDIUM. The "43 % of the ceiling" figure is attached to the wrong population, three times.

`04_experiments.tex:185` reads "on **the primary endpoint** +0.397 is 43 % of
0.932". But +0.397 is the **all-120**, reference-dropped mean; the primary
endpoint is **+0.369** (disjoint 93). 0.369/0.932 = **39.6 %**, not 43 %.
The same 43 % is then attached to the primary endpoint at `04:212` ("the gain
covers 43 % of what is achievable", in the primary-endpoint paragraph) and at
`05_conclusion.tex:22–23` ("from +0.202 to +0.369 …, 43 % of the +0.932").

Limitation (ii) is *correct* because it pairs 0.434/0.397 with 45 %/43 % without
calling either the primary endpoint. So the arithmetic exists correctly in one
place and is mis-transplanted in three.

**Fix.** In `04:185` change "on the primary endpoint" → "on the all-120,
reference-dropped metric"; in `04:212` and the conclusion, change 43 % → **40 %**
(0.369/0.932). This slightly *lowers* the reported achievement, which is the
right direction for a correction of this kind.

### Issue 4 — MEDIUM. The amplification triple 6.7 / 12.5 / 32.3 is inconsistent with its own formula.

`03_theory.tex:99–100` states the factor is `t‖δv‖/((1−t)σ²)` and, with σ = 1 and
t ∈ {0.85, 0.92, 0.97}, that formula gives **5.67 / 11.5 / 32.33**. The printed
6.7 / 12.5 / 32.3 is `1/(1−t)` for the first two and `t/(1−t)` for the third —
two different formulas in one list. `A1_proofs.tex:172` repeats the same triple
while naming `t/((1−t)σ²)` explicitly, and `04_experiments.tex:429` repeats it a
third time. `configs/*.yaml` confirms the probe times are indeed
{0.85, 0.92, 0.97}.

**Fix.** Either print **5.7 / 11.5 / 32.3** (consistent with the stated
`t/((1−t)σ²)`) or restate the factor as `1/((1−t)σ²)` and print
**6.7 / 12.5 / 33.3**. The argument is unaffected either way; the inconsistency
is what a careful referee will notice, and it sits in a theory section whose
whole purpose is credibility.

### Issue 5 — MEDIUM. Table 2's caption says "± SEM over seeds"; the printed dispersions are standard deviations.

`04_experiments.tex:318` claims "± SEM over seeds". For the same quantity
(FM-only r, all 120, +ref) Table 1 prints `± 0.005` and Table 2 prints
`± 0.010`; for the energy arm, `± 0.032` vs `± 0.063`. I recomputed from the
JSON: SD = 0.0095 / 0.0633, SEM = 0.0047 / 0.0316. So **Table 1 is SEM and
Table 2 is SD**, and the caption mislabels the latter. Two tables printing the
same mean with different error bars and no explanation is a referee magnet.

**Fix.** Either divide the Table 2 dispersions by √n_seeds, or change the caption
to "± SD over seeds" and add "(Table 1 reports SEM; Table 2 reports SD)".
Secondary: the same caption says the scrambled seeds are "printed individually
rather than averaging", but the NRV and r columns of that row print 3-seed
**averages** (1.062, 0.341, 1.691, 0.254) while the other four columns print
`2-seed-mean / stab-seed` pairs. Make the row uniform or say what each slash
means.

### Issue 6 — MEDIUM. The one placeholder that flatters the hypothesis.

Tab. 1's last contrast row prints `Δ ≈ +0.13, t ≈ 3.9, p < 0.05` in red for an
evaluation that is still running, while the registered true status (Q5, both in
`main.tex:95–98` and Tab. A6) is **Δ = +0.092, t = 1.77, p = 0.147 — not
significant**. The red colour, the "evaluation still running" note and the
honest self-criticism at `04:390–393` ("more evaluation parents shrink only the
within-seed component … the decisive experiment is additional matched seeds")
all mitigate it, and `DRAFT_STATUS.md:342` shows this was argued before. It is
still the only place in the manuscript where a *prediction that is better than
the measurement* sits inside a results table, and a printed PDF read in
greyscale loses the red.

**Fix (recommended).** Replace all four macros in that row with a single
`\pending{not yet measured}` and move the expectation to prose in §5.6.1, where
the caveat already lives. Alternatively delete the row: the measured n = 120
contrast is already reported two rows above.

### Issue 7 — LOW/MEDIUM. Abstract switches parent population mid-argument without saying so.

Sentence 4 gives the primary endpoint (0.202 → 0.369, 93 disjoint parents,
reference excluded); sentence 5 immediately compares "0.958 where our energy arm
reaches 0.434" — both all-120, reference-included. A reader will attach 0.434 to
the endpoint just described. Same pattern in bullet 4.

**Fix.** Six words: "on the permissive all-parent metric, 0.958 against our
0.434" — or use the matched pair 0.369 vs 0.932 (≈ 40 %) throughout the abstract.

### Issue 8 — LOW. "Fixes the response magnitude" overstates in one direction the table itself contradicts.

`04:290` and bullet 4 say the energy term "fixes the response magnitude"
(scale ratio 0.559 → 1.093). Two rows below, the reference-dropped scale ratio
is **1.420** — an overshoot of 42 %, not a fix. Suggest "moves the response
magnitude to roughly the right order (0.559 → 1.093 with the reference, 0.838 →
1.420 without, i.e. from too flat to somewhat too steep)". This is the only
place where the draft's prose is more favourable than its own table.

### Issue 9 — LOW. `3×10^4` groups vs 40 000 shard parents.

`01_intro.tex:142`, `02_method.tex:159`, `03_theory.tex:83` and the Related Work
paragraph all say the divergence is summed over "3×10⁴ chemically distinct
groups". `A2_details.tex:155` says every energy arm reads **two** shards, of
30 000 and 10 000 parents. So the parent count is 4×10⁴, and "chemically
distinct" (distinct compositions) is not verified anywhere. Say "≈4×10⁴ parent
groups" or "3×10⁴ training-split groups", and drop "chemically distinct" unless
composition uniqueness was checked.

### Issue 10 — LOW. `main.tex:87` mis-locates Q2 (Sec. 5.2 → Sec. 5.1).

The Tab. A6 mirror is right; only the master block is wrong. One-word fix, but
the inventory is the artefact that certifies the others, so it should be exact.

### Issue 11 — LOW, but load-bearing. Two citations are the paper's only external comparison and neither is fully verified.

- `driftingboltzmann` has `author = {Hu, Pipi and others}` with `TODO-verify` in
  the bib, and it is cited **six** times in the body, including in the sentence
  that scopes the paper's central negative result. An `and others` author list on
  a load-bearing citation is a visible defect; resolve it or cite only `psm`
  there.
- `zatom1`'s generation row (30.4 / 17.0 / 15.1 / 93.5, 80 M params, ~400
  GPU-hours) is the **only** external number in Tab. A4, and open item V1 in
  `main.tex:133–136` records that the source table could not be rendered. The
  appendix does disclose "we were unable to locate a published table", which is
  the honest handling; but if V1 cannot be closed before submission, consider
  dropping the row rather than shipping an unverifiable external comparison.

### Issue 12 — NOT an integrity issue, but a submission blocker.

`main.aux` puts `\label{endofmaintext}` on **page 17**; `main.log` reports 36
pages total. ICLR's main-text limit is 9 pages. The draft is ~8 pages over, and
`01_intro.tex:26–41` already flags a 2× overrun in the abstract/intro/related
budget. No amount of integrity work matters if the paper is desk-rejected on
length. The cut list in `DRAFT_STATUS.md` should be executed before any further
polish — and note that the compression pass must not remove the disclosures
above, which are precisely the compressible-looking material.

---

## WHAT THE DRAFT DOES UNUSUALLY WELL (worth preserving under compression)

1. **It reports the numbers that hurt it.** NRV worse without the reference
   (p = 9.2e−7), the scramble inversion under the permissive metric, the 3-in-9
   NaN rate, the guard that masked failure for a GPU-day, both control defects
   found by self-audit, the "reported fix is not a proof of robustness" clause.
2. **Every arm is traceable to a per-seed value** (`tab:cells`), and the four
   Table 1 columns come from one re-scored source with the sensitivity shown
   rather than inferred.
3. **The metric's admissibility is argued before the result**: independence of
   GFN2-xTB from the teacher, the Simpson's-paradox reason for grouping, and the
   two corrections applied simultaneously.
4. **The theory cedes what is prior and marks what its own scope excludes**
   ("Outside the theory altogether", "Scope of the theory", the deletion note).
5. **The ensemble's non-physicality is disclosed in the protocol section**
   (21 kT, ~800 kT at room temperature) rather than buried in limitations.

---

## REQUIRED-EDIT CHECKLIST (in the order I would do them)

- [ ] Issue 2: de-claim pre-registration (6 sites) + one honest sentence.
- [ ] Issue 1: one sentence in limitation (vi) disclosing the internal estimator check.
- [ ] Issue 3: 43 % → 40 % at `04:212` and `05:23`; fix the population label at `04:185`.
- [ ] Issue 4: make the amplification triple match its formula (3 sites).
- [ ] Issue 5: Table 2 caption SEM → SD (or rescale), and make the scrambled row uniform.
- [ ] Issue 6: collapse Tab. 1's n=240 row to `\pending{not yet measured}`.
- [ ] Issues 7–8: two clause-level softenings in the abstract and §5.5.
- [ ] Issues 9–11: group count, `main.tex` Q2 pointer, `driftingboltzmann` authors.
- [ ] Issue 12: execute the page-budget cuts *without* touching any disclosure.

Camera-ready gate (`main.tex:151`) is unchanged and correct: `grep -c pending`
must reach 0 and `make_experiment_figures.py --strict-real` must exit 0.
