# INTEGRITY AUDIT — `paper/` manuscript

Auditor: academic-integrity / placeholder auditor
Date: 2026-08-03
Scope: `paper/main.tex`, `paper/sections/*.tex`, `paper/figures/*.tex`,
`paper/figures/out/*.pdf`, cross-checked against `runs/eval_ours/*/`,
`configs/sweep/*.yaml`, `paper/SPINE.md`, `paper/DRAFT_STATUS.md`, and the
project FACTS block [A]–[G].

---

## VERDICT

**Conditionally safe — but NOT yet safe to submit, and three findings are
blocking.**

The placeholder machinery is genuinely good: every not-yet-real number is
wrapped in `\pending{}`, every one is listed in both the `main.tex` inventory
block and the reader-facing Appendix Table A4, the figures carry red
PLACEHOLDER stamps, and the wording discipline (no zero-shot claim, no
"outperform everyone", negative results in the main text) passes clean. The
`\pending` audit found **zero** unwrapped placeholders.

What the placeholder audit does *not* cover, and what this audit found, is a
different class of problem: **three black (i.e. asserted-as-real) numbers whose
provenance does not survive contact with the run directory.** Those are S1, S2,
S3 below. They are not placeholder violations — they are correctness /
attribution violations, and a determined reviewer with the artifact could find
them.

Safe to hand to the user **after** S1–S3 are resolved. Do not submit before
then.

---

## SEVERITY-RANKED FINDINGS

### S1 — BLOCKING. Table 2's "BGFM (ours)" row cannot be attributed to the model the paper's headline result is about

**Where:** `sections/04_experiments.tex:172` (table row), `:138–144` (prose),
`:157–164` (caption); `SPINE.md:477`; FACTS [E].

The row reads `43.0 / 32.0 / 23.0 / 100`. I searched every `validity.json` under
`runs/` (22 runs). The exact 4-tuple occurs in **exactly one** place:

```
runs/eval_ours/onpolicy_4m_epoch0/validity.json   43 / 32 / 23 / 100
```

The energy-only arm that carries the paper's entire calibration claim gives
something different:

```
runs/eval_ours/abl_a3_energy_only/validity.json   43 / 31 / 20 / 100
```

And `onpolicy_4m_epoch0` is an **on-policy-distillation checkpoint at epoch 0**
whose own Boltzmann correlation is

```
runs/eval_ours/onpolicy_4m_epoch0/boltz_independent.json
    mean_pearson_r = -0.0817   (n_groups = 30)
```

i.e. **anti-calibrated**. If that is the row's source, then Section 5.3 reports
generation quality from a model with r = −0.08 immediately after Section 5.2
reports calibration r = +0.430 from a different model, both under the single
label "BGFM (ours)". The paper also asserts on-policy distillation is a
*confound* elsewhere (`04_experiments.tex:266`, `A2_details.tex:101–106`), so
silently sourcing the headline generation row from an on-policy run would be
self-contradictory.

The manuscript transcribed FACTS [E] faithfully; the defect is upstream and was
never checked. Note that neither `SPINE.md §7.6` nor `DRAFT_STATUS.md` records a
run tag for this row — the provenance is undocumented at every level.

**Fix (choose one, then record the tag in SPINE §7.6):**
1. If the intended model is the energy-only arm, change the row to
   `43.0 / 31.0 / 20.0 / 100` (`abl_a3_energy_only`) and say so in the caption.
2. If the intended model really is the on-policy checkpoint, the paper must say
   which checkpoint each table describes and must not call both "BGFM"; the
   comparison in `:150–154` ("their Boltzmann calibrations differ by
   Δ = +0.355") then becomes false as written.
3. Either way, add a caption clause naming the run tag, and note that the row is
   a single run selected from a pool — `43.0` is the joint maximum over all 22
   evaluated runs on the headline column.

---

### S2 — BLOCKING. The Y-scrambling control arm is not configuration-matched, and the paper says twice that it is

**Where:** `sections/04_experiments.tex:295–299`
("The weight λ₂, the loss scale, the optimiser, the schedule and the seed count
are identical; the configuration diff between the two arms is three lines
(output directory, run name, shard path)"), repeated at
`sections/A2_details.tex:144–149`.

From the configs on disk:

| arm | seeds evaluated | `energy_b_parents` | `energy_loss_cap` |
|---|---|---|---|
| a3, true labels | s1, s2, s3, s5 | **4** (all) | **3000** (all) |
| a6, Y-scrambled | s2, s3 | 4 | 3000 |
| a6, Y-scrambled | **`stab_s5`** | **8** | **1500** |

(`configs/sweep/a3_energy_only_s{2,3,4,5}.yaml` vs
`configs/sweep/a6_energy_only_shuffled_s{2,3}.yaml` vs
`configs/sweep/a6_energy_only_shuffled_stab_s5.yaml`.)

So the third scrambled seed was trained under the **stabilised configuration**
(the very fix described in Sec. 5.7 / App. B.6) while no true-label seed was.
The "three-line diff" claim is false for one third of the scrambled arm.

This is not cosmetic. That third seed is the outlier (`0.4172`, vs `0.2843` and
`0.3155`). It is the seed that:
- raised the scrambled mean from 0.300 to 0.339,
- collapsed the Y-scrambling contrast from t = 3.63 to **t = 1.77 (n.s.)**, and
- changed the mechanism split from 63/37 to **74/26**.

In other words, the single number that decides both of the paper's second-order
claims comes from a differently-configured run, and the paper asserts
configuration identity. A reviewer who opens the artifact will find this.

**Fix:**
1. Delete the "three lines" sentence in both places, or restrict it to seeds
   s2/s3 explicitly.
2. Add one sentence to Sec. 5.6.2 and App. B.4: "the third scrambled seed was
   trained under the stabilised configuration (M = 8, cap = 1500); the other two
   and all four true-label seeds used M = 4, cap = 3000."
3. Report the contrast **both ways** (with and without `stab_s5`) — the
   like-for-like 2-seed comparison is the configuration-matched one, and hiding
   it while quoting only the 3-seed number is the wrong direction of selection.
4. Related: state which seeds were *dropped* and why. `configs/sweep/` contains
   `a3_..._s4`, `a6_..._s4`, `a6_..._s5`, `a6_..._stab_s4`, none of which appear
   in `runs/eval_ours/wide_*`. FACTS [F] says 3 of 9 energy seeds diverged. The
   paper reports the divergence count in the stability section but never
   connects it to seed attrition in the reported arms. That connection must be
   made explicit or it reads as survivorship selection on exactly the contested
   contrast.

---

### S3 — BLOCKING. `sections/02_method.tex:174` states a hyperparameter that no reported energy run used

**Where:** `sections/02_method.tex:171–179`: "Every reported energy run
therefore stabilises it in three layers: $M$ is raised from 1 to 8 …".

Every true-label energy run (`a3_*`) uses `energy_b_parents: 4`, and the
scrambled s2/s3 runs use 4. Only `a6_..._stab_s5` uses 8. So the sentence is
false for 6 of the 7 reported energy runs. Table A2
(`figures/tabA2_constants.tex:38`) is closer — it prints "1 → 4 → 8" — but the
method section overstates.

**Fix:** "M is raised from 1 to 4 in the reported arms, and to 8 in the
stabilised configuration described in App. B.6."

---

### S4 — HIGH. Figure 3(a) makes the force term look *helpful*, which inverts the paper's central negative claim

**Where:** `figures/out/fig3_ablation_bars.pdf`, panel (a); wrapper
`figures/fig3_ablation.tex`.

Panel (a) plots, on one axis: `no physics 0.075` (n = 120), `force only 0.109`
(n = 30), `force+energy 0.294` (n = 30), `energy Y-scrambled 0.339`,
`energy true 0.430`. The n = 30 round's own no-physics baseline, **+0.223**, is
not drawn.

A reader's eye therefore reads force-only (0.109) as *better* than no physics
(0.075) — the exact opposite of the claim the figure is captioned to support,
and exactly the cross-round comparison SPINE RULING R2 forbids. The caption
says "compare only with that round's own baseline (+0.223)", but the bar it
names is absent from the chart. A caption cannot undo what the bars show.

**Fix:** add the `+0.223` bar (same hatch as the other n = 30 bars) to panel
(a), or move both n = 30 bars into their own sub-panel with their own baseline.
This is the single highest-leverage figure fix in the paper.

---

### S5 — HIGH. Zatom-1: the citation is a stub, the non-convergence caveat is absent, and the comparison is the paper's only external win

**Where:** `refs.bib:121–126`; `sections/04_experiments.tex:141–144, 161, 174`;
`sections/01_intro.tex:159`; open item V1 in `main.tex:99`.

```bibtex
@article{zatom1,
  title  = {{Zatom-1}: ... Trained on {OMol25}},
  author = {Anonymous},
  journal = {arXiv preprint},
  year   = {2025}
}
```

The paper's sole external comparison — "BGFM exceeds Zatom-1 … on all four
columns" — currently cites an entry with **no author, no arXiv number, no
locatable source**. `RESEARCH_story.md:274` has the real identifiers
(arXiv:2602.22251 + repo). This must be fixed before anything is shown outside
the group.

On the non-convergence caveat there is a **direct conflict** the user must rule
on:
- FACTS says the Zatom-1 numbers must be honestly annotated as coming from a
  non-converged (80-epoch) model.
- SPINE ban B-11 / DRAFT_STATUS V1 forbid writing "non-converged" until a
  locatable author statement is found, and mandate the euphemism the manuscript
  currently uses ("at the training budget those authors report").

SPINE's rule is the more conservative one and I would keep it — but the current
wording is *too* soft: "at the training budget those authors report" reads to a
reviewer as a neutral protocol note, not as "this baseline is undertrained".
That gap is what makes "we exceed it on all four columns" feel stronger than the
evidence.

**Fix:**
1. Repair the bib entry (blocking).
2. Close V1. If the 80-epoch statement is locatable, cite it and say so.
3. If it is not locatable, strengthen the hedge without asserting
   non-convergence, e.g.: "reported by its authors at a training budget they
   describe as limited; we were unable to locate a table for a fully trained
   model, so this comparison should be read as a lower bound on that method."
4. Also close V2 (every cell of the capability table) — one wrong checkmark in
   Table 3 costs more than the table earns.

---

### S6 — HIGH. tmQM: the reported checkpoint is the best of three, and "the same checkpoints" is not established

**Where:** `sections/04_experiments.tex:186–197`
("We next evaluate **the same checkpoints**, without any adaptation …");
`sections/01_intro.tex:168–178`; `figures/tabA1_cells.tex:32–35`.

The reported `+0.365 / median +0.429 / 37% > 0.5 / 270-270 converged` matches
`runs/eval_ours/OOD_tmqm_energy_v3step30k/boltz_independent.json` exactly
(0.3648 / 0.4287 / 0.3667 / 270). Good — the number is real.

But on disk there are three energy-arm tmQM evaluations:

| run | mean r |
|---|---|
| `OOD_tmqm_energy_v3step30k` | **+0.365** ← reported |
| `OOD_tmqm_energy_v3bstep45k` | +0.329 |
| `v4_stable_tmqm` | **+0.091** |

and the reported one comes from the `v3_energy` run family, **not** from the
`wide_a3_energy_only_s*` checkpoints that produce the n = 120 headline. So "the
same checkpoints" is, on the evidence available to me, not accurate; and the
tmQM number is the maximum of three available energy-arm evaluations.

**Fix:** name the checkpoint explicitly ("the energy-supervised checkpoint of
App. B.3, evaluated without adaptation"), drop or qualify "the same
checkpoints", and either report the spread across available energy checkpoints
or state why one was selected. The force-vs-energy *sign flip* survives any of
these choices (+0.091 still beats −0.158), so the headline claim is safe — but
the selection must be visible.

---

### S7 — MEDIUM-HIGH. The abstract and intro bullet 4 pair a 74/26 split with a placeholder Δ = +0.13; the two are arithmetically incompatible

**Where:** `sections/01_intro.tex:31–33` (abstract), `:212–218` (bullet 4);
`sections/04_experiments.tex:319–330`; `figures/fig4_mechanism.tex:13`.

The manuscript states the split as ≈74% / ≈26% of the +0.355 total, then
annotates the 26% component with `Δ ≈ \pending{+0.13}`. But 26% of 0.355 is
**+0.092**, and +0.13 is **37%** of 0.355. The two numbers in the same
parenthesis describe different decompositions:
- 74/26 comes from the 3-seed scrambled mean (0.339) — the current real data,
  matching FACTS [C];
- +0.13 comes from the 2-seed mean (0.300) — which is exactly FACTS [G]'s
  63/37 split (+0.224 / +0.131).

So the manuscript has silently mixed FACTS [C] and FACTS [G]. It is internally
inconsistent, and **FACTS [G] itself is inconsistent with FACTS [C]** (G is
computed from superseded 2-seed data). The manuscript's 74/26 is the
self-consistent choice and is correctly footnoted at
`04_experiments.tex:325–327`, so I recommend keeping it — but the user must
ratify the deviation from FACTS [G], because the parent FACTS block says 63/37.

**Fix (one of):**
- (preferred) Keep 74/26, and either wrap the percentages in `\pending{}` too or
  restate as: "≈74% / ≈26% at n = 120 (Δ = +0.092); under the larger evaluation
  we project Δ ≈ \pending{+0.13}, which would move the split to ≈63% / ≈37%."
- Or revert to FACTS [G]'s 63/37 (2-seed) — but then the arm silently drops a
  seed, which is worse, and it would contradict `tab:cells`.

Also: the abstract asserts the three-quarters / one-quarter attribution in black
text with no signal that the second component is not statistically significant.
Add four words: "…the remainder, not yet significant at our current evaluation
size, to Boltzmann-specific pairing."

---

### S8 — MEDIUM. Table A2 contradicts the main text on two evaluation constants; `main.tex` claims a self-consistency that does not hold

**Where:** `figures/tabA2_constants.tex:37, 45` vs `sections/02_method.tex:108`,
`sections/04_experiments.tex:27`, `sections/A2_details.tex:157–159`;
`main.tex:103–104`.

| constant | Table A2 says | main text / App. B.5 say | disk |
|---|---|---|---|
| Hutchinson probes | 2 train / **2 eval** | 2 train / **4 eval** | launchers pass `--n_hutchinson 2` |
| eval perturbations K | 4 train / **8 eval** | K = **9** eval | `n_records/n_groups = 1080/120 = 9` |

`main.tex:104` asserts "The manuscript is self-consistent at 4 (Secs. 3, 5 and
App. B.5)" — it is not: Table A2 prints 2. And the `K = 8` entry is simply wrong;
the JSONs confirm K = 9 in every arm.

**Fix:** set Table A2 to `K = 4 train / 9 eval`, and resolve D1 (probes) against
the runs — the launcher evidence says the true value is 2, in which case three
places change (`02_method.tex:108`, `04_experiments.tex:27`,
`A2_details.tex:158`) plus Table A2.

---

### S9 — MEDIUM. Main text is 14 pages against ICLR's 9-page limit

`main.aux` gives `main text pages: 14` (PDF is 25 pages total). The SPINE
compression plan targeted 9. This is a desk-reject risk independent of content,
and the cut list in `DRAFT_STATUS.md §2.2` still needs sign-off. Flagged here
because no amount of integrity work matters if the paper is not reviewed.

---

### S10 — LOW-MEDIUM. "1080 scored geometries per arm" overstates by 1–2 records

`sections/04_experiments.tex:39` says n = 120 × K = 9 = "1080 scored geometries
per arm". The JSONs report `n_xtb_ok` of 1078 or 1079 in five of the eight
n = 120 arms — i.e. one or two xTB single points failed and were dropped. The
tmQM survivorship claim ("all 270 converged") is correct and verified; the
n = 120 statement is the one that is slightly off.

**Fix:** "1080 attempted geometries per arm (1078–1080 scored; xTB failures were
dropped)". Cheap, and it pre-empts a reviewer noticing an inconsistency in the
artifact.

---

### S11 — LOW. Stale comments that contradict the shipped artifact

- `figures/make_experiment_figures.py:716` — "The manuscript quotes 63% / 37%"
  is stale; the manuscript quotes 74/26 and the script default is already
  `--split seeds`.
- `figures/fig3_ablation.tex:3–4` and `figures/fig4_mechanism.tex:2–3` still
  carry "ACTION A1/A2: regenerate…" though `main.tex:96–97` records both DONE
  and the shipped PDFs confirm it (+0.355 / 10.8 / 0.002 and 74/26).
- Harmless to reviewers (comments do not compile), but they will mislead the next
  editor. Delete on the next pass.

---

## INTEGRITY CHECKS — ITEM BY ITEM

### 1. Placeholder wrapping and inventory coverage — **PASS**

`grep -n pending` over `main.tex sections/*.tex figures/*.tex` returns 26 hits.
Excluding the macro definitions, the inventory block and the comments, every
live `\pending{}` maps to an inventory row:

| ID | Live occurrences | In `main.tex` block | In App. Table A4 |
|---|---|---|---|
| P1 Δ ≈ +0.13 | `01_intro.tex:33, 217`; `04_experiments.tex:129, 305`; `A2_details.tex:271` | yes | yes |
| P2 t ≈ 3.9 | `01_intro.tex:217`; `04_experiments.tex:129, 305`; `A2_details.tex:274` | yes | yes |
| P3 p < 0.05 | `01_intro.tex:218`; `04_experiments.tex:129, 305`; `A2_details.tex:277` | yes | yes |
| P4 n = 240 | `01_intro.tex:218`; `04_experiments.tex:129, 304, 342`; `A2_details.tex:280` | yes | yes |
| P5 provisional | `04_experiments.tex:343`; `A2_details.tex:283` | yes | yes |
| P6 Symphony ×4 | `04_experiments.tex:175`; `A2_details.tex:286` | yes | yes |
| P7 EDM ×4 | `04_experiments.tex:176`; `A2_details.tex:289` | yes | yes |
| P8 GeoLDM ×4 | `04_experiments.tex:177`; `A2_details.tex:292` | yes | yes |
| P9 in progress | `04_experiments.tex:372`; `A2_details.tex:295` | yes | yes |
| P-F1 Fig. 2 points | `figures/fig2_boltzmann.tex:21` | yes | yes (`A2_details.tex:297`) |
| P-F3 | retired (Fig. A1 now reads 74/26, no red stamp) | closed | n/a |

Targeted checks the task called out:
- **Symphony / EDM / GeoLDM Table 2 numbers** — all 12 cells are
  `\pending{--}`, red, and the caption says so explicitly. No number is
  asserted. **PASS.**
- **n = 240 Y-scrambling statistics** — all four (Δ, t, p, n) are red
  everywhere they appear. **PASS.**
- **ΔF / ESS** — no number anywhere. `03_theory.tex:139–142` and
  `A1_proofs.tex:501, 564, 605–609` state explicitly that no free-energy
  accuracy and no effective sample size is measured. (`mean_ess_frac` exists in
  the eval JSONs and is correctly *not* reported.) **PASS.**
- **COV / MAT** — no COV/MAT table and no COV number. The only conditional-
  conformer number is AMR-R 3.03 Å vs 0.073 Å, stated in Limitations
  (`04_experiments.tex:362–367`) *against* our own interest with the task
  difference argued. This is the FACTS [P4] policy correctly executed. **PASS.**

Figures carry their own red stamps: Fig. 2 "PLACEHOLDER point positions … their
per-parent r and OLS slope are REAL (measured)"; Fig. 3(b) "PLACEHOLDER —
projected n=240" alongside the measured n = 120 line; Fig. A1 footnote naming
the 3-seed source and the superseded 2-seed split. The script has a
`--strict-real` gate and a LEDGER. This is above the standard of care I usually
see.

### 2. Placeholders presented as fact — **PASS, with one caveat (S7)**

No sentence claims a completed Symphony/EDM/GeoLDM comparison. The only
statements are `04_experiments.tex:371–372` ("retrains on OMol25 are `in
progress`") and the caption's "Red rows are retrains that have not produced
numbers yet". No "we outperform Symphony" anywhere. The one soft spot is the
abstract's black-text attribution built on a pending contrast — see S7.

### 3. Y-scrambling true status disclosed — **PASS (disclosed in five places)**

- `04_experiments.tex:301–311`: "Δ = +0.092 with Welch t = 1.77 and p = 0.147: a
  consistent point estimate, but not statistically significant", plus an
  explicit "we separate the two claims by strength" paragraph.
- `04_experiments.tex:339–343` (Limitations): the t = 3.63 → 1.77 history, and
  the n = 240 figures labelled `provisional`.
- `tab:boltzmann` (`:128`): the measured row `t = 1.77, p = 0.147 (n.s.)` is
  printed **above** the red projected row.
- `figures/fig3_ablation.tex:20–21` and the shipped PDF: measured n = 120 line
  shown next to the red projection.
- Appendix Table A4, rows P1–P3.

This requirement is met more thoroughly than the task demanded.

### 4. Wording discipline (FACTS 5 rules)

| Rule | Verdict | Evidence |
|---|---|---|
| (1) No "zero-shot to unseen elements" | **PASS** | `01_intro.tex:173–178` gives 0.34%, 0.7M/205M, 1.5–5.1×10⁴ per metal and says "we make no zero-shot claim". Repeated at `04_experiments.tex:188–192` and `tabA1_cells.tex:33`. Only three "unseen"/"zero-shot" hits in the whole manuscript, all in disclaimers. |
| (2) No "beat all baselines" | **PASS in wording, WEAK in substance** | `01_intro.tex:158–162` and `04_experiments.tex:146–149` both say "we claim no superiority over the broader 3D-generation literature". No hit for outperform/SOTA/surpass. But see S5 (Zatom-1 caveat too soft) and S1 (the winning row's provenance). |
| (3) Table 3 is capability, not performance | **PASS** | Caption: "Entries record what each model class *can be asked*, not how well it scores"; body text "the point is not that BGFM scores higher". V2 still open. |
| (4) Force-harmful and energy-instability in the main text | **PASS** | Force: abstract, intro finding 2, contribution bullet 3, §5.6.1, §5.4 (tmQM −0.158), Conclusion. Instability: contribution bullet 5, §5.7, App. B.6, with the masked-NaN-guard admission intact. Both are foregrounded, not buried. |
| (5) Y-scrambling status honest | **PASS** | See item 3. |

### 5. Have the real numbers been altered? — **NO. All verified against disk.**

| FACTS | manuscript | disk | ✓ |
|---|---|---|---|
| [A] energy seeds .377/.402/.525/.418 | `tabA1_cells.tex:23` | wide_a3_s1/s2/s3/s5 = .3768/.4025/.5246/.4181 | ✓ |
| [A] baseline .075/.083/.065/.077 | `tabA1_cells.tex:22` | wide_a1_s2/s3/s4/s5 = .0750/.0829/.0647/.0771 | ✓ |
| [A] r = 0.430 ± 0.033, 0.075 ± 0.004 | abstract, §5.2, Tab. 1 | recomputed: 0.4305 ± 0.0326, 0.0750 ± 0.0037 | ✓ |
| [A] Δ = +0.355, t ≈ 10.8 | throughout | consistent | ✓ |
| [B] force-only +0.109 vs +0.223 | §5.6.1, Tab. A3 | abl_a2 = 0.1093; abl_a1 = 0.2232 | ✓ |
| [B] force+energy +0.294 vs energy +0.420 | §5.6.1, Tab. A3 | abl_a4 = 0.2941; abl_a3 = 0.4203 | ✓ |
| [C] scrambled .284/.315/.417 → +0.339 | Tab. 1, Tab. A3 | .2843/.3155/.4172 → 0.3390 | ✓ |
| [C] Δ = +0.092, t = 1.77; earlier t = 3.63 | §5.6.2, §5.7 | consistent | ✓ |
| [C] within-parent std 77.6 kcal both arms | intro, §5.6.2, App. B.4, Fig. 3 | as stated | ✓ |
| [D] tmQM −0.158 / +0.365 / med +0.429 / 37% / 270-270 | §5.4, intro | OOD_tmqm_forceonly = −0.1584; v3step30k = 0.3648 / 0.4287 / 0.3667 / 270 | ✓ (but see S6) |
| [E] ours 43.0/32.0/23.0/100 | Tab. 2 | only `onpolicy_4m_epoch0` matches | ✗ **see S1** |
| [E] Zatom-1 30.4/17.0/15.1/93.5 | Tab. 2 | not verifiable — bib is a stub, V1 open | ⚠ **see S5** |
| [E] ΔE/atom 6.90 kcal, 3% failure | §5.3 | as stated (SPINE §7.6) | ✓ |
| [F] 48→974, 3/9 diverged, M=8, cap 1500, 2/2 stable | bullet 5, §5.7, App. B.6 | configs confirm cap 3000→1500, b_parents 4→8 | ✓ (but see S3) |
| [G] decomposition | 74/26, not [G]'s 63/37 | 74/26 is what the 3-seed data gives | ⚠ **see S7** |

Bond-free control row `35.5 / 23.0 / 16.0 / 100` verified as the mean of
`abl_a1_fm_only` (41/27/17/100) and `abl_a1_fm_only_s2` (30/19/15/100) — exactly
as documented in `main.tex:86–92`. Correctly **not** marked `\pending` (it was a
lookup, not a missing experiment). Derived quantities also check out: the
τ = 13.30 → 2.10 inversion of Prop. 1 and the 6.3× reduction are arithmetically
correct at r = 0.075 and 0.430, as are all five τ values in `A1_proofs.tex:349–353`.

**No fabricated or altered number was found.** Every black number in the paper
either matches a JSON on disk or matches FACTS. The three problems are
attribution (S1, S6), configuration matching (S2, S3), and one arithmetic
mismatch between two versions of the same decomposition (S7).

---

## RECOMMENDED ORDER OF WORK

1. **S1** — establish which checkpoint Table 2's BGFM row is. Nothing else in
   §5.3 can be trusted until this is settled.
2. **S2 / S3** — disclose the `stab_s5` configuration mismatch and seed
   attrition; fix the "three lines" and "M raised to 8" sentences.
3. **S5** — repair `refs.bib` (30 seconds), then close V1 and V2.
4. **S4** — add the +0.223 bar to Fig. 3(a).
5. **S6, S7, S8** — checkpoint provenance for tmQM; reconcile the 74/26 vs
   +0.13 arithmetic; fix Table A2.
6. **S9** — page budget.
7. **S10, S11** — cosmetics.

Items 1–3 are blocking. Once they are done, the placeholder discipline in this
manuscript is strong enough that swapping red numbers for real ones as the
n = 240 and baseline runs land is a safe, mechanical operation — the replacement
map in `DRAFT_STATUS.md §6` is already written and is accurate.
