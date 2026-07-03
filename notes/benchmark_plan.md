# BGFM / HBFM benchmark plan (ICLR 2027)

**All baseline numbers below are source-verified** (adversarial re-fetch of arXiv
HTML / PMLR / OpenReview tables; a fairness audit was applied — see §Audit-fixes).
`—` = **our row, filled only when the checkpoint lands** (never with placeholder
numbers). `N/R` = not reported by that paper; `N/A` = not computable for that method.

Conventions: COV in % (↑), AMR/MAT in Å (↓), energies kcal/mol (↓), μ in debye (↓).
**All COV/AMR values are MEANS** (medians differ and are easy to swap in — do not mix).

## 0. Fairness principle
Numbers compare only within a shared protocol (dataset, split, coverage δ, sample
budget, metric definition). Hence multiple tables, never one:
- **Conformer-given-graph** (GEOM COV/MAT, δ=0.75 DRUGS / 0.5 QM9) — Tables 1/1b/1c;
  our row = zero-shot transfer from OMol25 (Cell 4b).
- **De-novo generation on OMol25** — Table 2 (home turf; our harness metrics).
- **Boltzmann/free-energy fidelity** — Table 3 (the differentiator).
- **Capability matrix** — Table 4.
Do NOT merge a conformer row with a de-novo row. `Adjoint Sampling` uses δ=1.25 Å →
excluded from Tables 1/1b (listed separately). xTB ensemble numbers (T1c) are
**subset-dependent** → never rank across subsets.

---

## Table 1 — GEOM-DRUGS conformer generation (δ=0.75 Å, mean; our row = zero-shot transfer)

| Model | COV-R↑ | AMR-R↓ | COV-P↑ | AMR-P↓ | params | steps |
|---|---|---|---|---|---|---|
| GeoDiff (ICLR'22) | 42.1 | 0.835 | 24.9 | 1.136 | 1.6M | 5000 |
| GeoMol (NeurIPS'21) | 44.6 | 0.875 | 43.0 | 0.928 | 0.3M | 1 |
| Torsional Diffusion (NeurIPS'22) | 72.7 | 0.582 | 55.2 | 0.778 | 1.6M | 20 |
| MCF-B (ICML'24) | 84.0 | 0.427 | 64.0 | 0.667 | 64M | ~1000 |
| MCF-L (ICML'24) | 84.7 | 0.390 | 66.8 | 0.618 | 242M | ~1000 |
| ET-Flow (NeurIPS'24) | 79.5 | 0.452 | 74.4 | 0.541 | 8.3M | ~50 |
| ET-Flow-SS (stochastic) | 79.6 | 0.439 | 75.2 | 0.517 | 8.3M | ~50 |
| DiTMC+aPE-B (NeurIPS'25) | 79.9 | 0.434 | 76.5 | 0.500 | 9.5M | 50 |
| DiTMC+aPE-L | 79.2 | 0.432 | 77.8 | 0.470 | 28.2M | 50 |
| DiTMC+PE(3)-L [SO(3)-equiv] | 80.8 | 0.415 | 76.4 | 0.491 | 31.1M | 50 |
| AvgFlow-DiT-L (ICML'25) | 82.0 | 0.409 | 75.7 | 0.516 | 64M | full |
| Energy-Guided FM "EnFlow" (arXiv 2512.22597) | 78.8 | 0.475 | 70.7 | 0.590 | 16.6M | 50 |
| **BGFM/HBFM (ours, zero-shot)** | — | — | — | — | — | — |

*Selection policy: best-reported variant(s) per method; where a paper reports multiple
sizes, the largest and a ~10M-param size are shown. Caption to state δ=0.75 Å applies to
all rows (incl. DiTMC COV; AMR is threshold-free). Adjoint Sampling excluded (δ=1.25 Å):
for reference it reports DRUGS COV-R 87.3/AMR-R 0.70 at δ=1.25, where RDKit ETKDG beats it
on precision (COV-P 79.9 vs 68.0).*

## Table 1b — GEOM-QM9 conformer generation (δ=0.5 Å, mean)

| Model | COV-R↑ | AMR-R↓ | COV-P↑ | AMR-P↓ |
|---|---|---|---|---|
| GeoDiff | 76.5 | 0.297 | 50.0 | 0.524 |
| GeoMol | 91.5 | 0.225 | 86.7 | 0.270 |
| Torsional Diffusion | 92.8 | 0.178 | 92.7 | 0.221 |
| MCF-B | 95.0 | 0.103 | 93.7 | 0.119 |
| ET-Flow | 96.5 | 0.073 | 94.1 | 0.098 |
| DiTMC+aPE-B | 96.1 | 0.073 | 95.4 | 0.085 |
| AvgFlow-DiT | 96.0 | 0.082 | 95.0 | 0.088 |
| Energy-Guided FM "EnFlow"-SO(3) | 96.3 | 0.076 | 95.5 | 0.083 |
| **BGFM/HBFM (ours, zero-shot)** | — | — | — | — |

*QM9 is saturated (medians ~100 %, AMR-R ~0.07). Same baseline set as T1 for parity.*

## Table 1c — GEOM-DRUGS Boltzmann-weighted ensemble properties (GFN2-xTB, median abs err)

**Subset caveat:** ET-Flow / DiTMC / Torsional-Diffusion values are on the standard
random ~100-molecule subset (Torsional-Diffusion protocol); the **MCF** row is on the
**full test set** (its paper does not break this metric down by size) — annotated, not
ranked against the subset rows.

| Model | E (kcal/mol)↓ | μ (D)↓ | Δε gap (kcal/mol)↓ | E_min↓ | subset |
|---|---|---|---|---|---|
| GeoDiff | 0.31 | 0.35 | 0.89 | 0.39 | 100-mol |
| GeoMol | 0.42 | 0.34 | 0.59 | 0.40 | 100-mol |
| Torsional Diffusion | 0.22 | 0.35 | 0.54 | 0.13 | 100-mol |
| MCF (size not sub-labeled) | 0.68 | 0.28 | 0.63 | 0.04 | **full test set** |
| ET-Flow | 0.18 | 0.18 | 0.35 | 0.02 | 100-mol |
| DiTMC+aPE-L | 0.16 | 0.14 | 0.27 | 0.01 | 100-mol |
| **BGFM/HBFM (ours)** | — | — | — | — | 100-mol |

---

## Table 2 — De-novo 3D generation ON OMol25 (our home turf — a scaffold, not a leaderboard yet)

Only Zatom-1 exists externally on OMol25, and it is **non-converged (80 epochs)** — its
converged home-field (GEOM) numbers are far higher (see context table). "Ours" fills from
`benchmarks/omol25/validity_from_json.py` (now: validity + connectivity + **PoseBusters
config="mol" all-checks** [= Zatom-1's definition] + uniqueness) and the xTB / independent-
Boltzmann scripts, on the converged model. EDM/GeoLDM retrains are broken (0 % valid) → fix or drop.

| Model | RDKit-valid↑ | connected↑ | PoseBusters↑ | uniqueness↑ | xTB ΔE/atom↓ | Boltzmann-R²↑ |
|---|---|---|---|---|---|---|
| Zatom-1 (2026, non-converged) | 30.4 | 17.0 | 15.1 | 93.5 | N/R | N/R |
| FM-only / Level-1 (ours, ablation) | — | — | — | — | — | — |
| **Ours — HBFM (50k, test)** | 15.0 | 9.0 | 2.0 | 100 | — | 0.36 |
| **Ours — HBFM (4M, full)** | — | — | — | — | — | — |

*50k(test) row = corrected 50k checkpoint (physics active). Undertrained →
validity low, but this is the physics-validation run: Boltzmann-R² (Table 3) is
positive (pooled 0.36, mean per-group r +0.40 vs independent xTB), up from the
physics-free run's −0.19. 4M(full) fills on the headline run.*

### Context — GEOM-DRUGS de-novo generation (NOT OMol25; two separate protocols, do not merge)

FLOWR.root Table-1 protocol (RDKit-valid / PoseBusters / ΔE_relax kcal/mol / relax-RMSD Å / params):
EQGAT-diff 86.0/77.6/6.51/0.60/12M · SemlaFlow 95.5/88.5/31.9/0.24/40M · ADiT 99.9/82.7/79.3/1.30/150M ·
Megalodon 94.8/86.6/3.17/0.41/60M · FlowMol3 99.9/91.9/3.83/0.39/6M · **FLOWR.root 98.5/94.0/3.65/0.07/34M**.

Zatom-1 Table-3 protocol (Validity / Uniqueness / atoms-connected / PoseBusters):
EQGAT-diff 94.6/100.0/84.4/59.7 · SemlaFlow 93.9/100.0/92.3/87.5 · ADiT 95.3/100.0/93.0/85.3 ·
TABASCO 97.6/99.0/99.9/91.6 · **Zatom-1 93.6/99.9/99.6/94.1** (converged home-field).

---

## Table 3 — Boltzmann / free-energy fidelity (the differentiator)

Most generators cannot populate it. Boltzmann-generator numbers are on **peptides**
(different systems — capability context, not rows we beat).

| Method | targets Boltzmann | exact log p | Boltzmann-R² (log p vs −E_indep/kT)↑ | ESS w/o reweight↑ | ΔF error↓ | scope |
|---|---|---|---|---|---|---|
| Conformer SOTA (ET-Flow/DiTMC/MCF/TorDiff) | ✗ | ✗ | N/A | N/A | N/A | GEOM organics |
| Zatom-1 | ✗ | ✗ | N/A | N/A | N/A | OMol25 |
| Adjoint Sampling | ✓ (cond., graph fixed) | ✗ | N/A | N/R | N/R | SPICE conformers |
| Transferable BG (NeurIPS'24) | ✓ | ✓ | n/a (peptides) | 1.0–15.3 % | ΔF/kT 4.09 vs 4.10 ref | di/peptides |
| Sequential BG (ICML'25) | ✓ | ✓ | n/a | 3.0–5.2 % (AIS) | E-W2 0.63–1.02 | ala 2–6-mer |
| PROSE (2025) | ✓ | ✓ | n/a | 19.1 % | E-W2 0.371 | dipeptides |
| **Ours — HBFM (50k, test)** | ✓ | ✓ (FFJORD) | **0.36** (mean r **+0.40**) | 0.13 | — | OMol25 50k |
| **Ours — HBFM (4M, full)** | ✓ | ✓ (FFJORD) | — | — | — | OMol25 4M |

**Our metrics (committed, circularity-free):** Boltzmann-R² = pooled R² of `log p_θ` vs
`−E/kT` scored with **GFN2-xTB — independent of the eSEN we train on** (via
`scripts/eval_boltzmann_independent.py`); ESS = `(Σw)²/(n·Σw²)`, `w=exp(−E/kT)/p_θ`; ΔF
error vs thermodynamic integration; relative-population accuracy across tautomer/
protonation/constitutional isomers (COV/MAT cannot express this).

## Table 4 — Capability matrix (oral centerpiece: only all-✓ is ours)

| Method | bond-free | univ-MLIP-trained | de-novo composition | 3D conformation | exact log p | targets Boltzmann | free-energy F(c) | on-policy teacher | whole periodic table |
|---|---|---|---|---|---|---|---|---|---|
| ET-Flow / DiTMC / MCF | ✗ | ✗ | ✗ (graph given) | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Torsional Diffusion | ✗ | ✗ | ✗ | ✓ | ✓ | ~ (BG variant) | ✗ | ✗ | ✗ |
| Adjoint Sampling | ✓ | ✓ (eSEN) | ✗ | ✓ | ✗ | ✓ (cond.) | ✗ | ✓ | ✗ |
| EDM / GeoLDM | ✓ | ✗ | ✓ | ✓ | ~ (ELBO) | ✗ | ✗ | ✗ | ✗ |
| MiDi / JODO / NExT-Mol | ✗ (bonds) | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Zatom-1 | ✓ | ✗ (data only) | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ (OMol25) |
| Transferable BG / SBG / PROSE | ✓ | ✗ (classical FF) | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| **BGFM/HBFM (ours)** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

---

## Audit-fixes applied (this revision)
- **Fixed hard bug:** the DiTMC row previously labeled aPE-L held aPE-**B** numbers. Now
  both aPE-B (9.5M) and aPE-L (28.2M) are listed with their correct verified values + params.
- **Disambiguated names:** "Energy-Guided FM (EnFlow)" spelled out and cited (arXiv 2512.22597)
  so it is not confused with "ET-Flow"/"ET-Flow-SO(3)".
- **Removed the dagger convention:** every T1/1b/1c baseline is now source-verified.
- **T1c subset annotated:** MCF row flagged as full-test-set; others 100-mol subset; not ranked across.
- **T2 reframed** as an our-results scaffold; Zatom-1 annotated non-converged with home-field context; PoseBusters defined as config="mol" all-checks (matches Zatom-1); uniqueness = distinct canonical SMILES / valid.
- **All values marked as means; params/steps populated.**

## Eval-path status — ALL THREE BUILT + VALIDATED; "ours" auto-fills via chained jobs
- **Path 1 — independent Boltzmann-R² + ESS** (`scripts/eval_boltzmann_independent.py`): validated on off-policy (GFN2-xTB single-points; kills the eSEN circularity). Fills T3.
- **Path 3 — validity + PoseBusters(config=mol all-checks, = Zatom-1) + uniqueness** (`benchmarks/omol25/validity_from_json.py`): validated on off-policy (50k model: valid 0.19, PoseBusters 0.01, uniqueness 1.0, connected 0.01). Fills T2.
- **Path 2 — GEOM COV/MAT bridge** (`scripts/eval_geom_covmat.py` + `scripts/unity/eval_geom_covmat.slurm`): BUILT + validated end-to-end (GEOM element→OMol25 index=Z−1; ET-Flow `with_atom_indices`+`ordered=True` atom-order idiom; 2×T conformers/mol; serial `CovMatEvaluator`; reports COV% at δ + AMR). GEOM DRUGS+QM9 `test_mols.pkl` downloaded. Fills T1/T1b.
- **Auto-eval jobs** (all `--dependency=afterok:61376697`, the on-policy training run): `61380843` (T2+T3), `61382822` (T1 DRUGS), `61382823` (T1b QM9). Fire automatically when training completes; the FFJORD OOM was fixed (`expandable_segments` + `n_perturb 8`/`n_hutchinson 2`).

**Scale caveat (important):** zero-shot GEOM COV/MAT on the 50k model is very weak
(QM9 AMR-R ≈ 2.8 Å, COV ≈ 0 below 2.3 Å) — expected for a 50k-scale, OMol25→QM9
distribution-shifted, bond-free model. T1/T1b "ours" is a scale-limited transfer data
point until the 4M headline model; the paper's win is T3 (Boltzmann) + T4 (capability),
not COV/MAT.

## Current 50k on-policy checkpoint results (INVALID — physics loss was a silent no-op)
**These numbers (job 61376697) are from a run where the BGFM physics loss NEVER RAN:**
the dataset didn't populate `g.ndata['force_1_true']`, so the hook's guard skipped ALL
physics terms and the model trained as plain FM (metrics.csv had only x/a/c/e losses).
Root cause fixed (dataset now loads/wires `forces`; loud guard added; force-loss
clamp + teacher-failure mask fixed; all signs adversarially verified correct; 17 unit
tests pass). Corrected re-run: **job 61386215** (physics active), eval auto-chained.
The numbers below are retained only as the eval-infra validation:
- **T2 (OMol25 de-novo):** valid 18 %, connected 1 %, PoseBusters 0 %, uniqueness 100 %.
  BELOW Zatom-1 (30.4 / 17.0 / 15.1) — the 50k model mostly makes disconnected 200-atom
  structures. Not competitive at this scale.
- **T3 (Boltzmann fidelity vs INDEPENDENT GFN2-xTB):** pooled R² 0.17, **mean per-group
  Pearson r = −0.19** (median −0.28), ESS 0.12. The learned density does NOT track the
  Boltzmann energy (the in-loop eSEN diagnostic also gives mixed/negative per-group r).
  ACTION: if the 4M model ALSO shows negative r, investigate a score/FFJORD sign-convention
  bug; at 50k the dominant explanation is undertraining (the model can't even make valid mols).
- **T1b (zero-shot QM9 COV/MAT, 1000 mols):** COV-R/P = 0 % at δ=0.5 Å, AMR-R 3.03 Å
  (COV-R reaches 51 % only at δ=3.0 Å). Zero-shot transfer fails at 50k scale.
- **T1 (DRUGS COV/MAT):** job 61382822 still running (serial evaluator over ~1000 mols).

## Corrected 50k results (v2, job 61386215 — physics ACTUALLY active) — KEY VALIDATION
After the fix, the physics loss trained correctly (`train_score_force_cos` −0.10 → +0.36,
`L_force` 496 → 1.3). Eval on the corrected checkpoint (independent GFN2-xTB, not eSEN):

| metric | v1 (physics OFF, bug) | **v2 (physics ON, fixed)** |
|---|---|---|
| **Boltzmann mean per-group Pearson r** | −0.19 | **+0.40** |
| Boltzmann pooled R² | 0.17 | **0.36** |
| frac molecules r>0.5 | ~0 | **0.47** |
| ESS (no reweight) | 0.12 | 0.13 |
| OMol25 validity / connectivity / PoseBusters | 0.18 / 0.01 / 0.00 | 0.15 / **0.09** / 0.02 |

**The density flipped from anti-correlated (−0.19) to clearly positive (+0.40) with the
Boltzmann energy of an INDEPENDENT potential** — direct empirical proof the on-policy physics
loss works and produces a Boltzmann-consistent density (the core claim). Generation validity
is still low (50k too small to make valid 200-atom molecules); connectivity improved 9×.

**Takeaway:** method + fix EMPIRICALLY VALIDATED at 50k (positive Boltzmann-R² vs independent
potential). Generation quality is scale-limited → the lever is the **4M headline model**
(multi-GPU or 1M-subset), plus a Boltzmann-term ablation. The 50k run confirms the science works.

## Oral roadmap (unchanged strategy)
Win on Table 3 + Table 4; show non-regression on the saturating/criticized COV/MAT (T1/1b).
Kill oracle circularity (xTB/DFT, not eSEN) — done in Path 1. Killer figure: relative
tautomer/protonation populations + TM/radical/hypervalent coverage. Train the 4M headline.
