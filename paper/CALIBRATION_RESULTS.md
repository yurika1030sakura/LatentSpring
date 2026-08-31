# P2 — Calibration metrics and the oracle ceiling

Status: **run and complete** on all 11 existing wide-eval arms (n = 120 parent
molecules × 9 geometries each). Ceiling measured on the same 1080 geometries.

Scripts
- `scripts/analyze_calibration.py` — NRV / slope / T_eff / scale-ratio / r / rho
- `scripts/measure_oracle_ceiling.py` — eSEN teacher vs independent GFN2-xTB

Artifacts (holylabs, not in the repo)
- `runs/eval_ours/calibration/calibration_withref.{json,md}`
- `runs/eval_ours/calibration/calibration_dropref.{json,md}`
- `runs/eval_ours/ceiling_full/{ceiling_summary.json,ceiling_per_group.csv,esen_energies.jsonl}`

---

## 1. What the metrics are

For parent molecule *m* with K geometries, `u = log p_theta`, `E` = independent
GFN2-xTB energy in eV, `beta = 1/kT`, kT = 1.0 eV (the training config value):

| metric | definition | ideal |
|---|---|---|
| **NRV** | `Var_k[u + beta E] / Var_k[beta E]` | **0** |
| slope | OLS slope of `u` on `-E/kT` | 1 |
| T_eff | `kT / slope` = `-Var(E)/Cov(u,E)` (kT-invariant) | 1.0 eV |
| scale-ratio | `std(u) / std(beta E)` | 1 |
| NRV_min | `1 - r^2` — the best NRV *any* temperature rescaling could buy | 0 |
| r, rho | Pearson / Spearman of `u` vs `-E` | 1 |

Two identities make the results readable:

```
NRV(kT) = a*kT^2 + 2b*kT + 1,   a = Var(u)/Var(E),  b = Cov(u,E)/Var(E)
        -> parabola in kT, minimum 1 - r^2   at   kT* = -b/a
slope   = r * scale-ratio
```

So `slope` factorises into **direction** (`r`) and **magnitude** (`scale-ratio`).
Pearson r is blind to the magnitude factor entirely — that blindness is exactly
what P2 asks us to remove.

`NRV = 1` means log p is *no more useful than a constant*. `NRV > 1` means the
model's log p variation actively hurts relative to predicting a flat density.

---

## 2. Headline: all arms are un-calibrated (NRV >~ 1)

Seed-aggregated, kT = 1.0 eV, **all 9 geometries** (reference + 8 perturbations):

| arm | seeds | NRV median | NRV_min (=1-r^2) | slope median | scale-ratio | T_eff (eV) | T_eff (K) | r mean |
|---|---|---|---|---|---|---|---|---|
| A1 FM-only | 4 | **1.304 ± 0.045** | 0.844 ± 0.011 | 0.053 ± 0.009 | 0.559 | **19.3 ± 3.7** | 2.2e5 | 0.074 ± 0.010 |
| A3 energy | 4 | **1.211 ± 0.361** | 0.708 ± 0.060 | 0.491 ± 0.089 | 1.093 | **2.09 ± 0.41** | 2.4e4 | 0.434 ± 0.063 |
| A6 energy-shuffled | 2 | 1.107 ± 0.019 | 0.797 ± 0.013 | 0.227 ± 0.012 | 0.707 | 4.42 ± 0.23 | 5.1e4 | 0.303 ± 0.018 |
| A6 shuffled-stab | 1 | 0.974 | 0.738 | 0.325 | 0.794 | 3.08 | 3.6e4 | 0.418 |

**Reference-dropped** (8 perturbations only, the honest setting):

| arm | seeds | NRV median | NRV_min | slope median | scale-ratio | T_eff (eV) | r mean |
|---|---|---|---|---|---|---|---|
| A1 FM-only | 4 | **1.390 ± 0.073** | 0.830 ± 0.021 | 0.151 ± 0.023 | 0.838 | 6.75 ± 1.13 | 0.218 ± 0.037 |
| A3 energy | 4 | **2.024 ± 0.754** | 0.736 ± 0.030 | 0.548 ± 0.107 | 1.420 | **1.88 ± 0.38** | 0.397 ± 0.044 |
| A6 energy-shuffled | 2 | 1.689 ± 0.037 | 0.803 ± 0.024 | 0.212 ± 0.033 | 1.051 | 4.77 ± 0.75 | 0.233 ± 0.020 |
| A6 shuffled-stab | 1 | 1.695 | 0.785 | 0.284 | 1.152 | 3.53 | 0.296 |

### Reading

1. **No arm is Boltzmann-calibrated.** Median NRV is at or above 1 everywhere.
   A model that emitted a *constant* log p would score NRV = 1 exactly. The
   energy arm beats that bar on only ~42% of molecules (with reference) and ~27%
   (reference dropped).

2. **T_eff ~ 2 eV for the energy arm vs a 1.0 eV training target.** The learned
   density is roughly **2× too hot** — i.e. too flat, under-weighting the energy.
   FM-only is 19 eV, ~19× too hot: effectively no thermodynamic structure at all.
   Reporting T_eff in K (2.4e4 K vs 2.2e5 K) makes the gap vivid, but note that
   kT = 1.0 eV = 11 600 K is itself the *target*, so neither number is a physical
   temperature — the point is the ratio to the target.

3. **The energy loss fixes the magnitude, not just the sign.** scale-ratio goes
   0.56 → 1.09 (with ref) / 0.84 → 1.42 (dropped). The model's log-density
   dynamic range within a group becomes approximately correct — a fact Pearson r
   literally cannot express, and the strongest new positive result from P2.

### Paired per-molecule test (same 120 molecules, seed-median per molecule)

energy arm minus FM-only, Wilcoxon signed-rank:

| metric | with reference | reference dropped |
|---|---|---|
| **NRV** | Δ = −0.083, **p = 0.87 (n.s.)** | Δ = **+0.337, p = 9.2e−7 (WORSE)** |
| NRV_min (1−r²) | Δ = −0.132, p = 6.7e−10 (better) | Δ = −0.065, p = 4.7e−7 (better) |
| slope | Δ = +0.394, p = 1.3e−18 | Δ = +0.366, p = 7.9e−17 |
| scale-ratio | Δ = +0.383, p = 3.4e−19 | Δ = +0.471, p = 4.7e−19 |
| Pearson r | Δ = +0.340, p = 3.5e−19 | Δ = +0.210, p = 2.1e−12 |

**This is the paper-relevant result.** On correlation the energy arm wins
decisively (Δr = +0.34, p ~ 1e−19, matching the previously reported +0.360).
On the *calibration* metric the same comparison is **not significant with the
reference geometry, and significantly worse without it**.

The decomposition explains why: the energy loss adds a large, correctly-scaled
log-density variation of which only ~r² ≈ 19% of the variance is aligned with
energy. The remaining ~81% is injected noise, which NRV charges for and Pearson
does not. `NRV_min` — the NRV attainable after an optimal temperature rescale —
*does* improve significantly (0.83 → 0.74), confirming the signal is real; it is
the un-modelled residual that keeps total NRV above 1.

---

## 3. Oracle ceiling (eSEN teacher vs independent GFN2-xTB)

**Question this answers:** is r = 0.43 near the maximum achievable, or far from
it? We had no idea. Now we do.

Method: score all 1080 geometries with the eSEN teacher (fairchem OMol25,
`esen_sm_conserving_all.pt`, CPU, 1098 s, 0 failures), reuse the existing xTB
energies, and push `u = -E_eSEN / kT` — the log-density a **perfect** student
would have, up to the per-molecule constant every metric is invariant to —
through the identical calibration code.

| | with reference | reference dropped |
|---|---|---|
| ceiling Pearson r (mean) | **0.958** [95% CI 0.921, 0.978] | **0.932** [0.891, 0.959] |
| ceiling Pearson r (median) | 0.986 | 0.978 |
| ceiling Spearman (mean) | 0.941 | 0.923 |
| ceiling slope (mean / median) | 0.884 / 0.934 | 0.767 / 0.793 |
| ceiling T_eff (eV) | 1.07 | 1.26 |
| ceiling NRV (median) | **0.033** | **0.089** |
| ceiling NRV_min (mean) | 0.052 | 0.094 |
| groups with r > 0.8 | 98% | 92% |
| eSEN−xTB relative-energy RMSE | 5.11 eV mean / 3.01 eV median | 5.34 / 3.16 |

### 3.1 The main conclusion: r = 0.43 is nowhere near the ceiling

| metric | FM-only | energy arm | **ceiling** | energy / ceiling |
|---|---|---|---|---|
| Pearson r (with ref) | 0.074 | 0.434 | 0.958 | **45%** |
| Pearson r (dropped) | 0.218 | 0.397 | 0.932 | **43%** |
| slope (with ref) | 0.053 | 0.491 | 0.884 | 56% |
| slope (dropped) | 0.151 | 0.548 | 0.767 | 71% |
| NRV median (with ref) | 1.304 | 1.211 | 0.033 | 37× worse |

**The eSEN/xTB disagreement is not the bottleneck.** A student that matched the
teacher exactly would score r ≈ 0.96 and NRV ≈ 0.03 on our own reported metric.
We score 0.43 and 1.21. The missing 0.52 of correlation is genuinely unlearned
model capacity, not oracle noise — which is consistent with training having seen
only ~12% of one epoch, and means the headline number should be presented as an
early-training signal with large headroom, **not** as an approach to a limit.

The slope column is the more encouraging read: at 56–71% of ceiling, the energy
arm's *response magnitude* is much closer to correct than its *direction*.

### 3.2 The ceiling is high only because the ensemble is absurdly high-energy

The eSEN–xTB relative-energy disagreement is **3.0 eV RMSE (median per group)**.
The ceiling is nevertheless 0.96 purely because the perturbations span
**std(E) ≈ 20.7 eV** — the disagreement is only 17% of the signal.

Modelling the disagreement as additive noise of fixed scale σ_d = 3.0 eV, the
ceiling for an ensemble whose true per-group energy spread is σ_E would be
`r_ceiling ≈ σ_E / sqrt(σ_E² + σ_d²)`:

| ensemble spread σ_E | 0.026 eV (room-T) | 0.1 eV | 0.5 eV | 1 eV | 5 eV | 20 eV (ours) |
|---|---|---|---|---|---|---|
| projected r_ceiling | 0.01 | 0.03 | 0.16 | 0.32 | 0.86 | 0.99 |

**Consequence for P1 and P3:** GFN2-xTB cannot serve as the independent oracle on
any thermally-relevant ensemble. At the energy scales that separate real
conformational basins (0.1–1 eV), xTB's disagreement with eSEN would swamp the
signal and the measurable ceiling would collapse to r ≈ 0.03–0.32. The global
ensemble experiment (P1) and the fixed-RMSE-shell experiment (P3) will need
either a stronger independent reference (DFT single points) or an explicit,
directly-measured ceiling on their own ensemble.

Caveat on the projection: it assumes σ_d is independent of distortion magnitude,
which is pessimistic — the two potentials likely agree better on near-equilibrium
structures. It is an estimate that motivates measuring the ceiling directly on
each new ensemble, which `measure_oracle_ceiling.py` now makes a ~20-minute job.

---

## 4. Caveats that P2 exposes

1. **The evaluation is not thermally relevant.** Per-group `std(E_xTB)` is
   **20.7 eV** (median 15.0, max 75). At kT = 1.0 eV that is ~21 kT of spread;
   at room temperature it is ~800 kT. The σ = 0.15 Å isotropic Gaussian
   perturbations produce structures whose energy differences are dominated by
   bond stretching, not by any accessible thermal fluctuation. Any "Boltzmann
   fidelity" claim built on this ensemble is a claim about a regime no physical
   system samples. (Directly motivates P3 and P1.)

2. **NRV is heavy-tailed** — groups with small `Var(E)` or slope ≈ 0 blow up the
   plain mean (one FM-only seed has mean NRV = 7349 from a single group). Quote
   the **median**; the script also reports a 10% trimmed mean and bootstrap CIs.

3. **The reference geometry is a leverage point.** It is the only unperturbed
   structure and sits far below the other 8 in energy, so including it inflates
   every metric. All conclusions should be read off the reference-dropped table.

4. **kT = 1.0 eV is a training convenience, not a temperature.** T_eff should be
   reported as a *ratio to the training kT* (energy arm ≈ 2.0×) rather than in
   kelvin, to avoid implying a physical claim.

---

## 5. Recommended reporting change for the paper

Replace the headline "per-group Pearson r = 0.434 vs 0.074" with a table of:

| | FM-only | energy | ceiling |
|---|---|---|---|
| Pearson r | 0.074 | 0.434 | 0.958 |
| slope (ideal 1) | 0.053 | 0.491 | 0.884 |
| T_eff / kT_train (ideal 1) | 19.3 | 2.09 | 1.07 |
| **NRV (ideal 0)** | 1.304 | 1.211 | 0.033 |

and state plainly that the energy loss moves slope and T_eff a long way toward
calibration while leaving NRV above 1 — i.e. it learns the *direction and scale*
of the Boltzmann response but not yet a normalisable density. That is a weaker
but far more defensible claim than the correlation number alone supports, and it
is the claim the current 12%-of-one-epoch training budget actually licenses.

---

## 6. How to reproduce

```bash
export PYTHONNOUSERSITE=1
PY=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/omol25/bin/python
cd /n/home04/yulili/bgfm

ARGS=""
for d in /n/holylabs/woo_lab/Lab/yulili/bgfm/runs/eval_ours/wide_*/boltz_records.csv; do
  n=$(basename $(dirname $d)); ARGS="$ARGS --records ${n#wide_}=$d"
done
OUT=/n/holylabs/woo_lab/Lab/yulili/bgfm/runs/eval_ours/calibration
$PY scripts/analyze_calibration.py $ARGS --kT_eV 1.0 \
    --out_json $OUT/calibration_withref.json --out_md $OUT/calibration_withref.md
$PY scripts/analyze_calibration.py $ARGS --kT_eV 1.0 --drop_reference \
    --out_json $OUT/calibration_dropref.json --out_md $OUT/calibration_dropref.md

# ceiling (~25 min on 1 CPU; resumable via esen_energies.jsonl)
$PY scripts/measure_oracle_ceiling.py \
  --samples_json /n/holylabs/woo_lab/Lab/yulili/bgfm/runs/eval_ours/wide_a3_energy_only_s1/boltz1/boltzmann_samples.json \
  --records_csv  /n/holylabs/woo_lab/Lab/yulili/bgfm/runs/eval_ours/wide_a3_energy_only_s1/boltz_records.csv \
  --out_dir      /n/holylabs/woo_lab/Lab/yulili/bgfm/runs/eval_ours/ceiling_full
```

All 11 wide-eval arms share **byte-identical geometries** (verified: max position
difference 0.0 across `a1`, `a3`, `a6`), so one ceiling run applies to every arm.
