# P6 — Validation of the FFJORD log-likelihood estimator

**Headline: the Hutchinson trace estimator is fine. The ODE discretization is not.
`log p_θ` does not converge in `n_ode_steps`, and the headline per-group `r`
(and the energy-vs-FM-only arm gap) is a strong function of that hyperparameter.**

The BGFM headline number is a per-group Pearson `r = corr(log p_θ, −E_xTB/kT)` where
`log p_θ` comes from an FFJORD reverse-time Euler ODE (12 steps) with a stochastic
Hutchinson divergence (2 Rademacher probes). If that estimate carries
geometry-correlated noise or bias, `r` could be an artifact of the estimator rather
than a property of the learned density. This is the check a reviewer will demand.

## What was run

| item | value |
|---|---|
| script | `/n/home04/yulili/bgfm/scripts/validate_likelihood_estimator.py` |
| formatter | `/n/home04/yulili/bgfm/scripts/format_estimator_report.py` |
| launchers | `scripts/fasrc/validate_estimator_cpu.slurm` (energy arm), `validate_estimator_cpu_fmonly.slurm` (FM-only control), `validate_estimator.slurm` (GPU, full grid), `validate_estimator_reps.slurm` (8 repeats) |
| checkpoints | energy arm `runs/abl/a3_energy_only_s2/lightning_logs/version_30674363/checkpoints/midstep-step=30000.ckpt`; control `runs/abl/a1_fm_only_s2/.../midstep-step=30000.ckpt` |
| molecules | 10 held-out OMol25 val molecules, 8–22 atoms, **identical set for both arms** |
| geometries | 1 reference + 8 Gaussian perturbations, σ = 0.15 Å — **frozen**, generated from a per-group `torch.Generator` and cached to `geometries.json`, so every estimator setting sees byte-identical coordinates |
| grid | `n_ode_steps ∈ {4, 12, 48}` × `n_hutchinson ∈ {1, 2, 8}`, 3 independent repeats each |
| exact check | `divergence_exact_atomwise` (3N backward passes/step) on 3 molecules of 8/13/14 atoms, `n_ode_steps=12`, vs 6 Hutchinson repeats at each probe count |
| energies | GFN2-xTB single points — same binary and code path as `eval_boltzmann_independent.py` |
| results | `/n/holylabs/woo_lab/Lab/yulili/bgfm/runs/eval_ours/p6_estimator/{energy_s2_cpu,fmonly_s2_cpu}/` |

**Harness reproduces the published pipeline.** At the headline setting (12 steps,
2 probes) this harness gives mean per-group `r = 0.463 ± 0.050` on the energy arm,
against the published `0.424` for the same checkpoint (`runs/eval_ours/abl_a3_energy_only_s2/boltz_independent.json`,
n = 30 groups). Different molecule subset, same answer — the harness is measuring
the same quantity as the headline eval.

## 1. The Hutchinson trace estimator is NOT the problem

**Unbiased against exact divergence.** At `n_ode_steps=12`, mean `log p` bias of the
Hutchinson estimate relative to the exact atomwise divergence, over 6 repeats:

| group | n_atoms | probes=1 | 2 | 4 | 8 | (repeat SD at 8 probes) |
|---|---|---|---|---|---|---|
| 4 | 8 | −3.47 | +1.66 | +2.18 | −0.11 | 3.74 |
| 5 | 13 | +0.98 | +0.67 | −0.08 | +1.38 | 2.56 |
| 2 | 14 | +1.02 | +0.27 | −1.22 | +0.04 | 3.40 |

Every bias is within roughly one repeat-SD of zero, and the SD falls with probe
count as `1/√k` should. No detectable systematic error.

**Noise is not geometry-correlated.** Correlation of the per-record estimator SD
(measured on within-group-centred `log p`, the only part `r` sees) against geometry,
energy-arm:

| ODE steps | probes | corr(SD, n_atoms) | corr(SD, RMSD) | corr(SD, −E) pooled | mean within-group corr(SD, −E) |
|---|---|---|---|---|---|
| 12 | 2 | −0.057 | 0.162 | −0.061 | 0.024 |
| 12 | 8 | 0.082 | 0.140 | 0.049 | −0.233 |
| 48 | 8 | 0.261 | −0.006 | −0.058 | −0.084 |

Nothing systematic at the headline setting. Full table in `report.json`.

**`r` is flat in probe count at fixed ODE resolution.** Energy arm at 12 steps:
`r = 0.467 / 0.463 / 0.501` for 1 / 2 / 8 probes, while the estimator SD falls
7.75 → 5.97 → 2.97. The headline number is not probe-limited, and the
noise-attenuation correction barely moves it (0.516 / 0.494 / 0.510).

**Null control passes.** Pairing each group's `log p` with a *different* group's
energy vector (all cross-group pairings) gives mean `r ≈ −0.07` once the
unperturbed reference conformer is dropped. The pipeline cannot manufacture a
correlation from noise plus analysis choices.

**Common random numbers buy nothing.** A CRN cell (one probe pattern tiled across
the geometries of a group, via the new `xi_provider` hook) gave the same
within-group-centred SD as ordinary sampling (3.68 vs 3.68, FM-only 12 steps).
The trace noise is dominated by per-geometry Jacobian differences, not by the
shared probe draw.

## 2. The ODE discretization IS the problem

**`log p` does not converge in `n_ode_steps`.** Mean `log p` and the within-group
spread both grow without plateauing:

| n_ode_steps | mean log p (energy arm) | within-group signal SD | mean log p (FM-only) | signal SD |
|---|---|---|---|---|
| 4 | −65.5 | 7.7 | −86.0 | 4.4 |
| 12 | −114.3 | 16.3 | −108.9 | 7.8 |
| 48 | −199.9 | 42.3 | −126.2 | 15.4 |

A converged estimate would settle. Instead each refinement adds ≈ 50–85 nats and
roughly doubles the within-group spread. Mean bias relative to the 48-step/8-probe
cell is +134 (4 steps) and +86 (12 steps) — an order of magnitude larger than the
within-group signal the metric is built on.

**Likely cause.** `log_density_via_flow` integrates on a uniform midpoint grid
`t_k = 1 − (k+½)/n`, so the first evaluation sits at `t = 0.875` for n=4 but
`t = 0.9896` for n=48. The FM velocity/score is stiff as `t → 1` (the score carries
a `1/(1−t)` factor — see `score_from_fm_velocity`), so refining the grid pushes
sampling into the singular region and the divergence integral picks up an ever
larger contribution. The observed growth is roughly linear in `log n`, consistent
with a logarithmic divergence. **This is a hypothesis fitted to 3 grid points, not
a proof** — it should be confirmed by holding the first-node time fixed while
varying `n`, which this study did not do.

**Mitigating detail:** the discretization error is mostly a *per-molecule constant*.
Its within-group correlation with energy is small (−0.11 to +0.16 across all cells).
So the discretization does not directly inject an energy-shaped signal; it changes
`r` by reshaping the within-group spread, not by faking a correlation.

## 3. Consequence: the headline `r` and the arm gap depend on `n_ode_steps`

Mean per-group `r` on identical geometries, **reference conformer dropped**
(the unperturbed conformer is a leverage point — see below):

| cell | energy arm | FM-only | gap |
|---|---|---|---|
| 4 steps / 1 probe | 0.518 | 0.065 | **+0.452** |
| 4 steps / 2 probes | 0.628 | 0.259 | **+0.369** |
| 4 steps / 8 probes | 0.787 | 0.298 | **+0.489** |
| 12 steps / 1 probe | 0.375 | 0.187 | **+0.189** |
| **12 steps / 2 probes (headline)** | **0.347** | **0.163** | **+0.183** |
| 12 steps / 8 probes | 0.398 | 0.237 | **+0.161** |
| 48 steps / 1 probe | 0.298 | 0.242 | +0.056 |
| 48 steps / 2 probes | 0.307 | 0.319 | −0.012 |
| 48 steps / 8 probes | 0.321 | 0.271 | +0.050 |

With the reference conformer **included** the 48-step gap is negative in all three
probe settings (−0.084 to −0.116), i.e. the FM-only arm scores *higher*.

The gap is monotone in ODE resolution: +0.45 → +0.18 → +0.05. Since neither end of
the grid is converged, no single value is "the truth" — but the claim
"energy arm ≫ FM-only" as measured by this metric is **not robust to the
`n_ode_steps` choice**, and 12 was picked for cost, not for accuracy.

Per-group SEM is 0.04–0.10 on 10 molecules, so the 48-step gap (+0.05) is
indistinguishable from zero and the 12-step gap (+0.18) is ~2–3 SEM. The
qualitative collapse across resolutions is much larger than the sampling noise.

## 4. Two side findings worth handing to P2 and P3

**Calibration slope (P2).** The OLS slope of `log p` on `−E/kT` with kT = 1 eV is
0.91–1.02 for the energy arm at every resolution (1.35 at 48 steps/1 probe), versus
0.10–0.33 for FM-only at 4 and 12 steps. Slope is robust to zero-mean noise in
`log p`, so this discriminates the arms far more cleanly than `r` does, and it says
the energy arm's effective temperature is close to the kT = 1 eV it was trained at.
Note NRV is nonetheless large (5.3 at the headline cell, 31 at 48 steps): slope ≈ 1
with large residual scatter.

**Geometry-distance confound (P3).** At the headline cell, pooled within-group-centred
`r = 0.420`, but partial `r(log p, −E | RMSD) = 0.311`; `corr(−E, RMSD) = −0.60`.
Once the reference conformer is dropped the confound largely disappears
(pooled 0.325 → partial 0.315). So the RMSD confound rides almost entirely on the
unperturbed reference point, which is consistent with the team's existing
with-ref/drop-ref analysis. The null control makes the same point: shuffled-group
`r` is +0.13 with the reference included and −0.07 without it.

## Verdict

1. **Not an estimator-noise artifact.** Hutchinson is unbiased against exact
   divergence, its noise is uncorrelated with geometry and energy, `r` is flat in
   probe count, and the shuffled-energy null is ~0. The simple "the correlation is
   manufactured by noise" objection is answered.
2. **But the metric is not estimator-independent.** `log p` does not converge in
   `n_ode_steps`, and both `r` and the energy-vs-FM-only gap fall by ~4x from 4 to
   48 steps, reaching zero at 48. Reporting `r` at `n_ode_steps=12` without this
   caveat is not defensible.
3. **Recommended before the headline number is used:** fix the integration scheme
   (integrate to a fixed `t = 1 − ε` independent of `n`, or use a time change /
   adaptive solver, or score the smoothed marginal `p_{t*}` at a fixed `t* < 1`),
   then re-establish convergence in `n` before quoting any `r`. Prefer the
   calibration slope over `r` as the headline statistic — it is far more stable
   across estimator settings.

## Caveats on this study

- **n = 10 molecules, 8–22 atoms.** Small, and smaller than the headline eval's
  unrestricted set. Per-group SEM 0.04–0.10.
- One seed per arm (s2). The 4 other seeds were not run here.
- 3 repeats per cell; the 8-repeat job is still queued.
- The stiffness explanation for non-convergence is inferred from 3 grid points and
  the `1/(1−t)` structure of the score, not demonstrated by a controlled experiment.
- Runs were done on CPU (`serial_requeue`) because the GPU partitions were saturated
  by the Priority-0 training jobs; results are identical in kind, just slower.

## Still queued (will extend, not overturn, the above)

| job | what it adds |
|---|---|
| `p6_estimator` (GPU chain, `runs/eval_ours/p6_estimator/energy_s2`, `fmonly_s2`) | full 5×4 grid (adds 8 and 24 steps, 4 probes), 16 molecules up to 40 atoms |
| `p6_estimator_reps` | 8 repeats at 12/48 steps × 2/8 probes — tighter estimator-SD numbers |
| CRN cells `12:2`, `48:8` on the energy arm | variance-reduced cross-check of §1 |

Re-running the analysis is cheap and needs no GPU:

```bash
python scripts/validate_likelihood_estimator.py --analyze_only \
    --out_dir /n/holylabs/woo_lab/Lab/yulili/bgfm/runs/eval_ours/p6_estimator/energy_s2_cpu
python scripts/validate_likelihood_estimator.py --analyze_only --drop_ref --out_dir <same>
python scripts/format_estimator_report.py <out_dir> [<out_dir> ...]
```

## Code change made outside this script

`divergence_exact_atomwise` (`cfm_mol/bgfm_loss.py`) gained a `create_graph`
argument, and `log_density_via_flow` (`cfm_mol/bgfm_density.py`) now passes
`create_graph=for_training` to it. The old default built a retained second-order
graph across all 3N backward passes, which OOMs at eval time — the exact-divergence
reference path was effectively unusable before this. Default behaviour is unchanged
when the argument is omitted; `tests/test_bgfm_loss.py` and `tests/test_bgfm_density.py`
(26 tests) still pass.
