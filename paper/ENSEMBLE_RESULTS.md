# P1 — Global-ensemble (between-basin) Boltzmann evaluation

**Status: complete at n = 12 systems × 4 arms (2 seeds × {energy, FM-only}).
Every number below is copied from a file on disk — see §6 for provenance.
Jobs extending to all 60 systems are queued and write to the same paths.
One result is blocking (§4.2: the FFJORD estimator is not converged in ODE
steps), and it is stated as blocking rather than buried.**

## 0. One-paragraph summary

The model does **not** put Boltzmann-correct relative mass on different
conformational basins. Across 12 molecules × 3–8 xTB basins each and all four
checkpoints tested, the error on relative basin free energies is **32–92×
(median) the true spread**, and the model's log-density varies across basins by
**38–147× more than a Boltzmann distribution at the training temperature
allows**. The energy loss term reliably improves *within*-basin ranking
(paired p = 0.014, reproducing the existing headline) but gives **zero
improvement between basins** (p = 0.22–0.35) and is significantly **worse** on
calibration metrics (p = 0.001–0.037). Caveat that must be resolved before
publication: the FFJORD log-density is not converged in ODE steps, so the
qualitative conclusion holds at every resolution tested but the precise
magnitudes do not.

## 1. Why this experiment exists

Every Boltzmann-fidelity number BGFM currently reports is *within one basin*:
take one held-out geometry, jitter it with `sigma = 0.15 A` Gaussian noise, and
correlate `log p_theta` with `-E/kT` over that little cloud. That measures local
ranking only.

The Boltzmann property is far stronger. If `p_theta(x) = exp(-E(x)/kT)/Z` then

```
u(x) := log p_theta(x) + E(x)/kT = -log Z
```

is the **same constant everywhere**, not merely constant-ish inside each basin.
A model can achieve r ≈ 0.9 inside every basin and still be arbitrarily wrong
about `p(basin A) / p(basin B)`, because each basin is free to carry its own
offset. Relative basin populations are exactly what a generative model is used
for downstream (conformer populations, selectivity, binding poses), so this gap
is the difference between "learned a local energy gradient" and "learned the
Boltzmann distribution".

## 2. What was built

### 2.1 Reference ensembles — `scripts/build_reference_ensembles.py`

Option (c) of the three proposed (RDKit ETKDG → xTB optimise → RMSD cluster),
chosen because it does not depend on xtb MD/metadynamics stability.

1. Select OMol25 **val** molecules: 10–30 atoms, organic elements only
   (H C N O F S Cl Br P I), **single covalent fragment** (OMol25 val is full of
   non-covalent complexes — `CC.CC` etc. — which have no torsional basins),
   no radicals, ≥ 2 rotatable bonds.
2. RDKit `rdDetermineBonds` bond perception from the OMol25 xyz.
3. ETKDGv3 multi-conformer embedding (80 attempts, `pruneRmsThresh = 0.12`),
   **plus the OMol25 data geometry itself** as an extra starting point.
4. GFN2-xTB `--opt normal` on every starting geometry → local minima.
5. Reject any minimum whose canonical SMILES changed (xTB sometimes isomerises
   or dissociates) — a "basin" must be the same molecule.
6. Greedy clustering of the surviving minima by **heavy-atom `GetBestRMS`**
   (symmetry-aware), threshold 0.35 Å → basins.
7. Keep systems with ≥ 3 basins and ≥ 0.02 eV energy spread.

Gotcha found and fixed: **RDKit ETKDG with `randomSeed = 0` produces a single
conformer** (every embedding attempt returns identical coordinates, then
`pruneRmsThresh` removes the duplicates). The first build silently produced
2 starting geometries per molecule and 0 usable systems. Seeds are now mapped
onto a nonzero stream.

### 2.2 Evaluation — `scripts/eval_global_ensemble.py`

**Why density-on-reference-geometries and not model sampling.** FlowMol3's
sampler generates `(x, a, c, e)` jointly from the prior and resamples the CTMC
discrete channels at every step; there is no composition-conditioning API, so
`model.sample()` cannot be pinned to a fixed molecular formula without building
new conditional-sampling machinery (discrete-channel inpainting/guidance). We
therefore evaluate `log p_theta` directly on the reference basin geometries via
the FFJORD reverse-time ODE. This needs no new sampler and answers the question
(relative basin mass) more directly than sampling would at this sample budget.

Stage `logp` (GPU or CPU): for every basin minimum `x*_b`, evaluate `log p_theta`
at the minimum and at `m = 8` Gaussian perturbations `x*_b + delta`,
`delta ~ N(0, sigma^2)` drawn in the COM-free subspace (`sigma = 0.15 A`, the
same sigma as the existing local eval, so the two are directly comparable).
The perturbations double as importance samples from a **known** proposal.

Stage `analyze` (CPU, xtb): GFN2-xTB single points on every geometry, then:

* **(1) Pointwise across basins** — regress `log p_theta(x*_b)` on `-beta E(x*_b)`
  over basins: Pearson r, slope, `NRV = Var_b[log p + beta E] / Var_b[beta E]`
  (ideal 0), plus the partial correlation `r(log p, -beta E | R_g)` which kills
  the "the model just prefers compact geometries" confound.
* **(2) Basin mass** — self-normalised importance sampling with the *same*
  Gaussian proposal for every basin, so the proposal's normalising constant
  cancels:
  `log Zhat_theta(b) = LSE_i[log p_theta(x_i) - log q(x_i)] - log m`,
  `log Zhat_ref(b)   = LSE_i[-beta E(x_i)     - log q(x_i)] - log m`,
  with `log q(x_i) = -||delta_i||^2 / (2 sigma^2) + const`.
  Occupancies `P = softmax_b log Zhat`. Reported: KL, total variation, W1 on the
  basin-energy axis (eV), importance-weight ESS over basins, top-1 agreement,
  and the regression of all pairwise `Delta log Zhat_theta` on
  `Delta log Zhat_ref` (ideal slope 1).
* **(3) Within/between decomposition** — `u = log p_theta + beta E` split into
  within-basin and between-basin variance. The headline interpretable number is
  `dF_err = sqrt(Var_b[u_b]) * kT` in eV: **the error the model makes on relative
  basin free energies**. Its natural yardstick is `dF_true = sqrt(Var_b[beta E_b]) * kT`,
  the true spread — a model that is *flat across basins* scores
  `dF_err / dF_true = 1`, a Boltzmann model scores 0.
* **Effective temperature** `kT_eff = kT_eval / slope`, invariant to the
  (arbitrary) `kT_eval` used for reporting.
* **Confound check** — whether the model's top-1 basin is the basin the OMol25
  *data* geometry relaxes into (a memorisation model would always say yes).

### 2.3 Estimator work that was necessary to make the experiment possible

The inter-basin Boltzmann signal at `kT = 1 eV` (the training temperature) is
only `beta * Delta E ~ 0.1–0.7` nats, while the **Hutchinson** FFJORD estimator
error on a single `log p` measured here is **~6 nats** — 30× larger than the
effect. Two fixes were implemented:

1. **Common random numbers** (`cfm_mol/bgfm_loss.py::divergence_hutchinson`
   gained an optional `xi_provider`): tile one probe across all graphs in the
   batch, since they are all the same molecule. Measured: noise on log p
   *differences* drops only from 7.79 → 6.26 nats. **Not enough** — the error
   term `sum_{i!=j} xi_i xi_j J_ij` changes as fast as `J` does between
   geometries.
2. **Exact blocked divergence** (the default, `--exact_div`). The batched DGL
   graph is block-diagonal, so a probe equal to `sqrt(3N) * e_(i,c)` *replicated
   in every graph* returns, per graph, exactly `3N * J_(i,c),(i,c)`. Averaging
   the `3N` such probes — which is what `divergence_hutchinson` already does —
   yields the **exact** trace for every geometry at a cost of `3N` backward
   passes *total*, independent of how many geometries are in the batch.
   Stochastic error: **zero** (verified: repeat-to-repeat std = 0.000).
   Correctness verified against `divergence_exact_atomwise` on a synthetic
   block-diagonal vector field (max abs err 1.9e-6).

Without (2) this experiment is not measurable at all; that is worth a paragraph
in the paper's appendix.

## 3. Data actually built

`/n/holylabs/woo_lab/Lab/yulili/bgfm/processed_data/ensembles/`

| quantity | value |
|---|---|
| systems kept | 60 |
| basins total | 349 |
| basins / system | mean 5.8 (range 3–8) |
| atoms / system | mean 16.3 (range 10–29) |
| rotatable bonds | mean 3.7 |
| basin energy spread | mean 0.160 eV, median 0.115 eV, max 0.655 eV |
| min inter-basin heavy-atom RMSD | mean 0.53 Å, min 0.35 Å |
| build outcome | 72 ok / 9 topology change / 12 single basin / 10 degenerate |

## 4. Results

### 4.1 The headline number, and it does not depend on the estimator

`scripts/compare_ode_resolution.py`, energy arm (`a3_energy_only_s2`,
step 30000), 4 systems × 8 basins, kT = 1 eV (the training temperature):

| n_ode_steps | model basin log p spread (nats) | Boltzmann-allowed spread (nats) | ratio |
|---|---|---|---|
| 6  | 1.664 | 0.0553 | **38×** |
| 12 | 4.541 | 0.0553 | **147×** |

Read this as: a Boltzmann model at kT = 1 eV is only *allowed* to vary its
log-density across these basins by ≈ 0.06 nats, because the basins differ in
energy by only ≈ 0.06 eV. The model varies by 1.7–4.5 nats. **The learned
density is between one and two orders of magnitude too structured across
basins** — it is emphatically not assigning Boltzmann relative masses to
conformational basins.

Per-system detail (spread in nats):

| system | basins | ode6 | ode12 | Boltzmann | corr(ode6, ode12) |
|---|---|---|---|---|---|
| 8882  | 8 | 1.261 | 9.163 | 0.0229 | 0.727 |
| 9486  | 8 | 1.606 | 3.503 | 0.0358 | 0.819 |
| 17711 | 8 | 2.294 | 2.233 | 0.1156 | −0.064 |
| 18563 | 8 | 1.496 | 3.267 | 0.0468 | 0.656 |

### 4.2 …but the estimator is NOT converged, and that limits everything else

The last column above is a warning, not a footnote. The mean-centred basin
log p vectors at 6 and 12 ODE steps correlate only **0.53 on average
(min −0.06)**, and the spread itself grows 2.7× from 6 to 12 steps. The
*ordering and spacing of basins under the model is not yet a converged
quantity* at these settings.

What survives: the ratio to the Boltzmann-allowed spread is 38× at the coarser
setting and 147× at the finer one — the qualitative conclusion ("wrong by
orders of magnitude") holds at both, because the true spread is so small.
What does not survive: any specific number for the free-energy error, any
per-system ranking, and any close energy-arm vs control-arm comparison of these
between-basin quantities. **A convergence study (P6: ODE steps 4/8/12/24/48)
is a prerequisite for putting a between-basin number in the paper.** A 24-step
run is queued (`energy_s2_ode24`).

### 4.3 The between-basin failure is universal across arms

12 systems (69 basins), 4 arms, `midstep-step=30000`, kT = 1 eV, exact blocked
divergence, 12 ODE steps. `dF_err_min` = spread across basins of
`log p_theta(x*_b) + beta E(x*_b)` × kT — the model's error on relative basin
free energies. `dF_true_min` = the true spread, 0.058 eV.

| arm | median `dF_err_min / dF_true_min` | range |
|---|---|---|
| energy_s2  (`a3_energy_only_s2`) | **91.8×** | 17.6–401.8× |
| control_s2 (`a1_fm_only_s2`)     | **32.5×** | 6.9–84.7× |
| energy_s3  (`a3_energy_only_s3`) | **32.3×** | 9.7–155.8× |
| control_s3 (`a1_fm_only_s3`)     | **65.2×** | 25.2–196.5× |

Every arm, every system: the model's relative basin free energies are wrong by
1–2 orders of magnitude. Nothing in the current training signal teaches
between-basin mass.

### 4.4 The energy term's local win does NOT transfer to between-basin mass

Paired per-system test, pooled over both seeds (n = 24 paired systems,
energy arm − control arm; positive Δ = energy arm larger):

| metric | Δ (energy − control) | t | p | reading |
|---|---|---|---|---|
| **within-basin r** (the existing headline metric) | **+0.081** | 2.64 | **0.014** | energy arm better — reproduces the known local result |
| between-basin r (basin mass) | −0.143 | −1.26 | 0.22 | **no transfer** |
| `dF_err_min_eV` | +1.12 | 0.96 | 0.35 | **no transfer** |
| `dF_err_mass_ratio` | +1.24 | 3.08 | **0.005** | energy arm **worse** |
| `nrv_basin_mass` | +11.4 | 2.22 | **0.037** | energy arm **worse** |
| `nrv_within` (calibration, not correlation) | +2.84 | 3.68 | **0.001** | energy arm **worse** |

This is the result the advisor was asking for, and it is a negative one:

1. The energy term reliably improves **ordering** inside a basin (within-basin
   r, p = 0.014) — the existing +0.36 headline is real and reproduces here.
2. It gives **no improvement at all** between basins (r, p = 0.22;
   `dF_err_min`, p = 0.35).
3. On **calibration** metrics it is significantly *worse* — both between basins
   (`nrv_basin_mass`, p = 0.037) and, notably, **within** them
   (`nrv_within`, p = 0.001). `nrv_within` worse while within-basin r is better
   means the energy arm gets the *ordering* more right and the *scale* more
   wrong: its log p varies far too much relative to `beta E`. That is exactly
   the correlation-vs-calibration distinction P2 is about, showing up
   independently here.

Per-seed the sign of `dF_err_min` flips (s2: energy worse, p = 0.007; s3:
energy better, p = 0.029), so **the seed-to-seed spread exceeds the arm
difference** on that metric — do not report a per-seed direction. The metrics
that survive pooling are the three marked significant above.

### 4.5 Confound checks

* **Radius of gyration.** `r_pointwise` at basin minima is ≈ 0 for every arm
  (−0.10 to +0.42) and the partial correlation controlling for R_g is smaller
  still (0.01–0.20). The model's between-basin density is not tracking energy,
  and it is not simply tracking compactness either — it is tracking something
  else entirely.
* **Data-geometry memorisation — ruled out.** The basin that the OMol25 data
  geometry relaxes into is flagged (`from_data_basin`). The model's top-mass
  basin is that basin only 0 % / 9 % / 9 % / 36 % of the time
  (control_s2 / energy_s3 / control_s3 / energy_s2), i.e. at or below chance.
  The model is not merely reproducing training geometries; it is putting mass
  somewhere unrelated to both the energy and the data.
* **Top-1 basin agreement with the xTB reference** is at chance across arms
  (0.42 / 0.50 / 0.50 / 0.67, with 3–8 basins per system).

## 5. Caveats and honest limitations

1. **FFJORD convergence (blocking).** See 4.2. Between-basin log p differences
   change materially between 6 and 12 ODE steps. Fix before publication:
   n_ode_steps sweep to convergence (this is exactly P6's job), then re-run.
   Stochastic error is *not* the problem — exact blocked divergence makes the
   estimate deterministic (repeat std 0.000); the residual error is pure Euler
   discretisation of the reverse-time ODE.
2. **The importance-sampling basin-mass estimator is under-resolved at
   m = 8, sigma = 0.15 A.** Measured IS effective sample size fraction:
   `is_ess_frac_ref ≈ 0.19`, i.e. ~1.5 of 8 samples carry the weight, because
   a 0.15 A displacement costs 1–3 eV and the log-sum-exp is dominated by
   whichever perturbation happened to land lowest. Consequence: everything
   derived from `log Zhat` (KL, TV, W1, `ess_frac_basins`, `dF_err_mass`,
   `r_basin_mass`) is noisy, and the reported `dF_true_mass_eV ≈ 2.5 eV` is
   proposal noise, not a real free-energy spread. The **minima-based** metrics
   (`r_pointwise`, `dF_err_min_eV`, `dF_true_min_eV ≈ 0.06 eV`) do not use IS
   and are clean. A tighter-proposal variant (sigma = 0.08 A, m = 16) is queued
   (`*_is16`); the principled fix is m ≥ 64 or a harmonic reference.
3. **kT = 1 eV is the training temperature, and it makes the reference nearly
   flat.** At kT = 1 eV the true basin occupancies differ by only ~6 %, so
   KL/TV/W1 against the reference are dominated by the model's excess structure.
   `kT_eff` (= kT_eval / slope) is reported because it is invariant to this
   choice. Room-temperature numbers can be produced from the saved
   `global_ensemble_records.csv` without re-running xTB.
4. **Basin free energy is approximated at 0 K** (potential energy of the xTB
   minimum) for the minima-based metrics. No vibrational/rotational entropy, no
   symmetry numbers. Between conformers of the same molecule these corrections
   are typically ≪ 0.1 eV but not zero.
5. **The `--n_members` count per basin is proposal-biased** (it counts how many
   ETKDG starts fell into that basin) and is deliberately *not* used as an
   occupancy estimate.
6. **Systems are small and organic** (10–29 atoms, H/C/N/O/F/S/Cl/Br, neutral,
   single fragment). Nothing here tests transition metals or charged species —
   RDKit bond perception, which we need for ETKDG, does not survive them.
7. **We evaluate density, not samples.** FlowMol3 has no composition-conditioned
   sampler (see 2.2), so this measures whether the *learned density* places
   correct relative mass on basins, not whether ancestral sampling reproduces
   basin populations. Those coincide only if sampling is exact.
8. **n = 12 systems, 2 seeds per arm.** The CPU fallback was used because the
   GPU partitions are saturated by the 30 P0 training jobs; jobs extending to
   all 60 systems (and the GPU version, `scripts/fasrc/eval_global_ensemble.slurm`)
   are queued and resume into the same output directories.
9. **The per-seed direction of the arm difference on `dF_err_min` flips**
   (§4.4). Only the pooled, repeatedly-significant metrics should be quoted.

## 5b. What to do next (in priority order)

1. **P6 convergence sweep first.** Nothing between-basin should go in the paper
   until `n_ode_steps` is converged. Run `scripts/compare_ode_resolution.py`
   over 4/8/12/24/48 and find where the mean-centred basin log p vectors stop
   moving (target: corr > 0.99 between consecutive resolutions).
2. **Fix the IS proposal**: `--sigma 0.08 --n_perturb 16` runs are queued
   (`*_is16`); check `is_ess_frac_ref` climbs from 0.19 toward ≥ 0.5. If not,
   go to m = 64 or replace the Gaussian cloud with a harmonic reference.
3. **Extend to all 60 systems and all 6 P0 cells** (A_fmonly … F_both) once the
   P0 training jobs land — the eval takes ~3 CPU-hours per arm at n = 60.
4. **Then, and only then**, consider whether a between-basin training signal is
   worth adding (e.g. an energy-consistency term whose perturbations span
   basins rather than a 0.15 A cloud). The present result says the current
   loss provides none.

## 5c. Files produced

| path | contents |
|---|---|
| `/n/holylabs/woo_lab/Lab/yulili/bgfm/processed_data/ensembles/ensembles.json` | 60 systems, basin energies + metadata |
| `.../ensembles/geoms.npz` | basin geometries (float32) |
| `.../ensembles/build_log.json` | per-candidate accept/reject reason |
| `.../runs/eval_ours/global_ensemble_cpu/<arm>/basin_logp.json` | log p per geometry |
| `.../global_ensemble_cpu/<arm>/global_ensemble_results.json` | summary + per-system metrics |
| `.../global_ensemble_cpu/<arm>/global_ensemble_per_system.csv` | per-system table |
| `.../global_ensemble_cpu/<arm>/global_ensemble_records.csv` | per-geometry (log p, E) — re-analyse at any kT without re-running xTB |
| `.../global_ensemble_cpu/<arm>/global_ensemble.png` | basin-pair scatter + local-vs-global scatter |
| `.../global_ensemble_cpu/table_n12.md` | the 4-arm comparison table of §4.3–4.4 |

## 6. How to reproduce

```bash
source /n/home04/yulili/bgfm/scripts/fasrc/env.sh

# 1. reference ensembles (CPU, ~10 min on 16 cores)
sbatch /n/home04/yulili/bgfm/scripts/fasrc/build_ensembles.slurm

# 2+3. log p on basin representatives + xTB + metrics, one arm per job
sbatch -J ge_energy_s2 /n/home04/yulili/bgfm/scripts/fasrc/eval_global_ensemble_one.slurm \
    energy_s2 a3_energy_only_s2 a3_energy_only_s2 60
sbatch -J ge_control_s2 /n/home04/yulili/bgfm/scripts/fasrc/eval_global_ensemble_one.slurm \
    control_s2 a1_fm_only_s2 a1_fm_only_s2 60
# (GPU variant, all four arms in one job: scripts/fasrc/eval_global_ensemble.slurm)

# 4. comparison table
$FLOWMOL_PY /n/home04/yulili/bgfm/scripts/summarize_global_ensemble.py \
    --root /n/holylabs/woo_lab/Lab/yulili/bgfm/runs/eval_ours/global_ensemble_cpu \
    --arms energy_s2 control_s2 energy_s3 control_s3 \
    --paired energy_s2 control_s2
```

The `logp` stage flushes `basin_logp_partial.json` after every system and
resumes from it, so preemption on `gpu_requeue` costs at most one system.
