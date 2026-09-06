# Results

Metric files for every evaluation reported in the paper, mirrored from the run
tree (`runs/`, which lives on scratch and is not tracked).

## What is here

| directory | contents |
|---|---|
| `eval_ours/` | the primary Boltzmann-ordering evaluations: per-arm, per-seed |
| `eval_ours/calibration/` | calibration diagnostics (slope, NRV, scale ratio, per-group) |
| `eval_res/` | re-scorings of already-listed checkpoints at other integrator settings |
| `eval_sweep/`, `eval_p3/` | sweep and Paper-3 evaluations |
| `force_diag/` | the two force diagnostics and the lambda_1 calibration |
| `p1_surrogate_gradient/` | surrogate-gradient diagnostics |
| `dashboard_history/` | periodic run-status snapshots |

## The two files that matter for reproducing a headline number

- `*/boltz_independent_records.csv` — one row per (parent, perturbation):
  `group_id, pert_id, n_atoms, charge, log_p_theta, E_eV, negE_kT, xtb_ok`.
  Every reported correlation is computed from these.
- `eval_ours/calibration/calibration_dropref.json` — per-group `slope`, `nrv`,
  `pearson_r`, `scale_ratio`, from which the calibration table and the residual
  correlation follow.

Recipe for the primary endpoint: drop `pert_id == 0` (the data geometry), drop
rows with `xtb_ok` false, take the within-parent Pearson correlation between
`log_p_theta` and `negE_kT`, require at least three usable points, then average
over parents.

## What is not here

- Model weights (`*.ckpt`, 288 files, ~18.5 GB).
- `boltzmann_samples.json` (205 files, ~515 MB) and `basin_logp*.json`
  (~102 MB): Stage-1 dumps carrying the perturbed coordinates. Nothing in the
  paper is computed from the coordinates directly — the records CSVs above
  carry every field the metrics use — so they are omitted for size.

Both live under `/n/holylabs/woo_lab/Lab/yulili/bgfm/runs/`.
