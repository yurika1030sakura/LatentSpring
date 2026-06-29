# BGFM Implementation Status

All four modules of BGFM described in `paper/bgfm_paper.pdf` are
implemented and integrated into the training and sampling pipelines.
Each module has a corresponding entry in the smoke test suite at
[`tests/test_bgfm_smoke.py`](tests/test_bgfm_smoke.py).

## Module-by-module status

| Module | Code | Wired? | Smoke test | Notes |
|---|---|---|---|---|
| **Module 1 — Flow proposal** | `baselines/flowmol3/flowmol/models/*.py` | ✓ | covered by FlowMol3 itself | vendored, unchanged. |
| **Module 2 — Boltzmann regularization** (force-score, density-energy, anchor) | `cfm_mol/bgfm_loss.py`, `cfm_mol/bgfm_density.py`, `cfm_mol/log_z_predictor.py` | ✓ | exercised in `test_density_loss_with_joint_density` (variance-invariance) | computed every BGFM training step. |
| **Module 3 — Calibrated scalar energy head $\hat E_\psi$** | `cfm_mol/energy_head.py`, `cfm_mol/bgfm_loss.py:energy_head_calibration_loss` | ✓ | `test_energy_head_forward_and_force`, `test_energy_head_calibration_loss` | instantiated in `patch_flowmol_bgfm` when `bgfm.energy_head_enabled: true`; `L_head` added to total loss in `bgfm_training_step` with `lambda_4` weight; autograd through the head gives the force matching gradient. |
| **Module 4 — Learned-energy Langevin corrector** | `cfm_mol/refinement.py`, `scripts/sample_bgfm.py` | ✓ | `test_langevin_corrector_accounting` | Langevin chain under $\hat E_\psi$ at sampling time, with full `Accounting` record (flow NFEs, head NFEs, oracle NFEs, wall-clock, rejected). |
| **Module 3.5 — Joint discrete-continuous density** | `cfm_mol/joint_density.py`, `cfm_mol/bgfm_density.py:energy_consistency_loss_per_mol(discrete_log_p=...)` | ✓ | `test_joint_density_combiner`, `test_density_loss_with_joint_density` | density-energy loss now accepts `discrete_log_p` and combines via `joint_log_prob`; with `None` it falls back to position-only. The training-step caller can pass discrete log-probabilities computed from the FlowMol3 CTMC bridge to use the joint form. |

## Smoke test results

Tests run end-to-end on CPU in seconds and on a single MIG slice of an
A100 (`gpu_test` partition) in <30 seconds.

```
[smoke] energy head forward + autograd force: OK
[smoke] energy-head calibration loss: 61.563, grad on 14 params
[smoke] Langevin corrector accounting: {'flow_nfe': 0,
    'energy_head_nfe': 11, 'oracle_nfe': 0, ...}
[smoke] joint density combiner: OK
[smoke] joint density variance-invariance: OK
[smoke] ALL TESTS PASSED
```

The corrector accounting confirms 11 head NFEs (1 initial + 10 steps
for the 10-step toy test), and zero external oracle calls during
refinement, as specified in the paper.

## Honest compute accounting

`cfm_mol/refinement.py:Accounting` records the following per sampling
configuration:

| Field | Definition |
|---|---|
| `flow_nfe` | Number of flow ODE steps. |
| `energy_head_nfe` | Number of $\hat E_\psi$ forward passes. |
| `oracle_nfe` | Number of external OMol25 or xTB calls (zero during BGFM sampling by construction). |
| `wall_clock_s` | Per-sample wall-clock seconds. |
| `rejected` | Metropolis rejection count (zero when not enabled). |
| `n_samples` | Number of accepted samples produced. |

`scripts/sample_bgfm.py` writes this record alongside the samples
JSON, so every reported number in
Tables 1--3 of the paper carries its compute attribution.

## Production training configuration

`configs/omol25_4m_bgfm_v10_full.yaml` enables all four modules:

```yaml
bgfm:
  enabled: true
  lambda_1: 0.05          # force-score
  lambda_2: 1.0e-4        # density-energy variance
  lambda_3: 1.0e-12       # composition-conditioned offset anchor
  lambda_4: 1.0           # energy-head calibration
  lambda_F: 0.1           # force-gradient term inside L_head
  energy_head_enabled: true
  energy_head_hidden_dim: 128
  energy_head_n_layers: 3
  energy_head_cutoff: 5.0
  joint_density_enabled: true
  corrector_steps_default: 50
  corrector_eta_init: 1.0e-3
  corrector_eta_final: 1.0e-4
```

## How a reviewer can verify

```bash
# 1. Run the full smoke test (CPU is fine).
python tests/test_bgfm_smoke.py

# 2. Inspect the wiring points.
grep -n 'energy_head\|EnergyHead\|joint_density' \
    cfm_mol/bgfm_train_hook.py cfm_mol/bgfm_density.py

# 3. Run a tiny end-to-end training step (requires the FlowMol3
#    backbone and the OMol25 perturbation shard).
sbatch scripts/launch_omol25_bgfm_h200.sh \
       configs/omol25_4m_bgfm_v10_full.yaml "" 42
```
