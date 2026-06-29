# BGFM Paper → Code Map

This document maps each section of the paper
[`paper/bgfm_paper.pdf`](paper/bgfm_paper.pdf) to the source files that
implement it.

**Repository**: https://github.com/yurika1030sakura/bgfm

---

## 1. Repository orientation

```
bgfm/
├── paper/                                # LaTeX + compiled PDF
├── cfm_mol/                              # BGFM extension code
├── baselines/flowmol3/                   # vendored FlowMol3 backbone
├── configs/                              # training configs (yaml)
├── scripts/                              # training + evaluation entry points
└── notes/                                # method notes, plan documents
```

---

## 2. Method → code mapping

### §3.1 Module 1 — Flow proposal

| Paper element | Code |
|---|---|
| GVP-Transformer velocity field $v_\theta(x_t, t)$ | `baselines/flowmol3/flowmol/models/vector_field.py`, `gvp.py` |
| Linear interpolant on coordinates | `baselines/flowmol3/flowmol/models/flowmol.py` |
| CTMC priors on atom types / charges | `baselines/flowmol3/flowmol/models/ctmc_vector_field.py` |
| Flow-matching reconstruction $\mathcal{L}_{\rm FM}$ | `baselines/flowmol3/flowmol/models/flowmol.py` (`total_loss_weights`) |

### §3.2 Module 2 — Boltzmann regularization

| Paper element | Code | Line |
|---|---|---|
| Score-from-velocity (Eq.\,3) | `cfm_mol/bgfm_loss.py` | `score_from_fm_velocity` |
| $\mathcal{L}_{\rm force}$ (Eq.\,4) — cosine variant | `cfm_mol/bgfm_loss.py` | `force_loss`, `score_force_cosine` |
| $\mathcal{L}_{\rm dens}$ (Eq.\,5) — per-mol variance | `cfm_mol/bgfm_density.py` | `energy_consistency_loss_per_mol` |
| $\mathcal{L}_{\rm anchor}$ (Eq.\,6) | `cfm_mol/bgfm_density.py` | `energy_anchor_loss` |
| Composition-conditioned offset $a_\phi(c)$ | `cfm_mol/log_z_predictor.py` | `LogZPredictor` |
| Hutchinson trace estimator | `cfm_mol/bgfm_loss.py` | `divergence_hutchinson` |
| Offline perturbation precompute | `scripts/precompute_energy_perturbations.py` | `main` |
| In-training perturbation loader | `cfm_mol/perturbation_loader.py` | entire file |

### §3.3 Module 3 — Calibrated scalar energy head

| Paper element | Code | Line |
|---|---|---|
| Scalar head $\hat E_\psi(r, c)$ | `cfm_mol/energy_head.py` | `EnergyHead` |
| Joint energy + force evaluation via autograd | `cfm_mol/energy_head.py` | `energy_and_force` |
| Calibration loss $\mathcal{L}_{\rm head}$ (Eq.\,7) | `cfm_mol/bgfm_loss.py` | `energy_head_calibration_loss` |

### §3.4 Module 4 — Learned-energy Langevin corrector

| Paper element | Code | Line |
|---|---|---|
| Langevin corrector (Eq.\,8) | `cfm_mol/refinement.py` | `langevin_corrector` |
| Per-step size schedule | `cfm_mol/refinement.py` | `_step_schedule` |
| Honest NFE / wall-clock accounting | `cfm_mol/refinement.py` | `Accounting` dataclass |
| Inference-time sampler integration | `scripts/sample_bgfm.py` | top-level |

### §3.5 Joint discrete-continuous density estimator

| Paper element | Code | Line |
|---|---|---|
| $\log p_\theta(r \mid c)$ via FFJORD (Eq.\,10) | `cfm_mol/bgfm_density.py` | `log_density_via_flow` |
| $\log p_\theta(N)$ atom-count log-probability | `cfm_mol/joint_density.py` | `log_prob_atom_count` |
| Per-token CTMC path log-probability | `cfm_mol/joint_density.py` | `log_prob_discrete_token_path` |
| Per-graph discrete sum $\log p_\theta(c)$ (Eq.\,11) | `cfm_mol/joint_density.py` | `log_prob_discrete_state` |
| Joint combiner $\log p_\theta(x) = \log p_\theta(r \mid c) + \log p_\theta(c)$ | `cfm_mol/joint_density.py` | `joint_log_prob` |

### §3.6 Total objective and training

| Paper element | Code | Line |
|---|---|---|
| Total objective $\mathcal{L}$ (Eq.\,12) | `cfm_mol/bgfm_train_hook.py` | `bgfm_training_step` |
| Warm-up + ramp schedule on $\lambda_{1\dots 4}$ | `cfm_mol/bgfm_train_hook.py` | `_bgfm_schedule` |
| Lightning patcher: attach all modules onto the backbone | `cfm_mol/bgfm_train_hook.py` | `patch_flowmol_bgfm` |
| Optional temperature conditioning | `cfm_mol/kt_conditioning.py` | `patch_kT_conditioning` |
| Training entry | `scripts/run_train.py` | top-level |
| H200 launcher | `scripts/launch_omol25_bgfm_h200.sh` | top-level |

---

## 3. Experiments → code mapping

### §4.1 Standard unconditional generation (QM9 + GEOM-Drugs)

| Paper element | Code |
|---|---|
| 10k QM9 samples + EBMol-protocol metrics | `scripts/eval_qm9_ebmol_protocol.py` |
| 10k GEOM-Drugs samples + revised OpenBabel/RDKit + Vendi | `scripts/eval_geomdrugs_ebmol_protocol.py` |
| Sample-input producer (flow + corrector) | `scripts/sample_bgfm.py` |

### §4.2 Independent physical quality (GFN2-xTB / MMFF / DFT)

| Paper element | Code |
|---|---|
| xTB relaxation $\Delta E$, RMSD, step count, failure | `scripts/eval_xtb_relaxation.py` |
| MMFF94 and DFT subset extensions | extend `scripts/eval_xtb_relaxation.py` (oracle plugin slot) |

### §4.3 Quality–diversity–compute Pareto

| Paper element | Code |
|---|---|
| $\mathrm{VULS}_\tau$ throughput (Eq.\,13) | derived metric: combine `scripts/sample_bgfm.py` accounting + `eval_xtb_relaxation.py` |
| Compute-budget sweep | `--flow_nfe` and `--corrector_steps` flags on `scripts/sample_bgfm.py` |

### §4.4 Mechanism diagnostic (held-out perturbation families)

| Paper element | Code |
|---|---|
| FFJORD log-density on perturbation clouds | `scripts/eval_boltzmann_stage1.py` |
| Within-parent correlation / slope aggregation | `scripts/eval_boltzmann_stage2.py` |
| Per-slice analysis (charged / heteroatom / metal / size) | `scripts/eval_boltzmann_stage1.py --chemistry_slice` |
| Cross-oracle transfer (xTB, DFT) | combine Stage 2 output with `scripts/eval_xtb_relaxation.py` |

### §4.5 Ablations

| Paper element | Code |
|---|---|
| Per-module ablation configs | `configs/ablations/*` (one config per removed module) |
| Override flags (no config edit needed) | `scripts/run_train.py` with `--override_bgfm_lambda_{1,2,3,4}` |

### §4.6 Negative controls

| Paper element | Code |
|---|---|
| Shuffled energy / force / wrong $kT$ shard generator | `scripts/eval_negative_controls.py` |
| Re-train with shuffled shard | `scripts/launch_omol25_bgfm_h200.sh` + config pointing to shuffled shard |

### §4.7 Cross-model energy ranking

| Paper element | Code |
|---|---|
| BGFM density score $S_{\rm BGFM}^{\rm dens}$ (Eq.\,14) | `scripts/eval_cross_model_ranking.py` |
| BGFM energy-head score $\hat E_\psi$ | `scripts/eval_cross_model_ranking.py` (uses `cfm_mol/energy_head.py`) |
| Cross-model sample pool from EDM / GeoLDM / FlowMol3 / EBMol / BGFM | per-model sampler outputs collected in `runs/eval/cross_model/` |

---

## 4. Reproduction recipe

The end-to-end recipe is in [`README.md`](README.md). For a quick
check that everything imports:

```bash
python -c "
from cfm_mol import bgfm_loss, bgfm_density, bgfm_train_hook
from cfm_mol import energy_head, refinement, joint_density
from cfm_mol import log_z_predictor, kt_conditioning, perturbation_loader
print('all BGFM modules import')
"
```
