# BGFM Paper → Code Map

> Purpose: this document tells reviewers (and you) exactly **where in the codebase**
> each method/experiment from `bgfm_paper.tex` is implemented. Hand it together
> with the GitHub link to anyone reviewing the project.
>
> **GitHub**: https://github.com/yurika1030sakura/bgfm
> **Local checkout**: `/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/`
> **Tag/commit at submission time**: TBD (to add when final paper number lands)

All paths below are **relative to repo root**. Line numbers are pinned to
commit at the time of writing this doc; numbers may drift slightly.

---

## 1. Repository Layout (1-minute orientation)

```
bgfm/
├── cfm_mol/                          # Core BGFM extensions to FlowMol3
│   ├── bgfm_density.py               # FFJORD log p, per-mol variance, anchor
│   ├── bgfm_loss.py                  # Force loss, score-from-velocity, total
│   ├── bgfm_train_hook.py            # Patches FlowMol3 with BGFM losses
│   ├── kt_conditioning.py            # Temperature conditioning (v8b)
│   ├── log_z_predictor.py            # Invariant log-Z aux network
│   ├── perturbation_loader.py        # Loads precomputed perturbation shards
│   └── physics.py                    # OMol25 NP wrapper utilities
│
├── baselines/flowmol3/                # Vendored FlowMol3 (architecture backbone)
│   └── flowmol/models/
│       ├── vector_field.py           # GVP-Transformer
│       ├── gvp.py                    # GVP message passing
│       ├── ctmc_vector_field.py      # CTMC for atom-type / charge
│       └── flowmol.py                # Top-level LightningModule
│
├── configs/                           # All training configs (yaml)
│   ├── omol25_4m_bgfm.yaml            # FM baseline (no BGFM)
│   ├── omol25_4m_bgfm_energy_v7c_from_fm.yaml   # v7c: force only
│   ├── omol25_4m_bgfm_energy_v8a_room_T.yaml    # v8a: full BGFM @ room T
│   ├── omol25_4m_bgfm_energy_v8b_T_conditional.yaml # v8b: T-conditional
│   └── omol25_4m_bgfm_energy_v8c_energy_only.yaml   # v8c: energy + anchor (no force)
│
├── scripts/
│   ├── precompute_energy_perturbations.py        # Offline σ × OMol25 precompute
│   ├── run_precompute_energy.slurm
│   ├── run_train.py                              # Lightning training entry
│   ├── launch_omol25_bgfm_energy_h200.sh         # H200 training launcher
│   ├── eval_boltzmann_stage1.py                  # Level 1: FFJORD log p eval
│   ├── eval_boltzmann_stage2.py                  # Level 1: R² aggregation
│   ├── level2_relax_comparison.py                # Level 2: BFGS savings
│   ├── level3_bgfm_sample.py                     # Level 3: sampling
│   ├── level3_md_reference.py                    # Level 3: Langevin MD reference
│   └── level3_compare.py                         # Level 3: ensemble overlap
│
└── notes/
    ├── bgfm_method.md                # Internal method derivation notes
    ├── appendix_A_v4.tex             # Math appendix
    ├── paper_v2_bgfm_draft.md        # Paper draft (markdown)
    └── EXPERT_REVIEW_BGFM_2026-06.md # Expert-review framework doc
```

---

## 2. Paper Section → Code Mapping

### §3.1 Setup (configuration space)

| Paper element | Code |
|---|---|
| Bond-free 3D molecular representation | `baselines/flowmol3/flowmol/models/flowmol.py` (entire LightningModule) |
| OMol25 energy/force oracle | `cfm_mol/physics.py` (OMol25 wrapper); checkpoint loaded in `scripts/compute_omol25_energy.py:_load_calc` |
| Atom-type simplex (83 elements) | atom_map in any v8 config, e.g. `configs/omol25_4m_bgfm_energy_v8a_room_T.yaml:50-132` |

### §3.2 Flow Matching baseline

| Paper element | Code |
|---|---|
| GVP-Transformer architecture | `baselines/flowmol3/flowmol/models/vector_field.py`, `gvp.py` |
| Linear interpolant $x_t = (1-t)x_0 + t x_1$ | `baselines/flowmol3/flowmol/models/flowmol.py` (interpolation in training_step) |
| Standard FM loss (Eq. 2) | `baselines/flowmol3/flowmol/models/flowmol.py` (`total_loss_weights` in config) |

### §3.3 FFJORD log-density inside FM training (Eq. 3)

| Paper element | Code | Line |
|---|---|---|
| FFJORD integral (Eq. 3) | `cfm_mol/bgfm_density.py` | `log_density_via_flow:101` |
| Position-only velocity wrapper for FFJORD | `cfm_mol/bgfm_density.py` | `make_position_velocity_fn:73`, `v_fn:94` |
| Gaussian prior log p | `cfm_mol/bgfm_density.py` | `gaussian_prior_log_density:35` |
| Hutchinson trace estimator (Eq. 4) | `cfm_mol/bgfm_loss.py` | `divergence_hutchinson:172` |
| Exact-per-atom alternative | `cfm_mol/bgfm_loss.py` | `divergence_exact_atomwise:125` |

### §3.4 $\mathcal{L}_\mathrm{force}$ — score from velocity (Eq. 5–6)

| Paper element | Code | Line |
|---|---|---|
| Score-from-velocity (Eq. 5) | `cfm_mol/bgfm_loss.py` | `score_from_fm_velocity:56` |
| Force-consistency MSE (Eq. 6) | `cfm_mol/bgfm_loss.py` | `force_loss:249` |
| Cosine variant (used in v8) | `cfm_mol/bgfm_loss.py` | `score_force_cosine:292` |
| Direction-only loss (ablation) | `cfm_mol/bgfm_loss.py` | `force_direction_loss:314` |
| $t_\mathrm{eval}$ values | config: `bgfm.t_eval_values` |

### §3.5 $\mathcal{L}_\mathrm{energy}$ — per-mol variance (Eq. 7–8)

| Paper element | Code | Line |
|---|---|---|
| Per-mol variance loss (Eq. 8) | `cfm_mol/bgfm_density.py` | `energy_consistency_loss_per_mol:290` |
| Within-group variance helper | `cfm_mol/bgfm_density.py` | `within_group_variance_loss:234` |
| Naive (mean-form) version, only for reference | `cfm_mol/bgfm_density.py` | `energy_consistency_loss:186` |
| Variance form for loss-level integration | `cfm_mol/bgfm_loss.py` | `energy_loss_variance:359` |

### §3.6 $\mathcal{L}_\mathrm{anchor}$ — invariant log-$Z$ predictor (Eq. 9)

| Paper element | Code | Line |
|---|---|---|
| Invariant aux network | `cfm_mol/log_z_predictor.py` | `LogZPredictor:25` |
| Atom-type histogram + charge → log Z | `cfm_mol/log_z_predictor.py` | `LogZPredictor.forward:50` |
| Anchor loss term | `cfm_mol/bgfm_density.py` | `energy_anchor_loss:344` |
| Combined energy + anchor | `cfm_mol/bgfm_density.py` | `energy_consistency_loss_per_mol_with_anchor:403` |

### §3.7 Total objective + schedule (Eq. 10)

| Paper element | Code | Line |
|---|---|---|
| BGFM total loss (Eq. 10) | `cfm_mol/bgfm_loss.py` | `bgfm_total_loss:390` |
| Warmup/ramp schedule | `cfm_mol/bgfm_train_hook.py` | `_bgfm_schedule:29` |
| BGFM training step (one optimizer step) | `cfm_mol/bgfm_train_hook.py` | `bgfm_training_step:217` |
| Lightning patcher (turns FlowMol3 into BGFM) | `cfm_mol/bgfm_train_hook.py` | `patch_flowmol_bgfm:117` |

### §3.8 Offline perturbation precompute

| Paper element | Code | Line |
|---|---|---|
| Offline precompute script | `scripts/precompute_energy_perturbations.py` | `main:63` |
| OMol25 calculator load | `scripts/precompute_energy_perturbations.py` | `_load_calc:50` |
| SLURM launcher | `scripts/run_precompute_energy.slurm` | entire file |
| In-training shard loader | `cfm_mol/perturbation_loader.py` | entire file |

### §3.9 Temperature-conditional variant (v8b)

| Paper element | Code | Line |
|---|---|---|
| Sample $kT$ log-uniform per step | `cfm_mol/kt_conditioning.py` | `sample_kT:40` |
| $kT$ projection net (zero-init) | `cfm_mol/kt_conditioning.py` | `_kTProjection:57` |
| Patch FlowMol3 with $kT$ conditioning | `cfm_mol/kt_conditioning.py` | `patch_kT_conditioning:81` |
| Post-hook on scalar embedding | `cfm_mol/kt_conditioning.py` | `scalar_emb_post_hook:135` |

### §3.10 Training pipeline

| Paper element | Code | Line |
|---|---|---|
| Entry point | `scripts/run_train.py` | top-level |
| H200 launcher | `scripts/launch_omol25_bgfm_energy_h200.sh` | top-level |
| Top-level BGFM training integration | `cfm_mol/bgfm_train_hook.py` | `bgfm_training_step` + `patch_flowmol_bgfm` (`cfm_mol/bgfm_loss.py:compute_bgfm_step` is a stub — real assembly lives in the train hook) |

---

## 3. Experimental Pipeline → Code Mapping

### §4.1 Datasets, models, evaluation protocol

| Paper element | Code |
|---|---|
| OMol25 4M ingestion | `scripts/download_omol25_4m.slurm`, `scripts/preprocess_omol25.py` |
| Train/val split | `scripts/preprocess_omol25.slurm` |
| Atom-type vocabulary (83 elements) | `configs/omol25_4m_bgfm_energy_v8a_room_T.yaml:50-132` |
| FM baseline config | `configs/omol25_4m_bgfm.yaml` |
| BGFM-full v8a config | `configs/omol25_4m_bgfm_energy_v8a_room_T.yaml` |
| BGFM-T-cond v8b config | `configs/omol25_4m_bgfm_energy_v8b_T_conditional.yaml` |
| BGFM-energy-only v8c config | `configs/omol25_4m_bgfm_energy_v8c_energy_only.yaml` |

### §4.2 Experiment 1 — Standard unconditional generation (QM9 + GEOM-Drugs)

| Paper element | Code |
|---|---|
| QM9: 10k samples + EBMol QM9 metrics | `scripts/eval_qm9_ebmol_protocol.py` *(scaffolded)* |
| GEOM-Drugs: revised OpenBabel/RDKit + Vendi | `scripts/eval_geomdrugs_ebmol_protocol.py` *(scaffolded)* |
| NFE matching {930, 1370, 1810} / {1080, 1960, 3720, 7240} | CLI `--nfe` |

### §4.3 Experiment 2 — Independent physical oracle (GFN2-xTB / MMFF / DFT)

| Paper element | Code |
|---|---|
| xTB relaxation $\Delta E$, RMSD, step count, failure rate | `scripts/eval_xtb_relaxation.py` *(scaffolded)* |
| Pre-relax sample input | output of Experiment 1 scripts |
| MMFF + DFT subset | future extension of Experiment 2 script |
| Valid-connected-only AND all-samples protocol | both reported per the audit plan |

### §4.4 Experiment 3 — Quality–diversity–compute Pareto

| Paper element | Code |
|---|---|
| $\mathrm{VLU}_\tau$ throughput metric | derived in plotting notebook from Exp 1 + Exp 2 outputs |
| Compute budgets: BGFM ODE NFE, BGFM++ refinement steps, EBMol Langevin/PT steps | CLI `--nfe` on per-model scripts |

### §4.5 Experiment 4 — Boltzmann mechanism diagnostic (OMol25)

This is the OMol25 self-consistency metric. It is reported but **not the
headline**: it is too close to the training objective to make a
Boltzmann claim on its own. Demoted to mechanism verification per the
2026-06-29 audit.

| Paper element | Code | Line |
|---|---|---|
| Per-molecule FFJORD log p eval | `scripts/eval_boltzmann_stage1.py` | `main:56` |
| Sample perturbations + OMol25 E | `scripts/eval_boltzmann_stage1.py` | inside molecule loop |
| Per-mol OOM catch (added 2026-06) | `scripts/eval_boltzmann_stage1.py` | try/except around `log_density_via_flow` |
| Charge fix + T-conditional support | `scripts/eval_boltzmann_stage1.py` | `mol_charge`, `--kT`, kT projection install |
| $R^2$ aggregation | `scripts/eval_boltzmann_stage2.py` | top-level |
| Cross-oracle generalization (xTB, DFT subset) | extend Stage 2 to call xTB; see Exp 2 |
| Held-out perturbation families (torsion / stretch / bend / xTB-Langevin) | future shards via `scripts/precompute_energy_perturbations.py` |
| SLURM eval launcher | `scripts/run_boltzmann_eval.slurm` | entire file |

### §4.6 Experiment 5 — Ablations and negative controls

| Paper element | Code |
|---|---|
| Architecture ablations (force/energy/full/T) | `scripts/run_train.py` with `--override_bgfm_lambda_{1,2,3}` |
| Shuffled-energy / shuffled-force / wrong-kT shards | `scripts/eval_negative_controls.py` *(scaffolded)* |
| Random-regularizer / force-norm-only controls | extend `eval_negative_controls.py` |

### §4.7 Experiment 6 — Energy-based ranking and filtering

| Paper element | Code |
|---|---|
| Pooled-sample scorer $S(x)$ vs xTB $\Delta E$ | `scripts/eval_cross_model_ranking.py` *(scaffolded)* |
| Per-model samples (EDM, GeoLDM, FlowMol3, EBMol, FM, BGFM) | output of Experiment 1 + Experiment 2 |
| Strained perturbations + relaxed reference | constructed inline by Experiment 6 script |

### §4.8 Experiment 7 — Broad-chemistry OMol25 generalization

| Paper element | Code |
|---|---|
| Slice flag in eval | `scripts/eval_boltzmann_stage1.py --chemistry_slice` |
| Slices: neutral/charged, small/large, heteroatom-rich, metals, electrolytes | `--chemistry_slice` ∈ {organic_chnofs, transition_metal, halogenated, heavy_main_group, charged} |

### §4.9 OMol25 BFGS downstream (reported alongside Exp 2 with caveat)

| Paper element | Code |
|---|---|
| OMol25 BFGS step count to convergence | `scripts/level2_relax_comparison.py` |
| SLURM launcher | `scripts/run_level2_relax.slurm` |

Reported with caveat because OMol25 is also the training oracle.

### §4.10 Legacy MD-equivalent sampling efficiency

| Paper element | Code |
|---|---|
| BGFM K-step sampling | `scripts/level3_bgfm_sample.py` |
| Langevin MD reference | `scripts/level3_md_reference.py` |
| Energy + RDF compare | `scripts/level3_compare.py`, `scripts/level3_compute_energies.py` |
| SLURM launcher | `scripts/run_level3_efficiency.slurm` |

### §4.11 Multi-T evaluation (Appendix)

| Paper element | Code |
|---|---|
| Use v8b ckpt with explicit kT input | `scripts/eval_boltzmann_stage1.py --kT` |

---

## 4. Reproducing the Numbers

The minimal reproduction recipe for a reviewer with one A100/H200 80GB+ GPU:

```bash
# 1. Set up environment
git clone https://github.com/yurika1030sakura/bgfm
cd bgfm
conda env create -f environment.yml  # creates envs/flowmol and envs/omol25

# 2. Download OMol25 4M data
sbatch scripts/download_omol25_4m.slurm
sbatch scripts/preprocess_omol25.slurm

# 3. Precompute perturbations (the K=5 σ shards)
sbatch scripts/run_precompute_energy.slurm val 10000
sbatch scripts/run_precompute_energy.slurm train 30000

# 4. Train one BGFM variant (room T, full)
sbatch scripts/launch_omol25_bgfm_energy_h200.sh \
       configs/omol25_4m_bgfm_energy_v8a_room_T.yaml \
       "" 42

# 5. Evaluate Boltzmann correlation
sbatch scripts/run_boltzmann_eval.slurm \
       runs/omol25_4m_bgfm_energy_v8a_room_T/lightning_logs/.../checkpoints/last.ckpt \
       configs/omol25_4m_bgfm_energy_v8a_room_T.yaml

# 6. Downstream BFGS relaxation comparison
sbatch scripts/run_level2_relax.slurm <ckpt>

# 7. MD-equivalent sampling efficiency
sbatch scripts/run_level3_efficiency.slurm <ckpt>
```

Total compute on H200: $\sim$45–50 h per seed for training, plus
$\sim$2 h Level 1 eval, $\sim$10 h Level 2 BFGS comparison, $\sim$5 h
Level 3 MD reference.

---

## 5. Sanity Checks for Reviewers

These are the smoke tests reviewer can run in $<$30 min to verify the
math is implemented correctly:

| Test | What it checks | Code |
|---|---|---|
| FFJORD on a Gaussian | Recover analytic $\log p$ | `cfm_mol/test_fibre.py` |
| Score-from-velocity equivariance | Score is SE(3)-equivariant | `scripts/test_equivariance.py` |
| Variance loss is translation-invariant | $\mathrm{Var}(x+c) = \mathrm{Var}(x)$ | `cfm_mol/bgfm_density.py:within_group_variance_loss:234` (unit test inline) |
| Anchor breaks $\log Z$ symmetry | Loss drops once anchor enabled on $\log p \equiv c$ trivial solution | run smoke on v8a config with `lambda_3=0` then with `lambda_3 > 0` |
| OMol25 calculator returns sensible $F$, $E$ | $\|F\| < 5$ eV/Å on equilibrium ckpt | `scripts/compute_omol25_energy.py` on any test molecule |

---

## 6. Lineage of Configs (which paper number came from which run)

| Paper claim | Config | Run dir | Ckpt step | Note |
|---|---|---|---|---|
| $R^2 = 0.278$ (mixed force + small energy, $kT=1$ eV) | `omol25_4m_bgfm_energy_v7c_from_fm.yaml` | `runs/omol25_4m_bgfm_energy_v7c_from_fm/lightning_logs/version_19207322/` | 50000 | $\lambda_1=0.05$, $\lambda_2=0.001$, $\lambda_3=0$. **Not pure force-only** — see config header note. |
| $R^2 = $ [v8 pending] | `omol25_4m_bgfm_energy_v8a_room_T.yaml` | `runs/omol25_4m_bgfm_energy_v8a_room_T/lightning_logs/version_2014691{6,7,8}/` | 95000 (NaN at 100k) | Best 3 seeds; pending eval |
| T-conditional [v8 pending] | `omol25_4m_bgfm_energy_v8b_T_conditional.yaml` | `runs/omol25_4m_bgfm_energy_v8b_T_conditional/lightning_logs/version_20147027/` | 95000 (NaN at 98k) | Best variant |
| Energy-only ablation [pending] | `omol25_4m_bgfm_energy_v8c_energy_only.yaml` | `runs/omol25_4m_bgfm_energy_v8c_energy_only/lightning_logs/version_20147028/` | 55000 (NaN at 57k) | Force off |

---

## 7. Known Open Issues (transparent for reviewers)

| Issue | Where | Plan |
|---|---|---|
| v8 NaN at step 30–98k | L_anchor overflow at bf16 limit on outlier perturbations | v9 fix: soft-clamp anchor + NaN-skip grad. Implemented in branch `v9-nan-fix`. |
| Per-mol OOM on $>200$ atom molecules | FFJORD backward graph | OOM-skip + log skipped molecules (added 2026-06). Reported separately. |
| OMol25 NP coverage limits | Rare elements + spin states underrepresented | Inherits all OMol25 limitations. Documented in §6 (Limitations). |
| 5.9M-param model on 4M data | Small relative to Proteina (50M params, 21M data) | Scaling left to future work; this paper focuses on objective novelty. |

---

## 8. How to Hand This Off

To give an external expert reviewer a self-contained package:

1. **Send them**:
   - `bgfm_paper.tex` (the paper)
   - `iclr2026_extracted/iclr2026/iclr2026_conference.bib` (refs)
   - This `CODE_MAP.md`
   - Link to https://github.com/yurika1030sakura/bgfm
2. **Tell them**: "Each section of the paper has a corresponding source file in
   §2 of CODE_MAP.md. Equations in the paper are line-pinned. The repo can be
   reproduced from §4 of CODE_MAP.md."
3. **Ask them to focus on**: §3.5 (per-mol variance derivation) and §3.6
   (log-$Z$ anchor) — these are the two original contributions and the most
   worth a sanity-check on the math.

---

**Last updated**: 2026-06-09
**Maintainer**: Yuli Li (Harvard Woo Lab)
