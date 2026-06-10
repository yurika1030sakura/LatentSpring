# Boltzmann-Guided Flow Matching (BGFM)

A 3D molecular generator whose learned density is pushed to match the
**physical Boltzmann distribution** under a universal neural potential
(OMol25, 83 elements). Unifies flow matching + score matching + Boltzmann
generator ideas in a single training loop.

Target venue: **ICLR 2027** (submission ~2026-09-25).

> **Note:** earlier versions of this README described a reflected-diffusion /
> valence-constrained framing. That has been superseded. For day-to-day
> guidance see `CLAUDE.md` and `AGENTS.md`; for the method derivation see
> `notes/bgfm_method.md`; for proofs see `notes/appendix_A_v4.tex`.

## One-line pitch

Most 3D molecular generators only *imitate* a dataset — they produce molecules
that look valid but are not physically equilibrated, and collapse outside the
chemistry they were trained on. **BGFM trains a flow-matching generator so
that how likely it thinks a shape is matches how stable physics says that
shape is** (the Boltzmann law), using a single universal neural potential
(OMol25) as the physics signal. The model works across organic, organo-
metallic, and transition-metal chemistry from one trained checkpoint.

## Method at a glance

```
L_total = L_FM + λ₁·L_force + λ₂·L_energy   (+ optional λ₃·L_anchor)
```

- **L_FM** — standard flow matching: velocity MSE on positions + cross-entropy
  on element/charge.
- **L_force** — model's implied score must equal physics force `F/kT`. Read
  off a single velocity by closed form `s = (t·v − x) / ((1−t)·σ²)`. Cheap
  per-step signal.
- **L_energy** — within-molecule variance of `log p + E/kT` across K
  precomputed geometric perturbations. Cancels the per-molecule partition
  function. Optional invariant log-Z anchor prevents trivial-constant
  solutions.

**Bond-free.** Bonds are not generated (OMol25 has no bond labels).
Connectivity is recovered post-hoc with xyz2mol on the generated geometry.

**Architecture.** No FlowMol3 fork — `CTMCVectorField` is extended via runtime
monkey-patches in `cfm_mol/flow_model.py::patch_flowmol` (geometric hooks)
and `cfm_mol/bgfm_train_hook.py::patch_flowmol_bgfm` (training loss).

## Repo layout

```
bgfm/
├── cfm_mol/                       # OUR code
│   ├── bgfm_loss.py               # score, divergence, force / energy losses
│   ├── bgfm_density.py            # FFJORD log-density, per-mol energy variance
│   ├── bgfm_train_hook.py         # patch_flowmol_bgfm — training-step wrapper
│   ├── flow_model.py              # patch_flowmol — geometric hooks
│   ├── kt_conditioning.py         # temperature conditioning
│   ├── log_z_predictor.py         # invariant log-Z head for anchor loss
│   ├── perturbation_loader.py     # K-perturbation shard iterator
│   ├── data/omol25.py             # OMol25 LMDB → FlowMol3-style DGL graph
│   ├── domain.py / fibre*.py      # geometric checks + projections
│   └── projection.py              # dead under bond-free; reserved for Paper 2
├── configs/
│   ├── omol25_4m_bgfm.yaml        # primary BGFM training config
│   ├── omol25_4m_cfm.yaml         # Level-1 ablation (FM-only, bond-free)
│   ├── geom_cfm_bondfree.yaml     # ablation on FlowMol3's GEOM home field
│   └── omol25_4m_bgfm_energy_v8*.yaml   # energy-consistency variants
├── scripts/
│   ├── run_train.py               # entry point (loads cfg, applies patches, fits)
│   ├── preprocess_omol25.py       # OMol25 LMDB → FlowMol3-native .pt
│   ├── precompute_energy_perturbations.py   # K perturbations + OMol25 labels
│   ├── eval_boltzmann_stage1.py   # log p_theta on held-out molecules (envs/flowmol)
│   ├── eval_boltzmann_stage2.py   # OMol25 energies + R² (envs/omol25)
│   └── launch_omol25_bgfm_*.sh    # SLURM launchers (H100 / A100)
├── tests/                         # pytest: bgfm_loss, bgfm_density, loss weights
├── CLAUDE.md                      # authoritative project guide
└── AGENTS.md                      # AI-agent guidance (locked decisions, gotchas)
```

## Two-environment setup

Torch version conflict forces env separation — **do not merge them**.

- **`envs/flowmol/`** — torch 2.2, DGL, PyTorch Lightning, flowmol3. Used for
  **training and inference**.
- **`envs/omol25/`** — torch 2.8, fairchem-core 2.19. Used for **OMol25
  preprocessing and post-hoc energy evaluation only**.

```bash
source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh
conda activate /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/envs/flowmol  # or envs/omol25
```

## Quick start

```bash
# Tests (in envs/flowmol)
python -m pytest tests/ -v

# Preprocess OMol25 (in envs/omol25)
sbatch scripts/preprocess_omol25.slurm

# BGFM training (in envs/flowmol)
sbatch scripts/launch_omol25_bgfm_h100.sh    # primary; H100 80GB
sbatch scripts/launch_omol25_bgfm_a100.sh    # backup; A100 80GB

# Level-1 baseline (FM only, no BGFM)
sbatch scripts/launch_omol25_train_h200.sh

# Boltzmann evaluation (two-stage, cross-env)
conda activate envs/flowmol
python scripts/eval_boltzmann_stage1.py --checkpoint <ckpt> --config <cfg> ...
conda activate envs/omol25
python scripts/eval_boltzmann_stage2.py --samples_json <stage1_out>/boltzmann_samples.json ...
```

Launchers support SIGUSR1 checkpoint-and-resume on `gpu_requeue` preemption
and accept `[config] [resume_ckpt]` as positional args.

## Current results (Jun 2026)

Headline metric — **per-molecule perturbation R²** on held-out molecules:
jiggle one molecule into many shapes, ask the model how likely each is
(`log p`) and physics how stable each is (`−E/kT`), measure linear fit.

Best checkpoint (v7c, step 50k):

| Slice                | R²        | slope | frac r > 0.5 |
|----------------------|-----------|-------|--------------|
| Overall (≤50 atoms)  | **0.278** | 0.840 | 67.5%        |
| Organic (CHNOFS)     | **0.380** | 0.800 | 65%          |
| **Transition metal** | **0.501** | 1.40  | **80%**      |

vs flow-matching-only baseline R² = 0.091 → **~3× tighter** Boltzmann
alignment. The strongest slice is **transition metals**, validating the
universal-chemistry claim where it matters most (no prior physics-based
molecular generator handles metals).

## Key configs

| Config                              | Purpose                                         |
|-------------------------------------|-------------------------------------------------|
| `configs/omol25_4m_bgfm.yaml`       | Primary BGFM training (`λ₁=0.1, λ₂=0.0, kT=1.0`) |
| `configs/omol25_4m_cfm.yaml`        | Level-1 ablation (BGFM disabled)                |
| `configs/geom_cfm_bondfree.yaml`    | Cell (2): bond-free on FlowMol3's home field    |
| `configs/omol25_4m_bgfm_energy_v8*` | Energy-consistency variants (room-T, T-cond, energy-only) |

Key `bgfm:` knobs: `lambda_1/2/3`, `kT`, `kT_conditioning`, `force_loss_type`
(mse / cosine / norm_mse), `probe_mode` (path / endpoint), `t_eval_values`,
`force_correction_alpha`, `warmup_frac/ramp_frac`. See `CLAUDE.md` for the
full reference.

## Locked design decisions

These are settled — see `CLAUDE.md` for rationale and pivot dates:

1. **Flow matching, not reflected SDE.**
2. **Bond-free** (`total_loss_weights.e = 0`).
3. **BGFM three-term loss** with frozen physics labels and within-molecule
   variance form.
4. **`max_atoms = 200`** (covers BINAP-Pd-substrate TS complexes).
5. **OMol25 as primary training data**, not QM9/GEOM.

## Storage layout

- Code + small runs: `/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/`
- Preprocessed data + checkpoints: `/n/netscratch/ryl_lab/Lab/yulili_cfm_mol/`
  (90-day purge)
- Raw OMol25: `/n/netscratch/ryl_lab/Lab/omol25/` (4M) and
  `/n/netscratch/ryl_lab/Lab/omol25_100m/` (100M)
- Do **not** write to `/n/home04/yulili/` (95 GB quota, mostly full).

## License

Our code: MIT. Baselines retain their original MIT licenses.
