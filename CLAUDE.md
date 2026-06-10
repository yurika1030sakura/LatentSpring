# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status (Jun 2026)

**Target venue: ICLR 2027 (submission ~2026-09-25).** Paper title is
**Boltzmann-Guided Flow Matching (BGFM)**: a 3D molecular generator whose
learned density provably equals the Boltzmann distribution under a universal
neural potential (OMol25), unifying flow matching + score matching +
Boltzmann generator ideas.

The `README.md` is **out of date** — it describes a pre-pivot reflected-
diffusion/valence-constrained framing that has been replaced. When in
doubt, trust this file and `notes/bgfm_method.md`.

## Critical design decisions (do not re-litigate)

These are locked in — if a previous pivot resurfaces (reflected SDE,
bond supervision, valence tables), it is wrong:

1. **Flow matching, not reflected SDE** (pivoted 2026-04-19). O(Δt) TV
   convergence vs O(√Δt); clean theory.
2. **Bond-free** (pivoted 2026-04-21). No bond supervision; `total_loss_weights.e = 0`.
   Bonds are post-hoc (xyz2mol on generated geometry). Rationale: QM has no
   bond orders; supervising human abstractions corrupts training on TM /
   radical / hypervalent chemistry.
3. **BGFM three-term loss** (pivoted 2026-04-21):
   `L_total = L_FM + λ_1 L_force + λ_2 L_energy`. Force from OMol25 DFT
   labels, energy via variance form (no partition function needed).
   Boltzmann consistency theorem in `notes/appendix_A_v4.tex`.
4. **`max_atoms = 200`** (bumped from 120 on 2026-04-20) to cover
   BINAP-Pd-substrate TS complexes for downstream catalyst fine-tuning.
5. **OMol25 as primary training data** (not QM9/GEOM). OMol25 is
   QM-native: stores positions + atoms + DFT forces/energies, no bonds.
   83-element coverage.

## Architecture: monkey-patching FlowMol3

This codebase does NOT fork FlowMol3. Instead it extends it via runtime
monkey-patches applied to the `CTMCVectorField` instance after
`model_from_config(cfg)` returns. Understanding this pattern is essential:

**Layer 1 — Geometric constraints** (`cfm_mol/flow_model.py::patch_flowmol`):
Patches 5 methods on `model.vector_field` (the `CTMCVectorField`):
  - Hook 1 (gluing): wraps `sample_conditional_path` → retracts `x_t` onto the steric fibre after interpolation
  - Hook 2 (tangent): wraps the inner `vector_field` helper → tangent-projects velocity before Euler step
  - Hook 3 (retract): wraps `step` → stashes graph context, retracts `x_t` after Euler update, optionally applies BGFM score guidance at sampling time
  - Hook 4a (train_time_discrete): wraps `sample_conditional_path` → projects a_t + e_t onto valence manifold during training
  - Hook 4b (discrete_projection): wraps `integrate` → projects final a_1 + e_1 at sample time

Bond-free mode (`total_loss_weights.e == 0`) auto-disables hooks 4a/4b since they assume bond semantics.

**Layer 2 — BGFM training loss** (`cfm_mol/bgfm_train_hook.py::patch_flowmol_bgfm`):
Replaces `model.training_step` with a wrapper that:
  1. Optionally shifts FM target by `α * F(x₁)` (force-corrected FM, if `force_correction_alpha > 0`)
  2. Runs original FM training step → `fm_total`
  3. For each `t_eval` in `t_eval_values`: builds auxiliary graph on the conditional path (or endpoint), runs forward pass, derives score via `score_from_fm_velocity`, computes force loss
  4. Optionally computes energy-consistency loss via FFJORD reverse-time ODE (if `λ₂ > 0`)
  5. Combines: `total = fm_total + λ₁ * L_force + λ₂ * L_energy + λ₃ * L_anchor`

**Entry point** (`scripts/run_train.py`):
  1. `read_config_file` → pops `bgfm` sub-block from `cfg.mol_fm` (FlowMol3's `__init__` doesn't know about it)
  2. `model_from_config(cfg)` → constructs FlowMol3 `CTMCVectorField`
  3. `patch_flowmol(model, d_min)` → Layer 1
  4. `patch_flowmol_bgfm(model, bgfm_cfg)` → Layer 2
  5. Wraps `sample_analyzer.analyze` with error guard (early-training samples are degenerate)
  6. `pl.Trainer.fit(model, datamodule)`

## Two-environment architecture (important)

Torch version conflict forced env separation. Do not try to merge them.

- **`envs/flowmol/`** — torch 2.2, DGL, PyTorch Lightning, flowmol3.
  Used for **training and inference**.
- **`envs/omol25/`** — torch 2.8, fairchem-core 2.19. Used for
  **OMol25 data preprocessing and post-hoc energy evaluation only**.

Activation:
```bash
source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh
conda activate /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/envs/flowmol  # or envs/omol25
```

Cross-env communication (only needed for live OMol25 during training,
which we currently don't do — all OMol25 labels are precomputed in the
preprocess step): `scripts/omol25_worker.py` runs an XMLRPC server in
`envs/omol25`; `cfm_mol/physics_drift.py` is the client in `envs/flowmol`.

## Storage layout (netscratch vs holylabs — do not confuse)

- **Code + small runs**: `/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/` (2 PB fs, 28% used)
- **Preprocessed data + training checkpoints**: `/n/netscratch/ryl_lab/Lab/yulili_cfm_mol/` (4 PB, 81% used — scratch, 90-day purge)
- **Raw OMol25 downloads**: `/n/netscratch/ryl_lab/Lab/omol25/` (4M) and `/n/netscratch/ryl_lab/Lab/omol25_100m/` (100M)
- **Do NOT write to `/n/home04/yulili/`** (95 GB quota, already 78% full). All our intermediate files go to netscratch.

Configs point `output_dir` into `runs/` under holylabs and `processed_data_dir` into netscratch. If a future config deviates from this, fix it.

## Key files

### Method (the interesting code)

- `cfm_mol/flow_model.py::patch_flowmol` — 5 hooks on CTMCVectorField (see Architecture section above). Also contains `_apply_bgfm_score_guidance` for inference-time score-guided sampling with configurable schedule (late_linear, late_quadratic, constant).
- `cfm_mol/bgfm_train_hook.py::patch_flowmol_bgfm` — training-step monkey-patch (see Architecture section above). Key config knobs: `probe_mode` (path vs endpoint), `force_loss_type` (mse/cosine/norm_mse), `force_target_mode` (true/shuffle_atoms for negative-control ablation), `force_correction_alpha` (FM target shifting by α·F).
- `cfm_mol/bgfm_loss.py` — core loss math:
  - `score_from_fm_velocity(v_theta, x_t, t)` — closed-form score of FM marginal: `s = (t·v - x) / ((1-t)·σ²)`. Clamped at `SCORE_NORM_CAP=1000` per-atom.
  - `divergence_exact_atomwise` — O(3N) backward passes, reference for tests.
  - `divergence_hutchinson` — single backward pass per sample, used in training.
  - `force_loss` / `force_direction_loss` / `score_force_cosine` — L_force variants.
  - `energy_loss_variance` — L_energy = Var(log p + E/kT).
  - `bgfm_total_loss` — combines with warmup/ramp schedule.
- `cfm_mol/bgfm_density.py` — FFJORD-style log-density via reverse-time ODE + divergence accumulation (`log_density_via_flow`). Contains per-molecule energy-consistency losses:
  - `energy_consistency_loss_per_mol` — within-parent variance of (log p + E/kT) across K perturbations. Replaces the deprecated cross-batch form which was broken by per-molecule log Z spread.
  - `energy_consistency_loss_per_mol_with_anchor` — variance + anchor in one FFJORD pass.
  - `energy_anchor_loss` — L_anchor = mean((log p + E/kT + log_Z_pred)²) to prevent trivial-constant failure.
- `cfm_mol/log_z_predictor.py` — `LogZPredictor`: tiny (~few k params) invariant network predicting per-molecule log Z from atom-type counts + charge. Zero-initialized so model starts at no-op. Used by anchor loss.
- `cfm_mol/perturbation_loader.py` — `PerturbationLoader`: stateful iterator over pre-computed geometric perturbation shards (K perturbations per parent molecule). Iterated inside the training hook every `energy_every_k_steps`, separate from Lightning's main DataLoader.
- `cfm_mol/kt_conditioning.py` — temperature conditioning: `patch_kT_conditioning` adds residual kT projection to vector_field's scalar_embedding via forward hook. Zero-initialized final layer → no-op at init, safe to resume from any checkpoint. `sample_kT` draws log-uniform kT in [kT_min, kT_max].
- `cfm_mol/physics_drift.py` — inference-time drift (Level-1 OMol25 forces). Currently unused in training; reserved for Paper 2 scaffold-conditioned generation.
- `cfm_mol/projection.py` — valence/connectivity projections. **Dead code under bond-free**, kept for Paper 2/3 re-use.
- `cfm_mol/fibre_dgl.py` — DGL↔padded-tensor bridge. `retract_dgl` and `tangent_project_dgl` convert between FlowMol3's flat `(total_atoms, 3)` and the padded `(B, N, 3)` used by `cfm_mol/fibre.py`.
- `cfm_mol/domain.py` — admissible manifold: `valence_ok`, `steric_ok`, `connectivity_ok`, `default_d_min_table` (covalent-radius-based, scale=0.7).
- `cfm_mol/data/omol25.py` — OMol25 dataset adapter: ASE LMDB → FlowMol3-style DGL graph. Produces `x_1_true`, `a_1_true`, `c_1_true`, `e_1_true` ndata/edata matching `MoleculeDataset`.

### Preprocess

- `scripts/preprocess_omol25.py` — LMDB → FlowMol3-native .pt. Reads DFT forces/energies via `atoms.calc.results` (SinglePointCalculator pattern — **not** `atoms.info`). Stores `atom_types` as int8 indices (4× disk savings vs bool one-hot at 83 elements). Supports `--shard_size` for 100M-scale output.
- `scripts/write_omol25_valencies.py` — writes permissive `train_data_valencies_omol25.json` so FlowMol3's `SampleAnalyzer` doesn't refuse to construct.
- `scripts/precompute_energy_perturbations.py` — generates K geometric perturbations per parent molecule with OMol25 energies, stored as shards. Required when `λ₂ > 0`.

### Evaluation (two-stage Boltzmann eval)

- `scripts/eval_boltzmann_stage1.py` (in `envs/flowmol`) — loads checkpoint, generates M perturbations per held-out molecule, computes log p_theta via FFJORD, exports to JSON.
- `scripts/eval_boltzmann_stage2.py` (in `envs/omol25`) — reads Stage 1 JSON, computes OMol25 DFT energies for each perturbation, outputs correlation (log p vs -E/kT) and Boltzmann R² per molecule.
- `scripts/eval_cell_metrics.py` — computes ablation cell metrics (validity, uniqueness, novelty).
- `scripts/evaluate_validity.py` — xyz2mol-based validity check for bond-free outputs.

### Theory / paper drafts

- `notes/bgfm_method.md` — full derivation of the three BGFM losses, score formula, divergence estimator. Primary reference when editing `cfm_mol/bgfm_loss.py`.
- `notes/appendix_A_v4.tex` — formal proofs (8 theorems). Latest version; v1/v2/v3 are superseded.
- `notes/section3_bond_free_v1.md` — main-text Section 3 (Level-1 framing; needs v2 rewrite for BGFM).

### Configs

- `configs/omol25_4m_bgfm.yaml` — **primary training config** (bond-free, BGFM enabled, `λ_1=0.1, λ_2=0.0, kT=1.0`)
- `configs/omol25_4m_cfm.yaml` — Level-1 ablation baseline (same as bgfm but `bgfm.enabled=false`)
- `configs/geom_cfm_bondfree.yaml` — Cell (2) ablation: bond-free on FlowMol3's GEOM home field
- `configs/omol25_4m_bgfm_energy_v8*.yaml` — energy-consistency experiment variants (v8a room-T, v8b T-conditional, v8c energy-only)

## Common commands

### Preprocess OMol25 (in `envs/omol25`)

```bash
sbatch scripts/preprocess_omol25.slurm        # 4M, ~48 min, output 5.3 GB
sbatch scripts/preprocess_omol25_100m.slurm   # 100M, sharded --shard_size 5M, needs ~256 GB RAM
```

### Training (in `envs/flowmol`)

```bash
# BGFM (primary)
sbatch scripts/launch_omol25_bgfm_h100.sh     # gpu_requeue H100 80GB
sbatch scripts/launch_omol25_bgfm_a100.sh     # gpu A100 80GB (backup, ~2x slower)

# Level-1 baseline (no BGFM)
sbatch scripts/launch_omol25_train_h200.sh    # despite name, now H100 gpu_requeue
sbatch scripts/launch_omol25_level1_a100.sh   # A100 backup

# Smoke test (1 hour, --fast, validates pipeline before real training)
sbatch scripts/launch_omol25_smoke_h200.sh
```

All launchers support SIGUSR1 checkpoint-and-resume for gpu_requeue preemption. Pass `[config] [resume_ckpt]` as positional args.

### Unit tests

```bash
cd /n/holylabs/ryl_lab/Lab/yulili_cfm_mol
conda activate /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/envs/flowmol

# All tests
python -m pytest tests/ -v

# Individual test files
python -m pytest tests/test_bgfm_loss.py -v         # 11 tests, ~22s — score, divergence, force/energy loss
python -m pytest tests/test_bgfm_density.py -v       # FFJORD log-density, within-group variance
python -m pytest tests/test_flowmol_loss_weighting.py -v  # FlowMol3 loss weight broadcasting

# Single test by name
python -m pytest tests/test_bgfm_loss.py -v -k "test_divergence_exact_linear_vf"
```

### Boltzmann evaluation (two-stage, cross-env)

```bash
# Stage 1 (envs/flowmol): compute log p_theta for perturbations
conda activate envs/flowmol
python scripts/eval_boltzmann_stage1.py \
    --checkpoint <ckpt> --config <config> \
    --eval_data /n/netscratch/ryl_lab/Lab/yulili_cfm_mol/omol25_4m_processed \
    --n_molecules 60 --n_perturb 16 --sigma 0.15 \
    --out_dir runs/eval/boltzmann/<tag>

# Stage 2 (envs/omol25): compute OMol25 energies + correlation
conda activate envs/omol25
python scripts/eval_boltzmann_stage2.py \
    --samples_json runs/eval/boltzmann/<tag>/boltzmann_samples.json \
    --out_dir runs/eval/boltzmann/<tag>
```

### Ablation matrix for Paper 1

| Cell | Training data | Loss | Purpose |
|---|---|---|---|
| (0) | GEOM + bonds | FM + bond CE | FlowMol3 published (cite, don't re-run) |
| (2) | GEOM | FM only (bond-free) | Isolates bond-free variable on home field |
| (4a) | OMol25 + forces + energies | full BGFM | **Main result** |
| (4b) | Same ckpt as 4a | — | Zero-shot transfer to GEOM (strong OOD claim) |
| (4c) | Same ckpt | — | Eval on tmQM / kraken / hypervalent (killer figure) |

## Config knobs reference

Key `bgfm:` sub-block settings in training configs:

| Key | Values | Effect |
|---|---|---|
| `enabled` | true/false | Master switch for BGFM |
| `lambda_1` | float | Force loss weight |
| `lambda_2` | float | Energy loss weight (0 = disabled) |
| `lambda_3` | float | Anchor loss weight (prevents trivial-constant failure) |
| `kT` | float (eV) | Fixed temperature. 1.0 keeps F/kT tame; 0.025 = room-T |
| `kT_conditioning` | true/false | Sample kT per step, condition model on kT |
| `kT_min/kT_max` | float (eV) | Range for kT sampling when conditioning |
| `force_loss_type` | mse/cosine/norm_mse | MSE on full score, or direction-only variants |
| `force_target_mode` | true/shuffle_atoms | `shuffle_atoms` is negative-control ablation |
| `probe_mode` | path/endpoint | `path` evaluates score on conditional path; `endpoint` at x₁ |
| `t_eval_values` | list[float] | Times to probe score (e.g. [0.85, 0.92, 0.97]) |
| `force_correction_alpha` | float | FM target shifting: x₁ → x₁ + α·F(x₁). 0 = off |
| `warmup_frac/ramp_frac` | float | Schedule: FM-only warmup, then linear ramp to full λ |
| `energy_perturbation_shards` | list[path] | Required when λ₂ > 0 |
| `energy_every_k_steps` | int | Amortize expensive FFJORD (default: every step) |
| `energy_n_ode_steps` | int | Reverse-time ODE resolution (default: 8) |

## Training loop gotchas

1. **Python stdout buffering under SLURM**: always use `python -u` and `export PYTHONUNBUFFERED=1` in launchers, otherwise output only flushes at job end.
2. **FlowMol3's `SampleAnalyzer` needs a valency JSON** (`train_data_valencies_*.json` glob pattern). It raises FileNotFoundError at model init if absent. `write_omol25_valencies.py` produces a permissive 83-element one. `SampleAnalyzer.analyze()` is wrapped with `_safe_analyze` in `scripts/run_train.py` to swallow its RDKit-valence crashes on bond-free outputs.
3. **`n_atoms_histogram.pt` is a tuple `(values, counts)`, not a single tensor.** FlowMol3's `build_n_atoms_dist` unpacks two values; our preprocess writes both.
4. **Energy consistency term (`λ_2 > 0`) requires pre-computed perturbation shards** from `scripts/precompute_energy_perturbations.py`. The perturbation loader runs outside Lightning's DataLoader and iterates inside the training hook.
5. **Cross-batch energy variance is mathematically broken** — use per-molecule form only (`energy_consistency_loss_per_mol`). The cross-batch form picks up per-molecule log Z spread (~1e9), not Boltzmann deviation. The deprecated `energy_consistency_loss` in `bgfm_density.py` is kept only for unit test validation.
6. **Score formula diverges at t=1** — `score_from_fm_velocity` has a `(1-t)` denominator; always evaluate at `t_eval < 1` (e.g., 0.95). Per-atom norm capped at 1000.
7. **NaN guards in training hook** — both force and energy losses check `torch.isfinite` and skip contribution (with logging) if NaN, preventing optimizer poisoning.

## Things NOT to do

- **Don't re-introduce bond supervision** (setting `total_loss_weights.e > 0` on OMol25). It corrupts training on TMs because OMol25 has no bond labels; our pipeline stores empty bond tensors.
- **Don't install fairchem-core in `envs/flowmol`**. It pulls torch 2.8 + cudnn 9 and breaks DGL. Keep envs separate.
- **Don't delete `cfm_mol/projection.py` or valence code paths.** They're dead under bond-free but reserved for Paper 2 scaffold-conditioned fine-tuning.
- **Don't preprocess OMol25 data in `envs/flowmol`**. fairchem-core is only in `envs/omol25`.
- **Don't write OMol25 labels from `atoms.info` or `atoms.arrays['forces']`**. Use ASE's standard API: `atoms.get_potential_energy()` and `atoms.get_forces()` (OMol25 stores via `SinglePointCalculator`).
- **Don't pop `bgfm` from config before `model_from_config`** is called — `run_train.py` already handles this. Passing unknown keys to FlowMol3's `__init__` will crash.
