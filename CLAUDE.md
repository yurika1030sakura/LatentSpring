# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status (Apr 2026)

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

- `cfm_mol/flow_model.py::patch_flowmol` — 5 hooks that patch FlowMol3's CTMCVectorField. Geometric hooks (tangent/retract/gluing, `d_min` pair-distance) always active; bond-valence hooks (discrete_projection, train_time_discrete) auto-disabled when `total_loss_weights.e == 0` (bond-free).
- `cfm_mol/bgfm_train_hook.py::patch_flowmol_bgfm` — training-step monkey-patch that runs an auxiliary forward at (x_1, t_eval ~ 0.95), derives the implied score, and adds `λ_1 L_force`. Auto-invoked from `scripts/run_train.py` when `mol_fm.bgfm.enabled = true`.
- `cfm_mol/bgfm_loss.py` — score-from-FM-velocity (Gaussian-prior closed form), exact + Hutchinson divergence estimators, force/energy loss, warmup/ramp schedule. 11 unit tests in `tests/test_bgfm_loss.py`.
- `cfm_mol/physics_drift.py` — inference-time drift (Level-1 OMol25 forces). Currently unused in training; reserved for Paper 2 scaffold-conditioned generation.
- `cfm_mol/projection.py` — valence/connectivity projections. **Dead code under bond-free**, kept for Paper 2/3 re-use.

### Preprocess

- `scripts/preprocess_omol25.py` — LMDB → FlowMol3-native .pt. Reads DFT forces/energies via `atoms.calc.results` (SinglePointCalculator pattern — **not** `atoms.info`). Stores `atom_types` as int8 indices (4× disk savings vs bool one-hot at 83 elements). Supports `--shard_size` for 100M-scale output.
- `scripts/write_omol25_valencies.py` — writes permissive `train_data_valencies_omol25.json` so FlowMol3's `SampleAnalyzer` doesn't refuse to construct. (Validity metrics it reports during training are uninformative for bond-free; we use xyz2mol post-hoc instead.)

### Theory / paper drafts

- `notes/bgfm_method.md` — full derivation of the three BGFM losses, score formula, divergence estimator. Primary reference when editing `cfm_mol/bgfm_loss.py`.
- `notes/appendix_A_v4.tex` — formal proofs (8 theorems). Latest version; v1/v2/v3 are superseded.
- `notes/section3_bond_free_v1.md` — main-text Section 3 (Level-1 framing; needs v2 rewrite for BGFM).

### Configs

- `configs/omol25_4m_bgfm.yaml` — **primary training config** (bond-free, BGFM enabled, `λ_1=0.1, λ_2=0.0, kT=1.0`)
- `configs/omol25_4m_cfm.yaml` — Level-1 ablation baseline (same as bgfm but `bgfm.enabled=false`)
- `configs/geom_cfm_bondfree.yaml` — Cell (2) ablation: bond-free on FlowMol3's GEOM home field

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
python -m pytest tests/test_bgfm_loss.py -v   # 11 tests, ~22s
```

### Ablation matrix for Paper 1

| Cell | Training data | Loss | Purpose |
|---|---|---|---|
| (0) | GEOM + bonds | FM + bond CE | FlowMol3 published (cite, don't re-run) |
| (2) | GEOM | FM only (bond-free) | Isolates bond-free variable on home field |
| (4a) | OMol25 + forces + energies | full BGFM | **Main result** |
| (4b) | Same ckpt as 4a | — | Zero-shot transfer to GEOM (strong OOD claim) |
| (4c) | Same ckpt | — | Eval on tmQM / kraken / hypervalent (killer figure) |

## Training loop gotchas

1. **Python stdout buffering under SLURM**: always use `python -u` and `export PYTHONUNBUFFERED=1` in launchers, otherwise output only flushes at job end. We've been bitten — `scripts/preprocess_omol25.slurm` has this fixed.
2. **FlowMol3's `SampleAnalyzer` needs a valency JSON** (`train_data_valencies_*.json` glob pattern). It raises FileNotFoundError at model init if absent. `write_omol25_valencies.py` produces a permissive 83-element one. `SampleAnalyzer.analyze()` is wrapped with `_safe_analyze` in `scripts/run_train.py` to swallow its RDKit-valence crashes on bond-free outputs.
3. **`n_atoms_histogram.pt` is a tuple `(values, counts)`, not a single tensor.** FlowMol3's `build_n_atoms_dist` unpacks two values; our preprocess writes both.
4. **Energy consistency term (`λ_2 > 0`) is disabled in v1** because it requires a trajectory divergence integral (expensive + unstable). Enable in Week 3 grid search after force-only stability is confirmed.

## Things NOT to do

- **Don't re-introduce bond supervision** (setting `total_loss_weights.e > 0` on OMol25). It corrupts training on TMs because OMol25 has no bond labels; our pipeline stores empty bond tensors.
- **Don't install fairchem-core in `envs/flowmol`**. It pulls torch 2.8 + cudnn 9 and breaks DGL. Keep envs separate.
- **Don't delete `cfm_mol/projection.py` or valence code paths.** They're dead under bond-free but reserved for Paper 2 scaffold-conditioned fine-tuning.
- **Don't preprocess OMol25 data in `envs/flowmol`**. fairchem-core is only in `envs/omol25`.
- **Don't write OMol25 labels from `atoms.info` or `atoms.arrays['forces']`**. Use ASE's standard API: `atoms.get_potential_energy()` and `atoms.get_forces()` (OMol25 stores via `SinglePointCalculator`).
