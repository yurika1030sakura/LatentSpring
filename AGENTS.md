# AGENTS.md

This file provides guidance to the AI agent when working with code in this repository.

## Source of truth

- `CLAUDE.md` is the authoritative project guide — read it for architecture, design decisions, and command reference. Trust it over `README.md`, which is **out of date** (pre-pivot reflected-diffusion framing, replaced by BGFM).
- `notes/bgfm_method.md` — derivation of the three BGFM losses (primary reference when editing `cfm_mol/bgfm_loss.py`).
- `notes/appendix_A_v4.tex` — formal proofs (v1/v2/v3 are superseded).

## Locked design decisions (do not re-litigate)

1. **Flow matching, not reflected SDE.** Pivoted 2026-04-19.
2. **Bond-free.** `total_loss_weights.e = 0`. Do not re-introduce bond supervision on OMol25 — it has no bond labels.
3. **BGFM three-term loss:** `L_total = L_FM + λ₁·L_force + λ₂·L_energy` (+ optional `λ₃·L_anchor`).
4. **`max_atoms = 200`** (not 120).
5. **OMol25 is primary training data**, not QM9/GEOM.

## Architecture: monkey-patching, not forking

The code does **not** fork FlowMol3. It extends a `CTMCVectorField` instance via runtime patches after `model_from_config(cfg)`:
- `cfm_mol/flow_model.py::patch_flowmol` — 5 hooks (gluing, tangent projection, retract, discrete projections).
- `cfm_mol/bgfm_train_hook.py::patch_flowmol_bgfm` — wraps `training_step` to add force/energy/anchor losses.
- Entry point: `scripts/run_train.py` pops the `bgfm` sub-block from `cfg.mol_fm` **before** `model_from_config` (FlowMol3's `__init__` rejects unknown keys) — preserve this order.

`cfm_mol/projection.py` is dead code under bond-free but kept for Paper 2 — do not delete.

## Two-environment setup (do not merge)

Torch version conflict requires separation:
- `envs/flowmol/` — torch 2.2 + DGL + Lightning + flowmol3. **Training and inference.**
- `envs/omol25/` — torch 2.8 + fairchem-core 2.19. **OMol25 preprocessing and energy eval only.**

Do not install `fairchem-core` in `envs/flowmol` (pulls torch 2.8 + cudnn 9, breaks DGL).

## Storage layout

- Code + small runs: `/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/`
- Preprocessed data + checkpoints: `/n/netscratch/ryl_lab/Lab/yulili_cfm_mol/` (90-day purge)
- **Never write to `/n/home04/yulili/`** (95 GB quota, ~78% full).

## Commands

Tests (in `envs/flowmol`):
```bash
python -m pytest tests/ -v
python -m pytest tests/test_bgfm_loss.py -v -k "test_name"
```

Training: `sbatch scripts/launch_omol25_bgfm_h100.sh` (or `_a100.sh`). Launchers accept `[config] [resume_ckpt]` and support SIGUSR1 requeue.

Boltzmann eval is two-stage and crosses envs (`scripts/eval_boltzmann_stage1.py` in flowmol → `stage2.py` in omol25).

Primary configs: `configs/omol25_4m_bgfm.yaml` (BGFM main), `configs/omol25_4m_cfm.yaml` (Level-1 baseline).

## Gotchas

1. **SLURM stdout buffering** — always `python -u` and `export PYTHONUNBUFFERED=1` in launchers.
2. **`SampleAnalyzer` requires `train_data_valencies_*.json`** — `scripts/write_omol25_valencies.py` writes a permissive 83-element one. `_safe_analyze` in `run_train.py` swallows its RDKit crashes on bond-free outputs.
3. **`n_atoms_histogram.pt` is a tuple `(values, counts)`** — not a single tensor.
4. **Score formula `s = (t·v − x) / ((1−t)·σ²)` diverges at t=1** — always evaluate at `t < 1` (e.g. 0.95). Per-atom norm capped at 1000.
5. **Cross-batch energy variance is broken** — use `energy_consistency_loss_per_mol` only. The cross-batch form picks up per-molecule log Z spread (~1e9).
6. **OMol25 labels live in `atoms.calc.results`** (SinglePointCalculator), not `atoms.info` or `atoms.arrays['forces']`. Use `atoms.get_potential_energy()` / `atoms.get_forces()`.
7. **Energy term (`λ₂ > 0`) needs precomputed perturbation shards** from `scripts/precompute_energy_perturbations.py`; the loader runs outside Lightning's DataLoader inside the training hook.
8. **NaN guards** in the training hook skip force/energy contributions if non-finite — preserve them.
