# Running BGFM/HBFM on Harvard FASRC (Cannon) — this machine's setup

Companion to `notes/SETUP.md` (which is written for UMass Unity). This file records
how the `norman` branch is wired up on the Harvard **FASRC / Cannon** cluster, where
`$HOME` (`/n/home04/yulili`, 95 GB) is ~99 % full so **nothing large lives in the repo**.

## Storage model (repo in $HOME, big files in woo_lab)

| Path | What |
|---|---|
| `REPO  = /n/home04/yulili/bgfm` | git working tree only (code, configs, notes) |
| `STORE = /n/holylabs/woo_lab/Lab/yulili/bgfm` | envs, checkpoints, data, runs, logs (2 PB fs) |

The repo dirs `envs/ runs/ baselines/ data/ processed_data/ logs/ checkpoints/` are
**symlinks into `STORE`** and are git-ignored. Never put large files directly in the repo.

A second copy of this project exists at `/n/holylabs/ryl_lab/Lab/yulili_cfm_mol` — that's
a **different person on the `main` branch**. It was only used here as a reference for the
FASRC config (conda base, SLURM partitions, and to seed the two conda envs + data). Do not
write into it.

## What was set up (done)

- **Conda envs** cloned from the ryl_lab reference into `STORE/envs/{flowmol,omol25}`
  (isolated copies; editable installs re-pointed at THIS repo + `STORE/baselines/flowmol3`).
- **`baselines/flowmol3`** copied to `STORE/baselines/flowmol3` (code only; its 12 GB
  GEOM/QM9 `data/` dir was excluded — not needed for OMol25).
- **FlowMol3 force patch APPLIED** to `STORE/baselines/flowmol3/flowmol/data_processing/dataset.py`
  (`force_1_true` wiring verified present). Without it the physics loss silently no-ops.
- **Processed OMol25 4M** copied to `STORE/processed_data/omol25_4m_processed` (5.5 GB,
  has DFT forces+energies). Configs point here. Raw LMDB reused in place at
  `/n/netscratch/ryl_lab/Lab/omol25/train_4M` (only read by re-preprocessing).
- **Configs repointed** to woo_lab paths: `configs/omol25_4m_bgfm.yaml` (off-policy headline)
  and new `configs/omol25_4m_bgfm_onpolicy.yaml` (on-policy HBFM).
- **FASRC SLURM launchers** in `scripts/fasrc/` (H200 partition `gpu_h200`, account `woo_lab`).

## The one manual prerequisite (only for the ON-POLICY variant)

The **eSEN oracle checkpoint** (`esen_sm_conserving_all.pt`) is **not on this machine** and is
gated. It is needed ONLY for the on-policy HBFM teacher (and the optional eSEN eval diagnostic).
To enable on-policy training:
```
# request HF access to facebook/OMol25 + facebook/UMA, then:
export HF_TOKEN=...  HF_HUB_DISABLE_XET=1
# place the file at:
#   $STORE/checkpoints/omol25/esen_sm_conserving_all.pt
```
The **off-policy 4M headline run needs none of this** — it trains on the precomputed DFT
forces/energies already in the processed data.

## Launch on the H200 (`gpu_h200` partition, gres `gpu:nvidia_h200:1`)

```bash
cd /n/home04/yulili/bgfm

# 0. (recommended first) 1-hour smoke test — confirms env+data+forces on the H200.
#    In the log, you MUST see train_L_force / train_score_force_cos (physics ACTIVE);
#    if you see train_bgfm_no_force_skip, the force patch didn't take.
sbatch scripts/fasrc/smoke_bgfm_h200.slurm

# 1. THE 4M (FULL) HEADLINE RUN — off-policy BGFM, no oracle needed. This fills the
#    empty "Ours - HBFM (4M, full)" row in notes/benchmark_plan.md (Tables 2 & 3).
sbatch scripts/fasrc/train_bgfm_4m_h200.slurm
#    (auto-resumes from the latest checkpoint under output_dir on requeue.)

# 1b. On-policy HBFM variant (needs the eSEN ckpt above):
sbatch scripts/fasrc/train_bgfm_onpolicy_h200.slurm

# 2. Eval a finished checkpoint -> fills the "ours" rows:
sbatch scripts/fasrc/eval_h200.slurm \
  $STORE/runs/omol25_4m_bgfm/<...>/checkpoints/last.ckpt \
  configs/omol25_4m_bgfm.yaml ours_4m
```

Scale note (from `notes/SETUP.md`): full 4M is large even on an H200. A 1M subset
(add `--max_n_mols 1000000` to preprocessing, ~1 epoch overnight) or multi-GPU DDP is the
practical fast path to the headline; a `gpu_h200` node has 4× H200, so DDP `devices: 4` is
available if wanted.

## Env activation (interactive)

```bash
source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh
conda activate /n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol   # train/eval (torch 2.2 + DGL)
conda activate /n/holylabs/woo_lab/Lab/yulili/bgfm/envs/omol25    # preprocess + teacher (torch 2.8)
# or call the interpreters directly: $STORE/envs/{flowmol,omol25}/bin/python
```
The `flowmol` env's torch needs GPU-node CUDA libs on `LD_LIBRARY_PATH` (the launchers set
`LD_LIBRARY_PATH=$STORE/envs/flowmol/lib`); it will not import torch on a CPU-only login node.
