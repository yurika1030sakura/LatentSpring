#!/bin/bash
# Shared environment for Harvard FASRC (Cannon) runs of BGFM/HBFM.
# Source this from the FASRC SLURM launchers: `source scripts/fasrc/env.sh`.
#
# Layout on this machine (see notes/SETUP.md + CLAUDE.md storage section):
#   REPO  = the git working tree (lives on the small $HOME quota — no big files)
#   STORE = woo_lab holylabs (2 PB) — holds envs, checkpoints, data, runs, logs.
#           The repo dirs envs/ runs/ baselines/ data/ processed_data/ logs/
#           checkpoints/ are symlinks into STORE (kept out of git).
export REPO=/n/home04/yulili/bgfm
export STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm

# Keep conda's package cache OFF the full $HOME quota (woo_lab is fine for this).
export CONDA_PKGS_DIRS=$STORE/.conda_pkgs
# TMPDIR must be NODE-LOCAL (not NFS): the DataLoader/multiprocessing workers
# create temp dirs here, and cleaning them up on NFS throws benign-but-noisy
# ".nfs... Device or resource busy" tracebacks at teardown. Local /tmp avoids
# that and is faster. Falls back to $STORE/tmp only if no local scratch exists.
export TMPDIR="${SLURM_TMPDIR:-/tmp}/bgfm_${USER}_${SLURM_JOB_ID:-local}"
mkdir -p "$CONDA_PKGS_DIRS" "$STORE/logs"
mkdir -p "$TMPDIR" 2>/dev/null || { export TMPDIR=$STORE/tmp; mkdir -p "$TMPDIR"; }

source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh

# Direct interpreters (avoid `conda activate` fragility inside batch jobs).
export FLOWMOL_PY=$STORE/envs/flowmol/bin/python   # torch 2.2 + DGL + FlowMol3 (train/eval)
export OMOL25_PY=$STORE/envs/omol25/bin/python      # torch 2.8 + fairchem (preprocess + eSEN teacher)

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=$REPO
# Reduce CUDA fragmentation OOM (matters for the energy-term FFJORD + long runs).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
