#!/bin/bash
#SBATCH -J bgfm-oracle-gpu
#SBATCH -A ryl_lab
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?immutable source required}"
RUN_ROOT="${2:?unique output required}"
STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NVIDIA_TF32_OVERRIDE=0
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/oracle_gpu_mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/oracle_gpu_cache_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$SOURCE_ROOT"
exec "$STORE/envs/flowmol/bin/python" -u scripts/research/qualify_oracle_gpu.py \
 --panel /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/runs/jax_oracle_panel_v1 \
 --out "$RUN_ROOT" --oracle-python "$STORE/envs/omol25/bin/python" \
 --checkpoint "$STORE/checkpoints/omol25/esen_sm_conserving_all.pt"
