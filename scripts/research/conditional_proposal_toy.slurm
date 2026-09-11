#!/bin/bash
#SBATCH -J bgfm-conditional-toy
#SBATCH -A ryl_lab
#SBATCH -p sapphire,shared
#SBATCH --array=0-7%2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.err
set -euo pipefail
SOURCE_ROOT="${1:?immutable source required}"
RUN_ROOT="${2:?unique output required}"
OBJECTIVE="${3:-shape_jump}"
STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
KINDS=(nonlinear affine mala rwm)
KIND="${KINDS[$((SLURM_ARRAY_TASK_ID / 2))]}"
REPLICA=$((SLURM_ARRAY_TASK_ID % 2))
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/conditional_mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/conditional_cache_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$SOURCE_ROOT"
exec "$STORE/envs/flowmol/bin/python" -u scripts/research/train_conditional_proposal_toy.py \
  --out "$RUN_ROOT/${KIND}_s${REPLICA}" --kind "$KIND" --replica "$REPLICA" --objective "$OBJECTIVE"
