#!/bin/bash
#SBATCH -J bgfm-affine-confirm
#SBATCH -A ryl_lab
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?output required}"
BGFM_STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
LAB_RUNS=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/runs
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$BGFM_STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/bgfm_score_mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/bgfm_score_cache_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$SOURCE_ROOT"
AFFINE_RUN="${3:-species_affine_adapter_1000_v1}"
CONVEX_CONFIRMATION="${4:-species_entropy_adapter_confirmation_v1}"
exec "$BGFM_STORE/envs/flowmol/bin/python" -u scripts/research/confirm_species_affine.py \
 --runs-root "$LAB_RUNS" --out "$RUN_ROOT" --affine-run "$AFFINE_RUN" \
 --convex-confirmation "$CONVEX_CONFIRMATION"
