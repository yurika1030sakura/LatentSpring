#!/bin/bash
#SBATCH -J bgfm-dynamic-fix
#SBATCH --array=0-1%2
#SBATCH -A ryl_lab
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?output required}"
PROJECT="${3:?project required}"
FLOWMOL_ENV=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol
export PYTHONPATH="$SOURCE_ROOT" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$FLOWMOL_ENV/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/bgfm_dynamic_mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/bgfm_dynamic_cache_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$SOURCE_ROOT"
PROTOCOL="research/evidence/dynamic_tree_s${SLURM_ARRAY_TASK_ID}_v1.json"
"$FLOWMOL_ENV/bin/python" -u scripts/research/tree_prior_fm.py \
 --project "$PROJECT" --prior-run "$PROJECT/runs/tree_prior_fit_connected_v1/prior" \
 --out "$RUN_ROOT/s${SLURM_ARRAY_TASK_ID}/study" --protocol "$PROTOCOL" --methods fixed tree_learned tree_fixed local_learned \
 --reuse-fixed-run "$PROJECT/runs/dynamic_tree_v1/s${SLURM_ARRAY_TASK_ID}/study"
