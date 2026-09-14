#!/bin/bash
#SBATCH -J bgfm-tree-fm
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
PRIOR_RUN="${4:?prior fit run required}"
GROUP="${5:?method group required}"
FLOWMOL_ENV=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol
export PYTHONPATH="$SOURCE_ROOT" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$FLOWMOL_ENV/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/bgfm_tree_mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/bgfm_tree_cache_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$SOURCE_ROOT"
if [[ "$GROUP" == "fixed" ]]; then
  OPTIONS=(--methods gaussian fixed --include-warm)
elif [[ "$GROUP" == "learned" ]]; then
  OPTIONS=(--methods node pair)
else
  exit 2
fi
"$FLOWMOL_ENV/bin/python" -u scripts/research/tree_prior_fm.py \
 --project "$PROJECT" --prior-run "$PRIOR_RUN/prior" --out "$RUN_ROOT/study" \
 --protocol research/evidence/tree_prior_protocol_v1.json "${OPTIONS[@]}"
