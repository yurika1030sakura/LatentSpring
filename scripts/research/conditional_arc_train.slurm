#!/bin/bash
#SBATCH -J bgfm-conditional-arc-train
#SBATCH -A ryl_lab
#SBATCH -p sapphire,shared
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --array=0-3%4
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?unique output required}"
PROJECT="${3:?project required}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
METHODS=(work work_force)
METHOD="${METHODS[$((SLURM_ARRAY_TASK_ID/2))]}"
REPLICA="$((SLURM_ARRAY_TASK_ID%2))"
ARM="$RUN_ROOT/${METHOD}_s${REPLICA}"
export TMPDIR="$ARM/runtime/tmp" XDG_CACHE_HOME="$ARM/runtime/cache" MPLCONFIGDIR="$ARM/runtime/mpl"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
cd "$SOURCE_ROOT"
/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol/bin/python -u scripts/research/train_conditional_arc_energy.py \
  --project "$PROJECT" --out "$ARM" --objective "$METHOD" --replica "$REPLICA"
