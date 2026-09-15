#!/bin/bash
#SBATCH -J bgfm-fresh-physics-xtb
#SBATCH --array=0-1%2
#SBATCH -A ryl_lab
#SBATCH -p sapphire,shared
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?unique output required}"
PROJECT="${3:?project required}"
GENERATION="${4:?generation root required}"
STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH="$STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}"
export TMPDIR="$RUN_ROOT/runtime/tmp" MPLCONFIGDIR="$RUN_ROOT/runtime/mpl" XDG_CACHE_HOME="$RUN_ROOT/runtime/cache"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$SOURCE_ROOT"
"$STORE/envs/flowmol/bin/python" -u scripts/research/fresh_physics_xtb.py \
 --project "$PROJECT" --protocol "research/evidence/fresh_physics_s${SLURM_ARRAY_TASK_ID}_v1.json" \
 --generation "$GENERATION/s${SLURM_ARRAY_TASK_ID}/study" --out "$RUN_ROOT/s${SLURM_ARRAY_TASK_ID}/xtb" --workers 8
