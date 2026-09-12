#!/bin/bash
#SBATCH -J bgfm-eacf-direction-check
#SBATCH -A ryl_lab
#SBATCH -p sapphire,shared
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --array=0-3%2
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?unique output required}"
PROJECT="${3:?artifact root required}"
LAB=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
UPSTREAM="$LAB/external/eacf_20260910"
[ "$(git -C "$UPSTREAM" rev-parse HEAD)" = beafab1b1ccd2b770572daeef1cf15f3fe199c21 ]
git -C "$UPSTREAM" diff --quiet HEAD
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TMPDIR="$RUN_ROOT/runtime/tmp" MPLCONFIGDIR="$RUN_ROOT/runtime/mpl" XDG_CACHE_HOME="$RUN_ROOT/runtime/cache"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
PROFILES=(published compact)
STEMS=(eacf_joint eacf_compact)
PROFILE="${PROFILES[$((SLURM_ARRAY_TASK_ID/2))]}"
STEM="${STEMS[$((SLURM_ARRAY_TASK_ID/2))]}"
REPLICA=$((SLURM_ARRAY_TASK_ID%2))
ARGS=()
if [ "$REPLICA" = 0 ]; then ARGS+=(--check-jacobian); fi
cd "$SOURCE_ROOT"
"$STORE/envs/flowmol/bin/python" -u scripts/research/check_eacf_directional_interface.py \
  --run "$PROJECT/runs/${STEM}_condition_00_s${REPLICA}_v1" --table "$PROJECT/runs/chemical_policy_table_v1" \
  --transport-python "$LAB/external/eacf_env_20260910/bin/python" --upstream "$UPSTREAM" \
  --out "$RUN_ROOT/${PROFILE}_s${REPLICA}" "${ARGS[@]}"
