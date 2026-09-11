#!/bin/bash
#SBATCH -J bgfm-anneal
#SBATCH -A ryl_lab
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
#
# Matched annealing ablation: identical condition, kind, replica, seeds and step
# budget; the only difference is whether the training path is annealed. The
# TARGET is unchanged -- final evaluation uses physical kT; history records training kT.
set -euo pipefail
SRC="${1:?committed source snapshot required}"
RUN_ROOT="${2:?unique campaign output required}"
LAB_RUNS=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/runs
STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SRC"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/cache_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$SRC"
COND="${3:?condition index}"; KIND="${4:-convex}"; STEPS="${5:-1000}"
SWEEPS="${6:-1}"; ANNEAL="${7:-}"
OUT="$RUN_ROOT"
ARGS=(--source-root "$LAB_RUNS/species_breadth_source_full_v2" --condition-index "$COND"
      --out "$OUT" --kind "$KIND" --replica 0 --parity-target
      --steps "$STEPS" --eval-count 512 --sweeps "$SWEEPS")
[ -n "$ANNEAL" ] && ARGS+=(--anneal-from-kT "$ANNEAL" --anneal-fraction 0.5)
echo "out=$OUT cond=$COND kind=$KIND steps=$STEPS sweeps=$SWEEPS anneal=${ANNEAL:-none}"
exec "$STORE/envs/flowmol/bin/python" -u scripts/research/train_broad_entropy_adapter.py "${ARGS[@]}"
