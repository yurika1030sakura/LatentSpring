#!/bin/bash
#SBATCH -J bgfm-bench
#SBATCH -A ryl_lab
#SBATCH -p sapphire,shared
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-23%2
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.err
set -euo pipefail
SOURCE_ROOT="${1:?committed source snapshot required; use scripts/research/submit.py}"
RUN_ROOT="${2:?unique campaign output required}"
STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export LD_LIBRARY_PATH="$STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}" XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/c_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$RUN_ROOT"
cd "$SOURCE_ROOT"
mapfile -t GRID < "$SOURCE_ROOT/configs/research/particle_diagnostic_grid_v1.txt"
LINE="${GRID[${SLURM_ARRAY_TASK_ID:?array task required}]}"
read -r SYS KIND SWEEPS ANNEAL SEED <<< "$LINE"
TAG="${SYS}_${KIND}_sw${SWEEPS}_an${ANNEAL}_s${SEED}"
ARGS=(--system "$SYS" --kind "$KIND" --sweeps "$SWEEPS" --seed "$SEED"
      --steps 4000 --batch 256 --eval-samples 8192
      --out "$RUN_ROOT/${TAG}.json")
if [ "$ANNEAL" != "none" ]; then ARGS+=(--anneal-from-tau "$ANNEAL" --anneal-fraction 0.5); fi
exec "$STORE/envs/flowmol/bin/python" -u scripts/research/train_particle_benchmark.py "${ARGS[@]}"
