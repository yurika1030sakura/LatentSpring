#!/bin/bash
#SBATCH -J lspring-matched
#SBATCH --array=0-7%4
#SBATCH -A ryl_lab
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.err
set -euo pipefail
BASE_SOURCE="${1:?immutable base source}"
ENTRY_SOURCE="${2:?immutable current entrypoints and protocols}"
RUN_ROOT="${3:?run root}"
PROJECT="${4:?project data root}"
FLOWMOL_ENV=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol
export PYTHONPATH="$BASE_SOURCE" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$FLOWMOL_ENV/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/lspring_matched_mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/lspring_matched_cache_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
methods=(gaussian_fm harmonic_fm edm gaga)
seed=$((SLURM_ARRAY_TASK_ID / 4))
method="${methods[$((SLURM_ARRAY_TASK_ID % 4))]}"
cd "$BASE_SOURCE"
exec "$FLOWMOL_ENV/bin/python" -u "$ENTRY_SOURCE/scripts/research/run_matched_generators.py" \
 --project "$PROJECT" --protocol "$ENTRY_SOURCE/research/evidence/matched_generators_${method}_s${seed}_v1.json" \
 --out "$RUN_ROOT/s${seed}/${method}" --resume
