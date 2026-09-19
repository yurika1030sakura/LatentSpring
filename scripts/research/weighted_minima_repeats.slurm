#!/bin/bash
#SBATCH -J lspring-minima-repeats
#SBATCH -A ryl_lab
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE="${1:?source}"
PROJECT="${2:?project}"
OUT="${3:?out}"
FLOW_ENV=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol
export PYTHONPATH="$SOURCE" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$FLOW_ENV/lib:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/lspring_minima_repeats_${SLURM_JOB_ID}"
export MPLCONFIGDIR="$XDG_CACHE_HOME/mpl"
mkdir -p "$MPLCONFIGDIR"
cd "$SOURCE"
for seed in 260939 260940; do
 "$FLOW_ENV/bin/python" -u -m scripts.research.run_weighted_minima_native \
  --project "$PROJECT" --protocol "$SOURCE/research/evidence/weighted_minima_native_s${seed}_v1.json" --out "$OUT/s${seed}"
 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 "$FLOW_ENV/bin/python" -u -m scripts.research.score_native_minima \
  --project "$PROJECT" --protocol "$SOURCE/research/evidence/weighted_minima_native_xtb_s${seed}_v1.json" \
  --run "$OUT/s${seed}" --out "$OUT/s${seed}/xtb"
done
