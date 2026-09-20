#!/bin/bash
#SBATCH -J lspring-cross-head
#SBATCH -A ryl_lab
#SBATCH -p gpu_test
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
export LS_CROSS_SOURCE="${1:?source}" LS_CROSS_PROJECT="${2:?project}" LS_CROSS_OUT="${3:?out}" LS_CROSS_OFFSET="${4:-0}"
srun --export=ALL --gpu-bind=single:1 --kill-on-bad-exit=0 \
 --output="$LS_CROSS_PROJECT/logs/lspring-cross-worker-%j-%t.out" \
 --error="$LS_CROSS_PROJECT/logs/lspring-cross-worker-%j-%t.err" bash -c '
 set -euo pipefail
 export PYTHONPATH="$LS_CROSS_SOURCE" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
 export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
 export LD_LIBRARY_PATH=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol/lib:${LD_LIBRARY_PATH:-}
 export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/lspring_cross_${SLURM_JOB_ID}_${SLURM_PROCID}"
 mkdir -p "$XDG_CACHE_HOME"
 cd "$LS_CROSS_SOURCE"
 fit=$((SLURM_PROCID + LS_CROSS_OFFSET))
 exec /n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol/bin/python -s -u -m scripts.research.run_cross_generator_head \
 --project "$LS_CROSS_PROJECT" --protocol "$LS_CROSS_SOURCE/research/evidence/cross_generator_head_v1.json" \
 --fit "$fit" --out "$LS_CROSS_OUT/s${fit}" --resume
 '
