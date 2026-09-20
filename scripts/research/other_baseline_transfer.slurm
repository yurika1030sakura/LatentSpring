#!/bin/bash
#SBATCH -J lspring-other-transfer
#SBATCH -A ryl_lab
#SBATCH -p gpu_test
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --gres=gpu:3
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
export LS_OTHER_SOURCE="${1:?source}" LS_OTHER_PROJECT="${2:?project}" LS_OTHER_OUT="${3:?out}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol/lib:${LD_LIBRARY_PATH:-}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
srun --export=ALL --gpu-bind=single:1 --kill-on-bad-exit=0 \
 --output="$LS_OTHER_PROJECT/logs/lspring-other-worker-%j-%t.out" \
 --error="$LS_OTHER_PROJECT/logs/lspring-other-worker-%j-%t.err" bash -c '
 set -euo pipefail
 export PYTHONPATH="$LS_OTHER_SOURCE" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
 export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/lspring_other_${SLURM_JOB_ID}_${SLURM_PROCID}"
 mkdir -p "$XDG_CACHE_HOME"
 cd "$LS_OTHER_SOURCE"
 targets=(edm gaussian_fm harmonic_fm)
 target="${targets[$SLURM_PROCID]}"
 exec /n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol/bin/python -s -u -m scripts.research.run_other_baseline_transfer \
 --project "$LS_OTHER_PROJECT" --protocol "$LS_OTHER_SOURCE/research/evidence/other_baseline_transfer_v1.json" \
 --target "$target" --out "$LS_OTHER_OUT/$target"
 '
cd "$LS_OTHER_SOURCE"
export PYTHONPATH="$LS_OTHER_SOURCE"
/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol/bin/python -s -u -m scripts.research.audit_other_baseline_transfer \
 --project "$LS_OTHER_PROJECT" --protocol "$LS_OTHER_SOURCE/research/evidence/other_baseline_transfer_v1.json" \
 --run "$LS_OTHER_OUT" --out "$LS_OTHER_OUT/audit.json"
