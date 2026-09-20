#!/bin/bash
#SBATCH -J lspring-joint-transfer
#SBATCH -A ryl_lab
#SBATCH -p gpu_test
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
export LS_JOINT_SOURCE="${1:?source}" LS_JOINT_PROJECT="${2:?project}" LS_JOINT_OUT="${3:?out}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol/lib:${LD_LIBRARY_PATH:-}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
srun --export=ALL --gpu-bind=single:1 --kill-on-bad-exit=0 \
 --output="$LS_JOINT_PROJECT/logs/lspring-joint-worker-%j-%t.out" \
 --error="$LS_JOINT_PROJECT/logs/lspring-joint-worker-%j-%t.err" bash -c '
 set -euo pipefail
 export PYTHONPATH="$LS_JOINT_SOURCE" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
 export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/lspring_joint_${SLURM_JOB_ID}_${SLURM_PROCID}"
 mkdir -p "$XDG_CACHE_HOME"
 cd "$LS_JOINT_SOURCE"
 targets=(edm gaga);target="${targets[$((SLURM_PROCID / 2))]}";fit=$((SLURM_PROCID % 2))
 exec /n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol/bin/python -s -u -m scripts.research.run_joint_design_transfer \
 --project "$LS_JOINT_PROJECT" --protocol "$LS_JOINT_SOURCE/research/evidence/joint_design_transfer_v1.json" \
 --target "$target" --fit "$fit" --out "$LS_JOINT_OUT/$target/s$fit"
 '
cd "$LS_JOINT_SOURCE"
export PYTHONPATH="$LS_JOINT_SOURCE"
/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol/bin/python -s -u -m scripts.research.audit_joint_design_transfer \
 --project "$LS_JOINT_PROJECT" --protocol "$LS_JOINT_SOURCE/research/evidence/joint_design_transfer_v1.json" \
 --run "$LS_JOINT_OUT" --out "$LS_JOINT_OUT/audit.json"
