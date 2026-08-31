#!/bin/bash
#SBATCH -J p1_precB
#SBATCH -p gpu_requeue,gpu_h200
#SBATCH -A woo_lab
#SBATCH -N 1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h200:1 --mem=96G
#SBATCH -t 08:00:00
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/p1_precB-%x-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/p1_precB-%x-%j.err
#
# P1 TASK B only: repeat-scoring variance of log q on the frozen geometries of
# all 93 disjoint parents of the primary endpoint, production estimator setting.
# Needs a full H200 (the 20 GiB gpu_test MIG slice OOMs above ~130 atoms).
# Usage: sbatch -J <tag> scripts/fasrc/p1_precision_B.slurm <arm_tag> <ckpt> <config>
set -eo pipefail
source /n/home04/yulili/bgfm/scripts/fasrc/env.sh
TAG=${1:?arm tag}; CKPT=${2:?ckpt}; CFG=${3:?config}
export LD_LIBRARY_PATH=$STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}
export PATH=$STORE/envs/flowmol/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
OUT="$STORE/runs/eval_ours/p1_precision"; mkdir -p "$OUT"
echo "host $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv; date
"$FLOWMOL_PY" scripts/p1_estimator_precision.py --task B \
  --checkpoint "$CKPT" --config "$CFG" \
  --eval_data "$STORE/processed_data/omol25_4m_processed" \
  --samples_json "$STORE/runs/eval_ours/${TAG}/boltz1/boltzmann_samples.json" \
  --records_csv "$STORE/runs/eval_ours/${TAG}/boltz_records.csv" \
  --disjoint_json "$REPO/paper/figures/out/primary_endpoint.json" \
  --n_ode_steps 12 --n_hutchinson 2 \
  --taskB_n_parents 93 --taskB_repeat 16 --verify_production \
  --out_json "$OUT/${TAG}_taskB.json"
echo "## P1 TASK B DONE $TAG ##"; date
