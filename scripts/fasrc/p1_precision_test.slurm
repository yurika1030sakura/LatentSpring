#!/bin/bash
#SBATCH -J p1_prec
#SBATCH -p gpu_test
#SBATCH -A woo_lab
#SBATCH -N 1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --gres=gpu:1 --mem=64G
#SBATCH -t 11:00:00
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/p1_prec-%x-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/p1_prec-%x-%j.err
#
# P1 numerical validation of the log-density estimator.  NO TRAINING.
#   TASK A: exact divergence (3N backward passes) vs the production 2-probe
#           Hutchinson estimator, on the small parents of the reported population.
#   TASK B: repeat-scoring variance of log q on the frozen geometries of ALL 93
#           disjoint parents of the primary endpoint, at the production setting
#           n_ode_steps=12 / n_hutchinson=2.
# Usage: sbatch -J <tag> scripts/fasrc/p1_precision.slurm <arm_tag> <ckpt> <config>
set -eo pipefail
source /n/home04/yulili/bgfm/scripts/fasrc/env.sh
TAG=${1:?arm tag}; CKPT=${2:?ckpt}; CFG=${3:?config}
export LD_LIBRARY_PATH=$STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}
export PATH=$STORE/envs/flowmol/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
OUT="$STORE/runs/eval_ours/p1_precision"; mkdir -p "$OUT"
echo "host $(hostname)"; date; echo "tag=$TAG ckpt=$CKPT cfg=$CFG"
"$FLOWMOL_PY" scripts/p1_estimator_precision.py --task all \
  --checkpoint "$CKPT" --config "$CFG" \
  --eval_data "$STORE/processed_data/omol25_4m_processed" \
  --samples_json "$STORE/runs/eval_ours/${TAG}/boltz1/boltzmann_samples.json" \
  --records_csv "$STORE/runs/eval_ours/${TAG}/boltz_records.csv" \
  --disjoint_json "$REPO/paper/figures/out/primary_endpoint.json" \
  --n_ode_steps 12 --n_hutchinson 2 \
  --taskA_max_atoms 22 --taskA_n_parents 4 --taskA_n_geoms 0 --taskA_repeat 20 \
  --taskB_n_parents 93 --taskB_repeat 16 --verify_production \
  --out_json "$OUT/${TAG}.json"
echo "## P1 DONE $TAG ##"; date
