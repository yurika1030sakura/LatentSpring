#!/bin/bash
#SBATCH -J force_diag_b_value
#SBATCH -p gpu_requeue,gpu_h200
#SBATCH -A woo_lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=128G
#SBATCH -t 4:00:00
#SBATCH --requeue
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/force_diag_bval-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/force_diag_bval-%j.err
#
# ===========================================================================
# EXPERIMENT B -- gradient cosine. DIAGNOSTIC ONLY: no training, no manuscript.
# ===========================================================================
# No teacher worker needed: the force arm uses the dataset's DFT label force at
# x_1, and the value arm uses the precomputed perturbation ENERGIES.
set -eo pipefail
source /n/home04/yulili/bgfm/scripts/fasrc/env.sh

OUT=$STORE/runs/force_diag
mkdir -p "$OUT"
CKPT="${1:-$STORE/runs/p0/p0_A_fmonly_s1/lightning_logs/version_37027619/checkpoints/midstep-step=30000.ckpt}"
CONFIG="${2:-$REPO/configs/sweep/p0_A_fmonly_s1.yaml}"
SHARD="$STORE/processed_data/omol25_4m_processed/perturbation_val_n10000_s0.pt"
TAG="p0_A_fmonly_s1_step30000"

echo "host: $(hostname)"; nvidia-smi -L; date
echo "ckpt:   $CKPT"; echo "config: $CONFIG"; echo "shard:  $SHARD"
[ -f "$CKPT" ]  || { echo "ERROR: checkpoint not found"; exit 1; }
[ -f "$SHARD" ] || { echo "ERROR: perturbation shard not found"; exit 1; }

export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=$STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}
cd "$REPO"

"$FLOWMOL_PY" -u scripts/force_diag/exp_b_grad_cosine.py \
  --checkpoint "$CKPT" --config "$CONFIG" \
  --out "$OUT/expB_valuearm_${TAG}.json" \
  --t_values 0.80 0.85 0.90 0.92 0.95 0.97 \
  --n_minibatches 6 --batch_size 4 --max_atoms 40 --kT 1.0 --seed 0 \
  --shards "$SHARD" \
  --value_minibatches 12 --b_parents 4 --value_max_atoms 30 \
  --n_ode_steps 4 --n_hutchinson 2 \
  || echo "BROKE: expB"

date; echo DONE
