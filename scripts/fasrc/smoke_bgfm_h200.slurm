#!/bin/bash
#SBATCH -J bgfm_smoke_h200
#SBATCH -p gpu_h200
#SBATCH -A woo_lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/bgfm_smoke_h200-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/bgfm_smoke_h200-%j.err
#
# 1-hour pipeline validation BEFORE committing the multi-day 4M run.
# `--fast` shrinks the run; the point is to confirm on the H200 that:
#   (a) the flowmol env imports torch/DGL/CUDA on this node,
#   (b) the processed data loads and forces are wired,
#   (c) the log shows train_L_force / train_score_force_cos (physics ACTIVE) and
#       NOT train_bgfm_no_force_skip (which means the FlowMol3 patch didn't take).
# Usage: sbatch scripts/fasrc/smoke_bgfm_h200.slurm
set -eo pipefail
source /n/home04/yulili/bgfm/scripts/fasrc/env.sh
export LD_LIBRARY_PATH=$STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}
echo "host: $(hostname)"; nvidia-smi -L; date
cd "$REPO"
"$FLOWMOL_PY" -u scripts/run_train.py --config "$REPO/configs/omol25_4m_bgfm.yaml" --fast
echo "== smoke done; grep the log for train_L_force (should be present) =="
date
