#!/bin/bash
#SBATCH -J bgfm_4m_h200
#SBATCH -p gpu_requeue,gpu_h200
#SBATCH -A woo_lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=256G
#SBATCH -t 3-00:00:00
#SBATCH --requeue
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/bgfm_4m_h200-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/bgfm_4m_h200-%j.err
#SBATCH --signal=SIGUSR1@120
#
# ===========================================================================
# THE 4M (FULL) HEADLINE RUN  ->  fills the empty "Ours - HBFM (4M, full)" row
# in notes/benchmark_plan.md (Tables 2 & 3).
# ===========================================================================
# Plain (off-policy) BGFM: force loss on the PRECOMPUTED DFT forces/energies
# baked into the processed data. Needs NO live eSEN oracle -> runnable today.
# (For the on-policy HBFM variant, use train_bgfm_onpolicy_h200.slurm, which
# additionally needs the gated eSEN checkpoint.)
#
# Auto-resumes from the latest checkpoint under output_dir on SIGUSR1 requeue.
# Usage: sbatch scripts/fasrc/train_bgfm_4m_h200.slurm [config] [resume_ckpt]
set -eo pipefail
source /n/home04/yulili/bgfm/scripts/fasrc/env.sh

CONFIG="${1:-$REPO/configs/omol25_4m_bgfm.yaml}"
RESUME_CKPT="${2:-}"
export LD_LIBRARY_PATH=$STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}

echo "host: $(hostname)"; nvidia-smi -L; echo "config: $CONFIG"; date
cd "$REPO"

RESUME_ARG=""
if [ -n "$RESUME_CKPT" ] && [ -f "$RESUME_CKPT" ]; then
    RESUME_ARG="--resume_ckpt_path $RESUME_CKPT"
fi

"$FLOWMOL_PY" -u scripts/run_train.py --config "$CONFIG" $RESUME_ARG
date
