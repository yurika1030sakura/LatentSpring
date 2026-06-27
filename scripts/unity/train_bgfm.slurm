#!/bin/bash
#SBATCH -J omol25_bgfm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2080_ti:1
#SBATCH --account=pi_bsilva_umass_edu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH -o /home/renhaozhang_umass_edu/scratch_workspace/bgfm/logs/train_bgfm-%j.out
#SBATCH -e /home/renhaozhang_umass_edu/scratch_workspace/bgfm/logs/train_bgfm-%j.err
# BGFM (ours): FlowMol3 + physics loss, bond-free, OMol25 50k. ~9.3 GB @ batch 4 -> 2080ti.
set -eo pipefail
REPO=/home/renhaozhang_umass_edu/bgfm
SCR=/home/renhaozhang_umass_edu/scratch_workspace/bgfm
PY=$SCR/envs/flowmol/bin/python
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=$SCR/envs/flowmol/lib:${LD_LIBRARY_PATH:-}
echo "host $(hostname); $(nvidia-smi -L)"; date
cd "$REPO"
$PY -u scripts/run_train.py --config "$SCR/omol25_50k_bgfm.yaml"
date
