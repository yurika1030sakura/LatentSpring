#!/bin/bash
#SBATCH -J omol25_fm_only
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --account=pi_bsilva_umass_edu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH -o /home/renhaozhang_umass_edu/scratch_workspace/bgfm/logs/train_fm_only-%j.out
#SBATCH -e /home/renhaozhang_umass_edu/scratch_workspace/bgfm/logs/train_fm_only-%j.err
# FM-only baseline: same FlowMol3 architecture, BGFM loss OFF. The clean ablation.
set -eo pipefail
REPO=/home/renhaozhang_umass_edu/bgfm
SCR=/home/renhaozhang_umass_edu/scratch_workspace/bgfm
PY=$SCR/envs/flowmol/bin/python
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=$SCR/envs/flowmol/lib:${LD_LIBRARY_PATH:-}
echo "host $(hostname); $(nvidia-smi -L)"; date
cd "$REPO"
$PY -u scripts/run_train.py --config "$SCR/omol25_50k_fm_only.yaml"
date
