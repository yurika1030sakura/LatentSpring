#!/bin/bash
#SBATCH -J omol25_edm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2080_ti:1
#SBATCH --account=pi_bsilva_umass_edu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH -o /home/renhaozhang_umass_edu/scratch_workspace/bgfm/logs/train_edm-%j.out
#SBATCH -e /home/renhaozhang_umass_edu/scratch_workspace/bgfm/logs/train_edm-%j.err
# EDM baseline (EGNN diffusion), bond-free, OMol25 50k. ~5.2 GB @ batch 4 -> 2080ti.
set -eo pipefail
REPO=/home/renhaozhang_umass_edu/bgfm
SCR=/home/renhaozhang_umass_edu/scratch_workspace/bgfm
PY=$SCR/envs/edm/bin/python
export PYTHONUNBUFFERED=1
echo "host $(hostname); $(nvidia-smi -L)"; date
cd "$REPO/baselines/edm"
$PY -u main_qm9.py --dataset omol25 --datadir "$SCR/processed_data/omol25_50k" --no_wandb \
    --n_epochs 30 --batch_size 4 --num_workers 8 --nf 256 --n_layers 9 \
    --diffusion_steps 1000 --diffusion_loss_type l2 --lr 1e-4 --ema_decay 0.9999 \
    --n_stability_samples 0 --test_epochs 5 --exp_name omol25_50k_edm
date
