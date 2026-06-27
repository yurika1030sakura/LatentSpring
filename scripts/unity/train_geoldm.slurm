#!/bin/bash
#SBATCH -J omol25_geoldm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l4:1
#SBATCH --account=pi_bsilva_umass_edu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH -o /home/renhaozhang_umass_edu/scratch_workspace/bgfm/logs/train_geoldm-%j.out
#SBATCH -e /home/renhaozhang_umass_edu/scratch_workspace/bgfm/logs/train_geoldm-%j.err
# GeoLDM baseline (latent diffusion), bond-free, OMol25 50k. L4 (24 GB), batch 2; ~9 GB.
# Stage 1 = autoencoder (this job). Stage 2 (latent diffusion) = re-run with
# --train_diffusion --ae_path outputs/omol25_50k_geoldm_ae after this finishes.
set -eo pipefail
REPO=/home/renhaozhang_umass_edu/bgfm
SCR=/home/renhaozhang_umass_edu/scratch_workspace/bgfm
PY=$SCR/envs/edm/bin/python
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation on 2080ti
echo "host $(hostname); $(nvidia-smi -L)"; date
cd "$REPO/baselines/geoldm"
# batch 2: 200-atom fully-connected graphs make batch 4 exceed the 2080ti's ~10.5 GB
$PY -u main_qm9.py --dataset omol25 --datadir "$SCR/processed_data/omol25_50k" --no_wandb \
    --n_epochs 30 --batch_size 2 --num_workers 8 --nf 256 --n_layers 9 --latent_nf 4 \
    --diffusion_steps 1000 --diffusion_loss_type l2 --lr 1e-4 --ema_decay 0.9999 \
    --n_stability_samples 0 --test_epochs 5 --exp_name omol25_50k_geoldm_ae
date
