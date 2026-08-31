#!/bin/bash
#SBATCH -J p1_surrgrad
#SBATCH -p gpu_requeue,gpu_h200,gpu
#SBATCH -A woo_lab
#SBATCH -N 1 --ntasks-per-node=1 --cpus-per-task=4
#SBATCH --gres=gpu:1 --mem=96G
#SBATCH -t 4:00:00
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/p1_surrgrad-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/p1_surrgrad-%j.err
#
# P1: full-gradient vs frozen-trajectory surrogate gradient.
# No training; gradients only. See scripts/validate_surrogate_gradient.py.
set -eo pipefail
source /n/home04/yulili/bgfm/scripts/fasrc/env.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
OUT=$STORE/runs/p1_surrogate_gradient
mkdir -p "$OUT"
cd "$REPO"
date

echo "########## SELF-CHECK ##########"
"$FLOWMOL_PY" -u scripts/validate_surrogate_gradient.py --selfcheck

echo "########## SYNTHETIC ARM (CPU, float64) ##########"
CUDA_VISIBLE_DEVICES="" "$FLOWMOL_PY" -u scripts/validate_surrogate_gradient.py \
  --arm synthetic --n_reps 3 --M 3 --K 4 --n_atoms 6 \
  --n_ode_steps 1 2 4 12 --n_hutchinson 0 2 \
  --out "$OUT/synthetic.json"

echo "########## REAL ARM (checkpoint p0_D_energy_s1, step 30000) ##########"
CKPT=$STORE/runs/p0/p0_D_energy_s1/lightning_logs/version_37027659/checkpoints/last.ckpt
if [ ! -f "$CKPT" ]; then
  CKPT=$(ls -t $STORE/runs/p0/p0_D_energy_s*/lightning_logs/*/checkpoints/last.ckpt | head -1)
fi
echo "using $CKPT"
"$FLOWMOL_PY" -u scripts/validate_surrogate_gradient.py \
  --arm real --checkpoint "$CKPT" --config configs/sweep/p0_D_energy_s1.yaml \
  --n_reps 3 --M 2 --max_atoms 12 \
  --n_ode_steps 1 2 4 --n_hutchinson 2 \
  --out "$OUT/real.json"

date
echo DONE
