#!/bin/bash
#SBATCH -J p1_surrgrad_real2
#SBATCH -p gpu_requeue,gpu_h200,gpu
#SBATCH -A woo_lab
#SBATCH -N 1 --ntasks-per-node=1 --cpus-per-task=4
#SBATCH --gres=gpu:1 --mem=96G
#SBATCH -t 2:00:00
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/p1_surrgrad_real2-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/p1_surrgrad_real2-%j.err
#
# P1 real arm, RE-RUN under the current scripts/validate_surrogate_gradient.py
# (the first real-arm run, job 39958627, predates the 01:04 edit of that file,
# so its numbers are not from the same code version as the b_parents ladder).
# Adds n_ode_steps=12, the resolution the headline endpoint is evaluated at.
# No training; gradients only.
set -eo pipefail
source /n/home04/yulili/bgfm/scripts/fasrc/env.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
OUT=$STORE/runs/p1_surrogate_gradient
mkdir -p "$OUT"
cd "$REPO"
date
for CK in p0_D_energy_s2 p0_A_fmonly_s1; do
  CKPT=$(ls -t $STORE/runs/p0/$CK/lightning_logs/*/checkpoints/last.ckpt 2>/dev/null | head -1)
  [ -z "$CKPT" ] && { echo "no checkpoint for $CK, skipping"; continue; }
  for N in 1 2 4 12; do
    echo "########## $CK :: n_ode=$N :: $CKPT ##########"
    "$FLOWMOL_PY" -u scripts/validate_surrogate_gradient.py \
      --arm real --checkpoint "$CKPT" --config configs/sweep/p0_D_energy_s1.yaml \
      --n_reps 3 --M 2 --max_atoms 12 \
      --n_ode_steps "$N" --n_hutchinson 2 --fd_all_reps \
      --out "$OUT/real_v2_${CK}_n${N}.json" \
      || echo "BROKE: $CK n_ode=$N"
  done
done
date
echo DONE
