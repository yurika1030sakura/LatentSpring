#!/bin/bash
#SBATCH -J p1_surrgrad_tm
#SBATCH -p gpu_requeue,gpu_h200,gpu
#SBATCH -A woo_lab
#SBATCH -N 1 --ntasks-per-node=1 --cpus-per-task=4
#SBATCH --gres=gpu:1 --mem=96G
#SBATCH -t 4:00:00
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/p1_surrgrad_tm-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/p1_surrgrad_tm-%j.err
#
# P1, real arm at the TRAINING-MATCHED energy-term settings of
# configs/sweep/p0_D_energy_s*.yaml:
#   energy_n_ode_steps=4, energy_n_hutchinson=2, energy_b_parents=8,
#   energy_max_atoms_per_parent=50, kT=1.0.
# Ladder over b_parents to bracket where the fully differentiable reference
# runs out of GPU memory (the shipped surrogate trains fine at b_parents=8).
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
  for MP in 1 2 4 8; do
    echo "########## $CK :: b_parents=$MP :: $CKPT ##########"
    "$FLOWMOL_PY" -u scripts/validate_surrogate_gradient.py \
      --arm real --checkpoint "$CKPT" --config configs/sweep/p0_D_energy_s1.yaml \
      --n_reps 3 --M "$MP" --max_atoms 50 \
      --n_ode_steps 4 --n_hutchinson 2 \
      --out "$OUT/real_trainmatched_${CK}_M${MP}.json" \
      || echo "BROKE: $CK b_parents=$MP (see stderr for the exception)"
  done
done

date
echo DONE
