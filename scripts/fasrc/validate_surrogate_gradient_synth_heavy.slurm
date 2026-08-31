#!/bin/bash
#SBATCH -J p1_sg_synth_heavy
#SBATCH -p sapphire,shared,serial_requeue
#SBATCH -A woo_lab
#SBATCH -N 1 --ntasks-per-node=1 --cpus-per-task=4 --mem=150G
#SBATCH -t 2:00:00
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/p1_sg_synth_heavy-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/p1_sg_synth_heavy-%j.err
#
# The two exact-divergence cells the single-process run (job 39956590) was
# OOM-killed on.  One cell per process; the cost probe measured 46 GB peak RSS
# for n_ode=12 with exact divergence at 72 atoms.
set -eo pipefail
source /n/home04/yulili/bgfm/scripts/fasrc/env.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
OUT=$STORE/runs/p1_surrogate_gradient
mkdir -p "$OUT"
cd "$REPO"
date
for N in 4 12; do
  echo "########## synthetic cell n_ode=$N hutch=EXACT ##########"
  CUDA_VISIBLE_DEVICES="" timeout 5400 "$FLOWMOL_PY" -u \
    scripts/validate_surrogate_gradient.py \
    --arm synthetic --n_reps 3 --M 3 --K 4 --n_atoms 6 \
    --n_ode_steps "$N" --n_hutchinson 0 --fd_all_reps \
    --out "$OUT/synthetic_n${N}_h0.json" \
    || echo "BROKE synthetic n_ode=$N hutch=EXACT"
done
date
echo DONE
