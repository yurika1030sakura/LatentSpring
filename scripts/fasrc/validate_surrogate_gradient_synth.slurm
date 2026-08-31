#!/bin/bash
#SBATCH -J p1_surrgrad_synth
#SBATCH -p sapphire,shared,serial_requeue
#SBATCH -A woo_lab
#SBATCH -N 1 --ntasks-per-node=1 --cpus-per-task=8 --mem=180G
#SBATCH -t 4:00:00
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/p1_surrgrad_synth-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/p1_surrgrad_synth-%j.err
#
# P1 synthetic arm, ONE (n_ode, hutchinson) CELL PER PROCESS.
# The single-process version (job 39956590) was OOM-killed at n_ode=12 with
# exact divergence: the cost probe (job 39959842) measured 46 GB peak RSS for
# that cell alone, and the caching allocator's high-water mark accumulated
# across cells in one process.  Splitting keeps every cell independent.
# No training; gradients only.
set -eo pipefail
source /n/home04/yulili/bgfm/scripts/fasrc/env.sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
OUT=$STORE/runs/p1_surrogate_gradient
mkdir -p "$OUT"
cd "$REPO"
date
for HU in 2 0; do
  for N in 1 2 4 12; do
    echo "########## synthetic cell n_ode=$N hutch=$HU ##########"
    CUDA_VISIBLE_DEVICES="" timeout 3000 "$FLOWMOL_PY" -u \
      scripts/validate_surrogate_gradient.py \
      --arm synthetic --n_reps 3 --M 3 --K 4 --n_atoms 6 \
      --n_ode_steps "$N" --n_hutchinson "$HU" --fd_all_reps \
      --out "$OUT/synthetic_n${N}_h${HU}.json" \
      || echo "BROKE synthetic n_ode=$N hutch=$HU"
  done
done
date
echo DONE
