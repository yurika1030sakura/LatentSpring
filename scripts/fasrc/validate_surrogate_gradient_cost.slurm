#!/bin/bash
#SBATCH -J p1_surrgrad_cost
#SBATCH -p serial_requeue,shared,sapphire
#SBATCH -A woo_lab
#SBATCH -N 1 --ntasks-per-node=1 --cpus-per-task=4 --mem=180G
#SBATCH -t 3:00:00
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/p1_surrgrad_cost-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/p1_surrgrad_cost-%j.err
#
# P1 cost probe: wall time and peak RSS of the fully differentiable reference
# vs the shipped surrogate, ONE VARIANT PER PROCESS (ru_maxrss is a process
# high-water mark).  Locates where the fully differentiable version breaks.
set -eo pipefail
source /n/home04/yulili/bgfm/scripts/fasrc/env.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$REPO"
date
for NAT in 4 6; do
  for HU in 2 0; do
    for N in 1 2 4 12; do
      for V in surrogate full; do
        timeout 1800 "$FLOWMOL_PY" -u scripts/validate_surrogate_gradient.py \
          --memprobe --probe_variant "$V" --M 3 --K 4 --n_atoms "$NAT" \
          --n_ode_steps "$N" --n_hutchinson "$HU" 2>/dev/null \
          | grep MEMPROBE \
          || echo "BROKE variant=$V n_ode=$N hutch=$HU n_atoms_per_mol=$NAT (killed / timed out)"
      done
    done
  done
done
date
echo DONE
