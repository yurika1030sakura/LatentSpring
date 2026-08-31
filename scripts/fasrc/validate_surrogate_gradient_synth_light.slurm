#!/bin/bash
#SBATCH -J p1_sg_synth_light
#SBATCH -p shared,sapphire,serial_requeue
#SBATCH -A woo_lab
#SBATCH -N 1 --ntasks-per-node=1 --cpus-per-task=4 --mem=24G
#SBATCH -t 1:00:00
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/p1_sg_synth_light-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/p1_sg_synth_light-%j.err
set -eo pipefail
source /n/home04/yulili/bgfm/scripts/fasrc/env.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
OUT=$STORE/runs/p1_surrogate_gradient
mkdir -p "$OUT"
cd "$REPO"
date
for CELL in "1 2" "2 2" "4 2" "12 2" "1 0" "2 0"; do
  set -- $CELL; N=$1; HU=$2
  echo "########## synthetic cell n_ode=$N hutch=$HU ##########"
  CUDA_VISIBLE_DEVICES="" timeout 2400 "$FLOWMOL_PY" -u \
    scripts/validate_surrogate_gradient.py \
    --arm synthetic --n_reps 3 --M 3 --K 4 --n_atoms 6 \
    --n_ode_steps "$N" --n_hutchinson "$HU" --fd_all_reps \
    --out "$OUT/synthetic_n${N}_h${HU}.json" \
    || echo "BROKE synthetic n_ode=$N hutch=$HU"
done
date
echo DONE
