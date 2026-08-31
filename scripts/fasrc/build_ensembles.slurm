#!/bin/bash
#SBATCH -J build_ens
#SBATCH -p serial_requeue,sapphire
#SBATCH -N 1 --ntasks-per-node=1 --cpus-per-task=16 --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/build_ens-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/build_ens-%j.err
#
# P1 global-ensemble eval, step 1: build GFN2-xTB reference conformer ensembles.
# NOTE: absolute paths only ($0 is unusable -- SLURM copies the script to spool).
set -eo pipefail
source /n/home04/yulili/bgfm/scripts/fasrc/env.sh
export PATH=$STORE/envs/flowmol/bin:$PATH
export LD_LIBRARY_PATH=$STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=$STORE/processed_data/ensembles
mkdir -p "$OUT"
cd "$REPO"
date
"$FLOWMOL_PY" -u scripts/build_reference_ensembles.py \
  --out_dir "$OUT" \
  --min_atoms 10 --max_atoms 30 --min_rot 2 \
  --n_candidates 400 --n_systems 60 --n_confs 80 \
  --min_basins 3 --max_basins 8 --rms_thresh 0.35 --ediff_thresh_eV 0.02 \
  --n_workers 16 --seed 0
echo "## BUILD ENSEMBLES DONE ##"; date
