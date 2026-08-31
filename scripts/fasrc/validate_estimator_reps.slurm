#!/bin/bash
#SBATCH -J p6_estimator_reps
#SBATCH -p serial_requeue,shared
#SBATCH -A woo_lab
#SBATCH -N 1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -t 3:00:00
#SBATCH --requeue
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/p6_estimator_reps-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/p6_estimator_reps-%j.err
#
# P6 CPU insurance run. The GPU queue is saturated by the Priority-0 training
# jobs (fairshare 0), so this runs the SAME validation on serial_requeue with a
# reduced grid and smaller molecules. Identical script, identical frozen
# geometries protocol -- only the grid resolution differs.
#
# Chain this 4x with --dependency=afterany; cells are written one JSON at a time
# and skipped on re-run, so the chain accumulates coverage.
#
# Absolute paths only (SLURM copies the script to spool).
set -eo pipefail
source /n/home04/yulili/bgfm/scripts/fasrc/env.sh
export LD_LIBRARY_PATH=$STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}
export PATH=$STORE/envs/flowmol/bin:$PATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd "$REPO"
date

DATA=$STORE/processed_data/omol25_4m_processed
OUT=$STORE/runs/eval_ours/p6_estimator/energy_s2_cpu
CKPT=$STORE/runs/abl/a3_energy_only_s2/lightning_logs/version_30674363/checkpoints/midstep-step=30000.ckpt

$FLOWMOL_PY -u scripts/validate_likelihood_estimator.py \
  --checkpoint "$CKPT" --config configs/sweep/a3_energy_only_s2.yaml \
  --eval_data "$DATA" --out_dir "$OUT" \
  --n_molecules 10 --n_perturb 8 --sigma 0.15 --min_atoms 8 --max_atoms 22 \
  --ode_steps 12,48 --probes 2,8 --crn_cells 12:2,48:8 --n_repeat 8 \
  --exact_n_mols 0 \
  --device cpu --seed 0

echo "## P6 CPU ESTIMATOR VALIDATION DONE ##"; date
