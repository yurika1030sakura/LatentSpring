#!/bin/bash
#SBATCH -J p6_estimator
#SBATCH -p gpu_requeue,gpu_h200,gpu
#SBATCH -A woo_lab
#SBATCH -N 1 --ntasks-per-node=1 --cpus-per-task=4
#SBATCH --gres=gpu:1 --mem=32G
#SBATCH -t 3:00:00
#SBATCH --requeue
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/p6_estimator-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/p6_estimator-%j.err
#
# P6: validate the FFJORD likelihood estimator (ODE steps x Hutchinson probes,
# repeatability, geometry-correlated noise/bias, exact-divergence check, and the
# per-group r as a function of estimator precision).
#
# Short (3h) + chained on purpose: the shared queue is saturated with the
# Priority-0 training jobs, and short jobs backfill into gaps far sooner than
# a 12h reservation would. Submit this script 3x with --dependency=afterany so
# the chain never runs two copies concurrently against the same out_dir.
#
# Resumable by construction: validate_likelihood_estimator.py writes one JSON per
# (steps, probes, repeat) cell and skips cells that already exist, so a
# gpu_requeue preemption just picks up where it left off.
#
# NOTE: absolute paths only -- SLURM copies the script to spool, so $0/dirname $0
# would point at the spool copy.
set -eo pipefail
source /n/home04/yulili/bgfm/scripts/fasrc/env.sh
export LD_LIBRARY_PATH=$STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}
export PATH=$STORE/envs/flowmol/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
date

DATA=$STORE/processed_data/omol25_4m_processed
OUTROOT=$STORE/runs/eval_ours/p6_estimator

# --- arm 1: energy arm (the arm that produces the headline r ~= 0.43) --------
CKPT_E=$STORE/runs/abl/a3_energy_only_s2/lightning_logs/version_30674363/checkpoints/midstep-step=30000.ckpt
$FLOWMOL_PY -u scripts/validate_likelihood_estimator.py \
  --checkpoint "$CKPT_E" --config configs/sweep/a3_energy_only_s2.yaml \
  --eval_data "$DATA" --out_dir "$OUTROOT/energy_s2" \
  --n_molecules 16 --n_perturb 8 --sigma 0.15 --min_atoms 8 --max_atoms 40 \
  --ode_steps 4,8,12,24,48 --probes 1,2,4,8 --crn_cells 12:2,48:8 --n_repeat 3 \
  --exact_max_atoms 15 --exact_n_mols 4 --exact_n_geom 3 \
  --exact_ode_steps 12 --exact_hutch_reps 8 \
  --device cuda --seed 0
echo "## ARM energy_s2 DONE ##"; date

# --- arm 2: FM-only control (same estimator, weaker signal) ------------------
# Reduced grid so it still finishes inside the wall clock; the point is only to
# check whether estimator NOISE properties are model-independent.
CKPT_F=$(ls -t $STORE/runs/abl/a1_fm_only_s2/lightning_logs/version_*/checkpoints/midstep-step=30000.ckpt 2>/dev/null | head -1)
if [ -n "$CKPT_F" ]; then
  $FLOWMOL_PY -u scripts/validate_likelihood_estimator.py \
    --checkpoint "$CKPT_F" --config configs/sweep/a1_fm_only_s2.yaml \
    --eval_data "$DATA" --out_dir "$OUTROOT/fmonly_s2" \
    --n_molecules 16 --n_perturb 8 --sigma 0.15 --min_atoms 8 --max_atoms 40 \
    --ode_steps 12,48 --probes 2,8 --crn_cells 12:2 --n_repeat 3 \
    --exact_n_mols 0 \
    --device cuda --seed 0
  echo "## ARM fmonly_s2 DONE ##"
fi

# --- analysis variant: drop the unperturbed reference conformer --------------
for arm in energy_s2 fmonly_s2; do
  [ -d "$OUTROOT/$arm" ] || continue
  $FLOWMOL_PY -u scripts/validate_likelihood_estimator.py \
    --analyze_only --drop_ref --out_dir "$OUTROOT/$arm" > "$OUTROOT/$arm/report_dropref.txt" 2>&1 || true
  cp "$OUTROOT/$arm/report.json" "$OUTROOT/$arm/report_withref.json" 2>/dev/null || true
  mv "$OUTROOT/$arm/report.json" "$OUTROOT/$arm/report_dropref.json" 2>/dev/null || true
  # restore the with-reference report as the canonical one
  $FLOWMOL_PY -u scripts/validate_likelihood_estimator.py \
    --analyze_only --out_dir "$OUTROOT/$arm" > /dev/null 2>&1 || true
done

echo "## P6 ESTIMATOR VALIDATION DONE ##"; date
