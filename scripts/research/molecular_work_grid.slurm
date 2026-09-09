#!/bin/bash
#SBATCH -J bgfm-work-grid
#SBATCH -A ryl_lab
#SBATCH -p test
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?output required}"
BGFM_STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
export PYTHONPATH="$SOURCE_ROOT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/bgfm_work_grid_mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/bgfm_work_grid_cache_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$SOURCE_ROOT"
for WORK_SPEC in '32 0.1' '128 0.1' '512 0.1' '128 0.03' '128 0.3' '128 1.0'; do
 read -r WORK_STEPS WORK_NOISE <<< "$WORK_SPEC"
 CASE_ROOT="$RUN_ROOT/steps_${WORK_STEPS}_noise_${WORK_NOISE}"
 mkdir -p "$CASE_ROOT"
 if ! "$BGFM_STORE/envs/flowmol/bin/python" -u scripts/research/molecular_work_pilot.py propose \
  --config configs/sweep/a1_fm_only_s2.yaml \
  --checkpoint /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/runs/displacement_development_rho01_v2/training/last.ckpt \
  --source-row 5846 --seed 9021 --steps "$WORK_STEPS" --noise-scale "$WORK_NOISE" --out "$CASE_ROOT/proposals.pt"; then
   printf '%s\n' '{"failed":true,"stage":"propose","details":"see retained SLURM stderr"}' > "$CASE_ROOT/failure.json"
   continue
 fi
 if ! "$BGFM_STORE/envs/omol25/bin/python" -u scripts/research/molecular_work_pilot.py score \
  --proposals "$CASE_ROOT/proposals.pt" --out "$CASE_ROOT/scored.json" \
  --oracle "$BGFM_STORE/checkpoints/omol25/esen_sm_conserving_all.pt" --spin-policy minimum-parity; then
   printf '%s\n' '{"failed":true,"stage":"score","details":"see retained SLURM stderr"}' > "$CASE_ROOT/failure.json"
 fi
done
