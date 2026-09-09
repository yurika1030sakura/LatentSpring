#!/bin/bash
#SBATCH -J bgfm-work-molecule
#SBATCH -A ryl_lab
#SBATCH -p test
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?output required}"
BGFM_STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
export PYTHONPATH="$SOURCE_ROOT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/bgfm_work_mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/bgfm_work_cache_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$SOURCE_ROOT"
"$BGFM_STORE/envs/flowmol/bin/python" -u scripts/research/molecular_work_pilot.py propose \
 --config configs/sweep/a1_fm_only_s2.yaml \
 --checkpoint /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/runs/displacement_development_rho01_v2/training/last.ckpt \
 --source-row 5846 --out "$RUN_ROOT/proposals.pt"
exec "$BGFM_STORE/envs/omol25/bin/python" -u scripts/research/molecular_work_pilot.py score \
 --proposals "$RUN_ROOT/proposals.pt" --out "$RUN_ROOT/scored.json" \
 --oracle "$BGFM_STORE/checkpoints/omol25/esen_sm_conserving_all.pt" --spin-policy minimum-parity
