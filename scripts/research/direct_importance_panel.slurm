#!/bin/bash
#SBATCH -J bgfm-direct-importance
#SBATCH -A ryl_lab
#SBATCH -p test
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?output required}"
SOURCE_ROW="${3:-1137}"
PARTICLE_COUNT="${4:-1024}"
BGFM_STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
export PYTHONPATH="$SOURCE_ROOT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/bgfm_direct_mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/bgfm_direct_cache_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$SOURCE_ROOT"
exec "$BGFM_STORE/envs/flowmol/bin/python" -u scripts/research/direct_importance_panel.py \
 --pilot "/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/runs/molecular_symmetry_${SOURCE_ROW}_v1" \
 --source-row "$SOURCE_ROW" --oracle "$BGFM_STORE/checkpoints/omol25/esen_sm_conserving_all.pt" \
 --oracle-python "$BGFM_STORE/envs/omol25/bin/python" --out "$RUN_ROOT" --particles "$PARTICLE_COUNT"
