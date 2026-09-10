#!/bin/bash
#SBATCH -J bgfm-parity-array
#SBATCH --array=1-7%2
#SBATCH -A ryl_lab
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
PANEL_ROOT="${2:?output required}"
printf -v CONDITION_LABEL "%02d" "$SLURM_ARRAY_TASK_ID"
RUN_ROOT="$PANEL_ROOT/condition_$CONDITION_LABEL"
BGFM_STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
LAB_RUNS=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/runs
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$BGFM_STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/bgfm_species_mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/bgfm_species_cache_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$SOURCE_ROOT"
CONDITION_INDEX="$SLURM_ARRAY_TASK_ID"
exec "$BGFM_STORE/envs/flowmol/bin/python" -u scripts/research/run_broad_adapter_panel.py \
 --parity-target --source-root "$LAB_RUNS/species_breadth_source_full_v2" --out "$RUN_ROOT" --condition-index "$CONDITION_INDEX"
