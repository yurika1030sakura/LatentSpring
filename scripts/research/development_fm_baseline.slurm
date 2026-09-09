#!/bin/bash
#SBATCH -J bgfm-development-fm
#SBATCH -A ryl_lab
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?output required}"
BGFM_STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
LAB_RUNS=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/runs
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$BGFM_STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/bgfm_development_mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/bgfm_development_cache_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$SOURCE_ROOT"
"$BGFM_STORE/envs/omol25/bin/python" -u scripts/research/export_development_references.py \
 --manifest research/evidence/development_panel_v1.json \
 --raw "$BGFM_STORE/raw_data/omol25/v250514/official_validation/val_extracted/val" \
 --out "$RUN_ROOT/references.json"
exec "$BGFM_STORE/envs/flowmol/bin/python" -u scripts/research/sample_development_conditions.py \
 --config configs/sweep/a1_fm_only_s2.yaml --checkpoint "$LAB_RUNS/electronic_fm_global_v1/last.ckpt" \
 --manifest research/evidence/development_panel_v1.json --out "$RUN_ROOT/sampling" --samples 32 --batch 16 --seed 9063
