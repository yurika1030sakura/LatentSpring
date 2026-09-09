#!/bin/bash
#SBATCH -J bgfm-work-fm-control
#SBATCH -A ryl_lab
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?output required}"
WORK_RUN="${3:?completed work calibration required}"
FLOWMOL_ENV=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="$FLOWMOL_ENV/bin:$PATH" LD_LIBRARY_PATH="$FLOWMOL_ENV/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/bgfm_work_fm_mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/bgfm_work_fm_cache_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$RUN_ROOT"
cd "$SOURCE_ROOT"
"$FLOWMOL_ENV/bin/python" -u scripts/research/sample_work_fm_control.py --work-run "$WORK_RUN" --out "$RUN_ROOT/sampling"
exec "$FLOWMOL_ENV/bin/python" -u scripts/research/assess_work_panel.py \
 --samples fm16 "$RUN_ROOT/sampling/fm_midpoint_16_samples.pt" \
 --samples fm64 "$RUN_ROOT/sampling/fm_midpoint_64_samples.pt" \
 --out "$RUN_ROOT/assessment" --xtb-count 32 --seed 9059 --workers 4 --max-cycles 200
