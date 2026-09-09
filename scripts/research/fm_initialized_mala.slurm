#!/bin/bash
#SBATCH -J bgfm-fm-mala
#SBATCH -A ryl_lab
#SBATCH -p test
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?output required}"
BGFM_KT="${3:-1.0}"
BGFM_INITIAL_SAMPLES="${4:-/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/runs/work_fm_control_5846_v2/sampling/fm_midpoint_16_samples.pt}"
BGFM_REFERENCE="${5:-}"
BGFM_STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
FLOWMOL_ENV="$BGFM_STORE/envs/flowmol"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PATH="$FLOWMOL_ENV/bin:$PATH" LD_LIBRARY_PATH="$FLOWMOL_ENV/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/bgfm_mala_mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/bgfm_mala_cache_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$RUN_ROOT"
cd "$SOURCE_ROOT"
"$FLOWMOL_ENV/bin/python" -u scripts/research/fm_initialized_mala.py \
 --samples "$BGFM_INITIAL_SAMPLES" \
 --out "$RUN_ROOT/sampling" --oracle "$BGFM_STORE/checkpoints/omol25/esen_sm_conserving_all.pt" \
 --oracle-python "$BGFM_STORE/envs/omol25/bin/python" --steps 132 --proposal-std .1 --seed 9061 --oracle-batch-size 16 \
 --kT "$BGFM_KT" --scale-proposal-with-temperature
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ -n "$BGFM_REFERENCE" ]]; then
 "$FLOWMOL_ENV/bin/python" -u scripts/research/compare_work_reference.py --run "$RUN_ROOT/sampling" \
  --reference "$BGFM_REFERENCE" --out "$RUN_ROOT/reference_comparison.json" --unweighted
fi
exec "$FLOWMOL_ENV/bin/python" -u scripts/research/assess_work_panel.py \
 --samples initial_fm "$RUN_ROOT/sampling/initial_samples.pt" --samples mala "$RUN_ROOT/sampling/final_samples.pt" \
 --out "$RUN_ROOT/assessment" --xtb-count 32 --seed 9059 --workers 4 --max-cycles 200
