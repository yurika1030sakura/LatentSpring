#!/bin/bash
#SBATCH -J bgfm-long-mcmc
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
BGFM_KERNEL="${3:?kernel required}"
BGFM_STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
LAB_RUNS=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/runs
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PATH="$BGFM_STORE/envs/flowmol/bin:$PATH" LD_LIBRARY_PATH="$BGFM_STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/bgfm_lmc_mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/bgfm_lmc_cache_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$SOURCE_ROOT"
"$BGFM_STORE/envs/flowmol/bin/python" -u scripts/research/long_chain_mcmc.py --kernel "$BGFM_KERNEL" \
 --samples "$LAB_RUNS/work_fm_control_1137_v1/sampling/fm_midpoint_16_samples.pt" \
 --out "$RUN_ROOT/sampling" --oracle "$BGFM_STORE/checkpoints/omol25/esen_sm_conserving_all.pt" \
 --oracle-python "$BGFM_STORE/envs/omol25/bin/python"
"$BGFM_STORE/envs/flowmol/bin/python" -u scripts/research/compare_work_reference.py --run "$RUN_ROOT/sampling" \
 --reference "$LAB_RUNS/refined_triatomic_reference_1137_v2" --out "$RUN_ROOT/reference_comparison.json" --unweighted --stages final
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$BGFM_STORE/envs/flowmol/bin/python" -u scripts/research/assess_work_panel.py \
 --samples "$BGFM_KERNEL" "$RUN_ROOT/sampling/final_samples.pt" --out "$RUN_ROOT/xtb" --xtb-count 32 --seed 9059 --workers 4
