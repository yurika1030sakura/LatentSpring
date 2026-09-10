#!/bin/bash
#SBATCH -J bgfm-students-audit
#SBATCH -A ryl_lab
#SBATCH -p test
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?output required}"
FLOWMOL_ENV=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$SOURCE_ROOT"
export PATH="$FLOWMOL_ENV/bin:$PATH" LD_LIBRARY_PATH="$FLOWMOL_ENV/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/bgfm_work_eval_mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/bgfm_work_eval_cache_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$RUN_ROOT"
cd "$SOURCE_ROOT"
LAB_RUNS=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/runs
BGFM_SAMPLE_ARGS=()
for ARM in source uniform linear power; do
 VERSION=v2
 if [[ "$ARM" == source || "$ARM" == uniform ]]; then VERSION=v1; fi
 BGFM_SAMPLE_ARGS+=(--samples "${ARM}_ode" "$LAB_RUNS/forward_work_student_${ARM}_${VERSION}/ode_samples.pt")
 BGFM_SAMPLE_ARGS+=(--samples "${ARM}_stochastic" "$LAB_RUNS/forward_work_student_${ARM}_${VERSION}/stochastic_samples.pt")
done
"$FLOWMOL_ENV/bin/python" -u scripts/research/assess_work_panel.py \
 "${BGFM_SAMPLE_ARGS[@]}" --out "$RUN_ROOT" --xtb-count 32 --seed 9059 --workers 4 --max-cycles 200
"$FLOWMOL_ENV/bin/python" -u scripts/research/summarize_work_students.py \
 --runs-root "$LAB_RUNS" --assessment "$RUN_ROOT" --out "$RUN_ROOT/comparison.json"
