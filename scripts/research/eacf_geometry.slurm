#!/bin/bash
#SBATCH -J bgfm-eacf-xtb
#SBATCH -A ryl_lab
#SBATCH -p test
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?immutable source required}"
RUN_ROOT="${2:?unique output required}"
STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
LAB_RUNS=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/runs
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="$STORE/envs/flowmol/bin:$PATH" LD_LIBRARY_PATH="$STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/bgfm_mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/bgfm_cache_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$SOURCE_ROOT"
ARGS=()
for REPLICA in 0 1; do
  EXPORTED="$RUN_ROOT/export_s${REPLICA}"
  "$STORE/envs/flowmol/bin/python" -u scripts/research/export_eacf_geometry.py \
    --run "$LAB_RUNS/eacf_joint_condition_00_s${REPLICA}_v1" \
    --audit "$LAB_RUNS/eacf_joint_audit_s${REPLICA}_v1/audit.json" \
    --reference "$LAB_RUNS/parity_entropy_condition_00_v1/condition_00_convex_s${REPLICA}" \
    --out "$EXPORTED"
  ARGS+=(--samples "eacf_s${REPLICA}" "$EXPORTED/physical_samples.pt")
done
exec "$STORE/envs/flowmol/bin/python" -u scripts/research/assess_work_panel.py \
  "${ARGS[@]}" --out "$RUN_ROOT" --require-shared-parents \
  --xtb-count 32 --seed 9059 --workers 4 --max-cycles 200
