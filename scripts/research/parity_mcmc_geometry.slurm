#!/bin/bash
#SBATCH -J bgfm-mcmc-xtb
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
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mcmc_xtb_mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/mcmc_xtb_cache_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$SOURCE_ROOT"
ARGS=()
for KERNEL in mala hmc; do
  for REPLICA in 0 1; do
    RUN="$LAB_RUNS/parity_${KERNEL}_condition_00_s${REPLICA}_v1"
    "$STORE/envs/flowmol/bin/python" - "$RUN/results.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
if not r['complete'] or not r['proposal_replay_passed'] or r['engineering_only']:
    raise ValueError('Require completed, replayed production MCMC')
PY
    ARGS+=(--samples "${KERNEL}_s${REPLICA}" "$RUN/final_samples.pt")
  done
done
exec "$STORE/envs/flowmol/bin/python" -u scripts/research/assess_work_panel.py \
  "${ARGS[@]}" --out "$RUN_ROOT" --require-shared-parents \
  --xtb-count 32 --seed 9059 --workers 4 --max-cycles 200
