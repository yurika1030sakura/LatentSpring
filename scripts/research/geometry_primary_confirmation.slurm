#!/bin/bash
#SBATCH -J ls-geometry-primary
#SBATCH -A ryl_lab
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE="${1:?source}"; PROJECT="${2:?project}"; RUN="${3:?run}"
FLOW_ENV=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol
export PYTHONPATH="$SOURCE" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$FLOW_ENV/lib:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/ls_geometry_primary_${SLURM_JOB_ID}"
export MPLCONFIGDIR="$XDG_CACHE_HOME/mpl" TORCH_HOME="$XDG_CACHE_HOME/torch" HF_HOME="$XDG_CACHE_HOME/hf"
mkdir -p "$XDG_CACHE_HOME"
cd "$SOURCE"
"$FLOW_ENV/bin/python" -s -u -m scripts.research.run_geometry_primary_confirmation \
 --project "$PROJECT" --protocol "$SOURCE/research/evidence/geometry_primary_confirmation_v1.json" --out "$RUN"
exec "$FLOW_ENV/bin/python" -s -u -m scripts.research.audit_geometry_recovery \
 --project "$PROJECT" --protocol "$SOURCE/research/evidence/geometry_primary_confirmation_v1.json" --run "$RUN"
