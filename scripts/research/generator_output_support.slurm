#!/bin/bash
#SBATCH -J bgfm-generator-support
#SBATCH -A ryl_lab
#SBATCH -p sapphire,shared
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?unique output required}"
PROJECT="${3:?artifact root required}"
STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$SOURCE_ROOT"
"$STORE/envs/flowmol/bin/python" -u scripts/research/audit_generator_output_support.py \
  --runs-root "$PROJECT/runs" --out "$RUN_ROOT/results.json"
