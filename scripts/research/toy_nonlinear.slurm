#!/bin/bash
#SBATCH -J bgfm-nonlinear-toy
#SBATCH -A ryl_lab
#SBATCH -p test
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "${1:?source snapshot}"
exec /n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol/bin/python -u \
 scripts/research/toy_nonlinear_trace.py --out "${2:?output directory}"
