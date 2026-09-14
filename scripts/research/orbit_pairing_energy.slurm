#!/bin/bash
#SBATCH -J bgfm-orbit-energy
#SBATCH -A ryl_lab
#SBATCH -p sapphire,shared
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?unique output required}"
PROJECT="${3:?project required}"
INPUT="${4:?evaluation run required}"
AUDIT="${5:?audit file required}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$SOURCE_ROOT"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/bgfm_orbit_energy_cache_${SLURM_JOB_ID}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/bgfm_orbit_energy_mpl_${SLURM_JOB_ID}"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol/bin/python -u scripts/research/score_orbit_pairing.py \
 --run "$INPUT/evaluation" --audit "$AUDIT" --out "$RUN_ROOT/energy" \
 --protocol research/evidence/orbit_pairing_protocol_v1.json
