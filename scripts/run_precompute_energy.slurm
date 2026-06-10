#!/bin/bash
# Pre-compute K-perturbation tuples + OMol25 energies (envs/omol25).
# Output -> $OUT_DIR/perturbation_<split>.pt for the BGFM energy loss.
#
# Usage:
#   sbatch scripts/run_precompute_energy.slurm <split> <n_molecules> [start_index]
#     split: val | train
#     n_molecules: how many base molecules to perturb

#SBATCH -J pert_energy
#SBATCH -p gpu_requeue
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/eval/pert-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/eval/pert-%j.err

set -e
SPLIT="${1:?split: val|train}"
N_MOL="${2:?n_molecules}"
START="${3:-0}"

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
OUT_DIR=/n/netscratch/ryl_lab/Lab/yulili_cfm_mol/omol25_4m_processed
SRC="$OUT_DIR/${SPLIT}_data_processed.pt"
OUT="$OUT_DIR/perturbation_${SPLIT}_n${N_MOL}_s${START}.pt"

cd "$PROJ"
source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh
conda activate "$PROJ/envs/omol25"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

echo "host: $(hostname)"; nvidia-smi -L
echo "split=$SPLIT n_mol=$N_MOL start=$START"
echo "src=$SRC"
echo "out=$OUT"

python -u scripts/precompute_energy_perturbations.py \
    --src "$SRC" \
    --out "$OUT" \
    --atom_map_config configs/omol25_4m_bgfm.yaml \
    --n_molecules "$N_MOL" \
    --sigmas "${SIGMAS:-0.03,0.06,0.10,0.20,0.40}" \
    --start_index "$START"
echo "=== PERT DONE: $OUT ==="
