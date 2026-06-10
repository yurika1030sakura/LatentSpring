#!/bin/bash
# Boltzmann-consistency eval: chains Stage 1 (flowmol: perturbation
# log-density) and Stage 2 (omol25: energies + per-molecule correlation).
# This is the oral-defining experiment for BGFM.
#
# Usage:
#   sbatch scripts/run_boltzmann_eval.slurm <tag> <ckpt> <config> [n_mol] [n_perturb]

#SBATCH -J boltz_eval
#SBATCH -p gpu_test
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100_3g.20gb:1
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/eval/boltz-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/eval/boltz-%j.err

set -e
TAG="${1:?usage: tag ckpt config [n_mol] [n_perturb]}"
CKPT="${2:?}"
CONFIG="${3:?}"
N_MOL="${4:-60}"
N_PERTURB="${5:-16}"

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
OUT_DIR=$PROJ/runs/eval/boltzmann/$TAG
mkdir -p "$OUT_DIR" "$PROJ/runs/eval"
EVAL_DATA=/n/netscratch/ryl_lab/Lab/yulili_cfm_mol/omol25_4m_processed

cd "$PROJ"
source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh

echo "host: $(hostname)"; nvidia-smi -L
echo "tag=$TAG ckpt=$CKPT config=$CONFIG n_mol=$N_MOL n_perturb=$N_PERTURB"

echo "=== Stage 1: perturbation log-density (envs/flowmol) ==="
conda activate "$PROJ/envs/flowmol"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SLICE_ARG=""
[ -n "$CHEMISTRY_SLICE" ] && SLICE_ARG="--chemistry_slice $CHEMISTRY_SLICE"
python -u scripts/eval_boltzmann_stage1.py \
    --checkpoint "$CKPT" --config "$CONFIG" --eval_data "$EVAL_DATA" \
    --n_molecules "$N_MOL" --n_perturb "$N_PERTURB" --sigma 0.15 \
    --n_ode_steps 8 --n_hutchinson 2 --max_atoms_filter 50 $SLICE_ARG --out_dir "$OUT_DIR"
conda deactivate

echo "=== Stage 2: OMol25 energies + correlation (envs/omol25) ==="
conda activate "$PROJ/envs/omol25"
export PYTHONUNBUFFERED=1
python -u scripts/eval_boltzmann_stage2.py \
    --samples_json "$OUT_DIR/boltzmann_samples.json" \
    --out_csv "$OUT_DIR/boltzmann_correlation.csv" \
    --kT 1.0

echo "=== Boltzmann eval done: $OUT_DIR/boltzmann_correlation.csv ==="
