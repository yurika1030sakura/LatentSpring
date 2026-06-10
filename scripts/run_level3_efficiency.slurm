#!/bin/bash
# Level 3 efficiency comparison: BGFM K-step ensemble vs OMol25 Langevin MD.
# Three-stage pipeline:
#   Stage A (envs/omol25): Langevin MD reference for the test molecules.
#   Stage B (envs/flowmol): K-step BGFM sampling at fixed atom types.
#   Stage C (envs/omol25): OMol25 energies for BGFM samples + comparison.
#
# Usage:
#   sbatch scripts/run_level3_efficiency.slurm <tag> <bgfm_ckpt> <bgfm_config>
#     [n_md_steps] [K_steps] [n_samples_bgfm] [mol_indices]

#SBATCH -J level3_eff
#SBATCH -p gpu_requeue
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/eval/level3-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/eval/level3-%j.err

set -e
TAG="${1:?usage: tag bgfm_ckpt bgfm_config [n_md_steps] [K_steps] [n_samples] [mol_idx]}"
BGFM_CKPT="${2:?bgfm_ckpt}"
BGFM_CFG="${3:?bgfm_config}"
N_MD_STEPS="${4:-50000}"
K_STEPS="${5:-12}"
N_SAMPLES_BGFM="${6:-100}"
MOL_INDICES="${7:-0,7,42,100,2000}"

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
OUT=$PROJ/runs/eval/level3/$TAG
MD_DIR=$OUT/md
BGFM_DIR=$OUT/bgfm
CMP_DIR=$OUT/compare
mkdir -p "$MD_DIR" "$BGFM_DIR" "$CMP_DIR"

EVAL_DATA=/n/netscratch/ryl_lab/Lab/yulili_cfm_mol/omol25_4m_processed/val_data_processed.pt

cd "$PROJ"
source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh

echo "host: $(hostname)"; nvidia-smi -L
echo "tag=$TAG bgfm_ckpt=$BGFM_CKPT cfg=$BGFM_CFG"
echo "n_md_steps=$N_MD_STEPS K_steps=$K_STEPS n_samples_bgfm=$N_SAMPLES_BGFM"
echo "mol_indices=$MOL_INDICES"

echo "=== Stage A: Langevin MD reference (envs/omol25) ==="
conda activate "$PROJ/envs/omol25"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
python -u scripts/level3_md_reference.py \
    --eval_data "$EVAL_DATA" \
    --atom_map_config "$BGFM_CFG" \
    --mol_indices "$MOL_INDICES" \
    --n_md_steps "$N_MD_STEPS" --save_every 500 \
    --temperature 300 --dt_fs 0.5 --friction 0.01 \
    --out_dir "$MD_DIR"
conda deactivate

echo "=== Stage B: BGFM K-step sampling (envs/flowmol) ==="
conda activate "$PROJ/envs/flowmol"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u scripts/level3_bgfm_sample.py \
    --checkpoint "$BGFM_CKPT" --config "$BGFM_CFG" \
    --eval_data "$EVAL_DATA" \
    --mol_indices "$MOL_INDICES" \
    --K_steps "$K_STEPS" --n_samples "$N_SAMPLES_BGFM" \
    --out_dir "$BGFM_DIR"
conda deactivate

echo "=== Stage C: BGFM-sample OMol25 energies + comparison (envs/omol25) ==="
conda activate "$PROJ/envs/omol25"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
python -u scripts/level3_compute_energies.py --in_dir "$BGFM_DIR"
python -u scripts/level3_compare.py \
    --md_dir "$MD_DIR" --bgfm_dir "$BGFM_DIR" --out_dir "$CMP_DIR"

echo "=== Level 3 done: $CMP_DIR/level3_summary.csv ==="
