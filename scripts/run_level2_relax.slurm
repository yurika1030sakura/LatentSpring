#!/bin/bash
# Level 2 money-plot: DFT (OMol25) relax-steps comparison BGFM vs FM-only.
# Pipeline:
#   Stage A (envs/flowmol): sample N geometries each from BGFM + FM-only
#                            (one sample per molecule, many molecules)
#   Stage B (envs/omol25): BFGS relax each sample to F_max < tol, count steps
#
# Usage:
#   sbatch scripts/run_level2_relax.slurm <tag> <bgfm_ckpt> <bgfm_cfg> \
#         <fm_ckpt> <fm_cfg> [n_mols] [K_steps]

#SBATCH -J level2_relax
#SBATCH -p gpu_requeue
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/eval/level2-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/eval/level2-%j.err

set -e
TAG="${1:?usage: tag bgfm_ckpt bgfm_cfg fm_ckpt fm_cfg [n_mols] [K_steps]}"
BGFM_CKPT="${2:?}"
BGFM_CFG="${3:?}"
FM_CKPT="${4:?}"
FM_CFG="${5:?}"
N_MOLS="${6:-200}"
K_STEPS="${7:-50}"

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
OUT=$PROJ/runs/eval/level2/$TAG
BGFM_SAMPLES=$OUT/bgfm
FM_SAMPLES=$OUT/fm
RELAX_DIR=$OUT/relax
mkdir -p "$BGFM_SAMPLES" "$FM_SAMPLES" "$RELAX_DIR"

EVAL_DATA=/n/netscratch/ryl_lab/Lab/yulili_cfm_mol/omol25_4m_processed/val_data_processed.pt

# Build a mol_indices CSV of N_MOLS evenly-spaced indices into val shard
MOL_IDX=$(python3 -c "import sys; n=int(sys.argv[1]); print(','.join(str(int(i)) for i in range(0, 5000, max(1, 5000//n))[:n]))" "$N_MOLS")

cd "$PROJ"
source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh

echo "host: $(hostname)"; nvidia-smi -L
echo "tag=$TAG n_mols=$N_MOLS K=$K_STEPS"
echo "bgfm_ckpt=$BGFM_CKPT"
echo "fm_ckpt=$FM_CKPT"

echo "=== Stage A1: BGFM sampling (one geom per mol, $N_MOLS mols) ==="
conda activate "$PROJ/envs/flowmol"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u scripts/level3_bgfm_sample.py \
    --checkpoint "$BGFM_CKPT" --config "$BGFM_CFG" \
    --eval_data "$EVAL_DATA" \
    --mol_indices "$MOL_IDX" --K_steps "$K_STEPS" --n_samples 1 \
    --batch_per_pass 1 \
    --out_dir "$BGFM_SAMPLES"

echo "=== Stage A2: FM baseline sampling ==="
python -u scripts/level3_bgfm_sample.py \
    --checkpoint "$FM_CKPT" --config "$FM_CFG" \
    --eval_data "$EVAL_DATA" \
    --mol_indices "$MOL_IDX" --K_steps "$K_STEPS" --n_samples 1 \
    --batch_per_pass 1 \
    --out_dir "$FM_SAMPLES"
conda deactivate

echo "=== Stage B: OMol25 BFGS relax + comparison (envs/omol25) ==="
conda activate "$PROJ/envs/omol25"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
python -u scripts/level2_relax_comparison.py \
    --bgfm_samples "$BGFM_SAMPLES" \
    --fm_samples "$FM_SAMPLES" \
    --force_tol 0.05 --max_steps 200 \
    --max_samples_per_model "$N_MOLS" \
    --out_dir "$RELAX_DIR"

echo "=== Level 2 done: $RELAX_DIR/level2_summary.csv ==="
