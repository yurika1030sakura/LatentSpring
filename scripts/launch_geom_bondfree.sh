#!/bin/bash
# Paper 1 ablation Cell (2): Our model on GEOM without bond supervision.
# Isolates the "bond-free" variable vs FlowMol3's published GEOM numbers
# and our existing GEOM runs (which are bond-supervised).
#
# Uses the same GEOM data preprocessing as the existing 5 GEOM variants --
# only difference is total_loss_weights.e = 0 in the config. scripts/
# run_train.py auto-detects this and disables the valence/connectivity
# discrete-projection hooks.
#
# Expected runtime: ~3 A100-days.
# Expected outcome: val_x_loss similar to geom_full; val_e_loss undefined
# (no supervision). At inference, geometry quality should match bond-
# supervised runs to within ~5%; SMILES validity will be lower (requires
# post-hoc xyz2mol).

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
CONFIG=$PROJ/configs/geom_cfm_bondfree.yaml
OUTDIR=$PROJ/runs/geom_bondfree
mkdir -p "$OUTDIR"

sbatch --job-name="geom_bondfree" \
       --output="$OUTDIR/train-%j.out" \
       --error="$OUTDIR/train-%j.err" \
       -p gpu --gres=gpu:nvidia_a100-sxm4-80gb:1 \
       -N 1 --ntasks-per-node=1 --cpus-per-task=6 --mem=96G \
       -t 72:00:00 \
       --wrap "cd $PROJ && source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh && conda activate $PROJ/envs/flowmol && python scripts/run_train.py --config $CONFIG"

echo "submitted geom_bondfree (Cell 2 ablation). monitor: squeue -u \$USER | grep geom_bondfree"
