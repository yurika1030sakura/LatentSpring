#!/bin/bash
# Launch 5 GEOM-Drugs training variants for the ICLR primary paper
# experiments. Same ablation structure as launch_qm9_ablation.sh, but
# against the larger GEOM config (10-organic atom_map, longer training).
#
# Each variant ~3 GPU-days on a single A100-80GB. Run when GEOM data
# processed and QM9 variants have smoke-passed. Total ~15 GPU-days
# across the 5 runs if submitted in parallel.

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
CONFIG=$PROJ/configs/geom_cfm.yaml

declare -A VARIANTS=(
    [baseline]="--no-patch"
    [full]=""
    [no_tangent]="--no-tangent"
    [no_retract]="--no-retract"
    [no_gluing]="--no-gluing"
)

for variant in "${!VARIANTS[@]}"; do
    FLAG="${VARIANTS[$variant]}"
    OUTDIR=$PROJ/runs/geom_${variant}
    mkdir -p "$OUTDIR"
    echo "submitting geom_$variant  flag='$FLAG'  outdir=$OUTDIR"
    sbatch --job-name="geom_${variant}" \
           --output="$OUTDIR/train-%j.out" \
           --error="$OUTDIR/train-%j.err" \
           -p gpu --gres=gpu:nvidia_a100-sxm4-80gb:1 \
           -N 1 --ntasks-per-node=1 --cpus-per-task=6 --mem=96G \
           -t 72:00:00 \
           --wrap "cd $PROJ && source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh && conda activate $PROJ/envs/flowmol && python scripts/run_train.py --config $CONFIG $FLAG"
    sleep 1
done

echo ""
echo "launched 5 GEOM variants. monitor: squeue -u \$USER | grep geom_"
