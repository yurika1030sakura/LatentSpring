#!/bin/bash
# Launch 5 QM9 training variants for ICLR E3 ablation + in-distribution
# sanity. Each variant submits a separate SLURM job on gpu partition.
#
# Variants:
#   baseline    : vanilla FlowMol3 (--no-patch) — literature reference
#   full        : all 3 hooks on — our main method
#   no_tangent  : gluing + retract on, tangent off
#   no_retract  : gluing + tangent on, retract off
#   no_gluing   : tangent + retract on, gluing off
#
# Usage: bash scripts/launch_qm9_ablation.sh

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
CONFIG=$PROJ/configs/qm9_cfm.yaml

declare -A VARIANTS=(
    [baseline]="--no-patch"
    [full]=""
    [no_tangent]="--no-tangent"
    [no_retract]="--no-retract"
    [no_gluing]="--no-gluing"
)

for variant in "${!VARIANTS[@]}"; do
    FLAG="${VARIANTS[$variant]}"
    OUTDIR=$PROJ/runs/qm9_${variant}
    mkdir -p "$OUTDIR"
    echo "submitting $variant  flag='$FLAG'  outdir=$OUTDIR"
    sbatch --job-name="qm9_${variant}" \
           --output="$OUTDIR/train-%j.out" \
           --error="$OUTDIR/train-%j.err" \
           -p gpu --gres=gpu:nvidia_a100-sxm4-80gb:1 \
           -N 1 --ntasks-per-node=1 --cpus-per-task=4 --mem=64G \
           -t 24:00:00 \
           --wrap "cd $PROJ && source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh && conda activate $PROJ/envs/flowmol && python scripts/run_train.py --config $CONFIG $FLAG"
    sleep 1
done

echo ""
echo "launched 5 QM9 variants. monitor: squeue -u \$USER | grep qm9_"
