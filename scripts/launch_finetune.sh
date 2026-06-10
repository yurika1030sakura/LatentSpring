#!/bin/bash
# Fine-tune existing QM9 checkpoint with train-time discrete projection
# enabled. Much faster than retraining from scratch (~1 day vs 3-4).
#
# Uses the snapshotted baseline checkpoint (version_7 = qm9_baseline) and
# continues training with ALL 4 hooks (gluing + tangent + retract +
# train_time_discrete) active for ~3-5 more epochs.

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
CONFIG=$PROJ/configs/qm9_cfm.yaml
# Base checkpoint to resume from (qm9_baseline, vanilla-trained so clean)
CKPT=$PROJ/runs/eval/ckpt_snapshots/v7_baseline.ckpt

OUTDIR=$PROJ/runs/qm9_finetune_traintimedisc
mkdir -p "$OUTDIR"

sbatch --job-name=qm9_ft_td \
       --output=$OUTDIR/ft-%j.out --error=$OUTDIR/ft-%j.err \
       -p gpu --gres=gpu:nvidia_a100-sxm4-80gb:1 \
       -N 1 --ntasks-per-node=1 --cpus-per-task=4 --mem=64G \
       -t 24:00:00 \
       --wrap "cd $PROJ && source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh && conda activate $PROJ/envs/flowmol && python scripts/run_train.py --config $CONFIG --resume_from '$CKPT'"

squeue -u $USER | grep qm9_ft
