#!/bin/bash
# Sweep evaluate_validity across all 5 QM9 variants × 6 eval sets
# (in-distribution + 5 OOD slices). Each is a separate gpu_test job,
# ~5-10 min each. Total 30 evals.
#
# Requires: $CKPT_MAP env var OR checkpoints passed explicitly.
# For now, hard-code checkpoint paths here after identifying them.

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol

# Map: variant name -> checkpoint path. Fill in after mapping jobs to versions.
declare -A CKPT
CKPT[full]="$PROJ/runs/qm9_cfm/lightning_logs/version_4/checkpoints/epoch=5-step=86700.ckpt"
CKPT[baseline]=""
CKPT[no_tangent]=""
CKPT[no_retract]=""
CKPT[no_gluing]=""

# Map: eval-set name -> (processed_dir, config to build model, extra-flag-for-vanilla)
declare -A DATA
DATA[qm9]="$PROJ/data/qm9_processed"
DATA[tmqm_broad]="$PROJ/data/tmqm_broad_processed"
DATA[tmqm_narrow]="$PROJ/data/tmqm_narrow_processed"
DATA[tmqm_organic]="$PROJ/data/tmqm_organic_processed"
DATA[kraken]="$PROJ/data/kraken_processed"
DATA[hypervalent]="$PROJ/data/hypervalent_processed"

CONFIG=$PROJ/configs/qm9_cfm.yaml   # model built from this config
OUT=$PROJ/runs/eval
mkdir -p "$OUT"

for variant in full baseline no_tangent no_retract no_gluing; do
  ckpt="${CKPT[$variant]}"
  [ -z "$ckpt" ] && { echo "[skip] $variant: no checkpoint mapped"; continue; }
  for ds in qm9 tmqm_broad tmqm_narrow tmqm_organic kraken hypervalent; do
    dd="${DATA[$ds]}"
    [ -f "$dd/val_data_processed.pt" ] || { echo "[skip] $ds: no val data"; continue; }
    tag="${variant}_on_${ds}"
    if [ "$variant" = "baseline" ]; then
        VANILLA_FLAG="--vanilla"
    else
        VANILLA_FLAG=""
    fi
    echo "submit $tag"
    sbatch --job-name="ev_${tag}" \
           --output="$OUT/${tag}-%j.out" --error="$OUT/${tag}-%j.err" \
           -p gpu_test --gres=gpu:1 -N 1 --ntasks-per-node=1 --cpus-per-task=4 --mem=32G -t 1:00:00 \
           --wrap "cd $PROJ && source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh && conda activate $PROJ/envs/flowmol && PYTHONPATH=. python scripts/evaluate_validity.py --checkpoint '$ckpt' --config $CONFIG --eval_data $dd --n_samples 200 --n_timesteps 100 --batch_size 32 $VANILLA_FLAG --out $OUT/${tag}.csv"
    sleep 1
  done
done
echo "sweep launched. monitor: squeue -u \$USER | grep ev_"
