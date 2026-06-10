#!/bin/bash
# CPU watcher for shuffled-force BGFM checkpoint evals.
#
# Usage:
#   sbatch scripts/watch_and_submit_shuffled_evals.sh [n_samples]

#SBATCH -J watch_shuffle_eval
#SBATCH -p shared
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH -t 3-00:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/eval/bgfm_shuffled_control/logs/watch-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/eval/bgfm_shuffled_control/logs/watch-%j.err

set -euo pipefail

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
N="${1:-200}"
RUN_ROOT="$PROJ/runs/omol25_4m_bgfm_shuffled_from_fm"
OUT_ROOT="$PROJ/runs/eval/bgfm_shuffled_control"
MARK_DIR="$OUT_ROOT/submitted"
mkdir -p "$OUT_ROOT/logs" "$MARK_DIR"
cd "$PROJ"

steps=(50000 100000 150000 200000)

checkpoint_exists() {
  local step="$1"
  [ -d "$RUN_ROOT/lightning_logs" ] || return 1
  find "$RUN_ROOT/lightning_logs" -type f -path "*/checkpoints/midstep-step=${step}.ckpt" | grep -q .
}

echo "watching shuffled-force BGFM checkpoints, n_samples=$N"
while true; do
  all_submitted=1
  for step in "${steps[@]}"; do
    marker="$MARK_DIR/shuffled_${step}.jobid"
    if [ -f "$marker" ]; then
      continue
    fi
    all_submitted=0
    if checkpoint_exists "$step"; then
      jobid="$(sbatch --parsable --export=ALL,STEP="$step",N="$N" scripts/launch_shuffled_checkpoint_eval_a100.sh "$N")"
      echo "$jobid" > "$marker"
      echo "submitted shuffled STEP=$step job=$jobid"
    else
      echo "$(date -Is) waiting shuffled STEP=$step"
    fi
  done
  if [ "$all_submitted" = "1" ]; then
    echo "all shuffled evals submitted"
    exit 0
  fi
  sleep 600
done
