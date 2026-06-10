#!/bin/bash
# CPU watcher for clean BGFM-vs-FM-continuation checkpoint evals.
#
# It does not hold a GPU. It polls for mid-step checkpoints and submits the
# matching A100 physics eval exactly once per MODE/STEP.
#
# Usage:
#   sbatch scripts/watch_and_submit_fmseed_evals.sh [n_samples]

#SBATCH -J watch_fmseed_eval
#SBATCH -p shared
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH -t 3-00:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/eval/fmseed_vs_control/logs/watch-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/eval/fmseed_vs_control/logs/watch-%j.err

set -euo pipefail

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
N="${1:-200}"
OUT_ROOT="$PROJ/runs/eval/fmseed_vs_control"
MARK_DIR="$OUT_ROOT/submitted"
mkdir -p "$OUT_ROOT/logs" "$MARK_DIR"
cd "$PROJ"

steps=(50000 100000 150000 200000)
modes=(bgfm fm)

run_root_for_mode() {
  case "$1" in
    bgfm) echo "$PROJ/runs/omol25_4m_bgfm_dirft_from_fm" ;;
    fm) echo "$PROJ/runs/omol25_4m_cfm_continue_from_fm" ;;
    *) return 2 ;;
  esac
}

checkpoint_exists() {
  local mode="$1"
  local step="$2"
  local root
  root="$(run_root_for_mode "$mode")"
  [ -d "$root/lightning_logs" ] || return 1
  find "$root/lightning_logs" -type f -path "*/checkpoints/midstep-step=${step}.ckpt" | grep -q .
}

submit_eval() {
  local mode="$1"
  local step="$2"
  local marker="$MARK_DIR/${mode}_${step}.jobid"
  if [ -f "$marker" ]; then
    return 0
  fi
  local jobid
  jobid="$(sbatch --parsable --export=ALL,MODE="$mode",STEP="$step",N="$N" scripts/launch_fmseed_checkpoint_eval_a100.sh "$N")"
  echo "$jobid" > "$marker"
  echo "submitted MODE=$mode STEP=$step job=$jobid"
}

echo "watching for FM-seeded comparison checkpoints, n_samples=$N"
while true; do
  all_submitted=1
  for mode in "${modes[@]}"; do
    for step in "${steps[@]}"; do
      marker="$MARK_DIR/${mode}_${step}.jobid"
      if [ -f "$marker" ]; then
        continue
      fi
      all_submitted=0
      if checkpoint_exists "$mode" "$step"; then
        submit_eval "$mode" "$step"
      else
        echo "$(date -Is) waiting MODE=$mode STEP=$step"
      fi
    done
  done
  if [ "$all_submitted" = "1" ]; then
    echo "all requested evals submitted"
    exit 0
  fi
  sleep 600
done
