#!/bin/bash
#SBATCH -J adv_dash
#SBATCH -p serial_requeue,shared,sapphire
#SBATCH -A woo_lab
#SBATCH -N 1 --ntasks-per-node=1 --cpus-per-task=2 --mem=8G
#SBATCH -t 0:30:00
#SBATCH --requeue
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/adv_dash-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/adv_dash-%j.err
#
# Regenerate runs/ADVISOR_DASHBOARD.txt. Deliberately a SHORT one-shot job:
# we schedule several copies (via --dependency=afterany and via --begin) instead
# of running a long-lived sentinel, because a 3-day sentinel once timed out and
# took the whole eval batch with it.
# NOTE: absolute paths only -- SLURM copies this script into its spool dir.
set -eo pipefail
source /n/home04/yulili/bgfm/scripts/fasrc/env.sh
export LD_LIBRARY_PATH=$STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}
date
"$FLOWMOL_PY" -u /n/home04/yulili/bgfm/scripts/summarize_advisor_experiments.py \
  --store "$STORE" --out "$STORE/runs/ADVISOR_DASHBOARD.txt"
# keep a timestamped copy so progress over the batch is auditable
cp "$STORE/runs/ADVISOR_DASHBOARD.txt" \
   "$STORE/runs/dashboard_history/ADVISOR_DASHBOARD_$(date +%Y%m%d_%H%M%S).txt" 2>/dev/null || \
  { mkdir -p "$STORE/runs/dashboard_history" && cp "$STORE/runs/ADVISOR_DASHBOARD.txt" \
    "$STORE/runs/dashboard_history/ADVISOR_DASHBOARD_$(date +%Y%m%d_%H%M%S).txt"; }
echo "## DASHBOARD REFRESHED ##"; date
