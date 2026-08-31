#!/bin/bash
#SBATCH -J bgfm_sentinel
#SBATCH -p serial_requeue,shared
#SBATCH -A woo_lab
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 1-00:00:00
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/sentinel_eval_epoch0-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/sentinel_eval_epoch0-%j.err
#
# Watches for the epoch-0 checkpoint of the v2 headline run and submits the
# Boltzmann/validity eval on it the moment it lands. Session-independent.
# Args: $1 = training job id (for the version_<id> dir)
set -uo pipefail
REPO=/n/home04/yulili/bgfm
STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
TRAINJID="${1:?need training job id}"
CKDIR="$STORE/runs/omol25_4m_bgfm_onpolicy_v2/lightning_logs/version_${TRAINJID}/checkpoints"
CFG="$REPO/configs/omol25_4m_bgfm_onpolicy_v2.yaml"
echo "sentinel watching $CKDIR for epoch=0-*.ckpt"; date

for i in $(seq 1 576); do   # up to 24h, poll every 150s
  CKPT=$(ls -t "$CKDIR"/epoch=0-*.ckpt 2>/dev/null | head -1)
  if [ -n "$CKPT" ] && [ -f "$CKPT" ]; then
    echo "epoch-0 checkpoint found: $CKPT"; date
    # wait 60s to ensure the file is fully written, then submit the eval
    sleep 60
    eid=$(sbatch --parsable "$REPO/scripts/fasrc/eval_h200.slurm" \
          "$CKPT" "$CFG" onpolicy_4m_epoch0)
    echo "submitted epoch-0 eval -> job $eid"
    echo "results will land in $STORE/runs/eval_ours/onpolicy_4m_epoch0/"
    exit 0
  fi
  # bail out if the training job is gone AND still no epoch-0 ckpt
  st=$(sacct -j "$TRAINJID" --format=State -n 2>/dev/null | head -1 | tr -d ' ')
  case "$st" in
    FAILED|CANCELLED|TIMEOUT|COMPLETED|"")
      # give it one last look, else exit
      CKPT=$(ls -t "$CKDIR"/epoch=0-*.ckpt 2>/dev/null | head -1)
      [ -n "$CKPT" ] && continue
      echo "training job $TRAINJID is $st and no epoch-0 ckpt yet; exiting."; exit 0;;
  esac
  sleep 150
done
echo "sentinel timed out after 24h without an epoch-0 checkpoint."
