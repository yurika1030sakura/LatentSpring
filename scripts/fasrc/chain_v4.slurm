#!/bin/bash
#SBATCH -J v4_chain
#SBATCH -p serial_requeue,shared
#SBATCH -A woo_lab
#SBATCH -c 1 --mem=2G -t 2-00:00:00
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/chain_v4-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/chain_v4-%j.err
set -uo pipefail
REPO=/n/home04/yulili/bgfm
STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
TR=30480344
CK="$STORE/runs/omol25_4m_bgfm_onpolicy_v4_stable/lightning_logs/version_$TR/checkpoints"
echo "watching $CK"; date
for i in $(seq 1 1150); do
  c=$(ls "$CK"/midstep-step=30000.ckpt 2>/dev/null | head -1)
  if [ -n "$c" ] && [ -f "$c" ]; then
    sleep 60
    e=$(sbatch --parsable "$REPO/scripts/fasrc/eval_h200.slurm" "$c"           "$REPO/configs/omol25_4m_bgfm_onpolicy_v4_stable.yaml" v4_stable_step30k)
    echo "[$(date)] v4 step-30k -> eval job $e"
    # also the transition-metal (killer figure) eval on the same ckpt
    t=$(sbatch --parsable "$REPO/scripts/fasrc/eval_ood_boltzmann.slurm" "$c"           "$REPO/configs/omol25_4m_bgfm_onpolicy_v4_stable.yaml" v4_stable_tmqm           "$STORE/processed_data/ood_omol25map/tmqm_broad")
    echo "[$(date)] v4 step-30k -> tmQM eval job $t"
    exit 0
  fi
  st=$(sacct -j $TR --format=State -n 2>/dev/null | head -1 | tr -d ' ')
  case "$st" in FAILED|CANCELLED|TIMEOUT) echo "train job $st before ckpt"; exit 0;; esac
  sleep 150
done
