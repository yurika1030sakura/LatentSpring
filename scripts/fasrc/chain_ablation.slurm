#!/bin/bash
#SBATCH -J abl_chain
#SBATCH -p serial_requeue,shared
#SBATCH -A woo_lab
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 2-00:00:00
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/chain_ablation-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/chain_ablation-%j.err
#
# Unattended 5-cell BGFM ablation: launch all arms warm-started from the SAME
# epoch-0 checkpoint, then auto-eval each at its step-30000 checkpoint.
set -uo pipefail
REPO=/n/home04/yulili/bgfm
STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
WARM="$STORE/runs/omol25_4m_bgfm_onpolicy_v2/lightning_logs/version_29849595/checkpoints/epoch=0-step=146331.ckpt"
ARMS="a1_fm_only a2_force_only a3_energy_only a4_force_energy a5_shuffle_energy"
echo "=== ablation chain start $(date) ==="
[ -f "$WARM" ] || { echo "FATAL: warm ckpt missing"; exit 1; }

declare -A JID DONE
for a in $ARMS; do
  j=$(sbatch --parsable -J "$a" "$REPO/scripts/fasrc/sweep.slurm" "$a" "$WARM")
  JID[$a]=$j; DONE[$a]=0
  echo "launched $a -> job $j"
done

for i in $(seq 1 1150); do   # ~48h
  left=0
  for a in $ARMS; do
    [ "${DONE[$a]}" -eq 1 ] && continue
    CK="$STORE/runs/abl/$a/lightning_logs/version_${JID[$a]}/checkpoints"
    # final (30k) checkpoint: prefer last.ckpt once the job is done, else midstep-30000
    c=""
    st=$(sacct -j "${JID[$a]}" --format=State -n 2>/dev/null | head -1 | tr -d ' ')
    if [ "$st" = "COMPLETED" ]; then
      c=$(ls -t "$CK"/last.ckpt "$CK"/midstep-step=30000.ckpt 2>/dev/null | head -1)
    else
      c=$(ls "$CK"/midstep-step=30000.ckpt 2>/dev/null | head -1)
    fi
    if [ -n "$c" ] && [ -f "$c" ]; then
      sleep 45
      e=$(sbatch --parsable "$REPO/scripts/fasrc/eval_h200.slurm" "$c" \
            "$REPO/configs/sweep/${a}.yaml" "abl_${a}")
      echo "[$(date)] $a -> eval job $e (results: $STORE/runs/eval_ours/abl_${a}/)"
      DONE[$a]=1; continue
    fi
    case "$st" in
      FAILED|CANCELLED|TIMEOUT) echo "[$(date)] WARN: $a job ${JID[$a]} is $st with no ckpt"; DONE[$a]=1; continue;;
    esac
    left=1
  done
  [ "$left" -eq 0 ] && { echo "all ablation evals submitted"; break; }
  sleep 150
done
echo "=== ablation chain done $(date) ==="
