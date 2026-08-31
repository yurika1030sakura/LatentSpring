#!/bin/bash
#SBATCH -J bgfm_chain
#SBATCH -p serial_requeue,shared
#SBATCH -A woo_lab
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 3-00:00:00
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/chain_energy-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/chain_energy-%j.err
#
# UNATTENDED CHAIN for the lambda_2 (energy-term) experiment:
#   1. launch both energy arms (v3 balanced, v3b aggressive), warm-started from epoch-0
#   2. wait for each to save a checkpoint PAST full lambda (step 30000 > full-lambda ~19.3k)
#   3. auto-submit the Boltzmann/validity eval on that checkpoint
#   4. also eval the epoch-0 checkpoint if it lands within the window
# Fully session-independent.
set -uo pipefail
REPO=/n/home04/yulili/bgfm
STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
WARM="$STORE/runs/omol25_4m_bgfm_onpolicy_v2/lightning_logs/version_29849595/checkpoints/epoch=0-step=146331.ckpt"
LAUNCH="$REPO/scripts/fasrc/train_bgfm_onpolicy_h200.slurm"
EVAL="$REPO/scripts/fasrc/eval_h200.slurm"

declare -A CFG OUT JID
CFG[v3_energy]="$REPO/configs/omol25_4m_bgfm_onpolicy_v3_energy.yaml"
OUT[v3_energy]="$STORE/runs/omol25_4m_bgfm_onpolicy_v3_energy"
CFG[v3b_energy_hi]="$REPO/configs/omol25_4m_bgfm_onpolicy_v3b_energy_hi.yaml"
OUT[v3b_energy_hi]="$STORE/runs/omol25_4m_bgfm_onpolicy_v3b_energy_hi"

echo "=== chain start $(date) ==="
[ -f "$WARM" ] || { echo "FATAL: warm ckpt missing: $WARM"; exit 1; }

for arm in v3_energy v3b_energy_hi; do
  j=$(sbatch --parsable -J "$arm" "$LAUNCH" "${CFG[$arm]}" "$WARM")
  JID[$arm]=$j
  echo "launched $arm -> train job $j  (cfg ${CFG[$arm]})"
done

declare -A DONE30 DONEEP
for arm in v3_energy v3b_energy_hi; do DONE30[$arm]=0; DONEEP[$arm]=0; done

# poll for checkpoints past full-lambda, then epoch-0
for i in $(seq 1 1600); do   # ~66h at 150s
  allfinished=1
  for arm in v3_energy v3b_energy_hi; do
    CK="${OUT[$arm]}/lightning_logs/version_${JID[$arm]}/checkpoints"
    # (a) first checkpoint at/after step 30000 (full lambda reached ~19.3k)
    if [ "${DONE30[$arm]}" -eq 0 ]; then
      c=$(ls "$CK"/midstep-step=30000.ckpt 2>/dev/null | head -1)
      if [ -n "$c" ] && [ -f "$c" ]; then
        sleep 60
        e=$(sbatch --parsable "$EVAL" "$c" "${CFG[$arm]}" "${arm}_step30k")
        echo "[$(date)] $arm step-30k ckpt -> eval job $e  (results: $STORE/runs/eval_ours/${arm}_step30k/)"
        DONE30[$arm]=1
      fi
    fi
    # (b) epoch-0 checkpoint
    if [ "${DONEEP[$arm]}" -eq 0 ]; then
      c=$(ls -t "$CK"/epoch=0-*.ckpt 2>/dev/null | head -1)
      if [ -n "$c" ] && [ -f "$c" ]; then
        sleep 60
        e=$(sbatch --parsable "$EVAL" "$c" "${CFG[$arm]}" "${arm}_epoch0")
        echo "[$(date)] $arm epoch-0 ckpt -> eval job $e  (results: $STORE/runs/eval_ours/${arm}_epoch0/)"
        DONEEP[$arm]=1
      fi
    fi
    [ "${DONE30[$arm]}" -eq 1 ] && [ "${DONEEP[$arm]}" -eq 1 ] || allfinished=0
    # if the training job died and we got nothing, stop waiting on it
    st=$(sacct -j "${JID[$arm]}" --format=State -n 2>/dev/null | head -1 | tr -d ' ')
    case "$st" in
      FAILED|CANCELLED|TIMEOUT)
        if [ "${DONE30[$arm]}" -eq 0 ]; then echo "[$(date)] WARN: $arm train job ${JID[$arm]} is $st before a step-30k ckpt"; DONE30[$arm]=1; DONEEP[$arm]=1; fi;;
    esac
  done
  [ "$allfinished" -eq 1 ] && { echo "all evals submitted"; break; }
  sleep 150
done
echo "=== chain done $(date) ==="
