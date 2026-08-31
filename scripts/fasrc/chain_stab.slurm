#!/bin/bash
#SBATCH -J stab_chain
#SBATCH -p serial_requeue,shared
#SBATCH -A woo_lab
#SBATCH -c 1 --mem=2G -t 3-00:00:00
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/chain_stab-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/chain_stab-%j.err
set -uo pipefail
REPO=/n/home04/yulili/bgfm; STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
declare -A JID DONE
JID[a6_energy_only_shuffled_stab_s4]=36500890
JID[a6_energy_only_shuffled_stab_s5]=36489186
for a in "${!JID[@]}"; do DONE[$a]=0; done
for i in $(seq 1 3300); do   # ~7天
  left=0
  for a in "${!JID[@]}"; do
    [ "${DONE[$a]}" -eq 1 ] && continue
    CK=$(ls $STORE/runs/abl/$a/lightning_logs/version_${JID[$a]}/checkpoints/midstep-step=30000.ckpt 2>/dev/null | head -1)
    if [ -n "$CK" ]; then
      sleep 45
      e=$(sbatch --parsable -J "w_$a" $REPO/scripts/fasrc/eval_boltzmann_wide.slurm "$CK" "$REPO/configs/sweep/${a}.yaml" "wide_${a}" 120)
      echo "[$(date)] $a -> wide eval $e"; DONE[$a]=1; continue
    fi
    st=$(sacct -j ${JID[$a]} --format=State -n 2>/dev/null | head -1 | tr -d ' ')
    case "$st" in FAILED|CANCELLED|TIMEOUT) echo "[$(date)] $a is $st"; DONE[$a]=1; continue;; esac
    left=1
  done
  [ "$left" -eq 0 ] && { echo "done"; break; }
  sleep 300
done
