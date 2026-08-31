#!/bin/bash
#SBATCH -J seed_chain
#SBATCH -p serial_requeue,shared
#SBATCH -A woo_lab
#SBATCH -c 1 --mem=2G -t 3-00:00:00
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/chain_seeds-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/chain_seeds-%j.err
#
# 3天无人值守：训练 a1/a3/a6 的 seed 3/4/5，每个训完自动用 n=120 宽评测。
# 目的：把 real-vs-shuffled (Y-scrambling) 从 n=1~2 个种子变成 5 个种子的统计结论。
set -uo pipefail
REPO=/n/home04/yulili/bgfm
STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
WARM="$STORE/runs/omol25_4m_bgfm_onpolicy_v2/lightning_logs/version_29849595/checkpoints/epoch=0-step=146331.ckpt"
ARMS="a1_fm_only a3_energy_only a6_energy_only_shuffled"
echo "=== seed chain start $(date) ==="
declare -A JID DONE
for base in $ARMS; do for s in 3 4 5; do
  a="${base}_s${s}"
  j=$(sbatch --parsable -J "$a" "$REPO/scripts/fasrc/sweep.slurm" "$a" "$WARM" "$s")
  JID[$a]=$j; DONE[$a]=0
  echo "launched $a (seed $s) -> $j"
done; done

for i in $(seq 1 1700); do
  left=0
  for base in $ARMS; do for s in 3 4 5; do
    a="${base}_s${s}"
    [ "${DONE[$a]}" -eq 1 ] && continue
    CK="$STORE/runs/abl/$a/lightning_logs/version_${JID[$a]}/checkpoints/midstep-step=30000.ckpt"
    if [ -f "$CK" ]; then
      sleep 45
      # 评测前先验权重有限性（宽评测脚本内置了检查）
      e=$(sbatch --parsable -J "w_$a" "$REPO/scripts/fasrc/eval_boltzmann_wide.slurm" \
            "$CK" "$REPO/configs/sweep/${a}.yaml" "wide_${a}" 120)
      echo "[$(date)] $a -> wide eval (n=120) job $e"
      DONE[$a]=1; continue
    fi
    st=$(sacct -j "${JID[$a]}" --format=State -n 2>/dev/null | head -1 | tr -d ' ')
    case "$st" in FAILED|CANCELLED|TIMEOUT)
      echo "[$(date)] WARN $a is $st with no step-30k ckpt"; DONE[$a]=1; continue;; esac
    left=1
  done; done
  [ "$left" -eq 0 ] && { echo "all seed evals submitted"; break; }
  sleep 180
done
echo "=== seed chain done $(date) ==="
