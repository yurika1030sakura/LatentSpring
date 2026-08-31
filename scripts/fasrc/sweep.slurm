#!/bin/bash
#SBATCH -J bgfm_sweep
#SBATCH -p gpu_requeue,gpu_h200
#SBATCH -A woo_lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=192G
#SBATCH -t 1-00:00:00
#SBATCH --requeue
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/sweep-%x-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/sweep-%x-%j.err
#SBATCH --signal=SIGUSR1@120
#
# One capped-step sweep run. Auto-detects on-policy configs (starts the eSEN
# teacher worker on the port named in the config's onpolicy_rpc_url) vs
# off-policy (no worker). Submit one per config for a parallel sweep:
#   for c in fm_only bgfm_offpolicy hbfm_onpolicy hbfm_onpolicy_energy shuffle_control; do
#     sbatch -J $c scripts/fasrc/sweep.slurm $c
#   done
# Usage: sbatch -J <name> scripts/fasrc/sweep.slurm <config_basename_under_configs/sweep>
set -eo pipefail
source /n/home04/yulili/bgfm/scripts/fasrc/env.sh

NAME="${1:?usage: sbatch scripts/fasrc/sweep.slurm <config basename under configs/sweep/> [warm_ckpt]}"
WARM="${2:-}"                  # optional: warm-init WEIGHTS on FIRST start only (matched ablation)
SEED="${3:-}"                  # optional: RNG seed (run_train.py defaults to 42) -> for replicates
CONFIG="$REPO/configs/sweep/${NAME}.yaml"
[ -f "$CONFIG" ] || { echo "config not found: $CONFIG"; exit 1; }
CKPT="$STORE/checkpoints/omol25/esen_sm_conserving_all.pt"

# Warm-init on FIRST start only. On requeue, auto-resume full trainer state so
# global_step (and thus the step-based lambda schedule) is preserved.
RESUME_ARG=""
if [ -n "$WARM" ]; then
  OUTDIR=$(grep -E '^[[:space:]]*output_dir:' "$CONFIG" | head -1 | sed 's/.*output_dir:[[:space:]]*//')
  CKDIR="$OUTDIR/lightning_logs/version_${SLURM_JOB_ID}/checkpoints"
  if ls "$CKDIR"/*.ckpt >/dev/null 2>&1; then
    echo "requeue -> auto-resuming full state (ignoring warm ckpt)"
  elif [ -f "$WARM" ]; then
    RESUME_ARG="--resume_from $WARM"
    echo "first start -> warm-init from $WARM"
  fi
fi
echo "host: $(hostname)"; nvidia-smi -L; echo "config: $CONFIG"; date
cd "$REPO"

# --- start the eSEN teacher only if this config is on-policy ---
WORKER_PID=""
RPC=$(grep -oE 'onpolicy_rpc_url:[[:space:]]*http://[^[:space:]]+' "$CONFIG" | grep -oE 'http://[^[:space:]]+' || true)
if [ -n "$RPC" ]; then
  PORT=$(echo "$RPC" | sed -E 's#.*:([0-9]+).*#\1#')
  [ -f "$CKPT" ] || { echo "ERROR: on-policy config but oracle ckpt missing at $CKPT"; exit 1; }
  WLOG="$STORE/logs/omol25_worker-${SLURM_JOB_ID:-local}-${PORT}.log"
  echo "on-policy: starting eSEN teacher on port $PORT (log $WLOG)"
  (
    export LD_LIBRARY_PATH=$STORE/envs/omol25/lib:${LD_LIBRARY_PATH:-}
    exec "$OMOL25_PY" -u "$REPO/scripts/omol25_worker.py" --port "$PORT" --device cuda --ckpt "$CKPT"
  ) > "$WLOG" 2>&1 &
  WORKER_PID=$!
  trap 'echo "stopping teacher $WORKER_PID"; kill "$WORKER_PID" 2>/dev/null || true' EXIT
  for i in $(seq 1 180); do
    grep -q "serving on" "$WLOG" 2>/dev/null && { echo "teacher up after ${i}x2s"; break; }
    kill -0 "$WORKER_PID" 2>/dev/null || { echo "TEACHER DIED; tail:"; tail -30 "$WLOG"; exit 1; }
    sleep 2
  done
  grep -q "serving on" "$WLOG" || { echo "teacher never served"; tail -30 "$WLOG"; exit 1; }
else
  echo "off-policy config (no teacher worker needed)"
fi

# --- trainer ---
export LD_LIBRARY_PATH=$STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}
SEED_ARG=""; [ -n "$SEED" ] && SEED_ARG="--seed $SEED"
"$FLOWMOL_PY" -u scripts/run_train.py --config "$CONFIG" $RESUME_ARG $SEED_ARG
date
