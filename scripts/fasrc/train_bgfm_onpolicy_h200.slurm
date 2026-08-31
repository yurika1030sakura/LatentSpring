#!/bin/bash
#SBATCH -J bgfm_onpolicy_h200
#SBATCH -p gpu_requeue,gpu_h200
#SBATCH -A woo_lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=256G
#SBATCH -t 3-00:00:00
#SBATCH --requeue
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/bgfm_onpolicy_h200-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/bgfm_onpolicy_h200-%j.err
#SBATCH --signal=SIGUSR1@120
#
# ===========================================================================
# 4M (FULL) HEADLINE RUN — ON-POLICY HBFM variant.
# ===========================================================================
# Co-locates a live OMol25 eSEN *teacher* worker (omol25 env) and the trainer
# (flowmol env) on ONE H200. The teacher scores on-policy geometries so the
# force distillation targets the model's own samples, not just the dataset.
#
# PREREQUISITE (one-time, gated): download the eSEN oracle checkpoint and place
# it at $CKPT below. Needs HuggingFace access to facebook/OMol25 + facebook/UMA:
#   export HF_TOKEN=...  HF_HUB_DISABLE_XET=1
#   (fetch esen_sm_conserving_all.pt into $STORE/checkpoints/omol25/)
# Without it the worker will not start; use train_bgfm_4m_h200.slurm instead.
#
# Usage: sbatch scripts/fasrc/train_bgfm_onpolicy_h200.slurm [config]
set -eo pipefail
source /n/home04/yulili/bgfm/scripts/fasrc/env.sh

CONFIG="${1:-$REPO/configs/omol25_4m_bgfm_onpolicy.yaml}"
WARM="${2:-}"                  # optional: warm-init WEIGHTS from this ckpt on FIRST start only
EXTRA="${3:-}"                 # optional: extra run_train.py flags (e.g. --fast for a smoke)
CKPT="$STORE/checkpoints/omol25/esen_sm_conserving_all.pt"
# Derive the teacher port FROM THE CONFIG so worker and trainer can never disagree
# (and so concurrent on-policy runs on the same node use distinct ports).
PORT=$(grep -oE 'onpolicy_rpc_url:[[:space:]]*http://[^[:space:]]+' "$CONFIG" \
       | grep -oE '[0-9]+$' | head -1)
PORT="${OMOL25_PORT:-${PORT:-28900}}"
echo "host: $(hostname)"; nvidia-smi -L; echo "config: $CONFIG"; date

# Warm-init only on the FIRST start. On a requeue this job's own checkpoints
# exist, and run_train.py must auto-resume FULL trainer state (preserving
# global_step, which drives the step-based lambda schedule). Re-applying
# --resume_from on a requeue would reload the warm weights and reset the
# schedule to zero, silently throwing away all physics-phase training.
RESUME_ARG=""
if [ -n "$WARM" ]; then
  OUTDIR=$(grep -E '^[[:space:]]*output_dir:' "$CONFIG" | head -1 | sed 's/.*output_dir:[[:space:]]*//')
  CKDIR="$OUTDIR/lightning_logs/version_${SLURM_JOB_ID}/checkpoints"
  if ls "$CKDIR"/*.ckpt >/dev/null 2>&1; then
    echo "requeue detected (checkpoints exist in $CKDIR) -> auto-resuming full state, ignoring --resume_from"
  elif [ -f "$WARM" ]; then
    RESUME_ARG="--resume_from $WARM"
    echo "first start -> warm-init weights from $WARM (fresh global_step for the lambda schedule)"
  else
    echo "WARN: warm ckpt not found at $WARM; starting from scratch"
  fi
fi

if [ ! -f "$CKPT" ]; then
  echo "ERROR: eSEN oracle checkpoint not found at $CKPT"
  echo "       Download it (gated) or run the off-policy launcher train_bgfm_4m_h200.slurm."
  exit 1
fi

# --- 1. start the eSEN teacher worker (omol25 env) on the allocated GPU ---
WLOG="$STORE/logs/omol25_worker-${SLURM_JOB_ID:-local}.log"
(
  export LD_LIBRARY_PATH=$STORE/envs/omol25/lib:${LD_LIBRARY_PATH:-}
  exec "$OMOL25_PY" -u "$REPO/scripts/omol25_worker.py" \
       --port "$PORT" --device cuda --ckpt "$CKPT"
) > "$WLOG" 2>&1 &
WORKER_PID=$!
cleanup() { echo "stopping teacher worker $WORKER_PID"; kill "$WORKER_PID" 2>/dev/null || true; }
trap cleanup EXIT

# --- 2. wait until the worker is serving (fail fast if it dies) ---
echo "waiting for teacher worker on port $PORT ..."
for i in $(seq 1 180); do
  if grep -q "serving on" "$WLOG" 2>/dev/null; then echo "teacher up after ${i}x2s"; break; fi
  if ! kill -0 "$WORKER_PID" 2>/dev/null; then
    echo "TEACHER WORKER DIED before serving; tail of $WLOG:"; tail -30 "$WLOG"; exit 1
  fi
  sleep 2
done
grep -q "serving on" "$WLOG" || { echo "teacher worker did not come up in time"; tail -30 "$WLOG"; exit 1; }

# --- 3. run the trainer (flowmol env), pointed at the local teacher ---
export LD_LIBRARY_PATH=$STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}
cd "$REPO"
"$FLOWMOL_PY" -u scripts/run_train.py --config "$CONFIG" $RESUME_ARG $EXTRA
date
