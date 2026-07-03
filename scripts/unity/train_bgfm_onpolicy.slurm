#!/bin/bash
#SBATCH -J bgfm_onpolicy
#SBATCH --partition=gpupod-l40s
#SBATCH --account=pi_bsilva_umass_edu
#SBATCH --qos=gpu-quota-8
#SBATCH --gres=gpu:l40s:1
#SBATCH --constraint=x86_64
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH -o /home/renhaozhang_umass_edu/scratch_workspace/bgfm/logs/bgfm_onpolicy-%j.out
#SBATCH -e /home/renhaozhang_umass_edu/scratch_workspace/bgfm/logs/bgfm_onpolicy-%j.err
#
# BGFM Phase-1 ON-POLICY force distillation.
# Runs the trainer (envs/flowmol) AND a co-located OMol25 teacher worker
# (envs/omol25) on ONE L40S. Measured footprint: trainer ~13.5 GB (peak,
# nvidia-smi) + teacher ~0.6 GB => ~14 GB, well within the 46 GB L40S.
# Routed to gpupod-l40s (not the general `gpu` partition) because the trainer
# alone exceeds 12 GB and would OOM a 2080ti/1080ti/titan_x there.
set -eo pipefail
REPO=/home/renhaozhang_umass_edu/bgfm
SCR=/home/renhaozhang_umass_edu/scratch_workspace/bgfm
CONFIG=${1:-$SCR/omol25_50k_bgfm_onpolicy.yaml}
CKPT=$SCR/checkpoints/omol25/checkpoints/esen_sm_conserving_all.pt
PORT=${OMOL25_PORT:-5900}   # must match bgfm.onpolicy_rpc_url in the config
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
mkdir -p "$SCR/logs"
echo "host $(hostname); $(nvidia-smi -L)"; echo "config=$CONFIG"; date

# --- 1. start the OMol25 teacher worker (omol25 env) on the allocated GPU ---
WLOG=$SCR/logs/omol25_worker-${SLURM_JOB_ID:-local}.log
(
  export LD_LIBRARY_PATH=$SCR/envs/omol25/lib:${LD_LIBRARY_PATH:-}
  exec "$SCR/envs/omol25/bin/python" -u "$REPO/scripts/omol25_worker.py" \
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
export LD_LIBRARY_PATH=$SCR/envs/flowmol/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$REPO
cd "$REPO"
"$SCR/envs/flowmol/bin/python" -u scripts/run_train.py --config "$CONFIG"
date
