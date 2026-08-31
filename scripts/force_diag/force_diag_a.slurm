#!/bin/bash
#SBATCH -J force_diag_a
#SBATCH -p gpu_requeue,gpu_h200
#SBATCH -A woo_lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 3:00:00
#SBATCH --requeue
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/force_diag_a-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/force_diag_a-%j.err
#
# ===========================================================================
# EXPERIMENT A -- the t-sweep. DIAGNOSTIC ONLY: no training, no manuscript.
# ===========================================================================
# Co-locates the eSEN teacher worker (omol25 env) with the diagnostic
# (flowmol env) on one GPU, exactly as the on-policy training path does, so
# that teacher forces can be evaluated at the INTERMEDIATE point x_t.  The
# precomputed shards hold labels only at x_1.
#
# If the worker fails to come up, the diagnostic is still run with
# --skip_teacher so that parts (b) and (c) land; the JSON records the failure.
set -eo pipefail
source /n/home04/yulili/bgfm/scripts/fasrc/env.sh

OUT=$STORE/runs/force_diag
mkdir -p "$OUT"
CKPT="${1:-$STORE/runs/p0/p0_A_fmonly_s1/lightning_logs/version_37027619/checkpoints/midstep-step=30000.ckpt}"
CONFIG="${2:-$REPO/configs/sweep/p0_A_fmonly_s1.yaml}"
ESEN="$STORE/checkpoints/omol25/esen_sm_conserving_all.pt"
PORT=$((29000 + (${SLURM_JOB_ID:-0} % 900)))
TAG="p0_A_fmonly_s1_step30000"

echo "host: $(hostname)"; nvidia-smi -L; date
echo "ckpt:   $CKPT"
echo "config: $CONFIG"
echo "port:   $PORT"
[ -f "$CKPT" ] || { echo "ERROR: checkpoint not found"; exit 1; }

export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "$REPO"

# --- 1. eSEN teacher worker (omol25 env, same GPU) ----------------------
TEACHER_ARGS="--rpc_url http://127.0.0.1:$PORT"
WORKER_PID=""
if [ -f "$ESEN" ]; then
  WLOG="$STORE/logs/force_diag_a_worker-${SLURM_JOB_ID:-local}.log"
  (
    export LD_LIBRARY_PATH=$STORE/envs/omol25/lib:${LD_LIBRARY_PATH:-}
    exec "$OMOL25_PY" -u "$REPO/scripts/omol25_worker.py" \
         --port "$PORT" --device cuda --ckpt "$ESEN"
  ) > "$WLOG" 2>&1 &
  WORKER_PID=$!
  cleanup() { [ -n "$WORKER_PID" ] && kill "$WORKER_PID" 2>/dev/null || true; }
  trap cleanup EXIT
  echo "waiting for teacher worker on port $PORT ..."
  for i in $(seq 1 180); do
    grep -q "serving on" "$WLOG" 2>/dev/null && { echo "teacher up after $((i*2))s"; break; }
    kill -0 "$WORKER_PID" 2>/dev/null || { echo "TEACHER DIED:"; tail -30 "$WLOG"; break; }
    sleep 2
  done
  if ! grep -q "serving on" "$WLOG" 2>/dev/null; then
    echo "teacher did not come up; falling back to --skip_teacher"
    tail -30 "$WLOG" || true
    TEACHER_ARGS="--skip_teacher"
  fi
else
  echo "eSEN checkpoint missing at $ESEN -> --skip_teacher"
  TEACHER_ARGS="--skip_teacher"
fi

# --- 2. the sweep (flowmol env) -----------------------------------------
export LD_LIBRARY_PATH=$STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}

echo "########## main sweep: <=60 atoms, 24 minibatches x 8 ##########"
"$FLOWMOL_PY" -u scripts/force_diag/exp_a_t_sweep.py \
  --checkpoint "$CKPT" --config "$CONFIG" \
  --out "$OUT/expA_tsweep_${TAG}_a60.json" \
  --t_values 0.80 0.85 0.90 0.92 0.95 0.97 \
  --n_minibatches 24 --batch_size 8 --max_atoms 60 --kT 1.0 --seed 0 \
  $TEACHER_ARGS || echo "BROKE: main sweep"

echo "########## size robustness: <=150 atoms, 10 minibatches x 4 ##########"
"$FLOWMOL_PY" -u scripts/force_diag/exp_a_t_sweep.py \
  --checkpoint "$CKPT" --config "$CONFIG" \
  --out "$OUT/expA_tsweep_${TAG}_a150.json" \
  --t_values 0.80 0.85 0.90 0.92 0.95 0.97 \
  --n_minibatches 10 --batch_size 4 --max_atoms 150 --kT 1.0 --seed 1 \
  $TEACHER_ARGS || echo "BROKE: large-molecule sweep"

date; echo DONE
