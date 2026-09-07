#!/bin/bash
#SBATCH -J rescore
#SBATCH -p gpu_requeue,gpu_h200
#SBATCH -A woo_lab
#SBATCH -N 1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h200:1 --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/rescore-%x-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/rescore-%x-%j.err
# Usage: sbatch -J <arm>_n<N> rescore.slurm <arm> <n_ode_steps>
set -eo pipefail
source /n/home04/yulili/bgfm/scripts/fasrc/env.sh
ARM="${1:?arm}"; N="${2:?n_ode_steps}"
echo "host: $(hostname)"; nvidia-smi -L
if ! "$FLOWMOL_PY" -c "import torch; torch.zeros(8,device='cuda'); torch.cuda.synchronize()"; then
  echo "GPU PREFLIGHT FAILED"; sleep 60; scontrol requeue "$SLURM_JOB_ID"; exit 0
fi
echo "gpu preflight ok"
cd "$REPO"
export LD_LIBRARY_PATH=$STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}

CK=$(ls -t "$STORE/runs/abl/$ARM"/lightning_logs/version_*/checkpoints/midstep-step=30000.ckpt 2>/dev/null | head -1)
[ -z "$CK" ] && { echo "no ckpt for $ARM"; exit 1; }
SAMPLES="$STORE/runs/eval_ours/wide_$ARM/boltz1/boltzmann_samples.json"
[ -f "$SAMPLES" ] || { echo "no samples for $ARM"; exit 1; }
CFG="$REPO/configs/sweep/${ARM}.yaml"
[ -f "$CFG" ] || CFG="$REPO/configs/abl/${ARM}.yaml"
[ -f "$CFG" ] || { echo "no config for $ARM"; exit 1; }

echo "arm=$ARM n_ode_steps=$N"; echo "ckpt=$CK"; echo "cfg=$CFG"; date
"$FLOWMOL_PY" -u scripts/rescore/rescore_published.py \
  --samples "$SAMPLES" --checkpoint "$CK" --config "$CFG" \
  --eval_data "$STORE/processed_data/omol25_4m_processed" \
  --n_ode_steps "$N" --n_hutchinson 2 \
  --out "$STORE/runs/rescore/${ARM}__n${N}.json"
echo "## RESCORE DONE $ARM n=$N ##"; date
