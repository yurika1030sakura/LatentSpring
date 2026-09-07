#!/bin/bash
#SBATCH -J genstrain
#SBATCH -p gpu_requeue,gpu_h200
#SBATCH -A woo_lab
#SBATCH -N 1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_h200:1 --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/genstrain-%x-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/genstrain-%x-%j.err
#
# Estimator-free generation quality: sample from an arm, then measure how far
# each generated geometry sits from its own GFN2-xTB local minimum.  No log q
# anywhere in this pipeline, so nothing here depends on the density estimator.
#
# Usage: sbatch -J <arm> gen_strain.slurm <arm> [n_samples]
set -eo pipefail
source /n/home04/yulili/bgfm/scripts/fasrc/env.sh
ARM="${1:?arm}"; N="${2:-500}"
echo "host: $(hostname)"; nvidia-smi -L
if ! "$FLOWMOL_PY" -c "import torch; torch.zeros(8,device='cuda'); torch.cuda.synchronize()"; then
  echo "GPU PREFLIGHT FAILED"; sleep 60; scontrol requeue "$SLURM_JOB_ID"; exit 0
fi
echo "gpu preflight ok"
cd "$REPO"
export LD_LIBRARY_PATH=$STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}
export PATH=$STORE/envs/flowmol/bin:$PATH          # xtb binary
command -v xtb >/dev/null || { echo "xtb not on PATH"; exit 1; }

CK=$(ls -t "$STORE/runs/abl/$ARM"/lightning_logs/version_*/checkpoints/midstep-step=30000.ckpt 2>/dev/null | head -1)
[ -z "$CK" ] && { echo "no ckpt for $ARM"; exit 1; }
CFG="$REPO/configs/sweep/${ARM}.yaml"
[ -f "$CFG" ] || { echo "no config for $ARM"; exit 1; }
OUT="$STORE/runs/genstrain/$ARM"; mkdir -p "$OUT"

echo "arm=$ARM n_samples=$N"; echo "ckpt=$CK"; date
echo "## sampling ##"
"$FLOWMOL_PY" -u scripts/sample_to_json.py --checkpoint "$CK" --config "$CFG" \
  --eval_data "$STORE/processed_data/omol25_4m_processed" \
  --n_samples "$N" --n_timesteps 100 --batch_size 8 --device cuda \
  --out "$OUT/samples.json"

echo "## xtb relaxation ##"
"$FLOWMOL_PY" -u scripts/eval_xtb_relaxation.py --samples "$OUT/samples.json" \
  --out_csv "$OUT/xtb_relax.csv" --workers 16

echo "## GENSTRAIN DONE $ARM ##"; date
