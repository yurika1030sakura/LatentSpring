#!/bin/bash
#SBATCH -J clean_eval
#SBATCH -p gpu_requeue,gpu_h200
#SBATCH -A woo_lab
#SBATCH -N 1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h200:1 --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/clean_eval-%x-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/clean_eval-%x-%j.err
#
# 评测一个 CLEAN 臂: 自动定位该臂最新的有限权重 checkpoint, 跑 n=120 玻尔兹曼评测。
# 用 SLURM --dependency=afterany 链接到训练任务, 不依赖长驻哨兵(上次哨兵3天超时丢了全部评测)。
# 用法: sbatch --dependency=afterany:<trainjid> -J <arm> eval_clean_arm.slurm <arm> [n_mol]
set -eo pipefail
source /n/home04/yulili/bgfm/scripts/fasrc/env.sh
ARM=${1:?arm name}; NMOL=${2:-120}
# arm 可能落在 runs/p0 或 runs/clean 下, 自动定位(避免路径不匹配导致评测空跑)
RUNDIR=""
for root in "$STORE/runs/p0" "$STORE/runs/clean" "$STORE/runs/abl"; do
  [ -d "$root/$ARM" ] && { RUNDIR="$root/$ARM"; break; }
done
[ -z "$RUNDIR" ] && { echo "no run dir found for $ARM under runs/{p0,clean,abl}; exit 0"; exit 0; }
echo "rundir: $RUNDIR"
CFG="$REPO/configs/sweep/${ARM}.yaml"
export LD_LIBRARY_PATH=$STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}
export PATH=$STORE/envs/flowmol/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
echo "arm=$ARM n=$NMOL"; date

CKPT=$(ls -t "$RUNDIR"/lightning_logs/version_*/checkpoints/midstep-step=30000.ckpt 2>/dev/null | head -1)
[ -z "$CKPT" ] && CKPT=$(ls -t "$RUNDIR"/lightning_logs/version_*/checkpoints/*.ckpt 2>/dev/null | head -1)
[ -z "$CKPT" ] && { echo "no checkpoint for $ARM -- training produced none; exit 0"; exit 0; }
echo "ckpt: $CKPT"

# 权重有限性检查(教训: 曾对 NaN checkpoint 做过评测并据此误判)
"$FLOWMOL_PY" - "$CKPT" <<'PY'
import sys, torch
sd=torch.load(sys.argv[1], map_location="cpu", weights_only=False)["state_dict"]
bad=sum(int((~torch.isfinite(v)).sum()) for v in sd.values() if v.is_floating_point())
print(f"[ckpt-check] non-finite params: {bad}")
if bad: sys.exit(f"ABORT: checkpoint has {bad} non-finite params")
PY

OUT="$STORE/runs/eval_ours/clean_${ARM}"; mkdir -p "$OUT"
"$FLOWMOL_PY" scripts/eval_boltzmann_stage1.py --checkpoint "$CKPT" --config "$CFG" \
  --eval_data "$STORE/processed_data/omol25_4m_processed" \
  --n_molecules "$NMOL" --n_perturb 8 --sigma 0.15 \
  --n_ode_steps 12 --n_hutchinson 2 --device cuda --out_dir "$OUT/boltz1"
"$FLOWMOL_PY" scripts/eval_boltzmann_independent.py \
  --samples_json "$OUT/boltz1/boltzmann_samples.json" \
  --out_csv "$OUT/boltz_independent.csv" --out_json "$OUT/boltz_independent.json"
echo "## CLEAN EVAL DONE $ARM ##"; cat "$OUT/boltz_independent.json"; date
