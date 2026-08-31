#!/bin/bash
#SBATCH -J res_abl
#SBATCH -p gpu_requeue,gpu_h200
#SBATCH -A woo_lab
#SBATCH -N 1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h200:1 --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/res_abl-%x-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/res_abl-%x-%j.err
#
# 分辨率敏感性 ablation。纯评测: 复用已有 checkpoint, 不重训, 不改 bgfm_density.py。
#
# 背景: FFJORD 的积分网格 t_k = 1-(k+1/2)/n 把等效平滑尺度 eps 与步数 n 绑死(eps=1/(2n)),
# 而训练时 energy_n_ode_steps=4 (eps=0.125), 评测 headline 用 12 步 (eps=0.042)。
# 一个 10 分子/1 种子的初步检查暗示效应可能随分辨率衰减, 但功效不足以下结论。
# 本 ablation 用与主结果同等功效(120 分子 x 4-5 种子)测量该依赖关系。
#
# 关键: 固定 --seed 使各分辨率使用【完全相同的分子与扰动】-> 配对比较。
#
# 用法: sbatch -J <tag> eval_resolution_ablation.slurm <ckpt> <config> <tag> <n_ode_steps>
set -eo pipefail
source /n/home04/yulili/bgfm/scripts/fasrc/env.sh
CKPT=${1:?ckpt}; CFG=${2:?config}; TAG=${3:?tag}; NODE=${4:?n_ode_steps}
EVAL_SEED=12345          # 固定 -> 跨分辨率配对
OUT="$STORE/runs/eval_res/${TAG}__ode${NODE}"; mkdir -p "$OUT"
export LD_LIBRARY_PATH=$STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}
export PATH=$STORE/envs/flowmol/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
echo "tag=$TAG n_ode_steps=$NODE eval_seed=$EVAL_SEED"; date

# 权重有限性检查(教训: 曾对 NaN checkpoint 评测并据此误判)
"$FLOWMOL_PY" - "$CKPT" <<'PY'
import sys, torch
sd=torch.load(sys.argv[1], map_location="cpu", weights_only=False)["state_dict"]
bad=sum(int((~torch.isfinite(v)).sum()) for v in sd.values() if v.is_floating_point())
print(f"[ckpt-check] non-finite params: {bad}")
if bad: sys.exit(f"ABORT: {bad} non-finite params")
PY

"$FLOWMOL_PY" scripts/eval_boltzmann_stage1.py --checkpoint "$CKPT" --config "$CFG" \
  --eval_data "$STORE/processed_data/omol25_4m_processed" \
  --n_molecules 120 --n_perturb 8 --sigma 0.15 --seed "$EVAL_SEED" \
  --n_ode_steps "$NODE" --n_hutchinson 2 --device cuda --out_dir "$OUT/boltz1"

"$FLOWMOL_PY" scripts/eval_boltzmann_independent.py \
  --samples_json "$OUT/boltz1/boltzmann_samples.json" \
  --out_csv "$OUT/boltz_independent.csv" --out_json "$OUT/boltz_independent.json"
echo "## RES-ABL DONE $TAG ode=$NODE ##"; date
