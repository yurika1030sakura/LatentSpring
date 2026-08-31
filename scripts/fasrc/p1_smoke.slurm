#!/bin/bash
#SBATCH -J p1_smoke
#SBATCH -p gpu_requeue,gpu_h200
#SBATCH -A woo_lab
#SBATCH -N 1 --ntasks-per-node=1 --cpus-per-task=4
#SBATCH --gres=gpu:1 --mem=48G
#SBATCH -t 00:40:00
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/p1_smoke-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/p1_smoke-%j.err
set -eo pipefail
source /n/home04/yulili/bgfm/scripts/fasrc/env.sh
export LD_LIBRARY_PATH=$STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}
export PATH=$STORE/envs/flowmol/bin:$PATH
cd "$REPO"
CK=/n/holylabs/woo_lab/Lab/yulili/bgfm/runs/abl/a3_energy_only_s2/lightning_logs/version_30674363/checkpoints/midstep-step=30000.ckpt
OUT=$STORE/runs/eval_ours/p1_precision; mkdir -p "$OUT"
echo "host $(hostname)"; date
echo "=== SMOKE task B ==="
"$FLOWMOL_PY" scripts/p1_estimator_precision.py --task B \
  --checkpoint "$CK" --config configs/sweep/a3_energy_only.yaml \
  --eval_data "$STORE/processed_data/omol25_4m_processed" \
  --samples_json "$STORE/runs/eval_ours/wide_a3_energy_only_s2/boltz1/boltzmann_samples.json" \
  --records_csv "$STORE/runs/eval_ours/wide_a3_energy_only_s2/boltz_records.csv" \
  --disjoint_json paper/figures/out/primary_endpoint.json \
  --taskB_n_parents 3 --taskB_repeat 4 --taskB_max_atoms 20 --verify_production \
  --out_json "$OUT/smoke_B.json"
echo "=== SMOKE task A ==="
"$FLOWMOL_PY" scripts/p1_estimator_precision.py --task A \
  --checkpoint "$CK" --config configs/sweep/a3_energy_only.yaml \
  --eval_data "$STORE/processed_data/omol25_4m_processed" \
  --samples_json "$STORE/runs/eval_ours/wide_a3_energy_only_s2/boltz1/boltzmann_samples.json" \
  --records_csv "$STORE/runs/eval_ours/wide_a3_energy_only_s2/boltz_records.csv" \
  --disjoint_json paper/figures/out/primary_endpoint.json \
  --taskA_max_atoms 12 --taskA_n_parents 2 --taskA_n_geoms 2 --taskA_repeat 4 \
  --out_json "$OUT/smoke_A.json"
echo "## SMOKE DONE ##"; date
