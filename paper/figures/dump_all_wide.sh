#!/bin/bash
# Dump per-perturbation (log p, E_xTB) records for every n=120 arm.
# Needed for (a) real Figure 2 scatter points and (b) the pert_id != 0
# robustness recomputation (excluding the unperturbed reference geometry).
set -u
PY=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol/bin/python
export PATH=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol/bin:$PATH
cd /n/home04/yulili/bgfm
# wide_a1_fm_only (no suffix) is the fifth control seed, scored under the same
# protocol as the other four after the fact; it is dumped like any other arm.
for tag in wide_a3_energy_only_s1 wide_a1_fm_only_s2 wide_a3_energy_only_s2 \
           wide_a3_energy_only_s3 wide_a3_energy_only_s5 wide_a1_fm_only_s3 \
           wide_a1_fm_only_s4 wide_a1_fm_only_s5 wide_a1_fm_only \
           wide_a6_energy_only_shuffled_s2 \
           wide_a6_energy_only_shuffled_s3 wide_a6_energy_only_shuffled_stab_s5; do
  d=runs/eval_ours/$tag
  [ -f "$d/boltz_records.csv" ] && { echo "SKIP $tag"; continue; }
  echo "=== $tag $(date +%H:%M:%S)"
  $PY paper/figures/dump_boltz_records.py --run_dir "$d" 2>&1 | tail -2
done
echo "ALL DONE $(date)"
