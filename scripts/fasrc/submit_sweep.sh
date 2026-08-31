#!/bin/bash
# Submit the full capped-step sweep (5 jobs, one per config) to gpu_h200 / woo_lab.
# Each is a 1-GPU job; on a 4-GPU node up to 4 run in parallel. On-policy jobs
# self-start their eSEN teacher on distinct ports (28901/28902) so co-located jobs
# don't collide. Run from the repo root: bash scripts/fasrc/submit_sweep.sh
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
for c in fm_only bgfm_offpolicy hbfm_onpolicy hbfm_onpolicy_energy shuffle_control; do
  jid=$(sbatch --parsable -J "$c" "$HERE/sweep.slurm" "$c")
  echo "submitted $c -> job $jid"
done
echo ""
echo "watch:   squeue -u \$USER"
echo "logs:    /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/sweep-<name>-<jobid>.{out,err}"
echo "when done, eval each: sbatch scripts/fasrc/eval_h200.slurm <ckpt> configs/sweep/<name>.yaml <name>"
