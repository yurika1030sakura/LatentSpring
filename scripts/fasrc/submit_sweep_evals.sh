#!/bin/bash
# Chain an auto-eval job to each sweep training job. Each eval runs with
# --dependency=afterany:<trainjob> (fires when training ends; the eval script
# exits 0 gracefully if no checkpoint was produced) so the ours-rows fill
# without babysitting. Looks up each training job id by name.
#
# Usage: bash scripts/fasrc/submit_sweep_evals.sh
#   (optionally pass explicit ids: name=jobid ... to override the lookup)
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NAMES="fm_only bgfm_offpolicy hbfm_onpolicy hbfm_onpolicy_energy shuffle_control"

# optional explicit overrides: name=jobid
declare -A OVERRIDE
for kv in "$@"; do OVERRIDE[${kv%%=*}]=${kv#*=}; done

for c in $NAMES; do
  jid="${OVERRIDE[$c]:-}"
  if [ -z "$jid" ]; then
    # newest pending/running training job with this exact name
    jid=$(squeue -u "$USER" -h -n "$c" -o "%i" 2>/dev/null | sort -n | tail -1)
  fi
  if [ -z "$jid" ]; then
    echo "WARN: no training job found for '$c' -- submitting eval WITHOUT dependency (runs now)."
    eid=$(sbatch --parsable -J "$c" "$HERE/eval_sweep.slurm" "$c")
  else
    eid=$(sbatch --parsable --dependency=afterany:"$jid" -J "$c" "$HERE/eval_sweep.slurm" "$c")
    echo "eval $c -> job $eid   (afterany:$jid)"
  fi
done
echo ""
echo "results land in: /n/holylabs/woo_lab/Lab/yulili/bgfm/runs/eval_sweep/<name>/"
echo "  - validity.json (Table 2)   - boltz_independent.json (Table 3, GFN2-xTB R2/ESS)"
