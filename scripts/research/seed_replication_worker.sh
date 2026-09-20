#!/bin/bash
set -euo pipefail
SOURCE="${1:?source}"
PROJECT="${2:?project}"
RUN="${3:?run}"
OFFSET="${4:?fit offset}"
fit=$((SLURM_PROCID + OFFSET))
FLOW_PYTHON=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol/bin/python
cd "$SOURCE"
if (( fit >= 2 )); then
 for family in fm gaga; do
  "$FLOW_PYTHON" -u -m scripts.research.train_seed_replication \
   --project "$PROJECT" --protocol "$SOURCE/research/evidence/seed_replication_v1.json" \
   --fit "$fit" --family "$family" --out "$RUN/training/s${fit}/${family}"
 done
 "$FLOW_PYTHON" -u -m scripts.research.train_hydrogen_completion \
  --project "$PROJECT" --protocol "$SOURCE/research/evidence/seed_replication_hydrogen_v1.json" \
  --seed-index "$fit" --out "$RUN/hydrogen_training/s${fit}"
fi
exec "$FLOW_PYTHON" -u -m scripts.research.run_seed_replication \
 --project "$PROJECT" --protocol "$SOURCE/research/evidence/seed_replication_v1.json" \
 --fit "$fit" --out "$RUN/evaluation/s${fit}"
