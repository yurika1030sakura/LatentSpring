#!/bin/bash
set -euo pipefail
SOURCE="${1:?source}"
PROJECT="${2:?project}"
RUN="${3:?run}"
OFFSET="${4:?fit offset}"
fit=$((SLURM_PROCID + OFFSET))
FLOW_PYTHON=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol/bin/python
export PYTHONPATH="$SOURCE" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol/lib:${LD_LIBRARY_PATH:-}
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/lspring_seed_worker_${SLURM_JOB_ID}_${SLURM_PROCID}"
mkdir -p "$XDG_CACHE_HOME"
cd "$SOURCE"
"$FLOW_PYTHON" -s -u -c 'import sys,numpy,torch,json; assert sys.flags.no_user_site; assert numpy.__version__.startswith("1."); assert torch.cuda.device_count()==1; print(json.dumps(dict(numpy=numpy.__version__,numpy_path=numpy.__file__,torch=torch.__version__,device=torch.cuda.get_device_name())),flush=True)'
if (( fit >= 2 )); then
 for family in fm gaga; do
  "$FLOW_PYTHON" -s -u -m scripts.research.train_seed_replication \
   --project "$PROJECT" --protocol "$SOURCE/research/evidence/seed_replication_v1.json" \
   --fit "$fit" --family "$family" --out "$RUN/training/s${fit}/${family}"
 done
 "$FLOW_PYTHON" -s -u -m scripts.research.train_hydrogen_completion \
  --project "$PROJECT" --protocol "$SOURCE/research/evidence/seed_replication_hydrogen_v1.json" \
  --seed-index "$fit" --out "$RUN/hydrogen_training/s${fit}"
fi
exec "$FLOW_PYTHON" -s -u -m scripts.research.run_seed_replication \
 --project "$PROJECT" --protocol "$SOURCE/research/evidence/seed_replication_v1.json" \
 --fit "$fit" --out "$RUN/evaluation/s${fit}"
