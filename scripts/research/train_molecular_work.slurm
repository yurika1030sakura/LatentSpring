#!/bin/bash
#SBATCH -J bgfm-mean-work
#SBATCH -A ryl_lab
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?output required}"
MODE="${3:-joint}"
TRAIN_STEPS="${4:-100}"
TRAIN_BATCH="${5:-2}"
EVAL_PARTICLES="${6:-64}"
ORACLE_BATCH="${7:-1}"
TRAIN_SEED="${8:-9051}"
MEAN_PARAMETERIZATION="${9:-reference}"
BGFM_TARGET_KT="${10:-1.0}"
BGFM_TEMP_INITIALIZATION="${11:-unchanged}"
BGFM_SOURCE_ROW="${12:-5846}"
BGFM_NOISE_ANNEALING="${13:-0.0}"
BGFM_WORK_OBJECTIVE="${14:-mean_work}"
BGFM_GRADIENT_DIAGNOSTICS="${15:-0}"
BGFM_RESUME_RUN="${16:-}"
BGFM_PRECISION_KIND="${17:-none}"
BGFM_PRECISION_STRENGTH="${18:-32}"
BGFM_STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
export PYTHONPATH="$SOURCE_ROOT" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export LD_LIBRARY_PATH="$BGFM_STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/bgfm_mean_work_mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/bgfm_mean_work_cache_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$SOURCE_ROOT"
BGFM_TEMPERATURE_ARGS=()
if [[ -n "$BGFM_RESUME_RUN" ]]; then BGFM_TEMPERATURE_ARGS+=(--resume-work-run "$BGFM_RESUME_RUN"); fi
if [[ "$BGFM_TEMP_INITIALIZATION" == neutralize ]]; then
 BGFM_TEMPERATURE_ARGS+=(--neutralize-temperature-input)
elif [[ "$BGFM_TEMP_INITIALIZATION" != unchanged ]]; then
 echo 'Unknown temperature initialization mode' >&2
 exit 2
fi
exec "$BGFM_STORE/envs/flowmol/bin/python" -u scripts/research/train_molecular_work.py \
 --config configs/sweep/a1_fm_only_s2.yaml \
 --checkpoint /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/runs/electronic_fm_global_v1/last.ckpt \
 --metadata /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/runs/raw_metadata_replay_v1 \
 --oracle "$BGFM_STORE/checkpoints/omol25/esen_sm_conserving_all.pt" \
 --oracle-python "$BGFM_STORE/envs/omol25/bin/python" --out "$RUN_ROOT" \
 --precision-kind "$BGFM_PRECISION_KIND" --precision-strength "$BGFM_PRECISION_STRENGTH" --gradient-diagnostics "$BGFM_GRADIENT_DIAGNOSTICS" --objective "$BGFM_WORK_OBJECTIVE" --mode "$MODE" --steps "$TRAIN_STEPS" --path-steps 16 --batch "$TRAIN_BATCH" \
 --eval-particles "$EVAL_PARTICLES" --oracle-batch-size "$ORACLE_BATCH" --seed "$TRAIN_SEED" \
 --mean-parameterization "$MEAN_PARAMETERIZATION" --kT "$BGFM_TARGET_KT" --source-row "$BGFM_SOURCE_ROW" \
 --noise-annealing-power "$BGFM_NOISE_ANNEALING" \
 "${BGFM_TEMPERATURE_ARGS[@]}" --checkpoint-steps
