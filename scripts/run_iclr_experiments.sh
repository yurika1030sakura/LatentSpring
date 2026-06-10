#!/bin/bash
# Full ICLR-experiment pipeline.
#
# Runs (in order): sanity -> smoke test -> E3 ablation -> E1 OOD moat ->
# E2 convergence rate -> E4 NFE sweep -> results aggregation.
# Each step emits CSVs under $PROJ/runs/iclr_$(date +%F)/.
#
# Prerequisites:
#   - scripts/install_and_process.slurm has run successfully
#   - $PROJ/envs/flowmol/bin/python exists
#   - $PROJ/data/qm9_processed/*.pt exist
#
# Usage:
#   bash scripts/run_iclr_experiments.sh         # full pipeline
#   bash scripts/run_iclr_experiments.sh sanity  # just sanity checks
#   bash scripts/run_iclr_experiments.sh E1      # just the OOD moat

set -e

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
cd "$PROJ"

source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh
conda activate "$PROJ/envs/flowmol"

STEP=${1:-all}
RUN_DIR=$PROJ/runs/iclr_$(date +%F)
mkdir -p "$RUN_DIR"
LOG=$RUN_DIR/pipeline.log
touch "$LOG"

log() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }

run_sanity() {
    log "=== sanity: unit tests + equivariance + projection sweep ==="
    PYTHONPATH=. python -m cfm_mol.test_fibre 2>&1 | tee -a "$LOG"
    PYTHONPATH=. python scripts/test_equivariance.py 2>&1 | tee -a "$LOG"
    for np in 0.02 0.05 0.10 0.20; do
        log "projection noise_p=$np"
        PYTHONPATH=. python scripts/measure_projection_failure.py \
            --mode near_feasible --noise_p $np --n_mols 2000 \
            2>&1 | grep -E "init_valence|final_both|PASS|FAIL" | tee -a "$LOG"
    done
}

run_smoke() {
    log "=== smoke_test: patched FlowMol forward + backward on real QM9 batch ==="
    PYTHONPATH=. python scripts/smoke_test.py 2>&1 | tee -a "$LOG"
}

run_calibrate() {
    log "=== calibrate d_min on QM9 train split ==="
    PYTHONPATH=. python scripts/calibrate_d_min.py \
        --dataset qm9 \
        --processed_dir "$PROJ/data/qm9_processed" \
        --max_mols 20000 \
        2>&1 | tee -a "$LOG/../calibrate_d_min.log" | tee -a "$LOG"
}

run_e3_ablation() {
    log "=== E3 ablation: {full, -tangent, -retract, -gluing} on QM9 ==="
    CONFIG="$PROJ/configs/qm9_cfm.yaml"
    for variant in full no-tangent no-retract no-gluing; do
        FLAG=""
        case "$variant" in
            full)        FLAG="" ;;
            no-tangent)  FLAG="--no-tangent" ;;
            no-retract)  FLAG="--no-retract" ;;
            no-gluing)   FLAG="--no-gluing" ;;
        esac
        RUN=$RUN_DIR/ablation_${variant}
        mkdir -p "$RUN"
        log "submitting variant=$variant  flags=$FLAG"
        sbatch --job-name="cfm_${variant}" \
               --output="$RUN/train.out" \
               --export=ALL \
               --wrap "cd $PROJ && source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh && conda activate $PROJ/envs/flowmol && python scripts/run_train.py --config $CONFIG $FLAG" \
               -p gpu --gres=gpu:nvidia_a100-sxm4-80gb:1 -N 1 -n 8 --mem=64G -t 48:00:00 \
            | tee -a "$LOG"
    done
}

run_e1_ood() {
    log "=== E1 OOD moat: tmQM (broad+narrow) + kraken + hypervalent [+radicals] ==="
    CKPT="${CKPT:-$PROJ/runs/qm9_cfm/version_0/checkpoints/last.ckpt}"
    if [ ! -f "$CKPT" ]; then
        log "[ERR] checkpoint not found: $CKPT. Set CKPT env var or train first."
        return 1
    fi
    for slice in tmqm_broad tmqm_narrow kraken hypervalent; do
        DATA=$PROJ/data/${slice}_processed
        if [ ! -f "$DATA/val_data_processed.pt" ]; then
            log "  $slice: data not ready at $DATA  (skip)"
            continue
        fi
        OUT=$RUN_DIR/e1_ood_${slice}.csv
        log "  eval on $slice ..."
        PYTHONPATH=. python scripts/evaluate_validity.py \
            --checkpoint "$CKPT" \
            --config "$PROJ/configs/qm9_cfm.yaml" \
            --n_samples 500 \
            --out "$OUT" 2>&1 | tee -a "$LOG"
    done
}

run_e2_rate() {
    log "=== E2 convergence rate: dt sweep, our FM vs reflected-SDE ==="
    log "(not yet implemented; requires trained FM model + trained reflected SDE)"
    for dt in 0.1 0.05 0.02 0.01 0.005; do
        log "  dt=$dt  (not implemented)"
    done
}

run_e4_nfe() {
    log "=== E4 NFE efficiency: NFE=10,20,50,100 ==="
    CKPT="${CKPT:-$PROJ/runs/qm9_cfm/version_0/checkpoints/last.ckpt}"
    if [ ! -f "$CKPT" ]; then
        log "[ERR] checkpoint not found: $CKPT"
        return 1
    fi
    for nfe in 10 20 50 100 200; do
        OUT=$RUN_DIR/e4_nfe_${nfe}.csv
        log "  NFE=$nfe ..."
        PYTHONPATH=. python scripts/evaluate_validity.py \
            --checkpoint "$CKPT" \
            --config "$PROJ/configs/qm9_cfm.yaml" \
            --n_samples 500 \
            --n_timesteps $nfe \
            --out "$OUT" 2>&1 | tee -a "$LOG"
    done
}

case "$STEP" in
    all)
        run_sanity
        run_smoke
        run_calibrate
        run_e3_ablation
        run_e1_ood
        run_e2_rate
        run_e4_nfe
        ;;
    sanity) run_sanity ;;
    smoke)  run_smoke  ;;
    calibrate) run_calibrate ;;
    E3|ablation) run_e3_ablation ;;
    E1|ood) run_e1_ood ;;
    E2|rate) run_e2_rate ;;
    E4|nfe) run_e4_nfe ;;
    *) log "unknown step: $STEP"; exit 2 ;;
esac

log "done. outputs in $RUN_DIR"
