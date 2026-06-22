#!/usr/bin/env bash
# Run Table 4 training for one benchmark on two GPUs in parallel.
#
# Usage:
#   export ELAS_PATH=/kaggle/input/.../elasticity-20260508T172658Z-3-001
#   export DARCY_PATH=/kaggle/input/.../Darcy_421
#   bash run_table4.sh elas  "1 16 64 96 512"    "8 32 96 256 1024"
#   bash run_table4.sh darcy "1 16 64 256 1024"  "8 32 96 128 512"
#
# GPU0 and GPU1 each run their M-value list sequentially.
# Both GPUs run in parallel (suited for Kaggle T4 x2).

set -euo pipefail

TASK=${1:?Usage: run_table4.sh <elas|darcy> <GPU0_SLICES> <GPU1_SLICES>}
GPU0_SLICES=($2)
GPU1_SLICES=($3)

EPOCHS=300
mkdir -p logs checkpoints results

# ---- per-config training command ----
run_config() {
    local GPU=$1
    local M=$2

    echo "[$(date '+%H:%M:%S')] Starting ${TASK} M=${M} on GPU${GPU}"

    if [[ "$TASK" == "elas" ]]; then
        python exp_elas.py \
            --gpu "$GPU" \
            --model Transolver_Irregular_Mesh \
            --n-hidden 128 --n-heads 8 --n-layers 8 \
            --lr 0.001 --max_grad_norm 0.1 \
            --batch-size 1 \
            --slice_num "$M" \
            --epochs "$EPOCHS" \
            --unified_pos 0 \
            --data_path "${ELAS_PATH:?Set ELAS_PATH}" \
            --save_name "elas_M${M}" \
            2>&1 | tee "logs/elas_M${M}.log"
    else
        python exp_darcy.py \
            --gpu "$GPU" \
            --model Transolver_Structured_Mesh_2D \
            --n-hidden 128 --n-heads 8 --n-layers 8 \
            --lr 0.001 --max_grad_norm 0.1 \
            --batch-size 4 \
            --slice_num "$M" \
            --epochs "$EPOCHS" \
            --unified_pos 1 --ref 8 --downsample 5 \
            --data_path "${DARCY_PATH:?Set DARCY_PATH}" \
            --save_name "darcy_M${M}" \
            2>&1 | tee "logs/darcy_M${M}.log"
    fi

    echo "[$(date '+%H:%M:%S')] Finished ${TASK} M=${M} on GPU${GPU}"
}

# ---- sequential runner for one GPU ----
run_gpu() {
    local GPU=$1; shift
    for M in "$@"; do
        run_config "$GPU" "$M"
    done
}

# Launch both GPUs in parallel; each runs its list sequentially
run_gpu 0 "${GPU0_SLICES[@]}" &
PID0=$!
run_gpu 1 "${GPU1_SLICES[@]}" &
PID1=$!

wait $PID0 $PID1
echo "All ${TASK} runs complete."
