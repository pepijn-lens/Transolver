#!/bin/bash
#SBATCH --job-name=darcy_ablation_small
#SBATCH --partition=gpu-a100
#SBATCH --account=education-eemcs-courses-dsait4095
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=7G
#SBATCH --chdir=/scratch/plens/FRML/Transolver/PDE-Solving-StandardBenchmark
#SBATCH --output=results_darcy_ablation/slurm-%x-%j.out
#SBATCH --error=results_darcy_ablation/slurm-%x-%j.err

# Train Transolver on Darcy for the small slice counts (M=1,8,16,32,64,96),
# matching the Elasticity ablation set. All M values run sequentially inside
# this single job. The larger M (128/256/512/1024) are handled separately by
# darcy_ablation_slurm.sbatch / submit_darcy_ablation.sh.
#
# Submit with:
#   sbatch scripts/darcy_ablation_small_slurm.sbatch
# Override the M list via env var if needed:
#   sbatch --export=ALL,SLICE_NUMS="1 8 16" scripts/darcy_ablation_small_slurm.sbatch

set -euo pipefail

# ---------------------------------------------------------------------------
# Configurable paths (override via env vars before sbatch if needed).
# ---------------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-/scratch/plens/FRML/Transolver}"
BENCH_DIR="${BENCH_DIR:-${REPO_ROOT}/PDE-Solving-StandardBenchmark}"
DATA_PATH="${DATA_PATH:-/scratch/plens/FRML/data/darcy}"
RESULTS_DIR="${RESULTS_DIR:-${BENCH_DIR}/results_darcy_ablation}"
CONDA_ENV="${CONDA_ENV:-figure5a}"
EPOCHS="${EPOCHS:-300}"
SLICE_NUMS="${SLICE_NUMS:-1 8 16 32 64 96}"

cd "${BENCH_DIR}"
mkdir -p "${RESULTS_DIR}" ./checkpoints ./logs

# ---------------------------------------------------------------------------
# Module + python environment.
# ---------------------------------------------------------------------------
module purge
module load 2024r1 || module load 2023r1 || true
module load miniconda3
module load cuda || true

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# ---------------------------------------------------------------------------
# Sanity check on data presence before launching training.
# ---------------------------------------------------------------------------
for f in \
    "${DATA_PATH}/piececonst_r421_N1024_smooth1.mat" \
    "${DATA_PATH}/piececonst_r421_N1024_smooth2.mat"; do
    if [ ! -f "${f}" ]; then
        echo "Missing data file: ${f}" >&2
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Train each M sequentially.
# ---------------------------------------------------------------------------
for SLICE_NUM in ${SLICE_NUMS}; do
    SAVE_NAME="darcy_M${SLICE_NUM}"
    echo "[$(date '+%F %T')] Training Darcy M=${SLICE_NUM} for ${EPOCHS} epochs (save_name=${SAVE_NAME})"

    python exp_darcy.py \
        --gpu 0 \
        --model Transolver_Structured_Mesh_2D \
        --n-hidden 128 \
        --n-heads 8 \
        --n-layers 8 \
        --lr 0.001 \
        --max_grad_norm 0.1 \
        --batch-size 4 \
        --slice_num "${SLICE_NUM}" \
        --unified_pos 1 \
        --ref 8 \
        --downsample 5 \
        --eval 0 \
        --epochs "${EPOCHS}" \
        --save_name "${SAVE_NAME}" \
        --data_path "${DATA_PATH}" \
        2>&1 | tee "./logs/${SAVE_NAME}.log"

    # Keep a copy of the checkpoint alongside the logs.
    cp "./checkpoints/${SAVE_NAME}.pt" "${RESULTS_DIR}/${SAVE_NAME}.pt"

    echo "[$(date '+%F %T')] Done M=${SLICE_NUM}. Log: ./logs/${SAVE_NAME}.log  Ckpt: ${RESULTS_DIR}/${SAVE_NAME}.pt"
done

echo "[$(date '+%F %T')] All small-M Darcy ablation runs complete."
