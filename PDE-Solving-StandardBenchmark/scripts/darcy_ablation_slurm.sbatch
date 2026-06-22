#!/bin/bash
#SBATCH --job-name=transolver_darcy_ablation
#SBATCH --partition=gpu-a100
#SBATCH --account=education-eemcs-courses-dsait4095
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=7G
#SBATCH --chdir=/scratch/plens/FRML/Transolver/PDE-Solving-StandardBenchmark
#SBATCH --output=results_darcy_ablation/slurm-M%x-%j.out
#SBATCH --error=results_darcy_ablation/slurm-M%x-%j.err

# Train Transolver on Darcy for a single slice-count M (Table 4 ablation).
# Submit one job per M via:
#   sbatch --job-name=darcy_M128  --export=ALL,SLICE_NUM=128  scripts/darcy_ablation_slurm.sbatch
# or use scripts/submit_darcy_ablation.sh to fan out all four configs.

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
SLICE_NUM="${SLICE_NUM:?Set SLICE_NUM (e.g. --export=ALL,SLICE_NUM=128)}"
SAVE_NAME="${SAVE_NAME:-darcy_M${SLICE_NUM}}"

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
# Train.
# ---------------------------------------------------------------------------
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

echo "[$(date '+%F %T')] Done. Log: ./logs/${SAVE_NAME}.log  Ckpt: ${RESULTS_DIR}/${SAVE_NAME}.pt"
