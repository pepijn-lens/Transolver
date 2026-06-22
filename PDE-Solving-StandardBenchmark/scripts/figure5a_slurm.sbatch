#!/bin/bash
#SBATCH --job-name=transolver_fig5a
#SBATCH --partition=gpu-a100-small
#SBATCH --account=education-eemcs-courses-dsait4095
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=7G
#SBATCH --chdir=/scratch/plens/FRML/Transolver/PDE-Solving-StandardBenchmark
#SBATCH --output=results_figure5a/slurm-%j.out
#SBATCH --error=results_figure5a/slurm-%j.err

# Reproduce Figure 5(a) from Transolver: train M=64 on Elasticity, then plot
# slice weights of the last attention layer on original + resampled meshes.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configurable paths (override via env vars before sbatch if needed).
# ---------------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-/scratch/plens/FRML/Transolver}"
BENCH_DIR="${BENCH_DIR:-${REPO_ROOT}/PDE-Solving-StandardBenchmark}"
DATA_PATH="${DATA_PATH:-/scratch/plens/FRML/data}"
RESULTS_DIR="${RESULTS_DIR:-${BENCH_DIR}/results_figure5a}"
SAVE_NAME="${SAVE_NAME:-elas_figure5a}"
CONDA_ENV="${CONDA_ENV:-figure5a}"
EPOCHS="${EPOCHS:-500}"

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
    "${DATA_PATH}/elasticity/Meshes/Random_UnitCell_sigma_10.npy" \
    "${DATA_PATH}/elasticity/Meshes/Random_UnitCell_XY_10.npy"; do
    if [ ! -f "${f}" ]; then
        echo "Missing data file: ${f}" >&2
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Train.
# ---------------------------------------------------------------------------
python exp_elas.py \
    --gpu 0 \
    --model Transolver_Irregular_Mesh \
    --n-hidden 128 \
    --n-heads 8 \
    --n-layers 8 \
    --lr 0.001 \
    --max_grad_norm 0.1 \
    --batch-size 1 \
    --slice_num 64 \
    --unified_pos 0 \
    --ref 8 \
    --eval 0 \
    --epochs "${EPOCHS}" \
    --save_name "${SAVE_NAME}" \
    --data_path "${DATA_PATH}" \
    2>&1 | tee "${RESULTS_DIR}/train.log"

# Keep a copy of the checkpoint alongside the figure.
cp "./checkpoints/${SAVE_NAME}.pt" "${RESULTS_DIR}/${SAVE_NAME}.pt"

# ---------------------------------------------------------------------------
# Visualize Figure 5(a).
# ---------------------------------------------------------------------------
python visualize_figure5a.py \
    --gpu 0 \
    --checkpoint "./checkpoints/${SAVE_NAME}.pt" \
    --data_path "${DATA_PATH}" \
    --slice_num 64 \
    --n_hidden 128 \
    --n_layers 8 \
    --n_heads 8 \
    --sample_indices "${SAMPLE_INDICES:-0,1,2,3,4,5,6,7}" \
    --out_dir "${RESULTS_DIR}" \
    2>&1 | tee "${RESULTS_DIR}/visualize.log"

echo "Done. Outputs in ${RESULTS_DIR}/"
