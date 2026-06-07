#!/bin/bash
#SBATCH --job-name=transolver_aircraft
#SBATCH --partition=gpu-a100-small
#SBATCH --account=education-eemcs-courses-dsait4095
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --chdir=/scratch/plens/FRML/Transolver/Aircraft-Design
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Train Transolver on the aircraft surface dataset.
# On first run this also builds the .npz cache from the raw Tecplot meshes
# (CPU, ~5 min, idempotent: cached cases are skipped on subsequent runs).
#
# Resumable: a checkpoint (model + optimizer + scheduler + epoch) is written
# every epoch. If this job times out or is killed, just re-submit the same
# sbatch -- main.py auto-resumes from ./checkpoints/<save_name>.pt (--resume 1).

set -euo pipefail

# ---------------------------------------------------------------------------
# Configurable paths (override via env vars before sbatch if needed).
# ---------------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-/scratch/plens/FRML/Transolver}"
AIR_DIR="${AIR_DIR:-${REPO_ROOT}/Aircraft-Design}"
DATA_DIR="${DATA_DIR:-/scratch/plens/FRML/data/aircraft}"
CACHE_DIR="${CACHE_DIR:-/scratch/plens/FRML/data/aircraft_cache}"
SAVE_NAME="${SAVE_NAME:-aircraft_Transolver}"
CONDA_ENV="${CONDA_ENV:-figure5a}"
EPOCHS="${EPOCHS:-200}"
N_TEST_GEOM="${N_TEST_GEOM:-6}"
SEED="${SEED:-0}"

cd "${AIR_DIR}"
mkdir -p logs ./checkpoints "${CACHE_DIR}"

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
# Sanity check on data presence.
# ---------------------------------------------------------------------------
if [ ! -d "${DATA_DIR}" ]; then
    echo "Missing data dir: ${DATA_DIR}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Build the .npz cache if needed (idempotent; skips already-cached cases).
# ---------------------------------------------------------------------------
python dataset/preprocess.py \
    --data_dir "${DATA_DIR}" \
    --save_dir "${CACHE_DIR}" \
    --n_test_geom "${N_TEST_GEOM}" \
    --seed "${SEED}" \
    2>&1 | tee "logs/preprocess.log"

# ---------------------------------------------------------------------------
# Train.
# ---------------------------------------------------------------------------
python main.py \
    --gpu 0 \
    --save_dir "${CACHE_DIR}" \
    --n_hidden 256 \
    --n_layers 8 \
    --n_heads 8 \
    --mlp_ratio 2 \
    --slice_num 32 \
    --lr 0.001 \
    --max_grad_norm 0.1 \
    --epochs "${EPOCHS}" \
    --val_iter 10 \
    --eval 0 \
    --save_name "${SAVE_NAME}" \
    2>&1 | tee "logs/train.log"

echo "Done. Checkpoint: ./checkpoints/${SAVE_NAME}.pt"
