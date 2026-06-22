#!/usr/bin/env bash
# Submit one Darcy slice-count ablation job per M value.
#
# Usage:
#   bash scripts/submit_darcy_ablation.sh              # submits M in 128 256 512 1024
#   bash scripts/submit_darcy_ablation.sh 128 256      # custom M list
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_FILE="${SCRIPT_DIR}/darcy_ablation_slurm.sbatch"

MS=("$@")
if [ "${#MS[@]}" -eq 0 ]; then
    MS=(128 256 512 1024)
fi

mkdir -p "$(dirname "${SCRIPT_DIR}")/results_darcy_ablation"

for M in "${MS[@]}"; do
    echo "Submitting Darcy M=${M}..."
    sbatch \
        --job-name="darcy_M${M}" \
        --export=ALL,SLICE_NUM="${M}" \
        "${SBATCH_FILE}"
done
