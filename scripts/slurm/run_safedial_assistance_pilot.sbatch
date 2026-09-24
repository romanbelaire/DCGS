#!/usr/bin/env bash
#SBATCH --job-name=safedial-assistance-pilot
#SBATCH --account=pradeepresearch
#SBATCH --partition=researchshort
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/safedial-assistance-%j.out
#SBATCH --error=outputs/slurm/safedial-assistance-%j.err

set -euo pipefail
if (( $# != 0 )); then
  echo "Usage: sbatch $0 (fixed Zephyr pilot: IDs 1, 1217, 1236)" >&2
  exit 2
fi
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS
# Execute the shared environment/runner setup in this allocation. Its SBATCH
# comments are ignored by bash; there is no nested submission.
exec bash scripts/slurm/run_safedial_assistance.sbatch \
  outputs/safedial_baseline/zephyr_7b_beta_full \
  outputs/assistance/zephyr_pilot_v1 \
  --ids 1 1217 1236
