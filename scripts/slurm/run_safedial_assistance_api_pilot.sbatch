#!/usr/bin/env bash
#SBATCH --job-name=safedial-assistance-api-pilot
#SBATCH --account=pradeepresearch
#SBATCH --partition=researchshort
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/safedial-assistance-api-%j.out
#SBATCH --error=outputs/slurm/safedial-assistance-api-%j.err

set -euo pipefail
if (( $# != 0 )); then
  echo "Usage: sbatch $0 (GPT-4o-mini pilot: IDs 1, 1217, 1236; eight workers)" >&2
  exit 2
fi
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS
exec bash scripts/slurm/run_safedial_assistance_api.sbatch \
  outputs/safedial_baseline/zephyr_7b_beta_full \
  outputs/assistance/gpt4o_mini_zephyr_pilot_v2 \
  --ids 1 1217 1236
