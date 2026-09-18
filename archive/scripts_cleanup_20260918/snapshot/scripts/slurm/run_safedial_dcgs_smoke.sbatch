#!/usr/bin/env bash
#SBATCH --job-name=safedial-dcgs-smoke
#SBATCH --account=pradeepresearch
#SBATCH --partition=researchshort
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/safedial-dcgs-smoke-%j.out
#SBATCH --error=outputs/slurm/safedial-dcgs-smoke-%j.err

set -euo pipefail

module load Python/3.11.11-GCCcore-13.3.0

cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

.venv/bin/python scripts/run_safedial_dcgs.py \
  --ids 1 \
  --output-dir outputs/safedial_dcgs/zephyr_7b_beta_wildjailbreak_critic_smoke
