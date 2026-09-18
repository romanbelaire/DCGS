#!/usr/bin/env bash
#SBATCH --job-name=safedial-baseline-judge
#SBATCH --account=pradeepresearch
#SBATCH --partition=researchlong
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/safedial-baseline-judge-%j.out
#SBATCH --error=outputs/slurm/safedial-baseline-judge-%j.err

set -euo pipefail

module load Python/3.11.11-GCCcore-13.3.0

cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS

.venv/bin/python scripts/test_openai_connection.py \
  --model gpt-4o-mini

.venv/bin/python scripts/judge_safedial.py \
  --answers outputs/safedial_baseline/zephyr_7b_beta_full/answers.jsonl \
  --output-dir outputs/safedial_baseline/zephyr_7b_beta_full/judgments_gpt-4o-mini \
  --judge-model gpt-4o-mini \
  --parallel 4
