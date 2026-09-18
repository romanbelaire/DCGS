#!/usr/bin/env bash
#SBATCH --job-name=safedial-gpt4o-full
#SBATCH --account=pradeepresearch
#SBATCH --partition=researchlong
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/slurm/safedial-gpt4o-full-%j.out
#SBATCH --error=outputs/slurm/safedial-gpt4o-full-%j.err

set -euo pipefail
module load Python/3.11.11-GCCcore-13.3.0
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS

# CPU-only API generation; credentials loaded from .env without shell evaluation.
# Runner checks exact model access and locks outputs; saved turns are reused.
.venv/bin/python -u -B scripts/run_safedial_api.py \
  --model gpt-4o-2024-08-06 --model-id gpt-4o-baseline \
  --parallel 2 --seed 0 --temperature 0.7 --top-p 1.0 --max-new-tokens 1024 \
  --output-dir outputs/safedial_baseline/gpt4o_full
.venv/bin/python -B scripts/run_safedial_api.py \
  --output-dir outputs/safedial_baseline/gpt4o_full --validate-output
# Only fully successful dialogues enter answers.jsonl. Validation separately
# reports filtered turns and keeps benchmark completeness false when any remain.
.venv/bin/python -B scripts/judge_safedial.py \
  --answers outputs/safedial_baseline/gpt4o_full/answers.jsonl --dry-run
