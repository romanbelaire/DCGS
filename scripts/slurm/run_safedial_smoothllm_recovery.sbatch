#!/usr/bin/env bash
#SBATCH --job-name=safedial-smoothllm-recovery
#SBATCH --account=pradeepresearch
#SBATCH --partition=researchlong
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/safedial-smoothllm-recovery-%j.out
#SBATCH --error=outputs/slurm/safedial-smoothllm-recovery-%j.err

set -euo pipefail
module load Python/3.11.11-GCCcore-13.3.0
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

.venv/bin/python -u -B scripts/run_safedial_smoothllm_recovery.py --validate-only
.venv/bin/python -u -B scripts/run_safedial_smoothllm_recovery.py
.venv/bin/python -u -B scripts/run_safedial_smoothllm_recovery.py --audit-only --require-gpu
# A successful recovery exports only dialogue 344 (five turns). No paid API calls.
.venv/bin/python -u -B scripts/judge_safedial.py \
  --answers outputs/safedial_baseline/smoothllm_zephyr_recovery_344_turn5_v1/answers.jsonl --dry-run
