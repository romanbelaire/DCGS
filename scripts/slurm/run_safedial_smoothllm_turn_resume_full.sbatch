#!/usr/bin/env bash
#SBATCH --job-name=safedial-smoothllm-turn-resume
#SBATCH --account=pradeepresearch
#SBATCH --partition=researchlong
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/slurm/safedial-smoothllm-turn-resume-%j.out
#SBATCH --error=outputs/slurm/safedial-smoothllm-turn-resume-%j.err

set -euo pipefail
module load Python/3.11.11-GCCcore-13.3.0
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# Runner holds its target lock and takes the legacy lock during one-time import.
# Submit only after the current legacy SmoothLLM job has finished.
.venv/bin/python -u -B scripts/run_safedial_smoothllm_turn_resume.py \
  --model HuggingFaceH4/zephyr-7b-beta \
  --revision 892b3d7a7b1cf10c7a701c60881cd93df615734c \
  --model-id zephyr-7b-beta-smoothllm \
  --seed 0 --copies 8 --perturbation-percent 10 --max-new-tokens 1024 \
  --resume-from outputs/safedial_baseline/smoothllm_zephyr_full \
  --retry-errors --output-dir outputs/safedial_baseline/smoothllm_zephyr_full_turn_resume
.venv/bin/python -u -B scripts/validate_safedial_smoothllm_turn_resume.py \
  --output-dir outputs/safedial_baseline/smoothllm_zephyr_full_turn_resume --require-gpu
.venv/bin/python -u -B scripts/judge_safedial.py \
  --answers outputs/safedial_baseline/smoothllm_zephyr_full_turn_resume/answers.jsonl --dry-run
