#!/usr/bin/env bash
#SBATCH --job-name=safedial-smoothllm-full
#SBATCH --account=pradeepresearch
#SBATCH --partition=researchlong
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/slurm/safedial-smoothllm-full-%j.out
#SBATCH --error=outputs/slurm/safedial-smoothllm-full-%j.err

set -euo pipefail
module load Python/3.11.11-GCCcore-13.3.0
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p outputs/safedial_baseline/smoothllm_zephyr_full
exec 9>outputs/safedial_baseline/smoothllm_zephyr_full/.generation.lock
flock -n 9 || { echo 'Another SmoothLLM full writer is active' >&2; exit 1; }

.venv/bin/python -u -B scripts/run_safedial_smoothllm.py \
  --model HuggingFaceH4/zephyr-7b-beta \
  --revision 892b3d7a7b1cf10c7a701c60881cd93df615734c \
  --model-id zephyr-7b-beta-smoothllm \
  --seed 0 --copies 8 --perturbation-percent 10 --max-new-tokens 1024 \
  --retry-errors --output-dir outputs/safedial_baseline/smoothllm_zephyr_full
.venv/bin/python -u -B scripts/validate_safedial_smoothllm.py \
  --output-dir outputs/safedial_baseline/smoothllm_zephyr_full --require-gpu
.venv/bin/python -u -B scripts/judge_safedial.py \
  --answers outputs/safedial_baseline/smoothllm_zephyr_full/answers.jsonl --dry-run
