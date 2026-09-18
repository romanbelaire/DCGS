#!/usr/bin/env bash
#SBATCH --job-name=safedial-cat-full
#SBATCH --account=pradeepresearch
#SBATCH --partition=researchlong
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/slurm/safedial-cat-full-%j.out
#SBATCH --error=outputs/slurm/safedial-cat-full-%j.err

# User explicitly approved skipping the 12-dialogue check and 120-dialogue
# pilot after smoke 245201 and its five judge calls passed. Generation only.
set -euo pipefail
module load Python/3.11.11-GCCcore-13.3.0
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS

export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# Prevent overlapping invocations from writing the same resumable artifacts.
mkdir -p outputs/safedial_baseline/cat_zephyr_full
exec 9>outputs/safedial_baseline/cat_zephyr_full/.generation.lock
if ! flock -n 9; then
  echo "Another CAT full-run writer already holds the output lock." >&2
  exit 1
fi

.venv/bin/python -u -B scripts/prepare_safedial_cat.py --offline
.venv/bin/python -u -B scripts/run_safedial_baseline.py \
  --model HuggingFaceH4/zephyr-7b-beta \
  --revision 892b3d7a7b1cf10c7a701c60881cd93df615734c \
  --model-id zephyr-7b-beta-cat \
  --adapter-lock configs/safedial/cat.lock.json \
  --verify-adapter \
  --seed 0 --temperature 0.7 --top-p 1.0 --max-new-tokens 1024 \
  --retry-errors \
  --output-dir outputs/safedial_baseline/cat_zephyr_full
.venv/bin/python -u -B scripts/validate_safedial_generation.py \
  --output-dir outputs/safedial_baseline/cat_zephyr_full --require-adapter
.venv/bin/python -u -B scripts/judge_safedial.py \
  --answers outputs/safedial_baseline/cat_zephyr_full/answers.jsonl --dry-run
