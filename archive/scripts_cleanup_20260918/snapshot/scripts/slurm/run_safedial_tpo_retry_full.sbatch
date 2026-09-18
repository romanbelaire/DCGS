#!/usr/bin/env bash
#SBATCH --job-name=safedial-tpo-retry-full
#SBATCH --account=pradeepresearch
#SBATCH --partition=researchlong
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/slurm/safedial-tpo-retry-full-%j.out
#SBATCH --error=outputs/slurm/safedial-tpo-retry-full-%j.err

set -euo pipefail
module load Python/3.11.11-GCCcore-13.3.0
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

# All 2037 dialogues. Runner owns the output lock and resumes saved events.
# Uses the pinned RM's native 8192 positions, with an explicit startup GPU probe.
# At most two new-seed retries for empty generations; other errors stop.
.venv/bin/python -u -B scripts/run_safedial_tpo_full.py \
  --artifact-lock configs/safedial/tpo.lock.json \
  --empty-generation-retries 2 --seed 0 --sample-size 5 --max-iters 2 \
  --max-new-tokens 1024 --feedback-tokens 2048 \
  --output-dir outputs/safedial_baseline/tpo_zephyr_full_v8
.venv/bin/python -u -B scripts/validate_safedial_tpo_full.py \
  --output-dir outputs/safedial_baseline/tpo_zephyr_full_v8 --require-gpu
.venv/bin/python -u -B scripts/judge_safedial.py \
  --answers outputs/safedial_baseline/tpo_zephyr_full_v8/answers.jsonl --dry-run
