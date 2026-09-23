#!/usr/bin/env bash
#SBATCH --job-name=safedial-tpo-upstream-full
#SBATCH --account=pradeepresearch
#SBATCH --partition=researchlong
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/slurm/safedial-tpo-upstream-full-%j.out
#SBATCH --error=outputs/slurm/safedial-tpo-upstream-full-%j.err

set -euo pipefail
source /etc/profile.d/z00_lmod.sh
module load Python/3.11.11-GCCcore-13.3.0
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

# All 2037 dialogues / 10029 turns; same policy/settings as passed L40 smoke257858.
# Fresh outputs. Do not point this at v8 or import v8 turns.
# Resubmit this launcher to resume saved events after a time limit.
# Final judge command is input validation only; no paid API judging.
output_dir=outputs/safedial_baseline/tpo_zephyr_upstream_full_v1
# Always audit terminal failures after all turns have been processed.
if .venv/bin/python -u -B scripts/run_safedial_tpo_upstream_full.py \
  --artifact-lock configs/safedial/tpo.lock.json \
  --empty-generation-retries 0 --seed 0 --sample-size 5 --max-iters 2 \
  --max-new-tokens 1024 --feedback-tokens 2048 --output-dir "$output_dir"; then
  generation_status=0
else
  generation_status=$?
fi
if .venv/bin/python -u -B scripts/validate_safedial_tpo_upstream.py \
  --output-dir "$output_dir" --require-gpu; then
  validation_status=0
else
  validation_status=$?
fi
if (( generation_status != 0 )); then exit "$generation_status"; fi
if (( validation_status != 0 )); then exit "$validation_status"; fi
.venv/bin/python -u -B scripts/judge_safedial.py \
  --answers "$output_dir/answers.jsonl" --dry-run
