#!/usr/bin/env bash
#SBATCH --job-name=safedial-smoothllm-judge
#SBATCH --account=pradeepresearch
#SBATCH --partition=researchlong
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/slurm/safedial-smoothllm-judge-%j.out
#SBATCH --error=outputs/slurm/safedial-smoothllm-judge-%j.err

set -euo pipefail
module load Python/3.11.11-GCCcore-13.3.0
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS

# CPU-only paid API judging. Freeze an audited subset; retain failed coverage.
source_dir=outputs/safedial_baseline/smoothllm_zephyr_full_turn_resume
prepared_dir="$source_dir/judging_complete_dialogues_v1"
judge_dir="$prepared_dir/judgments_gpt-4o-mini"
mkdir -p "$judge_dir"
exec 9>"$judge_dir/.judge.lock"
if ! flock -n 9; then
  echo "Another SmoothLLM judge holds the output lock." >&2
  exit 1
fi

.venv/bin/python -u -B scripts/prepare_safedial_smoothllm_judging.py \
  --source-dir "$source_dir" --output-dir "$prepared_dir" \
  --dataset benchmark-artifacts/SafeDialBench-Dataset/data/complete/datasets_en.jsonl \
  --allow-failed-dialogue-ids 344

judge_args=(
  --answers "$prepared_dir/answers.jsonl"
  --output-dir "$judge_dir"
  --judge-model gpt-4o-mini --temperature 0.7 --max-tokens 2048 --parallel 2
)
.venv/bin/python -u -B scripts/judge_safedial.py "${judge_args[@]}" --dry-run
.venv/bin/python -u -B scripts/judge_safedial.py "${judge_args[@]}"
