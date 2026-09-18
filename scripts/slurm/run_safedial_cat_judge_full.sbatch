#!/usr/bin/env bash
#SBATCH --job-name=safedial-cat-judge
#SBATCH --account=pradeepresearch
#SBATCH --partition=researchlong
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/slurm/safedial-cat-judge-%j.out
#SBATCH --error=outputs/slurm/safedial-cat-judge-%j.err

set -euo pipefail
module load Python/3.11.11-GCCcore-13.3.0
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS

# CPU-only paid judging; successful judgments resume under the same manifest.
judge_dir=outputs/safedial_baseline/cat_zephyr_full/judgments_gpt-4o-mini
mkdir -p "$judge_dir"
exec 9>"$judge_dir/.judge.lock"
if ! flock -n 9; then
  echo "Another CAT judge holds the output lock." >&2
  exit 1
fi

.venv/bin/python -u -B scripts/validate_safedial_generation.py \
  --output-dir outputs/safedial_baseline/cat_zephyr_full --require-adapter
cp outputs/safedial_baseline/cat_zephyr_full/validation.json \
  "$judge_dir/generation_validation.json"

judge_args=(
  --answers outputs/safedial_baseline/cat_zephyr_full/answers.jsonl
  --output-dir "$judge_dir"
  --judge-model gpt-4o-mini --temperature 0.7 --max-tokens 2048 --parallel 2
)
.venv/bin/python -u -B scripts/judge_safedial.py "${judge_args[@]}" --dry-run
.venv/bin/python -u -B scripts/judge_safedial.py "${judge_args[@]}"
