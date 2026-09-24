#!/usr/bin/env bash
#SBATCH --job-name=safedial-vdcgs96-judge
#SBATCH --account=pradeepresearch
#SBATCH --partition=researchlong
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/safedial-vdcgs96-judge-%j.out
#SBATCH --error=outputs/slurm/safedial-vdcgs96-judge-%j.err

set -euo pipefail
source /etc/profile.d/z00_lmod.sh
module load Python/3.11.11-GCCcore-13.3.0
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS

# Native SafeDial scoring of VDCGS-96's final available answer export:
# 2,011 complete dialogues / 9,899 turns; excludes failed dialogues.
# CPU-only; makes paid GPT-4o-mini API calls. Identical submissions resume.
# Earlier frozen partial-judging caches are not imported into this scope.
run_dir=outputs/safedial_dcgs/zephyr_vdcgs_main_wildjailbreak_full_v3
judge_dir="$run_dir/judgments_gpt-4o-mini"
mkdir -p "$judge_dir"
exec 9>"$judge_dir/.judge.lock"
if ! flock -n 9; then
  echo "Another VDCGS-96 judge holds the output lock." >&2
  exit 1
fi
judge_args=(
  --answers "$run_dir/answers.jsonl"
  --output-dir "$judge_dir"
  --judge-model gpt-4o-mini --parallel 4
)
.venv/bin/python -u -B scripts/judge_safedial.py "${judge_args[@]}" --dry-run
.venv/bin/python -u -B scripts/judge_safedial.py "${judge_args[@]}"
