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
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  # Slurm executes a spool copy of this file, so BASH_SOURCE is not the checkout.
  repo_dir="${SLURM_SUBMIT_DIR:-$PWD}"
  if (( $# == 0 )); then
    set -- --source-dir outputs/safedial_baseline/smoothllm_zephyr_full_turn_resume
  fi
else
  repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [[ ! -f "$repo_dir/scripts/run_safedial_smoothllm_judging.py" ]]; then
  echo "Run/submit this launcher from the DCGS repository checkout." >&2
  exit 2
fi
cd "$repo_dir"

# Cluster modules are optional outside Slurm; Git Bash needs no module command.
if command -v module >/dev/null 2>&1; then
  module load Python/3.11.11-GCCcore-13.3.0
fi
if [[ -x .venv/bin/python ]]; then
  judge_python=.venv/bin/python
elif [[ -x .venv/Scripts/python.exe ]]; then
  judge_python=.venv/Scripts/python.exe
elif command -v python >/dev/null 2>&1; then
  judge_python=python
else
  judge_python=python3
fi

# Python owns a portable lock and chooses live Linux output or the saved snapshot.
exec "$judge_python" -u -B scripts/run_safedial_smoothllm_judging.py "$@"
