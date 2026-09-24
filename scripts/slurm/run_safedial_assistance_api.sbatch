#!/usr/bin/env bash
#SBATCH --job-name=safedial-assistance-api
#SBATCH --account=pradeepresearch
#SBATCH --partition=researchlong
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/safedial-assistance-api-%j.out
#SBATCH --error=outputs/slurm/safedial-assistance-api-%j.err

set -euo pipefail
if (( $# < 2 )); then
  echo "Usage: sbatch $0 RUN_DIR OUTPUT_DIR [runner options]" >&2
  exit 2
fi
source /etc/profile.d/z00_lmod.sh
module load Python/3.11.11-GCCcore-13.3.0
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS
assistance_run_dir="$1"
assistance_output_dir="$2"
shift 2
exec .venv/bin/python -u -B scripts/judge_safedial_assistance_api.py \
  --run-dir "$assistance_run_dir" --output-dir "$assistance_output_dir" \
  --parallel 8 "$@"
