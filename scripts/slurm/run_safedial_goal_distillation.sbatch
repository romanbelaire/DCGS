#!/usr/bin/env bash
#SBATCH --job-name=safedial-goal-distillation
#SBATCH --account=pradeepresearch
#SBATCH --partition=researchlong
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/safedial-goals-%j.out
#SBATCH --error=outputs/slurm/safedial-goals-%j.err

set -euo pipefail
source /etc/profile.d/z00_lmod.sh
module load Python/3.11.11-GCCcore-13.3.0
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS

# CPU-only; user submission makes paid API calls. Pass --ids/--output-dir for a pilot.
.venv/bin/python -u -B scripts/distill_safedial_goals.py "$@" --dry-run
.venv/bin/python -u -B scripts/distill_safedial_goals.py "$@"
