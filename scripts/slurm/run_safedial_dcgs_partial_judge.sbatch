#!/usr/bin/env bash
#SBATCH --job-name=safedial-dcgs-partial-judge
#SBATCH --account=pradeepresearch
#SBATCH --partition=researchlong
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/safedial-dcgs-partial-judge-%j.out
#SBATCH --error=outputs/slurm/safedial-dcgs-partial-judge-%j.err
set -euo pipefail
module load Python/3.11.11-GCCcore-13.3.0
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS
if (( $# != 1 )); then echo 'Pass one prepared snapshot directory' >&2; exit 2; fi
.venv/bin/python -u -B scripts/judge_safedial_partial.py run --snapshot "$1"
