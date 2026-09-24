#!/usr/bin/env bash
#SBATCH --job-name=safedial-assistance
#SBATCH --account=pradeepresearch
#SBATCH --partition=researchlong
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/safedial-assistance-%j.out
#SBATCH --error=outputs/slurm/safedial-assistance-%j.err

set -euo pipefail
if (( $# < 2 )); then
  echo "Usage: sbatch $0 RUN_DIR JUDGE_OUTPUT_DIR [runner options]" >&2
  exit 2
fi
source /etc/profile.d/z00_lmod.sh
module load Python/3.11.11-GCCcore-13.3.0
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
assistance_run_dir="$1"
assistance_output_dir="$2"
shift 2
.venv/bin/python -u -B scripts/judge_safedial_assistance.py \
  --run-dir "$assistance_run_dir" --output-dir "$assistance_output_dir" "$@"
