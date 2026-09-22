#!/usr/bin/env bash
#SBATCH --job-name=dcgs-api-smoke
#SBATCH --account=pradeepresearch
#SBATCH --partition=researchlong
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/dcgs-api-smoke-%j.out
#SBATCH --error=outputs/slurm/dcgs-api-smoke-%j.err

set -euo pipefail
module load Python/3.11.11-GCCcore-13.3.0
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
method="${1:-vdcgs}"
if (( $# > 1 )) || [[ "$method" != vdcgs && "$method" != rdcgs ]]; then
  echo 'Usage: sbatch scripts/slurm/run_dcgs_api_smoke.sbatch [vdcgs|rdcgs]' >&2
  exit 2
fi
.venv/bin/python -u -B scripts/run_dcgs_api_smoke.py \
  --method "$method" --bind 0.0.0.0 --port 0 \
  --duration 1800 --max-requests 3 \
  --output-dir "outputs/dcgs_api_smoke/$SLURM_JOB_ID"
