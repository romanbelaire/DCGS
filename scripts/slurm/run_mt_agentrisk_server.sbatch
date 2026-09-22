#!/usr/bin/env bash
#SBATCH --job-name=mt-agentrisk-server
#SBATCH --account=pradeepresearch
#SBATCH --partition=researchlong
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=outputs/slurm/mt-agentrisk-server-%j.out
#SBATCH --error=outputs/slurm/mt-agentrisk-server-%j.err
set -euo pipefail
module load Python/3.11.11-GCCcore-13.3.0
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
method="${1:-vdcgs}"
case "$method" in zephyr|vdcgs|rdcgs|smoothllm|tpo) ;; *) echo 'Unknown method' >&2; exit 2 ;; esac
if (( $# > 1 )); then echo 'Only one method argument is supported' >&2; exit 2; fi
.venv/bin/python -u -B scripts/serve_mt_agentrisk.py --method "$method" \
  --bind 0.0.0.0 --port 0 --duration 7200 --max-requests 40 \
  --output-dir "outputs/mt_agentrisk_server/$SLURM_JOB_ID"
