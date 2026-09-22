#!/usr/bin/env bash
#SBATCH --job-name=safedial-dcr-smoke
#SBATCH --account=pradeepresearch
#SBATCH --partition=researchshort
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/safedial-dcr-smoke-%j.out
#SBATCH --error=outputs/slurm/safedial-dcr-smoke-%j.err

set -euo pipefail
module load Python/3.11.11-GCCcore-13.3.0
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
output_dir="outputs/safedial_baseline/dcr_qwen_1.5b_smoke_v1"
args=(--artifact-lock configs/safedial/dcr.lock.json --ids 1 --seed 0
      --device cuda:0 --dtype bfloat16 --max-new-tokens 1024 --output-dir "$output_dir")
.venv/bin/python -u -B scripts/prepare_safedial_dcr.py --offline
.venv/bin/python -u -B scripts/run_safedial_dcr.py "${args[@]}" --validate-only
.venv/bin/python -u -B scripts/run_safedial_dcr.py "${args[@]}"
.venv/bin/python -u -B scripts/run_safedial_dcr.py \
  --output-dir "$output_dir" --audit-only --require-gpu
.venv/bin/python -u -B scripts/judge_safedial.py \
  --answers "$output_dir/answers.jsonl" --dry-run
