#!/usr/bin/env bash
#SBATCH --job-name=safedial-tpo-smoke
#SBATCH --account=pradeepresearch
#SBATCH --partition=researchshort
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm/safedial-tpo-smoke-%j.out
#SBATCH --error=outputs/slurm/safedial-tpo-smoke-%j.err

set -euo pipefail
module load Python/3.11.11-GCCcore-13.3.0
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

# V3 continues the audited calls from 249899; original V1/V2 output is preserved.
# The runner owns the nonblocking output lock and validates both artifact hashes.
.venv/bin/python -u -B scripts/run_safedial_tpo.py \
  --artifact-lock configs/safedial/tpo.lock.json \
  --ids 1 --seed 0 --sample-size 5 --max-iters 2 \
  --max-new-tokens 1024 --feedback-tokens 2048 \
  --output-dir outputs/safedial_baseline/tpo_zephyr_smoke_v3
.venv/bin/python -u -B scripts/validate_safedial_tpo.py \
  --output-dir outputs/safedial_baseline/tpo_zephyr_smoke_v3 --require-gpu
.venv/bin/python -u -B scripts/judge_safedial.py \
  --answers outputs/safedial_baseline/tpo_zephyr_smoke_v3/answers.jsonl --dry-run
