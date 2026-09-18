#!/usr/bin/env bash
#SBATCH --job-name=safedial-dcgs-two-stage-smoke
#SBATCH --account=pradeepresearch
#SBATCH --partition=researchshort
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/safedial-dcgs-two-stage-smoke-%j.out
#SBATCH --error=outputs/slurm/safedial-dcgs-two-stage-smoke-%j.err

set -euo pipefail
module load Python/3.11.11-GCCcore-13.3.0
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

.venv/bin/python -u -B scripts/run_safedial_dcgs_two_stage.py \
  --ids 1 --seed 0 --device cuda:0 \
  --artifact-lock configs/safedial/dcgs_two_stage.lock.json \
  --output-dir outputs/safedial_dcgs/zephyr_wildjailbreak_two_stage_smoke
.venv/bin/python -u -B scripts/validate_safedial_dcgs_two_stage.py \
  --output-dir outputs/safedial_dcgs/zephyr_wildjailbreak_two_stage_smoke --require-gpu
.venv/bin/python -u -B scripts/judge_safedial.py \
  --answers outputs/safedial_dcgs/zephyr_wildjailbreak_two_stage_smoke/answers.jsonl --dry-run
