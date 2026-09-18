#!/usr/bin/env bash
#SBATCH --job-name=safedial-rdcgs-full
#SBATCH --account=pradeepresearch
#SBATCH --partition=researchlong
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/slurm/safedial-rdcgs-full-%j.out
#SBATCH --error=outputs/slurm/safedial-rdcgs-full-%j.err

set -euo pipefail
module load Python/3.11.11-GCCcore-13.3.0
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

.venv/bin/python -u -B scripts/run_safedial_dcgs_methods.py \
  --method rdcgs --seed 0 --device cuda:0 \
  --artifact-lock configs/safedial/dcgs_two_stage_aug11.lock.json \
  --output-dir outputs/safedial_dcgs/zephyr_rdcgs_aug11_full
.venv/bin/python -u -B scripts/validate_safedial_dcgs_methods.py \
  --output-dir outputs/safedial_dcgs/zephyr_rdcgs_aug11_full --require-gpu
.venv/bin/python -u -B scripts/judge_safedial.py \
  --answers outputs/safedial_dcgs/zephyr_rdcgs_aug11_full/answers.jsonl --dry-run
