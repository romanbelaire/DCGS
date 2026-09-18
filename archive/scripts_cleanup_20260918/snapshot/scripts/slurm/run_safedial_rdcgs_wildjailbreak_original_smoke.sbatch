#!/usr/bin/env bash
#SBATCH --job-name=safedial-rdcgs-original-smoke
#SBATCH --account=pradeepresearch
#SBATCH --partition=researchshort
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/safedial-rdcgs-original-smoke-%j.out
#SBATCH --error=outputs/slurm/safedial-rdcgs-original-smoke-%j.err

set -euo pipefail
module load Python/3.11.11-GCCcore-13.3.0
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

.venv/bin/python -u -B scripts/run_safedial_dcgs_wildjailbreak.py \
  --method rdcgs --ids 1 --seed 0 --device cuda:0 --validate-only \
  --output-dir outputs/safedial_dcgs/zephyr_rdcgs_original_wildjailbreak_smoke
.venv/bin/python -u -B scripts/run_safedial_dcgs_wildjailbreak.py \
  --method rdcgs --ids 1 --seed 0 --device cuda:0 \
  --output-dir outputs/safedial_dcgs/zephyr_rdcgs_original_wildjailbreak_smoke
.venv/bin/python -u -B scripts/validate_safedial_dcgs_wildjailbreak.py \
  --output-dir outputs/safedial_dcgs/zephyr_rdcgs_original_wildjailbreak_smoke --require-gpu
.venv/bin/python -u -B scripts/judge_safedial.py \
  --answers outputs/safedial_dcgs/zephyr_rdcgs_original_wildjailbreak_smoke/answers.jsonl --dry-run
