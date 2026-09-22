#!/usr/bin/env bash
#SBATCH --job-name=safedial-dcr-prompt-smoke
#SBATCH --account=pradeepresearch
#SBATCH --partition=researchshort
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/safedial-dcr-prompt-smoke-%j.out
#SBATCH --error=outputs/slurm/safedial-dcr-prompt-smoke-%j.err

set -euo pipefail
if (( $# != 1 )) || [[ "$1" != training && "$1" != raw ]]; then
  echo "Usage: sbatch $0 {training|raw}" >&2
  exit 2
fi
prompt_format="$1"
if [[ "$prompt_format" == training ]]; then
  ids="2,3"
  output_dir="outputs/safedial_baseline/dcr_qwen_1.5b_prompt_training_ids2-3_smoke_v1"
else
  ids="1,2,3"
  output_dir="outputs/safedial_baseline/dcr_qwen_1.5b_prompt_raw_ids1-2-3_smoke_v1"
fi
module load Python/3.11.11-GCCcore-13.3.0
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
args=(--artifact-lock configs/safedial/dcr.lock.json --prompt-format "$prompt_format"
      --ids "$ids" --seed 0 --device cuda:0 --dtype bfloat16 --max-new-tokens 1024
      --output-dir "$output_dir")
.venv/bin/python -u -B scripts/run_safedial_dcr_prompt_ablation.py "${args[@]}" --validate-only
.venv/bin/python -u -B scripts/run_safedial_dcr_prompt_ablation.py "${args[@]}"
.venv/bin/python -u -B scripts/run_safedial_dcr_prompt_ablation.py \
  --output-dir "$output_dir" --audit-only --require-gpu
.venv/bin/python -u -B scripts/judge_safedial.py --answers "$output_dir/answers.jsonl" --dry-run
