#!/usr/bin/env bash
#SBATCH --job-name=safedial-dcgs-history-smoke
#SBATCH --account=pradeepresearch
#SBATCH --partition=researchshort
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/safedial-dcgs-history-smoke-%j.out
#SBATCH --error=outputs/slurm/safedial-dcgs-history-smoke-%j.err

set -euo pipefail
if (( $# != 1 )) || [[ "$1" != vdcgs && "$1" != rdcgs ]]; then
  echo "Usage: sbatch $0 {vdcgs|rdcgs}" >&2
  exit 2
fi
method="$1"
module load Python/3.11.11-GCCcore-13.3.0
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
artifact_lock="${DCGS_ARTIFACT_LOCK:-configs/safedial/dcgs_main.lock.json}"
output_dir="outputs/safedial_dcgs/zephyr_${method}_main_history_smoke_v1"
args=(--artifact-lock "$artifact_lock" --method "$method" --include-history
      --ids 1 --seed 0 --device cuda:0 --on-turn-error stop --output-dir "$output_dir")
.venv/bin/python -u -B scripts/run_safedial_dcgs_context.py "${args[@]}" --validate-only
if .venv/bin/python -u -B scripts/run_safedial_dcgs_context.py "${args[@]}"; then
  generation_status=0
else
  generation_status=$?
  .venv/bin/python -u -B scripts/run_safedial_dcgs_context.py \
    --output-dir "$output_dir" --audit-only --allow-incomplete || true
  exit "$generation_status"
fi
.venv/bin/python -u -B scripts/run_safedial_dcgs_context.py \
  --output-dir "$output_dir" --audit-only --require-gpu
.venv/bin/python -u -B scripts/judge_safedial.py --answers "$output_dir/answers.jsonl" --dry-run
