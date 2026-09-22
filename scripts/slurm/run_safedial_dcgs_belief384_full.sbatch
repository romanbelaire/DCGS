#!/usr/bin/env bash
#SBATCH --job-name=safedial-dcgs-belief384-full
#SBATCH --account=pradeepresearch
#SBATCH --partition=researchlong
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/slurm/safedial-dcgs-belief384-full-%j.out
#SBATCH --error=outputs/slurm/safedial-dcgs-belief384-full-%j.err

set -euo pipefail
if (( $# < 2 || $# > 3 )) || [[ "$1" != vdcgs && "$1" != rdcgs ]] ||
   [[ "$2" != history && "$2" != belief-only ]]; then
  echo "Usage: sbatch $0 {vdcgs|rdcgs} {history|belief-only} [prepared-output-directory]" >&2
  exit 2
fi
method="$1"
context_mode="$2"
module load Python/3.11.11-GCCcore-13.3.0
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
artifact_lock="${DCGS_ARTIFACT_LOCK:-configs/safedial/dcgs_main.lock.json}"
output_dir="${3:-outputs/safedial_dcgs/zephyr_${method}_main_${context_mode}_belief384_full_v1}"
args=(--artifact-lock "$artifact_lock" --method "$method" --seed 0 --device cuda:0
      --on-turn-error record-and-continue --output-dir "$output_dir")
if [[ "$context_mode" == history ]]; then
  args+=(--include-history)
fi
echo "Method=$method context=$context_mode belief_list_tokens=384 output=$output_dir"
.venv/bin/python -u -B scripts/run_safedial_dcgs_belief384.py "${args[@]}" --validate-only
if .venv/bin/python -u -B scripts/run_safedial_dcgs_belief384.py "${args[@]}"; then
  generation_status=0
else
  generation_status=$?
  .venv/bin/python -u -B scripts/run_safedial_dcgs_belief384.py \
    --output-dir "$output_dir" --audit-only --allow-incomplete || true
  exit "$generation_status"
fi
.venv/bin/python -u -B scripts/run_safedial_dcgs_belief384.py \
  --output-dir "$output_dir" --audit-only --require-gpu
.venv/bin/python -u -B scripts/judge_safedial.py --answers "$output_dir/answers.jsonl" --dry-run
