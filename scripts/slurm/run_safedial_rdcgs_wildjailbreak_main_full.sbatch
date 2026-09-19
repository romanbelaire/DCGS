#!/usr/bin/env bash
#SBATCH --job-name=safedial-rdcgs-main-full
#SBATCH --account=pradeepresearch
#SBATCH --partition=researchlong
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/slurm/safedial-rdcgs-main-full-%j.out
#SBATCH --error=outputs/slurm/safedial-rdcgs-main-full-%j.err

set -euo pipefail
module load Python/3.11.11-GCCcore-13.3.0
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
artifact_lock="${DCGS_ARTIFACT_LOCK:-configs/safedial/dcgs_main.lock.json}"

# Optional first argument: an already prepared, audited recovery directory.
# All 2037 dialogues / 10029 turns; main policy with trained LL reranking.
# Terminal model-output failures remain failed; later gold-history turns proceed.
# Infrastructure/integrity errors stop. No extra blank-response retries.
if (( $# > 1 )); then
  echo "Usage: sbatch $0 [prepared-output-directory]" >&2
  exit 2
fi
output_dir="${1:-outputs/safedial_dcgs/zephyr_rdcgs_main_wildjailbreak_full_v3}"

.venv/bin/python -u -B scripts/run_safedial_dcgs_wildjailbreak.py \
  --artifact-lock "$artifact_lock" --method rdcgs --seed 0 --device cuda:0 \
  --on-turn-error record-and-continue --validate-only --output-dir "$output_dir"
if .venv/bin/python -u -B scripts/run_safedial_dcgs_wildjailbreak.py \
  --artifact-lock "$artifact_lock" --method rdcgs --seed 0 --device cuda:0 \
  --on-turn-error record-and-continue --output-dir "$output_dir"; then
  generation_status=0
else
  generation_status=$?
  .venv/bin/python -u -B scripts/validate_safedial_dcgs_wildjailbreak.py \
    --output-dir "$output_dir" --allow-incomplete || true
  exit "$generation_status"
fi
.venv/bin/python -u -B scripts/validate_safedial_dcgs_wildjailbreak.py \
  --output-dir "$output_dir" --require-gpu
.venv/bin/python -u -B scripts/judge_safedial.py \
  --answers "$output_dir/answers.jsonl" --dry-run
