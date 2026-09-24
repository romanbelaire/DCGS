#!/usr/bin/env bash
#SBATCH --job-name=safedial-assistance-api-normalized-suite
#SBATCH --account=pradeepresearch
#SBATCH --partition=researchlong
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/safedial-assistance-api-normalized-suite-%j.out
#SBATCH --error=outputs/slurm/safedial-assistance-api-normalized-suite-%j.err

# Queue after the old suite; import/normalize saved responses into separate v2
# output directories, then call the API only for missing judgments.
# Six finished generation sources in sequence; eight simultaneous API requests
# total, no GPUs, same prompt/model/goals as the validated API pilot.
set -euo pipefail
if (( $# > 1 )) || { (( $# == 1 )) && [[ "$1" != --retry-errors ]]; }; then
  echo "Usage: sbatch $0 [--retry-errors]" >&2
  exit 2
fi
source /etc/profile.d/z00_lmod.sh
module load Python/3.11.11-GCCcore-13.3.0
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS

while IFS='|' read -r suite_name suite_source suite_output suite_guard; do
  echo "Starting assistance judging: $suite_name"
  if .venv/bin/python -u -B scripts/judge_safedial_assistance_api_normalized.py \
      --run-dir "$suite_source" --output-dir "$suite_output" \
      --llamaguard-dir "$suite_guard" --parallel 8 \
      --import-from "${suite_output%_v2}_v1" "$@"; then
    echo "Finished assistance and guard join: $suite_name"
  else
    suite_status=$?
    if (( suite_status != 2 )); then exit "$suite_status"; fi
    # Exit 2 is expected for missing generation, but not for API/parse failures
    # or incomplete work. Never let an old aggregate hide an interrupted run.
    if .venv/bin/python -B - "$suite_output" <<'PY'
import json
from pathlib import Path
import sys
folder = Path(sys.argv[1])
assistance = json.loads((folder / 'aggregate.json').read_text())
combined = json.loads((folder / 'combined_aggregate.json').read_text())
turns = [json.loads(line) for line in (folder / 'combined_turn_scores.jsonl').read_text().splitlines()]
inputs = [json.loads(line) for line in (folder / 'frozen/inputs.jsonl').read_text().splitlines()]
available = {r['input_sha256'] for r in inputs if r['generation_status'] == 'available'}
assert assistance['available_responses_judged'], 'Assistance errors or pending work; inspect before retrying'
assert not any(assistance['turn_counts'].get(k, 0) for k in ('judge_pending', 'judge_error'))
assert {t['input_sha256'] for t in turns} == {r['input_sha256'] for r in inputs}, 'Combined scope differs from frozen inputs'
assert all(t['complete'] for t in turns if t['input_sha256'] in available), 'Guard labels missing for available responses'
assert combined['incomplete_dialogues'] > 0, 'Unexpected exit 2; inspect the run'
print('All available responses judged and joined; source-generation gaps remain explicit.')
PY
    then
      echo "Finished available responses: $suite_name"
    else
      exit 2
    fi
  fi
done <<'RUNS'
Zephyr|outputs/safedial_baseline/zephyr_7b_beta_full|outputs/assistance/gpt4o_mini_zephyr_full_v2|outputs/llamaguard/zephyr_7b_beta_full_v2
CAT|outputs/safedial_baseline/cat_zephyr_full|outputs/assistance/gpt4o_mini_cat_full_v2|outputs/llamaguard/cat_zephyr_full_v2
GPT-4o|outputs/safedial_baseline/gpt4o_full|outputs/assistance/gpt4o_mini_gpt4o_full_v2|outputs/llamaguard/gpt4o_full_v2_available
SmoothLLM-primary|outputs/safedial_baseline/smoothllm_zephyr_full_turn_resume|outputs/assistance/gpt4o_mini_smoothllm_primary_full_v2|outputs/llamaguard/smoothllm_primary_full_v2_available
VDCGS-96|outputs/safedial_dcgs/zephyr_vdcgs_main_wildjailbreak_full_v3|outputs/assistance/gpt4o_mini_vdcgs96_full_v2|outputs/llamaguard/vdcgs96_full_v2_available
RDCGS-96|outputs/safedial_dcgs/zephyr_rdcgs_main_wildjailbreak_full_v3|outputs/assistance/gpt4o_mini_rdcgs96_full_v2|outputs/llamaguard/rdcgs96_full_v2
RUNS
echo "Suite finished: every available response judged across all six sources."
