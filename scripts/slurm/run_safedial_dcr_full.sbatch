#!/usr/bin/env bash
#SBATCH --job-name=safedial-dcr-full
#SBATCH --account=pradeepresearch
#SBATCH --partition=researchlong
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=outputs/slurm/safedial-dcr-full-%j.out
#SBATCH --error=outputs/slurm/safedial-dcr-full-%j.err

set -euo pipefail
if (( $# != 0 )); then
  echo "Usage: sbatch $0 (rerun identically to resume)" >&2
  exit 2
fi
module load Python/3.11.11-GCCcore-13.3.0
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
output_dir="outputs/safedial_baseline/dcr_qwen_1.5b_inference_format_full_v1"

# Pin the selected inference format without changing historical runner sources.
# The old CLI label 'training' selects this template; training parity is unknown.
.venv/bin/python -B - "$output_dir" <<'PY'
import ast
import hashlib
import json
from pathlib import Path
import sys

contract = json.loads(Path('configs/safedial/dcr_inference_format_full_v1.json').read_text())
checks = dict(contract['source_sha256'])
checks[contract['format_authority']] = contract['format_authority_sha256']
checks[contract['template']] = contract['template_sha256']
for name, expected in checks.items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
        raise ValueError(f'Frozen inference source/template mismatch: {name}')
tree = ast.parse(Path(contract['format_authority']).read_text())
template = next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'CHAT_TEMPLATE' for t in node.targets))
if template != Path(contract['template']).read_text():
    raise ValueError('Supplied inference format differs from runner template')
destination = Path(sys.argv[1]) / 'inference_format_contract.json'
destination.parent.mkdir(parents=True, exist_ok=True)
try:
    with destination.open('x') as handle:
        handle.write(json.dumps(contract, indent=2) + '\n')
except FileExistsError:
    if json.loads(destination.read_text()) != contract:
        raise ValueError('Existing output has a different inference contract')
print('Verified supplied inference template; training parity remains unverified.', flush=True)
PY

args=(--artifact-lock configs/safedial/dcr.lock.json --prompt-format training
      --seed 0 --device cuda:0 --dtype bfloat16 --max-new-tokens 1024
      --output-dir "$output_dir")
.venv/bin/python -u -B scripts/run_safedial_dcr_prompt_ablation.py "${args[@]}" --validate-only
.venv/bin/python -u -B scripts/run_safedial_dcr_prompt_ablation.py "${args[@]}"
.venv/bin/python -u -B scripts/run_safedial_dcr_prompt_ablation.py \
  --output-dir "$output_dir" --audit-only --require-gpu
.venv/bin/python -u -B scripts/judge_safedial.py --answers "$output_dir/answers.jsonl" --dry-run
