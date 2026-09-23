# Active SafeDial scripts

DCGS now uses main-method v3 with trained LL reranking. See the [current configuration and run commands](../README.md#current-dcgs-configuration) and [method/artifact guide](../docs/SAFEDIALBENCH_DCGS_MAIN.md). Earlier original-policy v2 results are historical; they are not new-method GPU validation.

Superseded experiments and one-time migrations were archived on 2026-09-18.
Use this directory for current work. See the [audit archive](../archive/scripts_cleanup_20260918/README.md)
for exact old files, hashes, run references, and historical tests.

| Task | Entry point | Validation / launcher |
| --- | --- | --- |
| Main WildJailbreak DCGS + trained LL critic | `run_safedial_dcgs_wildjailbreak.py --method vdcgs` or `rdcgs` | `validate_safedial_dcgs_wildjailbreak.py`; `slurm/run_safedial_{vdcgs,rdcgs}_wildjailbreak_main_{smoke,full}.sbatch` |
| Separate DCGS history experiment | `run_safedial_dcgs_context.py --method vdcgs --include-history` (also `rdcgs`) | Same runner `--audit-only`; `slurm/run_safedial_dcgs_history_smoke.sbatch {vdcgs\|rdcgs}`; [history guide](../docs/SAFEDIALBENCH_DCGS_HISTORY.md) |
| DCGS, 384-token belief lists | `run_safedial_dcgs_belief384.py` | `--audit-only`; `slurm/run_safedial_dcgs_belief384_full.sbatch {vdcgs\|rdcgs} {history\|belief-only}`; [384-token guide](../docs/SAFEDIALBENCH_BELIEF384.md) |
| TPO, legacy v8 full run | `run_safedial_tpo_full.py` | `validate_safedial_tpo_full.py`; `slurm/run_safedial_tpo_retry_full.sbatch` (v8) |
| TPO, upstream candidate handling | `run_safedial_tpo_upstream_full.py` | `validate_safedial_tpo_upstream.py`; `slurm/run_safedial_tpo_upstream_{smoke,full}.sbatch`; [policy guide](../docs/SAFEDIALBENCH_TPO_UPSTREAM_HANDLING.md) |
| SmoothLLM, resumable full run | `run_safedial_smoothllm_turn_resume.py` | `validate_safedial_smoothllm_turn_resume.py`; `slurm/run_safedial_smoothllm_turn_resume_full.sbatch` |
| SmoothLLM, single fresh-seed recovery of 344/index4 | `run_safedial_smoothllm_recovery.py` | `--validate-only` / `--audit-only --require-gpu`; `slurm/run_safedial_smoothllm_recovery.sbatch`; [supplemental policy](../docs/SAFEDIALBENCH_SMOOTHLLM_RECOVERY.md) |
| Zephyr baseline / CAT | `run_safedial_baseline.py` | `validate_safedial_generation.py`; Zephyr/CAT launchers in `slurm/` |
| DCR, Qwen2.5-1.5B + uploaded LoRA | `run_safedial_dcr.py` | `--validate-only` / `--audit-only --require-gpu`; `slurm/run_safedial_dcr_smoke.sbatch`; [DCR guide](../docs/SAFEDIALBENCH_DCR.md) |
| DCR full benchmark, supplied inference format | `run_safedial_dcr_prompt_ablation.py --prompt-format training` (legacy label) | `slurm/run_safedial_dcr_full.sbatch`; [format contract and resume](../docs/SAFEDIALBENCH_DCR.md#full-benchmark-using-the-supplied-inference-format) |
| DCR prompt-format diagnostic | `run_safedial_dcr_prompt_ablation.py --prompt-format training` or `raw` | `slurm/run_safedial_dcr_prompt_smoke.sbatch {training\|raw}`; [paired smoke guide](../docs/SAFEDIALBENCH_DCR_PROMPT_SMOKE.md) |
| DCR with literal Zephyr formatting | `run_safedial_dcr_zephyr_prompt.py` | `slurm/run_safedial_dcr_zephyr_smoke.sbatch`; [format semantics and smoke](../docs/SAFEDIALBENCH_DCR_PROMPT_SMOKE.md#additional-zephyr-style-diagnostic) |
| DCR with standard Qwen template | `run_safedial_dcr_qwen_prompt.py` | `slurm/run_safedial_dcr_qwen_smoke.sbatch`; [paired template smoke](../docs/SAFEDIALBENCH_DCR_PROMPT_SMOKE.md#standard-qwen-template-smoke) |
| GPT-4o baseline | `run_safedial_api.py` | Same entry point with `--validate-output`; GPT-4o launchers in `slurm/` |
| SmoothLLM successful-dialogue judging | `run_safedial_smoothllm_judging.py` (Windows/Linux) | CPU-only `slurm/run_safedial_smoothllm_judge_full.sbatch`; [coverage and resume](../docs/SAFEDIALBENCH_FULL_JUDGING.md) |
| Judging | `judge_safedial.py` | `--dry-run` checks inputs without paid calls; dedicated judge launchers in `slurm/` |
| LlamaGuard secondary safety evaluation | `judge_safedial_llamaguard.py` | `--prepare-only` freezes saved outputs without model execution; `slurm/run_safedial_llamaguard.sbatch`; [metrics, pilot and resume](../docs/SAFEDIALBENCH_LLAMAGUARD.md) |
| LlamaGuard full-dataset evaluation | `run_safedial_llamaguard_full.py` | Full ID/turn coverage, tokenizer-only `--preflight-only`, resumable GPU judging; `slurm/run_safedial_llamaguard_full.sbatch`; [full-run guide](../docs/SAFEDIALBENCH_LLAMAGUARD.md#full-dataset-runner) |
| Human adjudication export | `export_safedial_adjudication.py` | See `docs/SAFEDIALBENCH_HUMAN_ADJUDICATION.md` |
| Artifact preparation | `prepare_safedial_cat.py`, `prepare_safedial_tpo.py` | `--offline` validates existing assets |
| API connection check | `test_openai_connection.py` | Makes an API request; not part of offline tests |

The main-method v3 runner enables last-nonpadding-token HL pooling and trained
LL token-critic selection from five candidates. New launchers use separate
`*_main_wildjailbreak_{smoke,full}_v3` directories. CPU regression tests pass;
real model loading, GPU memory, throughput, and output validation remain pending
because no GPU is currently available. Preflight reports potential LL critic
context overflows against the 8,192-token limit. Full runs record such turns as
`ll_context_overflow` and continue; smoke runs stop. No truncation or extra retry
is introduced. Completed execution with failed turns returns 2, and only complete
dialogues enter the answer export. The user owns Slurm submissions; agents
must not submit or mutate jobs.

Historical original-DCGS A40 smokes 257599/257600 passed output, runtime, and
memory checks for v2 only. Their results and the
[original policy and recovery guide](../docs/SAFEDIALBENCH_DCGS_PARITY.md)
require the old method checkout (`4e2f493`). Do not resume those runs with v3.
The `safedial_dcgs_run_state.py` helper audits shared execution state; it does
not implement a separate DCGS policy.

## Why some older names remain

`run_safedial_dcgs.py` supplies history conversion and the model/critic loader
used by the current main-method adapter. Its old standalone launcher is archived.
The standalone entry point rejects trained LL reranking; use the active runner above.
`run_safedial_smoothllm.py` and `run_safedial_tpo.py` supply shared runner
functions used by their current full/resumable entry points. Their validators
are also shared. Removing these files would break current imports or pinned
source hashes.

The `safedial_*.py` modules and `tpo_vendor/` contain runtime dependencies.
Keep them at their present paths while runs pin their hashes. The DCGS integration
changes source hashes; historical manifests must be checked against their original
checkout.

## Offline checks

From the DCGS root:

```bash
module load Python/3.11.11-GCCcore-13.3.0
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -m unittest discover -s tests -v
```

The prior complete 146-test suite passed on Linux with CPU PyTorch and offline model access. Historical custom-DCGS tests remain in the
archive's complete 143-test suite; active TPO retry coverage remains here.

SmoothLLM judging adds five full-journal preparation tests and seven portable snapshot/locking tests. The seven portable tests pass on native Windows; all 37 related preparation, SmoothLLM audit/resume, and judge tests pass on Linux. The full suite was not repeated for these isolated helper/launcher additions. See the [local labelling commands](../docs/SAFEDIALBENCH_FULL_JUDGING.md#local-windows--git-bash--linux-without-slurm).
