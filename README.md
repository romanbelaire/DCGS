# SafeDialBench experiment status

Status checked **18 September 2026, 17:32 SGT (UTC+8)**. Turn counts captured at `2026-09-18T17:30:17.791710+08:00`.

This branch contains SafeDialBench generation, validation, and judging code plus saved results. A full benchmark has **2,037 dialogues and 10,029 turns**. Active-run counts are snapshots, not final results.

## Current benchmark runs

| Benchmark / method | Status | Successful generation turns | Validation / judging / remaining issue |
| --- | --- | ---: | --- |
| Zephyr baseline | Generation complete; judging incomplete | 10,029/10,029 | 10,028/10,029 turn judgments; one unresolved judge response. |
| GPT-4o baseline | Processing complete; filtered case excluded | 10,028/10,029 | One filtered turn in dialogue 1436. Judging complete for 2,036 exported dialogues / 10,024 turns, not the full benchmark. |
| CAT + Zephyr | Generation and judging complete | 10,029/10,029 | Validation passed; all 10,029 turn judgments complete. |
| SmoothLLM + Zephyr | Finished with one failed turn (256253) | 10,028/10,029 | All turns processed; dialogue 344, turn index 4 has an empty candidate. Exit 2; final validation/judging not reached. |
| TPO v8 + Zephyr | Stopped on parser failure (257501) | 281/10,029 | Dialogue 57, turn index 1: missing IMPROVED_VARIABLE opening tag. Exit 2; no final judge score. |
| VDCGS, original WildJailbreak | Running (257632) | 2,639/10,029 | One terminal all-SKIP belief failure: dialogue 327, turn index 2. Generation continues; judging pending. |
| RDCGS, original WildJailbreak | Running (257633) | 758/10,029 | No terminal failures recorded at this check; judging pending. |
| TPO upstream-handling smoke | Passed on L40 (257858) | 5/5 | 5/5 turns; 75 candidates; GPU audit and judge dry-run passed. No skipped candidates in this real smoke; skip handling has offline tests. Full run not launched. |

Counts include inherited results for resumed runs and deduplicate repeated records by dialogue/turn. A saved turn or completed Slurm job does not by itself establish successful validation or a final benchmark score. Turn indices in failure notes are zero-based.

## Smoke and historical runs

The directories below retain prior smoke results, failed attempts, or superseded policies. They are not additional completed full-benchmark results and must not be pooled with the current methods.

| Saved run directory | Saved successful turns | Recorded status |
| --- | ---: | --- |
| `cat_zephyr_smoke` | 5/5 | Historical/smoke validation passed |
| `smoothllm_zephyr_full` | 7,841/10,029 | Historical failure; 1 failed turn retained |
| `smoothllm_zephyr_smoke` | 5/5 | Historical/smoke validation passed |
| `tpo_zephyr_full` | 9/10,029 | Historical failure; 1 failed turn retained |
| `tpo_zephyr_full_v4` | 125/10,029 | Historical failure; 1 failed turn retained |
| `tpo_zephyr_full_v5` | 126/10,029 | Historical failure; 1 failed turn retained |
| `tpo_zephyr_full_v6` | 130/10,029 | Historical failure; 1 failed turn retained |
| `tpo_zephyr_full_v7` | 140/10,029 | Historical failure; 1 failed turn retained |
| `tpo_zephyr_smoke` | 2/5 | Historical failure; 1 failed turn retained |
| `tpo_zephyr_smoke_failed_249773` | 0/5 | Historical CUDA startup failure; no generated turns |
| `tpo_zephyr_smoke_v2` | 4/5 | Historical failure; 1 failed turn retained |
| `tpo_zephyr_smoke_v3` | 5/5 | Historical/smoke validation passed |
| `zephyr_7b_beta_smoke` | 62/62 | Smoke judging complete |
| `zephyr_7b_beta_wildjailbreak_critic_smoke` | 5/5 | Historical outputs; no final validation report |
| `zephyr_rdcgs_aug11_a5000_full` | 100/10,029 | Historical failure; 1 failed turn retained |
| `zephyr_rdcgs_aug11_a5000_full_retry_v1` | 100/10,029 | Superseded prepared continuation; inherited outputs |
| `zephyr_rdcgs_original_wildjailbreak_smoke` | 0/5 | Prepared/superseded; no generated turns |
| `zephyr_rdcgs_original_wildjailbreak_smoke_v2` | 5/5 | Historical/smoke validation passed |
| `zephyr_vdcgs_aug11_a5000_full` | 29/10,029 | Historical failure; 1 failed turn retained |
| `zephyr_vdcgs_aug11_a5000_full_retry_v1` | 29/10,029 | Superseded prepared continuation; inherited outputs |
| `zephyr_vdcgs_original_wildjailbreak_smoke` | 0/5 | Prepared/superseded; no generated turns |
| `zephyr_vdcgs_original_wildjailbreak_smoke_v2` | 5/5 | Historical/smoke validation passed |
| `zephyr_wildjailbreak_two_stage_smoke` | 5/5 | Historical/smoke validation passed |

Earlier A40 TPO smoke submission **257857** failed because L40 job **257858** already held the output lock. It is not a separate model result. Original VDCGS/RDCGS v2 smoke jobs **257599/257600** passed, 5/5 turns each; their original 1,500-token critic truncation remains part of the preserved policy.

## Results and reproducibility

- [Current status evidence](docs/verification/benchmark_status_20260918.json): per-run counts, failures, saved validations, judge aggregates, and scheduler states supporting this README.
- [Saved result snapshot](results/safedialbench/2026-09-18/README.md): answers, judgments, manifests, and compact turn records from 31 run directories. **This snapshot predates the status update above**; its live-run outputs may be less complete. See its [timestamp and checksums](results/safedialbench/2026-09-18/snapshot.json).
- [Script index](scripts/README.md): generation, validation, and Slurm entrypoints.
- [Upstream TPO handling](docs/SAFEDIALBENCH_TPO_UPSTREAM_HANDLING.md): skip-and-record policy and resume behavior.
- [Original DCGS policy](docs/SAFEDIALBENCH_DCGS_PARITY.md): original-code reuse, failure handling, and critic-context limitations.

The implementation test suite passed **130 tests** before this documentation-only update. Large saved JSONL files are gzip-compressed. Full model-call journals, checkpoints, caches, and the external dataset remain outside the committed result snapshot.

Root `handover.md`, `runbook.md`, `AGENTS.md`, `agents.md`, and `.agents.md` are local operational files and ignored by Git. Immutable archived audit copies remain part of the historical snapshot.
