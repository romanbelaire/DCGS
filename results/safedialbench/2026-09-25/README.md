# SafeDialBench saved results

Captured 2026-09-25T00:48:49.036701+08:00.

These are saved results, including historical and still-running experiments. They are not all final benchmark scores. Consult each saved validation and judge aggregate.

| Run | Exported dialogues | Unique recorded turns | Recorded turn errors |
| --- | ---: | ---: | ---: |
| cat_zephyr_full | 2037 | 10029 | 0 |
| cat_zephyr_smoke | 1 | 5 | 0 |
| dcr_qwen_1.5b_inference_format_full_v1 | 2007 | 9878 | 4 |
| dcr_qwen_1.5b_prompt_qwen_standard_ids1-2-3_smoke_v1 | 3 | 15 | 0 |
| dcr_qwen_1.5b_prompt_raw_ids1-2-3_smoke_v1 | 3 | 15 | 5 |
| dcr_qwen_1.5b_prompt_training_ids2-3_smoke_v1 | 2 | 10 | 0 |
| dcr_qwen_1.5b_prompt_zephyr_ids1-2-3_smoke_v1 | 3 | 15 | 0 |
| dcr_qwen_1.5b_smoke_v1 | 1 | 5 | 0 |
| gpt4o_full | 2036 | 10028 | 0 |
| smoothllm_zephyr_full | 1605 | 7842 | 1 |
| smoothllm_zephyr_full_turn_resume | 2037 | 10029 | 1 |
| smoothllm_zephyr_recovery_344_turn5_v1 | 1 | 0 | 0 |
| smoothllm_zephyr_smoke | 1 | 5 | 0 |
| tpo_zephyr_full | 1 | 10 | 1 |
| tpo_zephyr_full_v4 | 24 | 126 | 1 |
| tpo_zephyr_full_v5 | 24 | 127 | 1 |
| tpo_zephyr_full_v6 | 25 | 131 | 1 |
| tpo_zephyr_full_v7 | 27 | 141 | 1 |
| tpo_zephyr_full_v8 | 56 | 282 | 1 |
| tpo_zephyr_smoke | 0 | 3 | 1 |
| tpo_zephyr_smoke_failed_249773 | 0 | 0 | 0 |
| tpo_zephyr_smoke_v2 | 0 | 5 | 1 |
| tpo_zephyr_smoke_v3 | 1 | 5 | 0 |
| tpo_zephyr_upstream_full_v1 | 11 | 56 | 0 |
| tpo_zephyr_upstream_smoke_v1 | 1 | 5 | 0 |
| zephyr_7b_beta_full | 2037 | 10029 | 0 |
| zephyr_7b_beta_smoke | 12 | 62 | 0 |
| zephyr_7b_beta_wildjailbreak_critic_smoke | 1 | 5 | 0 |
| zephyr_rdcgs_aug11_a5000_full | 19 | 101 | 1 |
| zephyr_rdcgs_aug11_a5000_full_retry_v1 | 19 | 100 | 0 |
| zephyr_rdcgs_main_belief-only_belief384_full_v1 | 1464 | 7140 | 0 |
| zephyr_rdcgs_main_wildjailbreak_full_v3 | 2037 | 10029 | 0 |
| zephyr_rdcgs_main_wildjailbreak_smoke_v3 | 1 | 5 | 0 |
| zephyr_rdcgs_original_wildjailbreak_full_v2 | 623 | 3005 | 0 |
| zephyr_rdcgs_original_wildjailbreak_smoke | 0 | 0 | 0 |
| zephyr_rdcgs_original_wildjailbreak_smoke_v2 | 1 | 5 | 0 |
| zephyr_vdcgs_aug11_a5000_full | 5 | 30 | 1 |
| zephyr_vdcgs_aug11_a5000_full_retry_v1 | 5 | 29 | 0 |
| zephyr_vdcgs_main_belief-only_belief384_full_v1 | 2030 | 10022 | 0 |
| zephyr_vdcgs_main_history_smoke_v1 | 1 | 5 | 0 |
| zephyr_vdcgs_main_wildjailbreak_full_v3 | 2011 | 10003 | 0 |
| zephyr_vdcgs_main_wildjailbreak_smoke_v3 | 1 | 5 | 0 |
| zephyr_vdcgs_original_wildjailbreak_full_v2 | 1288 | 6272 | 0 |
| zephyr_vdcgs_original_wildjailbreak_smoke | 0 | 0 | 0 |
| zephyr_vdcgs_original_wildjailbreak_smoke_v2 | 1 | 5 | 0 |
| zephyr_wildjailbreak_two_stage_smoke | 1 | 5 | 0 |

Large JSONL files use standard gzip compression. `snapshot.json` lists file hashes, sources, per-run metadata, and saved judge aggregates.

Answer exports and journals are captured independently while jobs may advance. The compact `turn_status` files deduplicate repeated turn attempts using the latest record. Exported dialogues can contain errors in legacy runners; check the error fields. Original call journals, model weights, caches, credentials, and the external dataset are not included. This snapshot cannot be used as a resumable execution directory.

The `llamaguard/`, `assistance/`, and `safedial_goals/` directories preserve frozen inputs, configurations, raw evaluator records and saved aggregates. An aggregate may lag a captured live journal; no missing result is inferred. Older protocols and failed pilots are retained separately.

## Current experiment interpretation

- Finished 96-token VDCGS/RDCGS and baseline results retain generation gaps and native-judge refusals. See [score comparison](../../../docs/SAFEDIALBENCH_SCORE_COMPARISON.md).
- VDCGS-384 generation processed all 10,029 turns: 10,022 successful, seven terminal failures, 2,030 complete exported dialogues. Integrity and policy checks passed; historical GPU-memory evidence remains incomplete. Native judging is partial at capture; assistance has 10,021 valid labels and one truncated-response error. Its LlamaGuard snapshot is prepared with zero labels. Prepared combined aggregates are not final scores.
- DCR is actively generating; RDCGS-384 is awaiting its resume allocation. TPO was stopped by the user. These files are independently captured prefixes and must not be used to resume execution. See [scheduler observation](scheduler.txt).
- Full assistance/LlamaGuard results for six finished sources cover every available response. Their combined metric uses safe AND non-assistance on every turn; benign extracted goals limit its interpretation as a harmful-only defense score.
- Existing complete SmoothLLM native judging is preserved in the [September 19 archive](../2026-09-19-smoothllm-judging/README.md), rather than duplicated here.
- No generation, paid judging, or scheduler mutation was performed to create this archive.
