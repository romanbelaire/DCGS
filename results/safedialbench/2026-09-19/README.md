# SafeDialBench saved results

Captured 2026-09-19T01:30:35.167829+08:00.

These are saved results, including historical and still-running experiments. They are not all final benchmark scores. Consult each saved validation and judge aggregate.

| Run | Exported dialogues | Unique recorded turns | Recorded turn errors |
| --- | ---: | ---: | ---: |
| cat_zephyr_full | 2037 | 10029 | 0 |
| cat_zephyr_smoke | 1 | 5 | 0 |
| gpt4o_full | 2036 | 10028 | 0 |
| smoothllm_zephyr_full | 1605 | 7842 | 1 |
| smoothllm_zephyr_full_turn_resume | 2037 | 10029 | 1 |
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
| tpo_zephyr_upstream_smoke_v1 | 1 | 5 | 0 |
| zephyr_7b_beta_full | 2037 | 10029 | 0 |
| zephyr_7b_beta_smoke | 12 | 62 | 0 |
| zephyr_7b_beta_wildjailbreak_critic_smoke | 1 | 5 | 0 |
| zephyr_rdcgs_aug11_a5000_full | 19 | 101 | 1 |
| zephyr_rdcgs_aug11_a5000_full_retry_v1 | 19 | 100 | 0 |
| zephyr_rdcgs_original_wildjailbreak_full_v2 | 619 | 2982 | 0 |
| zephyr_rdcgs_original_wildjailbreak_smoke | 0 | 0 | 0 |
| zephyr_rdcgs_original_wildjailbreak_smoke_v2 | 1 | 5 | 0 |
| zephyr_vdcgs_aug11_a5000_full | 5 | 30 | 1 |
| zephyr_vdcgs_aug11_a5000_full_retry_v1 | 5 | 29 | 0 |
| zephyr_vdcgs_original_wildjailbreak_full_v2 | 1282 | 6237 | 0 |
| zephyr_vdcgs_original_wildjailbreak_smoke | 0 | 0 | 0 |
| zephyr_vdcgs_original_wildjailbreak_smoke_v2 | 1 | 5 | 0 |
| zephyr_wildjailbreak_two_stage_smoke | 1 | 5 | 0 |

Large JSONL files use standard gzip compression. `snapshot.json` lists file hashes, sources, per-run metadata, and saved judge aggregates.

Answer exports and journals are captured independently while jobs may advance. The compact `turn_status` files deduplicate repeated turn attempts using the latest record. Exported dialogues can contain errors in legacy runners; check the error fields. Original call journals, model weights, caches, credentials, and the external dataset are not included. This snapshot cannot be used as a resumable execution directory.

VDCGS has 5 terminal failures in its separate `failures.jsonl` ledger; the recorded-turn error column does not include that ledger. RDCGS has no recorded terminal failures. Both jobs still reported RUNNING at the scheduler check; a pause has not been verified. See `scheduler.txt` and the captured Slurm logs.
