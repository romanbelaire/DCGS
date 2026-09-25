# Latest SafeDialBench results — 25 September 2026

This incremental archive captures the saved results after DCR generation and
VDCGS-384 native judging finished overnight. Exact capture times are recorded
in `snapshot.json`. It contains eight updated data files (approximately 39 MiB);
753 unchanged files are referenced by path and hash in the earlier archives.

Base archives, already committed at `dfbc87713b81e4f38d9c8f68b0ca1190608bc3e4`:

- [September 25 full snapshot](../2026-09-25/README.md)
- [Completed partial judging](../2026-09-25-partial-judging/README.md)

For each entry in `files`, read `path` relative to this directory. For each entry
in `reused_files`, read `archive_path` relative to the repository root and verify
`archive_sha256`. `uncompressed_sha256` identifies the original captured content
in either case. The new capture was produced by
`scripts/snapshot_safedial_results.py`; files whose uncompressed hashes match the
base archives are referenced instead of duplicated. The `runs` list records the
latest complete inventory of run metadata, including unchanged historical runs.

## Current results

| Run | Saved progress | Remaining limitations |
| --- | --- | --- |
| DCR | All 2,037 dialogues exported; 10,025 successful turns out of 10,029; 2,033 error-free dialogues | Four existing empty-response failures at dialogue IDs 416, 1850, 1912 and 1915. Final judging remains. |
| VDCGS-384 native judging | 9,992/9,994 successful turn judgments; 2,028/2,030 fully judged dialogues; overall 3.4921/10 | Evaluator refusals at dialogue 1353, turn index 4, and dialogue 1869, turn index 5. Both refused all three attempts. Score is provisional. |
| VDCGS-384 generation | 10,022 successful turns and seven terminal failures; 2,030 complete dialogues | Unchanged from the base snapshot; historical GPU-memory evidence gap remains. |
| VDCGS-384 assistance | 10,021/10,022 available responses labeled | One truncated-label error; seven generation gaps. Unchanged from the base snapshot. |
| VDCGS-384 LlamaGuard | Zero labels | Not submitted; combined DSR unavailable. |
| RDCGS-384 generation | 1,464 complete dialogues; 7,140 successful turns | One terminal failure and 572 dialogues awaiting completion. Resume remains queued. |

Read-only scheduler inspection on 25 September at approximately 09:19 SGT found
no running jobs. RDCGS-384 job `302962` was pending node availability, with a
tentative start of 30 September 01:59:20 SGT, subject to change. DCR `302813`
ended at 02:00:21 with exit 2 because of the saved generation errors; native
judge `302964` ended at 01:36:46 with exit 1 because of the two failed judgments.
Assistance `303002` ended at 00:06:41 with exit 2.

## Verification and scope

`verification.json` records hash, gzip, JSON/JSONL and record-count validation
for all eight new files and all 753 referenced files. The three snapshot utility
tests passed. No model execution, paid API call or scheduler mutation was used.

The archives preserve incomplete results and raw evaluator failures. They are
not resumable execution checkpoints. Full generation call journals, model
weights, caches, credentials and the external dataset remain excluded, following
the existing snapshot workflow. Large JSONL files use standard gzip compression.
