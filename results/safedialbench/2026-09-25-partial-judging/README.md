# Completed partial SafeDial native judging

Supplement to the [September 25 main snapshot](../2026-09-25/README.md). These four historical partial judging datasets were omitted from that exporter and are retained here with exact source hashes, raw judgments, frozen answers, score aggregates and preparation provenance. They must not be added to final-run counts or treated as independent new results.

| Scope | Complete dialogues | Successful turn judgments |
| --- | ---: | ---: |
| dcgs_384tokens_20260922_v1/rdcgs | 497 | 2403 |
| dcgs_384tokens_20260922_v1/vdcgs | 923 | 4462 |
| dcgs_96tokens_20260920/rdcgs | 389 | 1886 |
| dcgs_96tokens_20260920/vdcgs | 457 | 2218 |

All four saved aggregates report complete=true, with zero unresolved judging errors. Large JSONL files are gzip-compressed. `snapshot.json` records checksums and original paths. Source files were verified unchanged during capture. No API calls, inference or job changes were performed.
