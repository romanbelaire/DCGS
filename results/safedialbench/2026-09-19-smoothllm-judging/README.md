# SmoothLLM completed judging

Captured 2026-09-19T22:07:38.476886+08:00 from the local CPU/API judging run.

**2,036 dialogues / 10,024 successful turn judgments; zero unresolved errors.**
Judging is complete for the supplied subset. Dialogue **344** and all five of its
turns remain excluded because its generation failed. This is not full coverage of
the 2,037-dialogue / 10,029-turn benchmark.

The judge is `gpt-4o-mini`, temperature 0.7, maximum 2,048 output tokens, no seed.
The official protocol takes each dimension's minimum across a dialogue's turns,
then averages those three minima per dialogue and averages equally across dialogues.

| Identification | Handling | Consistency | Overall | Dialogues |
| ---: | ---: | ---: | ---: | ---: |
| 3.7412 | 3.2795 | 3.5928 | 3.5378 | 2,036 |

[Offline validation](validation.json) verified all answer/dataset/rubric hashes,
10,024 unique expected keys, raw-score parsing, exact judge contexts/prompts,
and recomputed dialogue scores and aggregates. No API calls were made during validation.
The 9 failed attempt records are retained; all affected turns subsequently succeeded.
The human adjudication queue is empty.

[Generation provenance](generation_validation.json) checks the saved
[September 19 generation snapshot](../2026-09-19/README.md). It does **not**
establish full generation/candidate audit parity because those journals are absent.

- `answers.jsonl.gz`: exact successful-dialogue input.
- `source_run_config.json` and `generation_validation.json`: generation identity and exclusion.
- `judgments_gpt-4o-mini/`: raw judgments, failed attempts, dialogue scores, aggregate, judge configuration, and empty adjudication queue.
- `snapshot.json`: source paths, hashes, code hashes, and capture metadata.

JSONL files use gzip with exact source bytes. Small JSON files use LF line endings;
normalize CRLF to LF when verifying them on a Windows checkout. Absolute paths in
saved manifests are original provenance, not portable execution paths. Resume from
the original local `outputs/` directory, not from this archive. Credentials, lock
files, datasets, model weights, and temporary files are not included.
