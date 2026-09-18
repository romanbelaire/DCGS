# CAT and GPT-4o full-run judging

Prepared 2026-09-12 at the user's request. Both jobs are CPU-only, use the existing shared `gpt-4o-mini` rubric and judge, and make paid API requests. Each requests 2 CPUs, 8 GB RAM, and 48 hours in `researchlong`. Each permits two concurrent requests (four combined if both run together). No GPU is requested, so these jobs can run alongside SmoothLLM subject to scheduler and API capacity.

## Validated inputs and coverage

| Answers | Complete dialogues | Expected successful turn judgments | Coverage limitation |
|---|---:|---:|---|
| CAT | 2,037 | 10,029 | All generation passed validation |
| GPT-4o | 2,036 | 10,024 | Dialogue 1436 excluded because turn index 0 was content-filtered |

Total: **20,053 successful judgments** before retries. GPT-4o retains four additional successful turns from the excluded dialogue in its generation audit; those are not part of native complete-dialogue judging. The filtered turn is not retried or given a fabricated answer.

Each launcher revalidates generation before judging and saves a copy of that report as `generation_validation.json` inside its judge directory. For GPT-4o, the judge's `aggregate.json` may eventually say `complete=true` for the supplied 2,036 dialogues while generation coverage remains `complete=false` for the full benchmark. Report both coverages and retain the excluded ID. Do not call that a complete 2,037-dialogue result.

## Submit manually

From the DCGS project root:

```bash
sbatch scripts/slurm/run_safedial_gpt4o_judge_full.sbatch
sbatch scripts/slurm/run_safedial_cat_judge_full.sbatch
```

The user owns submissions under `AGENTS.md` / `.agents.md`. Save the returned IDs in the experiment logbook. No jobs were submitted during preparation.

The judge loads credentials through the existing `.env`/environment path. Settings match baseline judging: temperature 0.7, maximum 2,048 output tokens, no requested judge seed, and the same dataset, rubric, choice index, and default retry behavior. `gpt-4o-mini` is the judge for **both** answer models.

## Outputs, monitoring, and resume

- GPT-4o: `outputs/safedial_baseline/gpt4o_full/judgments_gpt-4o-mini/`
- CAT: `outputs/safedial_baseline/cat_zephyr_full/judgments_gpt-4o-mini/`
- Logs: `outputs/slurm/safedial-gpt4o-judge-<job-id>.out/.err` and `outputs/slurm/safedial-cat-judge-<job-id>.out/.err`.

Successful judgments are saved incrementally. Each launcher holds an exclusive lock on its own judge output directory, and the existing judge verifies the answer/dataset/rubric/settings manifest when resuming. Rerun the same launcher after an interruption; inspect persistent refusals before repeating paid retries. Do not run the judge directly into a directory while its batch job is active, since the lock is enforced by the launcher.

Inspect `aggregate.json`, `judgments.jsonl`, `failed_attempts.jsonl` if present, and the exported `human_adjudication.json` after completion. Job exit status alone does not establish complete judgment coverage. Keep baseline and smoke judgments in their existing directories.

Record progress and findings in [the experiment logbook](../EXPERIMENT_LOGBOOK.md).
