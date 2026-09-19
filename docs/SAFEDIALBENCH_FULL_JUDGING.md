# SafeDial result judging

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

## SmoothLLM successful-dialogue judging (2026-09-19)

The new CPU-only launcher audits the turn-resumable SmoothLLM run, freezes a separate answer subset, checks judge inputs, then makes paid `gpt-4o-mini` requests using the existing rubric/settings. It requests 2 CPUs, 8 GB RAM, 48 hours, no GPU, and two concurrent API requests. After transferring the new scripts to the execution checkout:

```bash
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS
mkdir -p outputs/slurm
sbatch scripts/slurm/run_safedial_smoothllm_judge_full.sbatch
```

Source: `outputs/safedial_baseline/smoothllm_zephyr_full_turn_resume/`. The preparation helper verifies dataset/source hashes, turn identities/seeds, gold history, successful candidate/vote audits, and answer/journal agreement. It holds the generation lock during export and never rewrites generation records. Only the known failed dialogue 344 may be excluded, and only if it still has a recorded failed turn; a repaired 344 is included. Other failed dialogues or missing turns stop preparation before API calls.

The September 19 snapshot implies 2,036 included dialogues / 10,024 turn judgments, excluding dialogue 344 and all its five turns. Actual coverage is recomputed on the execution host. This is a partial benchmark result until all generation failures are resolved.

- Frozen answers and exclusion/source-hash report: `outputs/safedial_baseline/smoothllm_zephyr_full_turn_resume/judging_complete_dialogues_v1/{answers.jsonl,generation_validation.json}`.
- Judgments and aggregate: `judging_complete_dialogues_v1/judgments_gpt-4o-mini/` under that source directory.
- Logs: `outputs/slurm/safedial-smoothllm-judge-<job-id>.{out,err}`.

Rerun the same `sbatch` command to resume successful judgments. The launcher locks its judge directory. Prepared inputs are immutable: if generation data or audited sources changed, it stops and requires a separate prepared/judgment directory rather than mixing results. The judge aggregate's `complete` refers only to the supplied subset; report the generation exclusion alongside it.

Offline verification: 30 relevant preparation, SmoothLLM audit/resume, and judge tests pass; new launcher passes `bash -n`. No model inference, paid judging, or job submission ran here. The default dataset/rubric files are absent locally, so the real-data dry-run executes on the host before paid calls.
