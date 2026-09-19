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

The portable launcher runs on Windows or Linux without a GPU. It prepares a separate successful-dialogue answer subset, checks judge inputs, then uses the existing `gpt-4o-mini` judge with temperature 0.7, maximum 2,048 output tokens, and two concurrent API requests. Credentials come from `OPENAI_API_KEY` in the project-root `.env` or environment. Install `openai` and `httpx` in the selected Python environment if needed: `python -m pip install openai httpx`.

### Local Windows / Git Bash / Linux, without Slurm

From the DCGS root, validate the committed September 19 snapshot first:

```bash
python scripts/run_safedial_smoothllm_judging.py --snapshot results/safedialbench/2026-09-19 --fetch-benchmark --dry-run
```

`--fetch-benchmark` downloads only missing public dataset/rubric files from SafeDialBench-Dataset revision `e242e5f3fcbf87f0e11e99d4563155da1e2e5a23`. It checks their SHA-256 hashes against the saved generation and CAT judging manifests before writing them. Existing files must match too. `--dry-run` creates and audits the frozen input subset but makes no judge API calls.

Start paid labelling with:

```bash
python scripts/run_safedial_smoothllm_judging.py --snapshot results/safedialbench/2026-09-19
```

This snapshot contains **2,036 successful dialogues / 10,024 turn judgments**, after excluding dialogue 344 and all five of its turns. The snapshot route verifies file checksums, generation source and dataset identity, answer/status hashes, turn seeds, and coverage. It cannot repeat the full SmoothLLM candidate/vote audit because the snapshot omits the full generation journal. The saved report explicitly sets `snapshot_answer_audit_passed=true` and `full_generation_audit_passed=false`.

Prepared answers, `generation_validation.json`, and `source_run_config.json` go into `outputs/safedial_judging/smoothllm_zephyr_full_turn_resume_snapshot_2026-09-19/`. Judgments and `aggregate.json` go into its `judgments_gpt-4o-mini/` subdirectory. The committed `results/` snapshot stays unchanged. Logs appear in the terminal; Slurm and `outputs/slurm` are not needed.

The Bash launcher also supports this mode, including in Git Bash:

```bash
bash scripts/slurm/run_safedial_smoothllm_judge_full.sbatch --snapshot results/safedialbench/2026-09-19
```

Without explicit input flags, direct execution uses the saved snapshot on Windows; on Linux it uses the default live run if its full `turns.jsonl` exists, otherwise the snapshot. Use explicit `--snapshot` or `--source-dir` for repeatable selection. Options also include `--parallel`, `--dataset`, `--prompts`, `--env-file`, and `--output-dir` (the frozen-input root, with judgments in a subdirectory).

### Full generation journal on Linux / HPC

For the live run at `outputs/safedial_baseline/smoothllm_zephyr_full_turn_resume/`, the existing preparation helper verifies dataset/source hashes, turn identities/seeds, gold history, successful candidate/vote audits, and answer/journal agreement. It holds the generation lock during export. This route requires Linux or WSL because the generation runner uses Linux locks.

Run directly on the execution host:

```bash
python scripts/run_safedial_smoothllm_judging.py --source-dir outputs/safedial_baseline/smoothllm_zephyr_full_turn_resume
```

Or submit from the DCGS root (2 CPUs, 8 GB RAM, 48 hours, no GPU):

```bash
mkdir -p outputs/slurm
sbatch scripts/slurm/run_safedial_smoothllm_judge_full.sbatch
```

Slurm defaults to the live source directory. Frozen inputs and the exclusion/source-hash report go into `judging_complete_dialogues_v1/` under that source; judgments go into its `judgments_gpt-4o-mini/` subdirectory. Slurm logs are `outputs/slurm/safedial-smoothllm-judge-<job-id>.{out,err}`.

### Resume and coverage

Both routes only exclude recorded failures in dialogue 344; a repaired 344 is included when preparing fresh input. Unexpected failed dialogues, missing turns, or invalid inputs stop preparation before paid calls. Actual coverage is recomputed from the chosen source.

Rerun the same command to resume. The portable launcher holds an OS lock on its judge directory, released when the process exits. Prepared inputs are immutable: changed generation data, audit inputs, or report provenance require a fresh `--output-dir`. Successful judgments are reused by the existing judge after manifest checks. Inspect persistent refusals before repeating paid retries. The judge aggregate's `complete` refers only to the supplied subset; report the generation exclusion alongside it.

Verification: the real snapshot dry-run passes on native Windows and through Git Bash with 2,036 dialogues / 10,024 requests, and both official benchmark files match the saved hashes. Seven portable regression tests pass on Windows; 37 related preparation, audit/resume, and judge tests pass on Linux. No GPU inference, paid judging, or job submission ran during this verification.

### Completed local result

The local SmoothLLM run finished on 2026-09-19 at 21:57 SGT. Offline verification at 22:07 SGT confirmed **10,024/10,024 successful judgments across 2,036 dialogues**, zero unresolved judge errors, and an empty human-review queue. Nine saved failed attempts across seven turns were recovered. Overall score: **3.5378** (Identification 3.7412, Handling 3.2795, Consistency 3.5928), using the official per-dialogue minimum-across-turns protocol. Dialogue 344 remains excluded, so this is complete for the supplied subset only.

The [committed result archive](../results/safedialbench/2026-09-19-smoothllm-judging/README.md) contains the exact prepared answers, generation/exclusion provenance, raw judgments and retry records, dialogue scores, aggregate, configuration, and checksum inventory. JSONL files are gzip-compressed. Verification reparsed every raw score, checked every prompt/context and expected key, and reproduced both dialogue scores and the aggregate exactly without API calls. Resume from the original local `outputs/` directory; the result archive is evidence, not an execution directory.
