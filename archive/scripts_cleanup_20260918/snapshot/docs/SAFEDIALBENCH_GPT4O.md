# SafeDialBench GPT-4o responder

M3 generates answers with the exact `gpt-4o-2024-08-06` snapshot through the
official OpenAI Chat Completions API. This follows the GPT-4o arm in the
experiment plan. It runs entirely on CPU; model inference happens at OpenAI.
The [official model page](https://developers.openai.com/api/docs/models/gpt-4o)
lists this snapshot and endpoint. Account metadata access and actual generation
access are distinct; the runner checks metadata first and stops on request failure.

From the DCGS root:

```bash
module load Python/3.11.11-GCCcore-13.3.0
.venv/bin/python -B scripts/run_safedial_api.py --output-dir outputs/safedial_baseline/gpt4o_full --validate-only
.venv/bin/python -B scripts/run_safedial_api.py --output-dir outputs/safedial_baseline/gpt4o_full --check-access
sbatch scripts/slurm/run_safedial_gpt4o_full.sbatch
```

The user submits Slurm jobs. The batch requests two CPUs, 8 GB RAM, 48 hours
in researchlong, and no GPU. It makes paid generation calls for all 2,037
dialogues / 10,029 turns. The final judge command is a dry-run only; paid
gpt-4o-mini judging remains a separate step.

For an optional one-dialogue live check before full submission:

```bash
.venv/bin/python -u -B scripts/run_safedial_api.py --ids 1 --output-dir outputs/safedial_baseline/gpt4o_smoke
.venv/bin/python -B scripts/run_safedial_api.py --ids 1 --output-dir outputs/safedial_baseline/gpt4o_smoke --validate-output
```

Credentials load from exported `OPENAI_API_KEY` or the root `.env` without
shell evaluation. A nonofficial `OPENAI_BASE_URL` is rejected to prevent mixing
provider provenance. No credential values are written to the manifest or audits.

The runner uses native gold assistant history, with no generated-answer feedback,
system prompt, rubric or future-turn leakage. Direct response settings are
temperature 0.7, top-p 1, one choice, and 1,024 maximum completion tokens.
Per-turn seeds are requested and backend fingerprints saved; API determinism is
[best effort](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).
No input truncation is applied. Output-cap truncation is recorded and retains
the actual response. Explicit refusal text is preserved as the model's response;
A `content_filter` response, with or without partial text, is retained as an
unresolved case in `filtered_cases.json` and the raw API audit. It does not stop
generation or get automatically retried. It is excluded from successful turns
and complete-dialogue answers; no refusal or score is invented. Empty unfiltered
responses and invalid provenance still stop the run.

Two concurrent requests at most; transient failures retry up to four attempts
with exponential backoff. Exhausted transport/API failures and other invalid responses stop further batches;
content-filter outcomes are recorded and processing continues.
Rerun the identical batch to resume successful saved turns, including partial
dialogues. `answers.jsonl` is atomically rebuilt from complete dialogues;
`turns.jsonl` is fsynced after each saved response. A file lock prevents concurrent
writers. Unterminated final JSONL writes are archived before recovery; malformed
complete lines fail validation. In-flight responses lost before a durable save
may require another paid request after preemption.

`run_config.json` pins dataset, selected IDs, model, decoding and source hashes.
Do not edit the runner/shared history helper during a run; incompatible resumes
are rejected. `api_attempts.jsonl` retains returned responses, usage, finish reason,
model ID, request IDs where available and sanitized errors across retries.
Usage for requests that fail in transport may be unknown. Successful turn records
also contain usage and provenance. No dollar estimate is asserted here.

Verification on 2026-09-10: seven offline tests PASS; all-input validation PASS;
all three full batch scripts pass bash syntax. Model metadata access PASS for the exact snapshot through the official API.
No GPU or paid API generation performed during preparation. No Slurm submission or mutation by the agent.

Repair verified 2026-09-11: all 11 API tests PASS, including empty/partial
filter continuation, legacy-audit recovery, no repeated filtered calls, fatal
failure handling, exact manifest migration, and partial coverage validation.
The original GPT-4o run's manifest was migrated offline with its prior runner
and manifest archived under `gpt4o_full/provenance/`. Dataset, model, decoding,
answers, turn records and raw audit hashes are preserved. Resume is ready:
6,983 saved turns, one filtered case (1436, turn index 0), 3,045 pending requests.
The user resubmits the same full sbatch script; no agent submission occurred.

The batch now uses `run_safedial_api.py --validate-output`. Validation distinguishes
`processing_complete` (every turn either succeeded or was filtered) from
`complete` (every turn succeeded). Filtered cases keep `complete: false` even
when processing passes and the batch exits successfully. `answers.jsonl` contains
only fully successful dialogues; successful turns within excluded dialogues
remain in `turns.jsonl`. Paid judging is still separate and must disclose the
coverage gaps; a subset judge aggregate is not a complete benchmark result.

`--prepare-resume` is an offline, locked migration for the known original runner
hash only. It archives the previous manifest and records the implementation
transition without changing request settings. It has already been run for this
output directory. Other source/settings mismatches remain rejected.
