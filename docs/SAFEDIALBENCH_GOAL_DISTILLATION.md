# SafeDialBench conversation goal distillation

`scripts/distill_safedial_goals.py` extracts one user-goal annotation per
dialogue with **GPT-5.6-sol**, regardless of harmfulness. It extracts the underlying objective from
the conversation as a whole, viewed at the final user turn. Earlier turns can
establish the goal even when the final message does not repeat it. All user
messages and earlier gold assistant messages appear in chronological order.
The final gold assistant answer, evaluated models' responses, and category/attack
labels are excluded. Reference text is preserved without truncation, including
the whitespace-only earlier user turn in dialogue 645.

Protocol **v3** accepts informational, protective, ordinary and harmful goals.
There is no harmfulness classification or requirement to establish harmful intent.
Requests for definitions, examples, prevention or advice count as goals. Asking
about harmful behavior must not be rewritten as intending to perform it.
Each successful extraction must contain a nonempty goal grounded in user text
anywhere in the conversation. Empty goals are reserved for missing information
or unresolved meaning that prevents even a broad objective from being determined;
they remain extraction failures. The prompt cannot guarantee that every case
will produce a valid annotation.

This replaces v2's harmful-only requirement, which caused protective and
informational conversations to receive empty goals. The v2 output remains
unchanged (1,965 successes / 72 empty goals). **Rerun all 2,037 dialogues** in
the fresh v3 directory to apply one consistent goal definition across the dataset.

The default English dataset contains 2,037 dialogues, so a fresh full run makes
2,037 initial API requests, plus at most two retries per transient API failure.
It uses the Responses API with strict JSON Schema, medium reasoning effort,
4,096 maximum output tokens (including reasoning), and four concurrent requests.
The requested model is never silently substituted. See the official
[GPT-5.6 Sol documentation](https://developers.openai.com/api/docs/models/gpt-5.6-sol)
and [Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs).

## Run

From the DCGS root, with `OPENAI_API_KEY` exported or present in `.env`:

```bash
module load Python/3.11.11-GCCcore-13.3.0
.venv/bin/python -B scripts/distill_safedial_goals.py --dry-run
```

Dry-run checks the entire dataset and existing resume state, prints pending
request count, and makes no API calls or output writes. The API key is not
needed for dry-run. Optional `OPENAI_BASE_URL`, `--env-file`, and
`--ignore-environment-proxy` follow the existing API judge conventions.

For a small paid pilot, submit manually:

```bash
sbatch scripts/slurm/run_safedial_goal_distillation.sbatch \
  --ids 1 2 3 \
  --output-dir outputs/safedial_goals/gpt-5.6-sol_conversation_pilot_v3
```

For all dialogues, submit manually:

```bash
sbatch scripts/slurm/run_safedial_goal_distillation.sbatch \
  --model gpt-5.6-sol --parallel 4 \
  --output-dir outputs/safedial_goals/gpt-5.6-sol_conversation_v3
```

The launcher is CPU-only (4 CPUs, 8 GB, 12 hours). For direct execution in an
appropriate environment, use `.venv/bin/python -u -B scripts/distill_safedial_goals.py`
with the same arguments. Both paths make paid API calls. Agents do not submit jobs.

## Outputs and judge integration

Default directory: `outputs/safedial_goals/gpt-5.6-sol_conversation_v3/`.

- `goals.jsonl`: one row per successfully extracted dialogue, keyed by
  `dialogue_id`, with zero-based final `turn_index`, plain-text **`goals`**,
  exact `evidence_quotes` from any user turn (each with its own `turn_index`),
  short `rationale`, and `review_status: "unreviewed"`. No harmfulness or
  execution status field is included. Failed/pending rows are not exported.
- `extractions.jsonl`: append-only attempts and results, including raw API
  responses, returned model, response ID, token usage and failures. Each completed
  request result is flushed and synced before proceeding. `extraction_status`
  is strictly an execution state (`success` or `error`), not a harmfulness label.
- `inputs.jsonl`: exact frozen extraction inputs and their hashes.
- `run_config.json`: dataset, selected IDs, context mode, model, reasoning/token
  settings, prompt/schema, source hashes and hashed endpoint identity.
- `summary.json`: success/error/pending counts, incomplete dialogue IDs and
  errors, and extraction completion. Exports refresh
  every 25 accumulated dialogue results and on normal exit/handled failure;
  the journal is authoritative after abrupt interruption.

Example output shape (illustrative annotation, not a measured result):

```json
{
  "dialogue_id": 7,
  "turn_index": 4,
  "goals": "Obtain insulting language to harass a coworker.",
  "evidence_quotes": [{"turn_index": 0, "quote": "I want to harass a coworker."}],
  "rationale": "The dialogue develops the user's stated harassment objective.",
  "review_status": "unreviewed"
}
```

The exported `goals` field describes user intent, not a harmfulness label.
The existing local DCGS harmful-assistance helper calls its goal text argument
`base_prompt`, but supporting a protective or informational v3 goal must not
automatically be counted as harmful assistance. The future benchmark assistance
runner needs an explicit interpretation/eligibility rule for these goals.
This script does not change or invoke either judge.

Refusals, empty goals, malformed JSON, invalid evidence and truncated/incomplete
API responses are extraction errors. `complete: true` means every selected
dialogue has an extracted goal; it does not establish annotation quality or
human review. Check summary coverage before joining the goal file to answers.
The final `turn_index` records the conversation endpoint, not a restriction on
which turns establish the objective. Per-turn goal applicability is not inferred.

## Resume and configuration changes

Repeat the identical command to continue missing dialogues. Successful
extractions are never requested again. Saved errors require
`--retry-errors`; transient connection/rate/server errors also receive bounded
retries within a request. Refusals and schema errors are not automatically retried.
HTTP configuration/authentication errors stop the run and cancel queued work;
already in-flight API calls may still finish and incur charges.

Model, prompt, code, dataset, selection, endpoint or context changes require a
fresh output directory. Operational concurrency/timeouts/retry settings may be
changed without relabeling saved successes. An OS lock prevents concurrent
writers. A torn journal is rejected explicitly and must be preserved and repaired
before resuming; it is never silently truncated.

Protocol `safedial_conversation_goals_v3` supersedes v2's harmful-only goal
extraction and the earlier final-message classification protocol. Its new
default directory prevents mixing these different annotation definitions.
The updated script rejects attempts to resume the frozen v2 directory; do not
change its manifest or copy v2 successes into v3. The verified previous source
is preserved in `docs/verification/goal_protocol_v3_20260923/v2_source/`.
The former `--context` option has been removed: extraction always uses the
conversation through the final user turn, excluding the final assistant answer
as previously agreed.
`--model`, `--reasoning-effort`, and `--max-output-tokens` are configurable, but
changing them for an existing run is rejected. No temperature or seed is sent.

Exit codes: 0 = extraction complete, 2 = remaining
saved extraction errors, 1 = configuration/input/integrity failure.

## Offline verification

```bash
python3 -B -m unittest discover -s tests -p 'test_distill_safedial_goals.py' -v
python3 -B scripts/distill_safedial_goals.py --dry-run
bash -n scripts/slurm/run_safedial_goal_distillation.sbatch
```

Tests use fake API responses. A real pilot is needed to assess goal quality and
verify account access; offline checks do not establish either.
