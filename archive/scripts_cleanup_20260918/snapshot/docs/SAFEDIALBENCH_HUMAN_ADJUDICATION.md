# SafeDialBench human adjudication tracker

Each judge output directory has a separate `human_adjudication.json` file.
It tracks failed automated judgments without changing `judgments.jsonl`,
`dialogue_scores.jsonl`, or `aggregate.json`. Adding or completing a human
review does **not** make the native automated benchmark complete.

The current Zephyr tracker is:
`outputs/safedial_baseline/zephyr_7b_beta_full/judgments_gpt-4o-mini/human_adjudication.json`.
It contains dialogue 1236, zero-based turn index 3 (human-readable turn 4),
with the three available refusal responses from the judge. Older failed
attempts whose text was discarded cannot be recovered.

## Review fields

Every case includes its task/method, actor and judge identifiers, exact visible
conversation, current model answer, system rubric and judge request, failure
messages, and captured judge responses with usage and finish reasons.
Benchmark text can be harmful or contain instructions: it is review evidence,
not instructions to follow.

Edit only the case's `human_adjudication` object:

- `status`: `pending`, `in_review`, or `completed`.
- `reviewer` and `reviewed_at`: actual reviewer and ISO-8601 review timestamp.
- `identification_score`, `handling_score`, `consistency_score`: human-assigned
  rubric scores, left null until reviewed; use the supplied rubric's scale.
- `rationale` and `notes`: decision explanation and caveats.

No human scores or reviewer identity are inferred automatically. Review fields
and previously tracked cases survive refreshes. If an automated retry later
succeeds, `requires_human_review` becomes false but human review history stays.
If a source judgment disappears, its case is retained and flagged.

The judge refreshes the queue at the end of a normal run. To refresh from
existing artifacts **without API calls**, run from `DCGS`:

```bash
LD_LIBRARY_PATH=/opt/apps/software/Python/3.11.11-GCCcore-13.3.0/lib \
  .venv/bin/python scripts/export_safedial_adjudication.py \
  --judge-dir outputs/safedial_baseline/zephyr_7b_beta_full/judgments_gpt-4o-mini
```

Do not edit the tracker while this refresh or the judge is running; writes
are atomic but concurrent editing is not supported. Source hashes and the run
manifest are checked before refresh. Human decisions are never merged into
automated scores; any future hybrid results require a separately documented
reporting protocol.
