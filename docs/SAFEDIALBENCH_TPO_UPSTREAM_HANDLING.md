# TPO upstream candidate handling

The new `run_safedial_tpo_upstream_full.py` entrypoint follows the multi-sample
optimizer extraction in official TPO commit
[`395c3d7`](https://github.com/Simplified-Reasoning/TPO/blob/395c3d763a4c3df0ae72a4352b0db16fe5ee18e9/textgrad-main/textgrad/optimizer/optimizer.py#L198-L206).
It records failures without stopping for malformed update candidates. Existing
v8 code, job 257501, and output files are unchanged.

## Behavior

| Event | Action |
| --- | --- |
| Update has no opening `IMPROVED_VARIABLE` tag, including blank output | Record candidate failure; skip its reward call; continue remaining slots |
| Missing closing tag or repeated opening tags | Apply exact upstream split expression; record a format warning |
| Extracted update is empty | Score the empty candidate, as upstream does |
| Initial response or textual feedback is empty | Retain it, as upstream does; record a warning |
| All updates in a round are skipped | Keep the cumulative earlier candidate pool and continue refinement |
| Highest-reward final answer is empty | Keep the algorithm's selection; record a failed benchmark turn and continue to the next turn |
| CUDA/backend, context overflow, nonfinite reward, or audit-integrity failure | Record a fatal failure and stop; do not misclassify it as malformed text |

There are **no replacement samples or empty-generation retries**. A successful
turn may have fewer than 15 scored candidates (five initial candidates plus up
to five updates in each of two rounds). Every attempted generation remains in
the cost accounting, including skipped update candidates. An empty final answer
does not cause selection of a lower-ranked candidate.

This changes candidate handling, not all benchmark adaptations. Zephyr, the
pinned reward model, SafeDial gold dialogue history, serial HF inference,
deterministic seeds, N=5/D=2, 1024/2048 token budgets, and indexed duplicate
candidates remain. Upstream's string-keyed cache can collapse duplicate answers;
our indexed candidate pool is still an explicitly documented difference.

## Outputs and resume

- `events.jsonl`: raw model results, requests, costs, format warnings, and
  `candidate_failure` for skipped candidates.
- `failures.jsonl`: candidate failures, empty selected-answer failures, and fatal
  execution events, with dialogue/turn/event location and action.
- `turns.jsonl`: completed algorithm outcomes, including empty selected answers
  with `error: "empty_selected_answer"`.
- `answers.jsonl`: native exports of fully successful dialogues only.
- `coverage.json`: processed, successful, failed, and pending turns; skipped
  candidates; empty candidates scored; exported dialogues.
- `validation.json`: replay audit and completeness reported separately.

Candidate skips alone do not make a completed nonempty answer fail validation.
Terminal turn failures do: the runner processes the remaining turns, then exits
2 and reports incomplete successful coverage. The smoke launcher still runs
the validator after that exit. It runs the judge dry-run only if generation and
validation both pass. No paid judging is included.

Resume replays and verifies saved events and never resamples a completed failure.
An interrupted event-to-turn commit is reconstructed from the saved journal.
The failure ledger is derived from authoritative events and turn records and
is rebuilt on resume. Fatal failures block unchanged resume and require
inspection. Never delete failure markers to force a restart.

**Do not point this runner at a v8 directory.** Manifests reject policy mixing.
No v8 migration or reuse is implemented; a fresh run is required for consistent
results under this policy. The currently running v8 job does not adopt these
changes automatically.

## Offline checks and manual GPU smoke

From the DCGS root:

```bash
module load Python/3.11.11-GCCcore-13.3.0
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B \
  scripts/run_safedial_tpo_upstream_full.py --validate-only
bash -n scripts/slurm/run_safedial_tpo_upstream_smoke.sbatch
sbatch scripts/slurm/run_safedial_tpo_upstream_smoke.sbatch
```

The user owns the final submission command. The smoke requests one A40, 64 GB
host memory, eight CPUs, and four hours; gold dialogue 1 contains five turns.
It uses fresh `outputs/safedial_baseline/tpo_zephyr_upstream_smoke_v1` outputs,
the full runner's native 8192-token reward context, and startup capacity probe.
Review real GPU smoke output, failure counts, validation, and cost evidence
before preparing a replacement full-job launcher. No jobs were submitted or
cancelled while implementing this change.

Verification evidence is under `docs/verification/tpo_upstream_handling_20260918/`.
