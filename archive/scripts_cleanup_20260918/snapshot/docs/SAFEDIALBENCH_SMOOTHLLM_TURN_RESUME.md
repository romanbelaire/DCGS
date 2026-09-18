# SmoothLLM turn-level continuation

Prepared 2026-09-15. Use this launcher for the next allocation after legacy job
252143 ends. The running job still uses dialogue-level resume. Its source files,
output directory and original launcher remain unchanged.

```bash
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS
sbatch scripts/slurm/run_safedial_smoothllm_turn_resume_full.sbatch
```

The new launcher requests the same one L40, 64 GB RAM, eight CPUs and 48 hours.
It invokes `run_safedial_smoothllm_turn_resume.py` with `--retry-errors` and:

- Source: `outputs/safedial_baseline/smoothllm_zephyr_full/`
- New output: `outputs/safedial_baseline/smoothllm_zephyr_full_turn_resume/`
- Logs: `outputs/slurm/safedial-smoothllm-turn-resume-<JOBID>.{out,err}`

All Slurm execution remains user-owned. Do not submit an overlapping legacy or
continuation job. The new runner locks its destination and takes the legacy
writer lock for import. If the old job is still writing, import stops before
model loading or copying results. It does not stop or modify the old job.

## Resume behavior

Each completed or handled-error turn is appended and fsynced immediately.
On resume, the runner checks saved run identifiers, seeds, gold history and
successful candidate/vote/cost audits, then reuses successful turns unchanged.
Only missing turns and, with `--retry-errors`, failed turns invoke the model.
A failed turn is attempted once per invocation; later missing turns continue.
Without the retry flag, saved errors remain visible and the invocation returns
failure rather than declaring a complete successful run.

Each retried turn still generates eight candidates, unless a candidate fails
earlier. Checkpointing is per turn: an interruption inside a turn can replay
that turn's candidates. Successful neighboring turns are retained. Completed
dialogue answers are rebuilt from the turn journal, including after interruption
between saving the last turn and exporting the dialogue. Error-bearing answers
remain explicitly marked and cannot pass final generation validation.

For a five-turn dialogue with one failed turn, a successful retry takes eight
model calls, rather than regenerating the four good turns plus the failed turn.
The existing empty-candidate error can still repeat; this change does not alter
the candidate failure policy or count empty outputs as votes.

## Import and provenance

On the first invocation, the importer reads the old output while holding its
writer lock, so it imports the final state available at that time. No stale
live-run snapshot is prepared as benchmark output. Subsequent invocations reuse
the new output without importing again.

The import verifies the legacy dataset/configuration/source manifest, checks
all successful turn audits, and archives exact original files and pinned source
copies under `provenance/legacy/`. Original source files remain unchanged.
`continuation.json` records file hashes, counts and old/new run IDs. Turn records
change only their run/answer identifiers. Duplicate historical entries use the
latest record; raw source journals remain in the archive. A partial last append
is recovered only in the staging copy, preserving original source/archive bytes.
Interrupted or failed partial imports without a manifest are rejected for
inspection; they are not silently mixed with a fresh run.

The new runner pins its own hash and resume policy in a new manifest. It preserves
model revision, eight-copy/10% perturbation settings, greedy decoding, seeds,
1,024-token cap, gold conversation history, voting and candidate selection.
Superseded turn/error attempts and their candidate costs are copied to
`provenance/turn_journals/` before journal compaction. Recovered partial bytes are
retained under `recovery/`. Current-run runtime stats and invocation records stay
separate from archived legacy runtime measurements. Abruptly terminated jobs may
lack final invocation timing; retain Slurm accounting for allocation cost.

The batch runs the dedicated source/identifier/coverage/audit validator and judge
**dry-run** after successful generation. Paid judging is separate.

## Verification

- All **82 repository tests pass**, including interruption after a saved turn,
  retrying just one failed turn, repeated failure while later turns continue,
  stale/missing answer export recovery, no-op resume, saved-data tamper rejection,
  partial append recovery, import lock protection and legacy record preservation.
- Full-input validate-only: **2,037 dialogues / 10,029 turns / 80,232 candidates**.
- Batch `bash -n` passes.
- Offline audit checked all saved successful turns in a read-only live snapshot.
  A separate fixture imported actual dialogue 344: four successful real turns
  were unchanged, eight injected calls replaced the failed fifth turn, and the
  5-turn / 40-candidate audit and no-op resume passed. Fixture outputs are under
  `/tmp`, never in benchmark output. See the adjacent turn-resume review JSON.
- Original SmoothLLM and TPO runtime hashes remain unchanged. No real model/API
  calls or Slurm mutations were performed. New-runner GPU execution remains
  pending the user's next allocation.
