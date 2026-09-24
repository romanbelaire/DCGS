# Resuming VDCGS 96 after job 296226

Job 296226 stopped during the CPU replay audit, before loading the generation
backend. The first failing saved record is dialogue 13, turn index 4 (fifth
turn). The saved final response and replayed final response are equal.

The original `replay_trace` hands an event's `result` dictionary directly to
the original policy. When an all-SKIP belief attempt is retried, the policy
replaces an entry in its `raw_outputs` list. That list aliases the saved event's
`result.texts`. The checker therefore mutates its own in-memory input evidence
and reports `Original-policy selection/audit mismatch`. The files on disk are
not changed by that failed audit.

`scripts/safedial_dcgs_replay_isolation.py` gives the original replay function
a deep copy of the trace. It applies to successful turns, failure replay, and
interrupted-call reconciliation. Every original request, context, policy, and
journal comparison still runs. Generation functions, source-pinned files,
manifests, seeds, saved responses, and terminal failure policy are unchanged.
The new entrypoint logs SHA-256 hashes of the two added Python files for
provenance. Use it for subsequent audits as well as generation resumes.

## Verification

Evidence is under `docs/verification/vdcgs96_resume_20260923/`:

- `audit.json`: full saved-state audit with the actual tokenizer, source and
  prospective manifest checks, coverage, and before/after journal hashes.
- `first_failure.json`: reproduction on dialogue 13, turn index 4, including
  identical response and belief selection after the fix.
- `artifacts.json`: actor and both critic artifact hashes checked against the
  existing artifact lock.
- Three regression tests reproduce the pre-fix failure, check repeatable
  non-mutating replay, reject response tampering, and cover exhausted retries
  and incomplete traces. The 23 existing original-policy/recovery tests also
  pass with replay isolation enabled.

The full CPU audit passed in 388 seconds: all 8,974 successful turns and 17
terminal failures validated, no blocking failure, no uncommitted completed
turns, and the native answer export matches. All six saved file hashes were
unchanged. No model inference or paid API calls were made.

## Resume command

Run from the DCGS project root after reviewing `audit.json`:

```bash
sbatch scripts/slurm/run_safedial_vdcgs_replay_isolated_full.sbatch
```

This uses the existing `zephyr_vdcgs_main_wildjailbreak_full_v3` directory,
one L40S, 64 GB RAM, 8 CPUs, and a 48-hour researchlong allocation. Submission
is user-owned per `.agents.md`. No job was submitted during preparation.

The 17 recorded terminal output failures remain failed and are not retried.
There are 1,038 remaining turns after the 8,974 saved successful turns and
17 terminal failures. One interrupted call has unknown cost; its unfinished
turn resumes with its original seed under the existing interruption policy. A final exit code 2 can still mean generation exhausted
all pending work with those terminal failures preserved; inspect coverage and
logs instead of equating that exit code with this startup audit bug.

For a later standalone validation using the fix:

```bash
module load Python/3.11.11-GCCcore-13.3.0
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B \
  scripts/run_safedial_dcgs_replay_isolated.py validate \
  --output-dir outputs/safedial_dcgs/zephyr_vdcgs_main_wildjailbreak_full_v3 \
  --allow-incomplete
```

The validation command writes the normal `validation.json`; it does not run
model inference. The old launcher still has the replay bug and should not be
used to resume this saved run.
