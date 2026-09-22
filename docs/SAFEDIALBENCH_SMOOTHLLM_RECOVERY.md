# SmoothLLM single-attempt supplemental recovery

The original run has 10,028 successful turns and one failed turn: dialogue 344,
turn index 4 (fifth turn). Four original-seed attempts produced an empty decoded
first candidate after 1,024 tokens. Latest original retry: job 258863, exit 2.

This recovery is a **separate supplemental experiment**, not an exact completion
of the original fixed-seed protocol. It makes one predeclared fresh-seed attempt
at that turn. It never changes the source journals or existing judging results.

## Run

From the DCGS root, submit manually:

```bash
sbatch scripts/slurm/run_safedial_smoothllm_recovery.sbatch
```

The launcher requests one L40, 64 GB RAM, eight CPUs and one hour in
`researchlong`. It performs a read-only source preflight, generation, GPU/audit
checks and a judge **dry-run**. It makes no paid API calls. Agents do not submit
or mutate Slurm jobs; give the resulting job ID to the agent for inspection.

Optional preflight, without GPU inference or creating the recovery directory:

```bash
module load Python/3.11.11-GCCcore-13.3.0
.venv/bin/python -B scripts/run_safedial_smoothllm_recovery.py --validate-only
```

## Predeclared policy

- Target: dialogue 344, index 4 only. Original seed: `644269963`.
- Recovery seed: **`1844738519`**, derived before any generation as
  `int(stable_id(original_seed, "smoothllm-single-fresh-seed-recovery-v1", 1, length=16), 16) % (2**31-1)`.
- Eight fresh candidates, 10% current-user random character swaps, the same
  pinned Zephyr model, greedy decoding, 1,024-token cap and gold dialogue history.
- The new seed drives perturbations, candidate seeds and the existing random
  selection within the majority. It is not a change to decoding temperature.
- All eight candidates must be valid before the unchanged voting rule applies.
  Accept the selected answer regardless of its safety score. No selection based
  on judge results, no seed search, no candidate retries or dropping blank votes.
- Any candidate failure stops this attempt and preserves its partial audit.
  A failure or infrastructure error returns exit 2 and exports no answer.
- A durable marker is written before model loading. Completed invocations only
  re-audit/re-export on rerun, with no model calls. An interrupted invocation
  with no final result refuses automatic retry. Do not delete the marker or use
  another output directory to obtain another draw; inspect and document any
  future policy change first.

## Artifacts and preservation

Source (read-only):
`outputs/safedial_baseline/smoothllm_zephyr_full_turn_resume/`.

Separate output:
`outputs/safedial_baseline/smoothllm_zephyr_recovery_344_turn5_v1/`.

- `run_config.json`: policy, seed, original source manifest and SHA256 hashes.
- `original_dialogue.json`: all five original records, including the failure.
- `attempt_started.json`: single-attempt guard.
- `candidate_calls.jsonl`: durable returned candidate results and their costs.
- `result.json`: recovered turn or error, candidate/voting audit, GPU/runtime data.
- `dialogue_turns.jsonl`: on success, four original records unchanged plus the
  recovered fifth record. Original records retain their original provenance;
  the recovered record has distinct recovery identifiers and seed metadata.
- `answers.jsonl`: on success, **one complete five-turn dialogue**, not a full
  benchmark export. On failure it is empty.
- `validation.json`: separate audit status and recovery success flag.

The runner checks all original successful-turn audits, coverage, original source
hashes and dataset integrity before inference. It holds a read-only shared source
lock and an exclusive output lock; a concurrent original writer is rejected.
Source file hashes are checked again before exporting the recovery result.

Logs: `outputs/slurm/safedial-smoothllm-recovery-<JOBID>.{out,err}`.

Saved-result audit (no generation):

```bash
.venv/bin/python -B scripts/run_safedial_smoothllm_recovery.py --audit-only --require-gpu
```

## Reporting and judging

Keep the original incomplete generation record and the existing 2,036-dialogue
judging aggregate as the primary result. If recovery succeeds, its exported
five-turn dialogue can be judged separately later. All five turns need judging
because the original successful-dialogue judge subset excluded dialogue 344.
Any combined 2,037-dialogue aggregate must be stored and labelled separately as
including this supplemental one-attempt recovery. The launcher does not create
or modify an aggregate or judge any response.

Offline tests cover success with exactly eight calls, preservation of successful
neighbours, blank-candidate failure without voting, interrupted/failed attempt
guards, source/seed/export tampering, read-only preflight and source locking.
GPU recovery passed on 2026-09-20: user-submitted job `258867` completed with
exit 0 after 2m24s on lagoon/L40. All eight candidates were nonempty; final
GPU/audit checks and the five-turn judge dry-run passed. Original source hashes
and the four preserved dialogue turns match exactly. Paid judging and a
supplemental combined aggregate remain pending.
