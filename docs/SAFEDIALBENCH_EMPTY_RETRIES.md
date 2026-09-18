# Bounded empty-generation recovery

> Script cleanup (2026-09-18): superseded launchers and one-time migration
> scripts referenced below are now in the [audit archive](../archive/scripts_cleanup_20260918/README.md).
> Use the [current script index](../scripts/README.md) for active entry points.
> The original text below is retained as experiment history.

**2026-09-18 DCGS status:** the user now requires parity with the original
method across benchmarks. The DCGS continuations below are preserved adapted
experiments, not the final recommended runs. Do not submit them for that
comparison. Reference selection and required changes are tracked in
[DCGS parity review](SAFEDIALBENCH_DCGS_PARITY.md). This note does not change TPO.

DCGS policy v3 and TPO policy v8 add up to **two extra attempts** for an empty
sampled generation. Attempt zero retains the original prompt, seed, settings,
and token budget. Each retry uses a distinct deterministic seed derived from
the original seed and retry number. The first valid result fills the original
candidate slot; successful calls are reused, and subsequent slots proceed as
before. There are still five response candidates per DCGS turn and fifteen
response candidates per full TPO turn.

Every attempt has a separate fsynced journal event, including failed attempts.
Token and timing totals include those failures. Resuming after interruption
replays saved attempts and makes only the next missing call. Repeated resumes,
including `--retry-errors`, do not reset an exhausted empty-generation budget.
Three empty attempts leave an explicit failed turn and stop the job.

Recovery applies only to sampled generation with valid input/token accounting
and empty or whitespace-only decoded output (also response-tag-only DCGS
output). Backend errors, OOM, truncation, nonfinite scores, malformed TPO update
tags, and other errors still stop for inspection. A substantive answer followed
by trailing whitespace remains valid. The decoder now records generated token
IDs, raw decoded text before stripping, and its EOS/token-cap/other stop reason.
Historical calls retain their original records; missing historical token IDs
are not fabricated.

The CLI option is `--empty-generation-retries 2` (default); `0` disables retries.
The policy and implementation hashes are pinned in each manifest. Recovery
changes the effective generation policy and its cost, so use the prepared new
continuations instead of writing into the old run directories.

## Prepared continuations

| Method | New directory under `outputs/` | Preserved successful turns | Preserved events |
| --- | --- | ---: | ---: |
| VDCGS | `safedial_dcgs/zephyr_vdcgs_aug11_a5000_full_retry_v1` | 29 | 473 |
| RDCGS | `safedial_dcgs/zephyr_rdcgs_aug11_a5000_full_retry_v1` | 100 | 2,214 |
| TPO | `safedial_baseline/tpo_zephyr_full_v8` | 140 | 4,783 |

Each import preserves the original event requests/results, including the failed
attempt, and all successful responses. Only run/answer identity changes for
the versioned continuation. The failed turn is represented by its event journal
so the next invocation automatically starts the first new-seed retry. Each
directory has `continuation.json` and verified copies of the original run,
nested provenance, and original source. Original benchmark outputs are intact.
Original runtime measurements are archived; new GPU evidence is required for
the continuation and must not be inferred from archived measurements.

From the DCGS root, submit manually:

```bash
sbatch scripts/slurm/run_safedial_vdcgs_a5000_retry_full.sbatch
sbatch scripts/slurm/run_safedial_rdcgs_a5000_retry_full.sbatch
sbatch scripts/slurm/run_safedial_tpo_retry_full.sbatch
```

DCGS requests two A5000s per job; TPO requests one A100. Each requests 48 hours
in `researchlong`. Launchers run their existing capacity probe, generation,
full GPU/output audit, then judge dry-run only. No paid judging is included.
Resubmit the same launcher after a timeout to resume. Inspect any saved error
before further action; do not change seeds or remove failed events manually.
The old full launchers still point to the old manifests and are not suitable
for continuing under the new source/policy identity.

## Verification and limits

All 135 repository tests pass. Dedicated cases cover retry exhaustion, distinct
seeds and hash collisions, interruption before/after each attempt, reuse of
successful calls, unchanged candidate counts, all-attempt costs, first-valid
selection, non-retryable failures, tag-only output, diagnostics, and audit
tampering. Three real-checkpoint offline continuations pass native-output and
event audits and byte-identical no-op resume. Only injected fixture calls in
temporary directories filled the missing generations; they are not benchmark
model results. Full dataset artifact/tokenizer preflight results are recorded
under `docs/verification/empty_retry_20260918/`.

The retry implementation has not yet run on a GPU. Different seeds can recover
empty samples but cannot guarantee success. No Slurm jobs were submitted or
mutated during implementation.

SmoothLLM uses greedy decoding. Its active runtime files and policy were left
unchanged (all four pinned hashes verified). Random-seed retries would not
address its persistent blank candidate; it needs a separate validated policy.

Reproducible import: `scripts/prepare_safedial_empty_retry.py` accepts method,
source directory, fresh output directory, and original source snapshot. The
real-checkpoint offline check is
`docs/verification/empty_retry_20260918/review_continuations.py`.
