# Original-DCGS reliability review — 2026-09-18

Implementation update: these recommendations were subsequently implemented in
harness v2. See [verification](../original_dcgs_reliability_v2_20260918/verification.json)
and the [current guide](../../SAFEDIALBENCH_DCGS_PARITY.md). The investigation
below remains the pre-fix record.

Scope: inspect and recommend changes, without modifying runtime sources,
prepared manifests, experiment outputs, or Slurm jobs. Both original-WildJailbreak
smoke folders currently contain only run_config.json, preflight.json and .lock;
there are no actual generated-candidate truncation statistics yet.

## Critic context

Original `src/value/value_function.py:437` sets max_seq_length=1500 and encodes
`Observation: {history}\nHigh-Level Context: {belief}` with truncation enabled.
The actual pinned Zephyr tokenizer has **left** truncation and left padding.
The factory explicitly sets padding side; truncation side comes from the
checkpoint tokenizer. Do not assume that lack of a factory override means
right truncation. This review's preliminary right-truncation concern was
disproved by inspecting actual tokenizer behavior.

The tokenizer-only full-dataset probe used real SafeDial gold histories and
two fixed, short synthetic beliefs. Of 10,029 turns, 3,292 exceeded the cutoff
for at least one probe belief. Dialogue 1, zero-based turn 4, measured 1742 and
1743 tokens. Maximum probe input length was 8000. The two beliefs never became
identical encoded critic inputs. Left truncation retains their trailing text
and removes older history. These are diagnostic counts, not observed model
candidate statistics. Exact actual counts depend on generated belief lengths.
See `critic_context_review.json` and reproducible `inspect_context.py`.

Confirmed reporting weaknesses:

- Preflight returns synthetic scores immediately, without measuring critic
  contexts. It reports the cutoff but no affected turns or dropped lengths.
- Runtime records per-score untruncated lengths and truncation flags, but the
  validator trusts those fields and treats missing flags as an empty list.
- `critic_truncated_inputs` sums score evaluations, including repeated Q,
  Q_min and regret evaluations and abandoned invocations. It is not a count
  of distinct affected turns or candidate contexts.
- The generic `input_truncated_turns` field reads a top-level flag that the
  original adapter never sets, so it can report zero despite critic truncation.

Recommended fix: preserve the original 1500-token encoding, truncation side,
pooling and scoring. Improve diagnostics only. Preflight should show synthetic
critic-length estimates separately from actor prompt capacity. Runtime/audit
should report before/after/dropped token lengths, truncation side, distinct
affected turns and candidate contexts, plus per-head call totals. Validate
these using the actual tokenizer; missing metadata must be unknown or an audit
failure, not zero truncations. Separate `policy_parity_passed` from
`critic_full_context`; a faithful run can have the first true and second false.
Keep successful-turn context counts distinct from abandoned-attempt costs.

Increasing the cutoff, changing truncation side, or summarizing history would
change the method. Such alternatives belong in a separate, consistently
configured cross-benchmark experiment, not an unlabelled SafeDial fix.

## Failure and resume

Injected no-model probes confirmed that a blank final response on turn 2 of a
three-turn dialogue saves turn 1 and then stops. Existing failure.json blocks
all subsequent work on resume, including the untouched third turn. An injected
backend-initialization error creates the same blocking marker. The failure
file has only invocation/error/audit, without explicit turn, stage, category
or recovery decision. See `resume_review.json`.

The adapter wraps several backend exceptions as PolicyFailure, so that class
alone cannot distinguish an invalid model output from infrastructure failure.
The GPU validator also requires nonzero peak memory on every invocation; this
would reject a recovered history containing an initialization failure before
CUDA was available. A new recovery design must handle that explicitly.

Recommended execution-layer changes, preserving per-turn DCGS behavior:

1. Keep fail-fast for smoke. Record typed failure categories and explicit
   dialogue/turn/stage/seed/request/invocation identifiers and original causes.
   Schema/hash/journal corruption and unusable CUDA state remain fatal.
2. Provide an audited recover-into-new-directory operation after inspection.
   Preserve the source run, its failures, journals, runtime evidence and hashes.
   Replay-validate and copy successful turns; preserve their seeds and outputs.
   Require identical method/checkpoint/prompt/generation settings and record any
   harness revision explicitly rather than bypassing source checks.
3. Retry interrupted/transient execution only with the original turn seed and
   configuration, after addressing the infrastructure cause. Do not add new
   seeds or regenerate until success for a completed blank greedy response or
   an exhausted original belief retry budget. Retain all attempt costs and
   explicitly unknown costs from killed in-flight calls.
4. For a full run, predeclare a record-and-continue execution policy for known
   terminal model-output failures. Persist the failed turn and continue other
   counterfactual gold-history turns. This changes batch orchestration, not
   candidate generation/scoring/selection. Keep failures visible in coverage;
   never silently drop them from the benchmark denominator or invent responses.
   Distinguish `execution_finished` from `complete_without_failures`. Native
   complete-dialogue export can remain strict; incomplete coverage is not a
   final benchmark result.
5. Separate integrity, coverage and GPU-evidence validation. Validate GPU data
   for invocations that executed model work; preserve earlier failed startup
   evidence without pretending it had measurable GPU execution.

Useful acceptance checks for a later implementation: exact original requests
and selected answers unchanged; smoke turn 5 clearly reports critic truncation;
validator rejects missing/tampered diagnostics; failed initialization can be
recovered with preserved provenance; known terminal failures are never retried
implicitly; later turns can proceed under the declared full-run policy; saved
successes remain byte-identical; counts expose every failed/unattempted turn.

No runtime fix was implemented in this investigation.
