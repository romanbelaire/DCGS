# SafeDialBench two-stage WildJailbreak RDCGS

> Script cleanup (2026-09-18): superseded launchers and one-time migration
> scripts referenced below are now in the [audit archive](../archive/scripts_cleanup_20260918/README.md).
> Use the [current script index](../scripts/README.md) for active entry points.
> The original text below is retained as experiment history.

Updated 2026-09-17: two-stage GPU smoke256826 COMPLETED on analog/A10040GB,
exit0:0,7m09s allocation. All five turns and both selection stages passed the
full validator; no input truncations. Peak allocated GPU memory was27.38GiB
(29,393,395,712 bytes); generation/loading runtime385.99s. No paid judging.
Separate VDCGS/RDCGS full runners are now prepared; see
[full-method launch instructions](SAFEDIALBENCH_DCGS_METHODS.md).
The remaining sections describe the preserved historical smoke policy.

The user selected the August11 high-level weights. Local
`value_function_final.pt` and `value_function_final_aug11.pt` are byte-identical:
SHA-256 `b1ecbe5461dfefe0edf9a2f6fb83277cf494d9744deea0cf56bb58d4bbf6d75d`.
Thus smoke256826 already used the requested weights, including its regret head.
The explicit dated lock for subsequent experiments is
`configs/safedial/dcgs_two_stage_aug11.lock.json`. Pass it with `--artifact-lock`
and use a fresh output directory: changing the stored path changes the manifest
even though the model bytes are identical. The original lock/launcher/output
manifest remain unchanged so the completed smoke can still validate/resume.
Identity evidence: `docs/verification/dcgs_two_stage_20260917/aug11_checkpoint_identity.json`.
No repeat smoke is required solely for this equivalent filename change.

## Run the GPU smoke

From DCGS:

```bash
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS
sbatch scripts/slurm/run_safedial_dcgs_two_stage_smoke.sbatch
```

This requests one A100, 64 GB host RAM, eight CPUs and four hours in
researchshort. It evaluates dialogue 1 (five turns), then validates the full
selection audit and runs a judge dry-run. It does not make paid judge calls.
User owns Slurm execution. Do not submit overlapping writers to the same output.

Output: `outputs/safedial_dcgs/zephyr_wildjailbreak_two_stage_smoke/`.
Logs: `outputs/slurm/safedial-dcgs-two-stage-smoke-<JOBID>.out` and `.err`.
The runner also defaults to dialogue 1 when invoked without a selection.
Job256826 was submitted by the user and completed successfully.

Offline preflight, if needed:

```bash
module load Python/3.11.11-GCCcore-13.3.0
.venv/bin/python -B scripts/run_safedial_dcgs_two_stage.py --ids 1 --validate-only
```

## Frozen experiment inputs

`configs/safedial/dcgs_two_stage.lock.json` pins all Zephyr weight/tokenizer
files and both critic checkpoints by SHA-256. The run manifest additionally
pins the complete local source and prompt files used by the runner.

- Backbone: HuggingFaceH4/zephyr-7b-beta, revision
  `892b3d7a7b1cf10c7a701c60881cd93df615734c`.
- High-level critic:
  `outputs/wildjailbreak_hierarchical_regret_critic/hierarchical_regret_critic/checkpoints/value_function_final.pt`.
  All five high-level heads load strictly; selection uses Q and regret.
- Low-level critic: `models/wildjailbreak_ll_token_critic/ll_token_critic.pt`.
  Both trained `harm_head` and `follow_head` load strictly and are frozen.
  Uses the fetched upstream `LLTokenCritic` Shapley scoring implementation.

The user explicitly chose the WildJailbreak token path despite identical CARES
bytes. The manifest retains that selection and does not invent missing training
dataset provenance. See `models/ll_token_critic_provenance.json`.

Generation and high-level scoring use bfloat16 Zephyr, matching the recorded
high-level checkpoint precision. The contextual token scorer uses a separate
float16 Zephyr encoder and float32 token heads, matching its upstream encoder
specification. Both copies are frozen. The unused random token head in the old
ValueFunction is never instantiated or called by this runner.

## Two selection stages

Every evaluated turn starts from the benchmark's prior gold assistant answers
and current user message. Neither the current gold answer nor future turns or
evaluation labels enter a generation/scoring request.

1. Generate five nominal and five adversarial intent proposals with the existing
   repository templates and numbered-list parser. Each side has a 512-token cap,
   temperature 0.8, top-p 0.9. Unparseable intent slots remain recorded as SKIP;
   all-SKIP fails visibly. There are no hidden proposal retries.
2. Score each valid intent with the frozen Q and regret heads. Select using
   seeded softmax over `0.8 * Q - 0.2 * regret`. Repeated texts retain distinct
   candidate IDs and probability mass. Heads score one unpadded input at a time
   using the repository observation/intent prefix and mean-pooled last hidden state.
3. Generate five independent response samples serially, each conditioned on the
   gold history and selected intent. Each has its own deterministic seed,
   temperature 0.7, top-p 1.0, top-k 0, and 1,024-token cap. The existing single
   response template, RESPONSE prefill/parser and word-suppression processor are
   used. Raw output and parsed candidate are both preserved.
4. Score each response with the contextual token critic. Its prefix contains
   observation and selected intent; only action-token residuals enter the trained
   heads. The Shapley score is the mean over action tokens of
   `0.5 * (harm_head + follow_head)`. Offset mappings include any token crossing
   the prefix/action boundary. Record the action span for every score.
5. Select the maximum token score, breaking ties by the first candidate index.
   Duplicate responses stay indexed. This follows the downloaded critic's
   argmax example; it does not use the old response-softmax implementation.

Serial independent response proposals replace the old multi-response numbered
list and its repeated fallback answers. This is an explicit versioned native
SafeDial adaptation, not a claim of byte-identical legacy generation. Both
stages retain the repository's raw template encoding and separately tokenized
prefill (no additional chat-template wrapping). Nominally each turn performs
seven generations, ten Q/regret scoring calls and five token-scoring calls.
A Q/regret call shares one encoder forward for its two heads.

## Context policy and limitations

No input truncation is allowed. Each actual generation/scoring input is checked
before a model call. Generation uses Zephyr's native 32,768-token context with
room reserved for the requested completion. Both critic contexts are capped at
8,192 tokens. Preflight validates known prompts and checks placeholders for
not-yet-generated intents/responses; actual dynamic inputs are checked again.

The old high-level path truncated at 1,500 tokens. Even dialogue 1's fifth turn
has a 1,735-token placeholder critic input, so this runner explicitly extends
that limit to 8,192 using the native backbone capacity. It preserves all history;
quality beyond the legacy training/input window is not established by a capacity
check. This adaptation is pinned in the policy. Inputs exceeding either critic
limit stop for inspection. A successful short smoke alone does not establish
full-dataset context coverage or peak memory at the longest inputs.

## Persistence and verification

`events.jsonl` fsyncs every generation/scoring result, including raw output,
request, seed, token counts, latency, score, cap flag and token span. Successful
calls replay on resume, including calls from an interrupted unfinished turn.
`turns.jsonl` contains the full candidate/selection audit and all-call costs.
Only complete error-free dialogues are exported to judge-compatible
`answers.jsonl`. Errors stop with exit code 2 and remain saved; retries require
explicit `--retry-errors`. A parsing failure may replay unchanged and needs
inspection rather than blind retries. Failed event attempts remain in the journal.

The manifest and exclusive output lock prevent mixing runs or concurrent writers.
Partial final JSONL appends are archived before recovery. Completed turn audits
and their journal entries must agree before any new work is performed.
The validator replays both selection stages and checks history, IDs, seeds,
costs, answer coverage, journal consistency and source/checkpoint hashes.

`model_loading.json` records strict loading, both backbone dtypes and frozen
parameter counts. `runtime_stats.json` records actual calls, wall time and peak
allocated/reserved GPU memory for the current invocation; `invocations.jsonl`
preserves earlier invocation measurements. No-op resume retains real runtime
measurements. Artifact hashing/preflight time is outside reported GPU runtime.

Preparation verification:

- All 99 repository tests passed, including context leakage, both selection
  stages, duplicate IDs/ties, nonfinite scores, interruption/retry/no-op resume,
  manifest/history/journal tampering, and separate frozen backbone loading.
- Real artifact and tokenizer preflight passed for one dialogue/five turns.
  Maximum intent prompt 2,045 tokens; placeholder response prompt 1,820;
  placeholder high-level input 1,735; placeholder token-critic input 1,743.
- Real seven-head CPU loading and real-tokenizer action-span scoring with
  synthetic encoder residuals passed. No real backbone inference occurred.
- Native dialogue 1 with 110 injected fixture calls passed the full offline
  audit, with byte-identical no-op resume. Fixture output is isolated in /tmp.
- Slurm launcher bash syntax passed. Actual GPU smoke256826 passed; see the result above.

Evidence and reproducible review:
`docs/verification/dcgs_two_stage_20260917/`.
