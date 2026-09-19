# DCGS parity with the original benchmark implementation

**Historical v2 documentation.** The active runner has migrated to [main-method v3 with trained LL reranking](SAFEDIALBENCH_DCGS_MAIN.md). Reproduce this page's original policy using checkout `4e2f493`.

2026-09-18: the user requires the original DCGS approach to remain consistent
across benchmarks. This supersedes recommending the custom DCGS empty-retry
continuations as the final method. Existing outputs and prepared continuations
remain preserved evidence of an adapted policy. No jobs have been cancelled,
submitted, or otherwise changed.

## Verified differences

| Component | Original local src path | Adapted SafeDial full methods |
| --- | --- | --- |
| HL failures | main.py retries the entire failed episode pool when every belief is SKIP, up to3 total attempts; optional iterative mode retries individual beliefs | custom stage generation and newly added per-empty-generation retries |
| HL iterative mode | opt-in; default false; when enabled, regret candidate allocation also differs | currently one list per nominal/adversarial side |
| LL generation when pool enabled | LowLevelAgent.generate_ll_candidates: one numbered-list call,128*K token budget | K independent serial samples,1024 tokens each |
| LL missing candidates | one greedy single-response generation,128 tokens; same fallback copied into missing slots | stop on failed slot, or new custom retry policy |
| LL selection when pool enabled | src.main._generate_or_select_ll_action uses softmax sampling from a string-keyed score mapping | first-index argmax of contextual harm/follow scores |
| Duplicate handling | string-keyed Q/LL score dictionaries merge identical strings | indexed duplicates remain separate |
| LL conditioning | config-controlled ll_action_belief_only; default true | gold history plus selected intent |
| LL scorer | ValueFunction.predict_ll_candidate_scores uses its token_critic_head and action-only encoding | separately downloaded trained LLTokenCritic harm/follow heads with contextual encoding |
| Prompt and score contexts | original helpers include their own budgets/truncation | full native context guards with no truncation |

The original local safedial_dcgs.json sets environment_type=wildjailbreak,
ll_candidate_rerank=false, ll_action_belief_only=true and a freeform HL policy.
It therefore selects a single greedy LL response, not a candidate pool. This
file is a prior SafeDial adapter config and is not evidence of the configuration
used by the user's other benchmark runs. The repository contains multiple
branches; source availability alone does not identify the reference experiment.

The downloaded src/value/ll_token_critic.py is not called by the original
src/main.py/ValueFunction selection path. ValueFunction initializes a separate
token head, while its checkpoint loading shown in this version loads Q/V and
regret-related heads. Do not silently replace the explicitly chosen trained
harm/follow scorer with that legacy head and call this method parity.

Original src files still match the snapshot made before the retry fix. An
offline probe of the exact original LL methods (AST extracted, injected model
outputs) confirms that0/2/5 parsed initial candidates produce one pool-generation
call and at most one fallback, with duplicates filling missing slots. Evidence:
verification/original_dcgs_parity_20260918/original_ll_fallback_probe.json.

## Implemented SafeDial adapter (2026-09-18)

The user selected **WildJailbreak** and explicitly requested maximum reuse of
original DCGS code. The new entry point is
`scripts/run_safedial_dcgs_wildjailbreak.py`, backed by
`scripts/safedial_dcgs_wildjailbreak.py`. It calls the original functions directly:

- `src.main.batch_generate_beliefs_for_episodes`
- `src.main.batch_compute_q_values_for_episodes`
- `src.main._select_high_level_belief`
- `src.main._generate_or_select_ll_action`
- Original `FreeformHighLevelAgent`, `LowLevelAgent`, `PromptManager`,
  `EpisodeState`, `ValueFunction` and `batch_generate` implementations.

The existing `initialize_dcgs` loader is reused. Generation/scoring boundaries
record requests and results without replacing the original parsers, selection,
retry loops, or generation helper. This adapter is serial; its scoped audit
patches must not be used concurrently in one process.

Configuration basis: the local `safedial_dcgs.json` WildJailbreak branch plus
original `BaseConfig` defaults. The user identified the benchmark, not a saved
historical run configuration; exact historical-run equivalence is therefore
not established. Effective settings are saved in each manifest. VDCGS disables
regret and adversarial pool expansion; RDCGS keeps them enabled. Both use the
user-selected August 11 high-level checkpoint and pinned Zephyr revision.

This branch uses five nominal beliefs (plus five adversarial beliefs for
RDCGS), non-iterative 96-token belief-list generation at temperature 0.8,
original all-SKIP retries, original Q/regret selection, and one greedy
128-token low-level response conditioned on belief only. LL reranking is off;
the separate downloaded token critic is unused. Original critic truncation at
1500 tokens is retained and recorded, rather than extended for SafeDial.

SafeDial-specific adaptation is limited to gold-history conversion, using the
current user message as the adversarial base-prompt anchor, independent turn
seeds, audit/resume, and native answer export. The current reference answer is
never passed to the policy. No new per-empty-call retries are added.

The current execution harness is protocol v2. Completed turns are reused.
Interrupted turns restart with their original seed and preserve call-start and
result journals. If a completed response/failure was journaled before a crash
but not yet committed as a turn, replay commits it without generating again.
Killed in-flight calls can have unknown costs, which are reported explicitly.
No-op resume does not load the actor or rewrite completed outputs. Prior custom
DCGS outputs and v1 manifests cannot be imported silently into v2.

Verification: all 117 active repository tests pass, including direct original
policy parity, context-metadata tampering, typed failures, fresh-directory
recovery, interruption reconciliation, preserved seeds/records, and source
locking. Both real artifact/tokenizer preflights pass (one dialogue/five turns).
Native five-turn injected fixtures pass replay and byte-identical no-op resume.
A spy on the original critic encoder confirms that diagnostics hash the exact
1500 retained tokens. Original batch_generate instrumentation also passes.
Evidence: `verification/original_dcgs_reliability_v2_20260918/`. These initial
checks were offline; the subsequent A40 GPU smoke review is recorded below.

From the DCGS root, the user can submit the prepared one-A100 smoke runs:

```bash
sbatch scripts/slurm/run_safedial_vdcgs_wildjailbreak_original_smoke.sbatch
sbatch scripts/slurm/run_safedial_rdcgs_wildjailbreak_original_smoke.sbatch
```

Each launcher runs preflight, generation, original-policy/native-output audit
with `--require-gpu`, and a judge dry-run. Output folders are
`outputs/safedial_dcgs/zephyr_{vdcgs,rdcgs}_original_wildjailbreak_smoke_v2`.
Both launchers pass `bash -n`. The user subsequently submitted A40 jobs257599
and257600; both passed the new GPU smoke gate. Historical custom-method smoke
results are not the basis for this gate. TPO and SmoothLLM remain unchanged.

## V2 context diagnostics

The original critic still uses its 1500-token cutoff and the pinned tokenizer's
left truncation. This retains the trailing belief while dropping older history.
Changing the cutoff, tokenizer side, or history representation is a separate
method change, not part of this reliability fix.

Preflight reports `estimated_critic_context` for synthetic beliefs. Both smoke
preflights identify dialogue1/index4 as truncated (1743 input tokens,243 dropped).
Runtime records before/retained/dropped lengths, side and encoded-token hashes.
The validator recomputes these using the tokenizer and rejects missing or
altered metadata. Final reports distinguish:

- `policy_parity_passed` and `complete_without_failures` from `critic_full_context`;
- distinct `critic_truncated_turns` and `critic_truncated_candidate_contexts`;
- repeated per-head call/input totals for Q, Q_min and regret;
- successful-turn context from failed/interrupted-attempt context and costs.

A passing smoke may legitimately report `critic_full_context: false`. This is
original-policy truncation, not a failed execution. The misleading generic
`input_truncated_turns: 0` field is no longer emitted by this validator.

## V2 failure policy and recovery

Default `--on-turn-error stop` remains fail-fast and is used by both smoke
launchers. `failures.jsonl` records category, code, original exception, stage,
dialogue/turn, seed, invocation, request and complete available audit. A blocking
`failure.json` is backed by that ledger; deleting the marker cannot reset it.

The optional `--on-turn-error record-and-continue` must be selected when creating
a run and is pinned in its manifest. It records exhausted original belief pools
or completed blank final responses as terminal outcomes, then processes later
independent gold-history turns. It never retries those outputs. Infrastructure,
unknown-code/schema and journal-integrity errors stop execution in either mode.
The full launchers below use record-and-continue following the passed GPU smoke.

`execution_finished` means every selected turn has an outcome;
`complete_without_failures` requires every turn to have a successful answer.
Any terminal failures keep exit status2 and full validation `passed: false`.
Only fully successful dialogues enter native answers.jsonl. Do not treat a
success-only subset as the full benchmark or drop failures from its denominator.
`--allow-incomplete` audits partial results but does not change those statuses.

After inspecting a v2 failure and resolving its infrastructure cause, prepare
an identical-policy continuation in a fresh directory. This is a CPU/tokenizer
operation; it does not start inference. For example, for the RDCGS smoke:

```bash
module load Python/3.11.11-GCCcore-13.3.0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python -B scripts/validate_safedial_dcgs_wildjailbreak.py \
  --output-dir outputs/safedial_dcgs/zephyr_rdcgs_original_wildjailbreak_smoke_v2 \
  --allow-incomplete
.venv/bin/python -B scripts/run_safedial_dcgs_wildjailbreak.py \
  --method rdcgs --ids 1 --seed 0 --device cuda:0 --on-turn-error stop \
  --recover-from outputs/safedial_dcgs/zephyr_rdcgs_original_wildjailbreak_smoke_v2 \
  --output-dir outputs/safedial_dcgs/zephyr_rdcgs_original_wildjailbreak_smoke_v2_recovery1 \
  --recovery-reason 'Describe the inspected failure and the resolved cause here'
```

Use the actual inspection reason. Recovery locks the source, validates every
trace, preserves a complete source snapshot and hashes, copies successful turns
byte-for-byte, and records the acknowledged failures. Method, source hashes,
weights, selection, seeds and execution policy must be identical. It refuses
occupied destinations, active writers, damaged journals and integrity failures.
It supports v2-to-v2 recovery; a code/policy revision needs a separately reviewed
migration rather than bypassing manifest checks.

Run the continuation only inside a user-owned GPU allocation, using the same
runner arguments and the fresh `--output-dir`, with `--recover-from` omitted.
The normal smoke launchers still target their original v2 folders, so a recovery
allocation must explicitly target the new directory. No recovery GPU job is
submitted automatically. A terminal blank remains failed and is never retried;
recovery can only process other unattempted turns in that case.

The GPU audit checks loading/memory for invocations contributing successful
turns. Earlier startup failures need not invent a GPU peak. Unknown historical
model-work memory and unknown killed-call costs are shown explicitly; missing
GPU evidence for successful work still fails the GPU gate.

## A40 smoke allocation

The same original-policy smoke scripts can request one A40 instead of an A100
by passing `--gres=gpu:a40:1` to `sbatch`. This changes only the allocation;
model precision, checkpoint, prompts, seeds and validation remain unchanged.
The runner uses one GPU; requesting three GPUs will not accelerate this serial
adapter. Each method should receive one GPU. A40 peak memory and runtime still
need to pass the smoke gate; hardware changes do not promise identical outputs.

As of the 2026-09-18 check, user-submitted A100 jobs257594/257595 were pending.
To replace these specific queued jobs, the user can run:

```bash
scancel 257594 257595
sbatch --gres=gpu:a40:1 scripts/slurm/run_safedial_vdcgs_wildjailbreak_original_smoke.sbatch
sbatch --gres=gpu:a40:1 scripts/slurm/run_safedial_rdcgs_wildjailbreak_original_smoke.sbatch
```

These reuse the existing v2 output directories, which contained only prepared
preflight/config files at that check. Do not submit duplicate writers while the
old jobs are active. If an old job has since produced runtime results or failed,
inspect its state before using these replacement commands; failure recovery
must follow the audited procedure above. Agents do not execute these commands.

## Reviewed A40 smokes and full runs (2026-09-18)

Jobs257599/257600 completed with exit0. Both audits were rerun with
`--require-gpu` and passed source identity, original-policy replay, native
answers, checkpoint loading and memory evidence. Each saved5/5 nonempty turns,
with no failed or interrupted calls and no custom empty retries.

| Measurement | VDCGS | RDCGS |
| --- | --- | --- |
| Whole Slurm job | 1m57s | 2m00s |
| Recorded model invocation | 41.607s | 65.505s |
| Peak allocated GPU memory | 14.25GiB | 14.30GiB |
| Generation calls | 10 | 15 |
| Non-SKIP beliefs per turn | 2,3,2,2,2 | 4,4,4,4,4 |
| Critic-truncated turns | 1/5 | 1/5 |
| Maximum critic tokens dropped | 289 | 298 |

All HL calls used the full96-token budget; all LL calls used128 tokens.
Many responses visibly end mid-sentence; some include unfilled placeholders,
and RDCGS turn5 answers about finances after selecting an off-topic belief.
These are material response-quality limitations, not evidence of faithful
implementation failing its replay checks. Keep the original budgets, parser,
SKIP padding, sampling and1500-token critic cutoff for the requested benchmark
consistency. This one-dialogue smoke establishes operational readiness only;
it does not establish safety/quality scores or full-dataset success. Only judge
dry-runs have been performed.

Prepared full launchers each request one A40,64GB host RAM,8CPUs and48hours on
researchlong, with all2037 dialogues/10029 turns, seed0 and the smoke's original
DCGS model settings. They use distinct `*_full_v2` folders and predeclare
`--on-turn-error record-and-continue`. Terminal outputs remain failed with no
new samples; infrastructure/integrity failures stop. Partial audits preserve
nonzero generation exit codes; complete runs require the GPU audit and judge
dry-run. No paid judging or automatic job submission is included.

```bash
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS
sbatch scripts/slurm/run_safedial_vdcgs_wildjailbreak_original_full.sbatch
sbatch scripts/slurm/run_safedial_rdcgs_wildjailbreak_original_full.sbatch
```

Outputs: `outputs/safedial_dcgs/zephyr_{vdcgs,rdcgs}_original_wildjailbreak_full_v2`.
Logs: `outputs/slurm/safedial-{vdcgs,rdcgs}-original-full-<job-id>.{out,err}`.
The launchers repeat full synthetic preflight before inference; real generated
context is checked dynamically. Serial throughput varies by dialogue;48hours
is an allocation limit, not a completion guarantee. Identical relaunch resumes
saved turns after a scheduler interruption without a fatal recorded failure.
A saved fatal failure requires the inspected fresh-directory recovery above.

For a full RDCGS infrastructure recovery, after inspecting and fixing its cause:

```bash
module load Python/3.11.11-GCCcore-13.3.0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python -B scripts/validate_safedial_dcgs_wildjailbreak.py \
  --output-dir outputs/safedial_dcgs/zephyr_rdcgs_original_wildjailbreak_full_v2 \
  --allow-incomplete
.venv/bin/python -B scripts/run_safedial_dcgs_wildjailbreak.py \
  --method rdcgs --seed 0 --device cuda:0 --on-turn-error record-and-continue \
  --recover-from outputs/safedial_dcgs/zephyr_rdcgs_original_wildjailbreak_full_v2 \
  --output-dir outputs/safedial_dcgs/zephyr_rdcgs_original_wildjailbreak_full_v2_recovery1 \
  --recovery-reason 'Replace this with the inspected failure and resolved cause'
sbatch scripts/slurm/run_safedial_rdcgs_wildjailbreak_original_full.sbatch \
  outputs/safedial_dcgs/zephyr_rdcgs_original_wildjailbreak_full_v2_recovery1
```

The final submission is user-owned. For VDCGS replace both the method and path
names. The optional positional launcher argument selects the prepared recovery
folder without editing pinned runtime code. Never submit concurrent writers
for the same output directory, or change policy/source to bypass a failed run.

Full-dataset CPU preflights passed for both methods:2037 dialogues/10029 turns,
zero model calls, maximum actor prompt+budget8301(VDCGS)/8398(RDCGS), below
32768. Synthetic critic contexts truncate on3292 turns in both variants;
maximum8000 input tokens/6500 dropped. This is estimated from synthetic beliefs,
not a measured full-generation failure count. Preflight repeats on submission
and can spend several minutes with only the tokenizer startup line in the log.
Evidence: `verification/original_dcgs_full_launch_20260918/` (smoke review,
full preflights, manifests and launcher snapshots). Both shell syntax checks
passed; no runtime source was changed and no Slurm submission was performed.
