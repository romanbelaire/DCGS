# DCGS parity with the original benchmark implementation

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

Completed turns are reused. An interrupted turn restarts from its original
seed, preserving call-start and result journals; a killed in-flight call may
have unknown cost and is counted separately. A saved policy failure blocks
unchanged resume. Fully completed resume does not load the actor or rewrite
outputs. Prior custom-method outputs cannot be imported into these fresh runs.

Verification: all 143 repository tests pass, including eight new tests for
direct original-code parity, all-SKIP recovery/exhaustion, empty LL behavior,
gold-history isolation, audit tampering, interruption/resume and failure gates.
Both real-tokenizer/artifact smoke preflights pass (one dialogue, five turns).
Injected native five-turn fixtures pass full replay/native output validation
and byte-identical no-op resume. Instrumentation around the real batch_generate
also matches its unwrapped output using a token-returning model stub. Original
src and prior runtime sources match the preserved snapshot (94 files).
Evidence: `verification/original_dcgs_parity_20260918/adapter_review.json`,
`tests.log`, and `preserved_source_review.json`. These checks do not perform
model inference; actual GPU loading, answers, runtime and memory remain pending.

From the DCGS root, the user can submit the prepared one-A100 smoke runs:

```bash
sbatch scripts/slurm/run_safedial_vdcgs_wildjailbreak_original_smoke.sbatch
sbatch scripts/slurm/run_safedial_rdcgs_wildjailbreak_original_smoke.sbatch
```

Each launcher runs preflight, generation, original-policy/native-output audit
with `--require-gpu`, and a judge dry-run. Output folders are
`outputs/safedial_dcgs/zephyr_{vdcgs,rdcgs}_original_wildjailbreak_smoke`.
Both launchers pass `bash -n`. No jobs were submitted. Full replacement runs
must wait for the new GPU smoke gates; historical smoke results use another
policy and cannot satisfy this gate. TPO preparation and SmoothLLM are preserved.
