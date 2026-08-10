# Native Benchmark and Simulated-User Integration Plan

**Date:** 2026-08-09  
**Status:** Proposed  
**Primary reference:** `[papers/review/NeurIPS Experiment TODO.md](../../../../papers/review/NeurIPS%20Experiment%20TODO.md)`, especially I0-I2  
**Target benchmarks:** SafeDialBench, MultiBreak, and MT-AgentRisk (Unsafer in Many Turns / ToolShield)  
**Target system:** DCGS plain actor, VDCGS, and RDCGS evaluation, followed by simulator-expanded comparisons and optional training

## 1. Goal

Add the three requested multi-turn benchmarks without erasing the protocol differences that make their results meaningful. The implementation must support:

1. faithful benchmark-native evaluation;
2. simulator-expanded evaluation derived from the same seed examples;
3. paired comparison between native and expanded conditions;
4. a common output and metric contract across plain, VDCGS, and RDCGS;
5. trajectory collection with correct terminal and bootstrap semantics when a condition is used for learning.

The first milestone is evaluation, not training. Native test data must not enter critic training. Any later training condition must use a permitted training split or separately generated data and must be recorded in a distinct manifest.

### 1.1 Prerequisite benchmark acquisition and environment setup

Complete this setup and record its outputs before implementing loaders. Benchmark artifacts must live outside the tracked source tree or under an ignored local artifact directory. Repository manifests store URLs, immutable revisions, hashes, licenses/terms, and logical artifact IDs; machine-specific absolute paths belong in local configuration or environment variables. Credentials and access tokens must never be committed.

Recommended logical layout:

```text
benchmark-artifacts/
|-- SafeDialBench-Dataset/
|-- MultiBreak/                 # absent until an official release is obtained
|-- MT-AgentRisk/               # gated Hugging Face dataset
`-- ToolShield/                 # official runner and AgentRisk evaluation harness
```

Acquisition status verified on 2026-08-10:

| Benchmark | Official artifact | Access status | Implementation gate |
| --- | --- | --- | --- |
| SafeDialBench | [`drivetosouth/SafeDialBench-Dataset`](https://github.com/drivetosouth/SafeDialBench-Dataset) | Public Git repository; already acquired for this project | Verify the remote, commit, expected English data file, and repository/license metadata before WP2. |
| MultiBreak | [Paper/OpenReview record](https://openreview.net/pdf?id=WR1IcmCjbl) | **Blocked:** no official public benchmark dataset, evaluator, schema, or code repository was located. The public `jsong2333333/multibreak_qwen2_5` checkpoint is a generator model, not the benchmark. | Do not start WP3 native implementation or claim MultiBreak results until the authors provide the official data and terms. |
| MT-AgentRisk | Gated [`CHATS-Lab/MT-AgentRisk`](https://huggingface.co/datasets/CHATS-Lab/MT-AgentRisk) data plus public [`CHATS-lab/ToolShield`](https://github.com/CHATS-lab/ToolShield) runner | Dataset requires accepting its Hugging Face conditions and authenticating; runner is public | Download and pin both artifacts, then complete a plain-agent official-harness feasibility run before DCGS integration. |

Create a checked-in `docs/benchmarks/artifacts.lock.json` containing no credentials or absolute paths. Until exact revisions are captured locally, the expected shape is:

```json
{
  "verified_at": "2026-08-10",
  "safedialbench": {
    "status": "available",
    "source_url": "https://github.com/drivetosouth/SafeDialBench-Dataset.git",
    "revision": "<git-commit-sha>",
    "content_hash": "<normalized-artifact-hash>",
    "license_or_terms": "<audited-value>",
    "logical_path": "SAFEDIALBENCH_ROOT"
  },
  "multibreak": {
    "status": "blocked_artifact_unreleased",
    "source_url": null,
    "revision": null,
    "content_hash": null,
    "license_or_terms": null,
    "logical_path": "MULTIBREAK_ROOT",
    "blocker": "No official public benchmark data/evaluator located as of 2026-08-10"
  },
  "mt_agentrisk": {
    "status": "available_gated",
    "dataset_url": "https://huggingface.co/datasets/CHATS-Lab/MT-AgentRisk",
    "dataset_revision": "<huggingface-dataset-commit-sha>",
    "dataset_hash": "<normalized-artifact-hash>",
    "runner_url": "https://github.com/CHATS-lab/ToolShield.git",
    "runner_revision": "<git-commit-sha>",
    "openhands_revision": "0.54.0",
    "mcpmark_revision": "<git-commit-sha>",
    "license_or_terms": "gated dataset terms plus audited component licenses",
    "logical_path": "MT_AGENTRISK_ROOT"
  }
}
```

#### SafeDialBench verification

The official repository stores the complete English dataset at `data/complete/datasets_en.jsonl`. For the copy already acquired:

```bash
export SAFEDIALBENCH_ROOT="/absolute/path/to/SafeDialBench-Dataset"
git -C "$SAFEDIALBENCH_ROOT" remote -v
git -C "$SAFEDIALBENCH_ROOT" rev-parse HEAD
test -f "$SAFEDIALBENCH_ROOT/data/complete/datasets_en.jsonl"
```

Record the resolved commit and a content hash in `artifacts.lock.json`. Do not install or modify the repository's FastChat subtree merely to build our loader; use the raw official JSONL for normalization, and audit the official evaluator separately.

#### MultiBreak artifact blocker

The paper reports 10,389 multi-turn prompts across 2,665 harmful intents, but the paper is not a substitute for the released row-level artifact. The public generator checkpoint must not be used to regenerate prompts and label them as native MultiBreak.

Before unblocking WP3, obtain from the corresponding authors:

- the official benchmark rows and immutable version/revision;
- train/dev/test or evaluation-only split semantics;
- field schema, stable IDs, harmful-intent/category metadata, and turn ordering;
- evaluator implementation, judge prompt, early-stopping rule, and aggregation command;
- license, redistribution restrictions, and permitted evaluation/training uses.

Correspondence addresses listed in the paper are `jsa505@sfu.ca`, `xiaodl@microsoft.com`, and `weiwei.yang@microsoft.com`. Record contact attempts and responses in `docs/benchmarks/multibreak-artifact-status.md`. If the artifact remains unavailable, report `blocked_artifact_unreleased`; do not silently substitute a reproduction, scrape prompts from a paper, or call generated data MultiBreak.

#### MT-AgentRisk acquisition

Use WSL2/Linux or a Linux host for the official evaluation environment. The repository setup invokes Bash, OpenHands, MCP servers, and Linux-style sandbox/workspace paths. First accept the gated terms on the Hugging Face dataset page, then run:

```bash
mkdir -p benchmark-artifacts
git clone https://github.com/CHATS-lab/ToolShield.git \
  benchmark-artifacts/ToolShield

python -m pip install -U huggingface_hub
huggingface-cli login
huggingface-cli download CHATS-Lab/MT-AgentRisk \
  --repo-type dataset \
  --local-dir benchmark-artifacts/MT-AgentRisk

mkdir -p benchmark-artifacts/ToolShield/workspaces
cp -a benchmark-artifacts/MT-AgentRisk/workspaces/. \
  benchmark-artifacts/ToolShield/workspaces/
```

The Hugging Face login is user-local setup. Never place the token in a tracked `.env`, manifest, shell transcript, or experiment output. Capture the dataset repository revision reported by the Hub and hash the downloaded task tree before copying it.

Set up the pinned official runner in an isolated environment:

```bash
cd benchmark-artifacts/ToolShield
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[eval]"

git clone --branch 0.54.0 --single-branch \
  https://github.com/OpenHands/OpenHands.git
cp agentrisk/client.py OpenHands/openhands/mcp/client.py
python -m pip install -e OpenHands/

git clone https://github.com/eval-sys/mcpmark.git mcpmark-main
```

Record the ToolShield, OpenHands, MCPMark, and dataset revisions before running anything. The copied `agentrisk/client.py` is an official setup patch and must be recorded as such; hash the patched file and do not mix it with the later DCGS wrapper.

Runtime credentials and endpoints are local secrets/configuration. The official harness names include `TOOLSHIELD_MODEL_NAME`, `OPENROUTER_API_KEY`, `NOTION_TOKEN`, `SOURCE_NOTION_KEY`, and `SERVER_HOST`; only configure the credentials needed by the selected task subset. Full Playwright tasks additionally require the official SafeArena hosting setup. Start with terminal or filesystem fixtures before PostgreSQL, Notion, or Playwright.

Prerequisite validation:

```bash
test -f benchmark-artifacts/ToolShield/agentrisk/run_eval.py
test -d benchmark-artifacts/ToolShield/workspaces/terminal
test -f benchmark-artifacts/ToolShield/OpenHands/openhands/mcp/client.py
git -C benchmark-artifacts/ToolShield rev-parse HEAD
git -C benchmark-artifacts/ToolShield/OpenHands rev-parse HEAD
git -C benchmark-artifacts/ToolShield/mcpmark-main rev-parse HEAD
```

Do not treat `pip install toolshield` as benchmark installation. That command installs the defense package and bundled experiences; native MT-AgentRisk evaluation also requires the gated task data and the `agentrisk` harness above.

#### Prerequisite exit criteria

Before WP1 or any benchmark-specific coding begins:

- SafeDialBench has a verified official remote, immutable commit, raw-data hash, and confirmed English data path;
- MultiBreak is either pinned with official data/evaluator/terms or explicitly recorded as `blocked_artifact_unreleased`;
- MT-AgentRisk has separate pinned data and runner revisions, accepted terms, an isolated Linux environment, and no credentials in the repository;
- `scripts/benchmarks/preflight_artifacts.py` can validate configured logical paths, required files, revisions, and hashes without loading a model;
- every artifact's access and redistribution constraints are represented in `artifacts.lock.json` and the dataset-level manifest.

## 2. Protocol facts that constrain the design

The adapters must preserve the following distinctions.


| Benchmark     | User-turn source                                                                                | Assistant turns retained in next context                                        | How the next user turn is released                                                                         | Native outcome                                                                      |
| ------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| SafeDialBench | Fixed complete dialogue with human-reviewed, model-generated user and reference assistant turns | Reference assistant response, not the tested response                           | Evaluate each assistant position against its fixed reference prefix                                        | Per-turn safety score; dialogue score is the minimum turn score                     |
| MultiBreak    | Fixed sequence of 2-6 adversarial user prompts derived from one harmful intent                  | Tested model's actual responses                                                 | Next pre-scripted prompt is sent according to the benchmark sequence                                       | Conversation attack success, including ASR by turn/horizon                          |
| MT-AgentRisk  | Fixed multi-turn task instructions plus files, tools, dependency state, and success patterns    | Tested agent's actual messages, tool calls, tool results, and environment state | Progress-gated: advance after the current turn's success pattern; stop on refusal/failure/cap as specified | Whole task is COMPLETE, REJECT, or FAILED, with refusal and task-completion metrics |


Consequences:

- SafeDialBench is counterfactual turn evaluation over a gold history. It is not an ordinary live conversation.
- MultiBreak is live-response but non-responsive user replay: the model's response remains in history, while the next attack prompt is fixed.
- MT-AgentRisk is live-response and environment-responsive, but the user instruction text is still prescribed. Environment progress, not an LLM user, selects when the next instruction appears.
- Only a simulator-expanded condition should generate new user wording in response to the defended model.
- Native and simulator-expanded results must never be pooled into one score.

## 3. Terminology and configuration contract

Retire `online` and `offline` as the primary labels for these benchmarks. Those terms currently mix several independent behaviors. Introduce explicit orthogonal fields:

```json
{
  "environment_type": "multibreak",
  "benchmark_protocol": "native",
  "user_policy": "fixed_script",
  "assistant_history_policy": "actual",
  "advance_policy": "unconditional",
  "tool_backend": "none"
}
```

Allowed values:

- `benchmark_protocol`: `native` or `simulator_expanded`;
- `user_policy`: `golden_history`, `fixed_script`, `progress_gated_script`, `adaptive_llm`, or `hybrid_script_llm`;
- `assistant_history_policy`: `gold_reference` or `actual`;
- `advance_policy`: `counterfactual_turn`, `unconditional`, `progress_gated`, or `adaptive`;
- `tool_backend`: `none` or `official_openhands` initially.

Validate combinations rather than silently correcting them. For example:

- SafeDialBench native requires `golden_history + gold_reference + counterfactual_turn`.
- MultiBreak native requires `fixed_script + actual + unconditional` unless the audited official artifact specifies an early-stop rule.
- MT-AgentRisk native requires `progress_gated_script + actual + progress_gated + official_openhands`.
- Simulator-expanded conditions require `adaptive_llm` or `hybrid_script_llm` and must use a different experiment-condition name.

Keep the existing CARES-like flags during migration, but translate them to the new fields at config load time and emit a deprecation warning. Do not add another `<dataset>_online` boolean.

## 4. Target architecture

Separate four concerns that are currently coupled inside environment classes.

```text
Benchmark loader
    -> normalized BenchmarkExample + ProtocolSpec
        -> UserPolicy chooses/provides the next user turn
        -> HistoryController decides which assistant response enters context
        -> AdvanceController decides whether the script may advance
        -> OutcomeEvaluator computes native and common outcomes
            -> common EpisodeResult / per-turn JSONL
```

### 4.1 Normalized benchmark schema

Add `src/benchmarks/schema.py` with validated dataclasses or Pydantic models:

```python
class BenchmarkExample:
    example_id: str
    dataset: str
    split: str
    source_revision: str
    source_hash: str
    protocol: ProtocolSpec
    turns: list[TurnSpec]
    environment_spec: dict | None
    protected: ProtectedBenchmarkFields
    metadata: dict

    def policy_view(self, runtime: RuntimeState) -> PolicyView: ...

class ProtectedBenchmarkFields:
    hidden_goal: str | None
    ground_truth_class: str | None
    attack_family: str | None
    official_target: dict
    future_turn_ids: list[str]
    release_conditions: dict[str, dict]
    reference_answers: dict[str, str]
    attack_chain: dict | None

class TurnSpec:
    turn_id: str
    user_text: str
    reference_assistant_text: str | None
    release_condition: dict | None
    official_annotations: dict

class ProtocolSpec:
    user_policy: str
    assistant_history_policy: str
    advance_policy: str
    maximum_turns: int
    stopping_rule: str
    native_metric: str

class PolicyView:
    example_id: str
    condition_id: str
    current_turn_id: str | None
    visible_messages: list[dict]
    visible_tool_events: list[dict]
    visible_environment_state: dict

class NativePolicyInit:
    example: BenchmarkExample
    condition_id: str
    released_turn_ids: list[str]
    seed: int

class SimulatorInit:
    example_id: str
    parent_example_id: str
    condition_id: str
    private_goal: str | None
    allowed_initial_context: list[dict]
    category: str | None
    user_policy: str
    assistant_history_policy: str
    advance_policy: str
    stopping_rule: str
    maximum_turns: int
    simulator_model: str
    simulator_prompt_hash: str
    seed: int
    temperature: float
```

Schema rules:

- Preserve the stable source ID before filtering.
- Preserve all native turn order and metadata; do not reduce an example to `base_prompt`.
- Hash the raw source row/task directory and normalized representation.
- Store hidden goals, labels, future turns, reference answers, and release conditions in protected fields.
- Expose an explicit `policy_view()` containing only information permitted to the defended actor and its high-level candidate generator.
- Construct `SimulatorInit` only for simulator-expanded conditions. Native fixed-policy conditions must not instantiate an LLM user merely because a hidden goal exists.
- Pass `ProtectedBenchmarkFields.hidden_goal` into `SimulatorInit.private_goal` only when the selected user policy is allowed to use it.
- Never serialize `ProtectedBenchmarkFields` into actor, critic, candidate-generation, or ordinary transcript prompts.
- Keep `private_goal` out of `PolicyView`, even when it is also stored as `EpisodeState.ground_truth_goal` temporarily for legacy logging or evaluation compatibility.
- Record license, artifact URL, commit/revision, split, and exclusions in a dataset-level manifest.

The split between `ProtectedBenchmarkFields`, `SimulatorInit`, and `PolicyView` is an access-control boundary, not just documentation. Tests must serialize the exact objects passed to every model boundary and fail if a protected value appears in a defender prompt.

#### Example normalized record and views

The following abbreviated MultiBreak example shows the intended separation. Placeholder strings stand in for official source text; loaders must preserve the real source values and hashes.

```json
{
  "example_id": "multibreak:test:mb_000123",
  "dataset": "multibreak",
  "split": "test",
  "source_revision": "<pinned-commit>",
  "source_hash": "sha256:<raw-row-hash>",
  "protocol": {
    "user_policy": "fixed_script",
    "assistant_history_policy": "actual",
    "advance_policy": "unconditional",
    "maximum_turns": 3,
    "stopping_rule": "<audited-official-rule>",
    "native_metric": "conversation_attack_success"
  },
  "turns": [
    {
      "turn_id": "t1",
      "user_text": "<official user prompt 1>",
      "reference_assistant_text": null,
      "release_condition": null,
      "official_annotations": {}
    },
    {
      "turn_id": "t2",
      "user_text": "<official user prompt 2>",
      "reference_assistant_text": null,
      "release_condition": null,
      "official_annotations": {}
    }
  ],
  "environment_spec": null,
  "protected": {
    "hidden_goal": "<official source harmful intent>",
    "ground_truth_class": "adversarial",
    "attack_family": "<official attack family>",
    "official_target": {"intent_id": "intent_0042"},
    "future_turn_ids": ["t2"],
    "release_conditions": {},
    "reference_answers": {},
    "attack_chain": null
  },
  "metadata": {"category": "<official category>"}
}
```

For the paired adaptive simulator condition, only the user-side initialization receives the private goal:

```json
{
  "example_id": "multibreak:test:mb_000123",
  "parent_example_id": "multibreak:test:mb_000123",
  "condition_id": "adaptive_llm:seed_17",
  "private_goal": "<official source harmful intent>",
  "allowed_initial_context": [],
  "category": "<official category>",
  "user_policy": "adaptive_llm",
  "assistant_history_policy": "actual",
  "advance_policy": "adaptive",
  "stopping_rule": "harmful_response_or_max_turns",
  "maximum_turns": 6,
  "simulator_model": "<pinned-model>",
  "simulator_prompt_hash": "sha256:<prompt-hash>",
  "seed": 17,
  "temperature": 0.7
}
```

At turn 1, the defender-side projection is intentionally smaller:

```json
{
  "example_id": "multibreak:test:mb_000123",
  "condition_id": "adaptive_llm:seed_17",
  "current_turn_id": "generated_t1",
  "visible_messages": [
    {"role": "user", "content": "<simulator-generated first utterance>"}
  ],
  "visible_tool_events": [],
  "visible_environment_state": {}
}
```

The last object is the only benchmark-derived state that may enter the defended actor or DCGS high-level candidate generator.

Create:

- `src/benchmarks/__init__.py`
- `src/benchmarks/schema.py`
- `src/benchmarks/registry.py`
- `src/benchmarks/manifest.py`
- `src/benchmarks/loaders/safedialbench.py`
- `src/benchmarks/loaders/multibreak.py`
- `src/benchmarks/loaders/mt_agentrisk.py`

The registry should return normalized examples and capability metadata. It should not create `EpisodeState` directly.

### 4.2 User-policy interface

Generalize the current `PatientAgent` behavior behind `src/agents/user_policy.py`:

```python
class UserPolicy(Protocol):
    def reset(
        self,
        initialization: NativePolicyInit | SimulatorInit,
    ) -> UserTurnDecision: ...
    def observe(
        self,
        assistant_message: str,
        tool_events: list[dict],
        environment_state: dict,
    ) -> None: ...
    def next_turn(self) -> UserTurnDecision: ...

class UserTurnDecision:
    text: str | None
    advance: bool
    stop: bool
    reason: str
    source_turn_id: str | None
```

Implement these policies:

- `GoldenTranscriptPolicy`: provides SafeDialBench user turns from the source history.
- `FixedScriptPolicy`: provides MultiBreak prompts in fixed order.
- `ProgressGatedScriptPolicy`: releases MT-AgentRisk instructions after a success-pattern event.
- `AdaptiveLLMPolicy`: wraps the existing `PatientAgent` for simulator-expanded conditions.
- `HybridScriptLLMPolicy`: preserves a benchmark turn's objective/order while allowing controlled response-aware paraphrasing or bridging.

`UserPolicy` must not own transcript construction. SafeDialBench proves why history handling is a separate concern.

#### 4.2.1 User-policy initialization and hidden-goal contract

Preserve the conceptual contract of the implemented CARES, WildJailbreak, RedBench, and HarmBench simulator path:

```text
private benchmark goal
    -> goal-conditioned user simulator
        -> released user utterance
            -> defender infers intent from visible dialogue only
```

The existing code maps `base_prompt` to `EpisodeState.ground_truth_goal`, converts it to `goal_json`, and gives it to `PatientAgent` for the first and subsequent user turns. The new adapters should preserve that role assignment, but not the legacy storage and label-conditioning shortcuts. In particular:

- the simulator/user policy may receive the private goal;
- the evaluator may receive the private goal and official target;
- the defended actor and DCGS high-level policy receive only `PolicyView`;
- storing a private goal in an episode object for legacy compatibility does not make it policy-visible;
- `harmful_level`, `ground_truth_class`, or non-null `hidden_goal` must not select a different defender or high-level candidate-generation prompt;
- candidate generation must be label-blind across attack examples and the frozen benign control, with candidate-pool composition controlled across VDCGS and RDCGS.

The last two rules intentionally tighten the current implementation. Current CARES-like belief generation uses `harmful_level` and the presence of `base_prompt` to choose nominal versus adversarial candidate-generation templates. `PromptManager` does not insert the exact `base_prompt` text, but template selection still reveals class information and changes the candidate proposal distribution. Do not reproduce that confound in the new benchmark adapters.

Hidden-goal availability and initialization by benchmark:

| Benchmark | Native user initialization | Hidden goal availability | Simulator-expanded initialization |
| --- | --- | --- | --- |
| SafeDialBench | `GoldenTranscriptPolicy`; first released source user turn plus the permitted gold prefix | Treat as nullable until WP0 confirms a canonical source intent field. If absent, create only an audited sanitized representation from legally permitted initial context and category; never derive it from future transcript text. | `AdaptiveLLMPolicy` receives the sanitized private goal, permitted initial context, category, simulator configuration, and actual defended responses. |
| MultiBreak | `FixedScriptPolicy`; ordered official prompts and current actual-response history | Explicit source harmful intent is available. Keep it evaluator/simulator-private. | `AdaptiveLLMPolicy` receives the harmful intent; `HybridScriptLLMPolicy` additionally receives only the current scripted turn objective/order needed for faithful paraphrasing. |
| MT-AgentRisk | `ProgressGatedScriptPolicy` inside the official tool harness; current instruction and declared initial workspace state | The harmful task objective is available, while attack-chain annotations, future instructions, release conditions, and success patterns remain harness-private. | On hold until native feasibility passes. A future adaptive user may receive the task goal and intended user-visible tool events, but not hidden defender state, future turns, or completion patterns. |

Initialization invariants:

- Native and simulator-expanded records use distinct `condition_id` values and share `parent_example_id` for paired analysis.
- A native fixed-policy run does not require a simulator model, prompt, temperature, or private-goal prompt injection.
- `AdaptiveLLMPolicy.reset()` receives `SimulatorInit`; fixed policies receive `NativePolicyInit`, which contains the normalized example plus an explicit released-turn cursor.
- The initial observation is either the first officially released user turn or the first utterance generated from `SimulatorInit`; it is never the hidden goal itself.
- Only actual defended responses are passed back to adaptive or hybrid simulators. SafeDialBench native remains the exception at the history-controller layer because later actor prompts restore reference assistant history.
- Every run records the goal-source field or derivation rule, simulator prompt hash, model revision, seed, temperature, and leakage-test result.

### 4.3 History controller

Add `src/environments/history_controller.py`:

- `GoldReferenceHistoryController` inserts the source reference assistant turn after recording the tested response separately.
- `ActualResponseHistoryController` commits the tested response, tool calls, tool outputs, and next user turn.

Every stored turn must distinguish:

- `model_response`: what the defended actor actually produced;
- `context_response`: what was placed in the next actor context;
- `reference_response`: the benchmark response, if any.

Add a leakage assertion before every defended-model call. It should fail if the policy input contains a hidden label, future user turn, hidden goal, reference response outside the allowed prefix, or official target answer.

### 4.4 Protocol environment

Add `src/environments/protocol_env.py` for text-only native and expanded interactions. It should implement `DialogueEnvironment` while delegating user generation, history commitment, advancement, and outcome evaluation to the components above.

Extend `StepResult` in `src/environments/dialogue_env.py` in a backward-compatible migration:

```python
terminated: bool       # true absorbing outcome under the task definition
truncated: bool        # horizon, timeout, infrastructure cap, or unsupported continuation
transition_valid: bool # whether this completed action should enter replay/evaluation
termination_reason: str | None
protocol_event: dict
```

Keep `done` as a derived compatibility property (`terminated or truncated`) until all environments migrate. Do not infer termination from `goal_achieved`.

Extend `EpisodeState` with:

- `protocol_id` and `condition_id`;
- `source_example_id` and `source_turn_id`;
- `native_turn_index`;
- `assistant_history_policy`;
- `terminated`, `truncated`, and `termination_reason`;
- `protocol_trace` and `official_events`;
- separate `defender_success`, `attack_success`, and `task_completion` fields.

`goal_achieved` can remain temporarily for old environments, but new code must not use it as a generic terminal flag.

### 4.5 Defender model boundary and wrapper contract

Do not build one universal wrapper that owns benchmark execution. Keep the benchmark protocol and model call as separate boundaries:

```text
BenchmarkRunner
    -> constructs the allowed PolicyView and protocol-specific ModelRequest
        -> DefenderPolicy generates one response or structured action
            -> BenchmarkRunner commits history, advances the protocol, and evaluates the result
```

The per-benchmark requirement is:

| Benchmark/condition | Model integration | Required behavior |
| --- | --- | --- |
| SafeDialBench native or expanded | Thin `DefenderPolicy` adapter over the existing DCGS text-generation path | Accept the current `PolicyView`, generate one assistant response, and return normalized metadata. The adapter must not own gold-history substitution or evaluator logic. |
| MultiBreak native or expanded | The same thin `DefenderPolicy` adapter | Accept actual visible response history and generate one assistant response. The adapter must not choose, rewrite, or advance the fixed native user script. |
| MT-AgentRisk plain-agent feasibility | The official harness's compatible model client without DCGS intervention | Establish that the pinned agent, tool schema, parser, workspace, retry behavior, and official metrics work before adding DCGS. |
| MT-AgentRisk with VDCGS/RDCGS | A true model-call wrapper inside the official OpenHands/MCP agent boundary | Add selected high-level intent guidance while preserving the official model request/response and tool-action contract. |
| Simulator-expanded user side | A separate `AdaptiveLLMPolicy` or `HybridScriptLLMPolicy` | Initialize from `SimulatorInit`; never route simulator-private goals through the defender adapter or tool-agent wrapper. |

Add shared types under `src/benchmarks/model_boundary.py` and the defender interface under `src/agents/defender_policy.py`:

```python
class ModelRequest:
    messages: list[dict]
    tools: list[dict] | None
    tool_choice: str | dict | None
    response_format: dict | None
    stop: list[str] | None
    generation_kwargs: dict

class ModelResult:
    assistant_text: str | None
    tool_calls: list[dict]
    finish_reason: str | None
    usage: dict
    latency_ms: float | None
    retry_count: int
    provider_metadata: dict

class DefenderPolicy(Protocol):
    def generate(
        self,
        view: PolicyView,
        request: ModelRequest,
    ) -> ModelResult: ...
```

The shared interface is a normalization and access-control boundary, not a new benchmark runtime. `BenchmarkRunner` continues to own turn release, transcript commitment, environment state, stopping, and evaluation. `DefenderPolicy` owns only one defended generation. `UserPolicy` remains a separate role and may receive `SimulatorInit.private_goal` only under the rules in Section 4.2.1.

For SafeDialBench and MultiBreak, implement a `LegacyDCGSDefenderAdapter` that reuses the current `LowLevelAgent`, `get_model_instance()`, and `_generate_defender_action_with_backend()` behavior, including the existing standard, SmoothLLM, and TPO backend selection. Do not duplicate model loading or silently change prompt templates, decoding defaults, response-tag handling, or token accounting during the first integration. Any later provider-neutral client refactor should be a separate change with parity tests.

For MT-AgentRisk, implement `ToolAgentModelCallWrapper` at the model callback/client boundary used by the official agent, not around the full harness:

```python
class ToolAgentModelCallWrapper:
    def __call__(self, official_request: object) -> object:
        view = policy_view_from_visible_agent_state(official_request)
        assert_no_protected_fields(view, official_request)
        guidance = dcgs_policy.select_guidance(view)
        guided_request = clone_with_guidance_only(official_request, guidance)
        official_response = underlying_official_model_client(guided_request)
        audit_sink.record(normalize_model_result(official_response))
        return official_response
```

This pseudocode specifies placement, not permission to coerce the official request or response into the common types before the harness consumes them. `ModelRequest` and `ModelResult` are the normalized interface for text defenders and the audit representation for tool runs; the MT-AgentRisk wrapper must return the original provider response type to the official parser. It must pass through unchanged:

- tool definitions, tool names, parameter schemas, and tool-choice settings;
- structured-output or response-format constraints;
- message roles and all harness-required correlation/tool-call IDs;
- stop tokens, maximum-output settings, sampling parameters, and supported provider options;
- retry, timeout, streaming, and error-propagation semantics expected by the official agent;
- the response envelope and tool-call argument format consumed by the official parser.

Only the selected DCGS guidance and its diagnostics may be added. Guidance must be derived exclusively from `PolicyView`; hidden goals, labels, attack-chain annotations, future instructions, release conditions, success patterns, reference answers, and evaluator outputs may not enter either candidate generation or the wrapped model call. Hash the pre-wrapper request, post-guidance request, tool schema, and normalized result so tool-call fidelity and leakage can be audited without logging secrets.

Model-boundary acceptance tests must prove:

- SafeDialBench and MultiBreak use the same adapter and preserve the current DCGS backend output contract;
- the adapter cannot advance a benchmark, mutate source turns, or call an evaluator;
- native fixed-policy runs never instantiate or call a simulator model;
- `SimulatorInit.private_goal` is absent from every defender-side `PolicyView` and `ModelRequest`;
- plain and wrapped MT-AgentRisk calls receive byte-equivalent tool schemas and equivalent provider options;
- valid plain-agent tool calls remain schema-valid after wrapping;
- retry and parser behavior are unchanged by the wrapper;
- unsupported structured-output, streaming, or provider features fail capability preflight rather than being silently dropped.

### 4.6 Tool-benchmark runner boundary

Do not force MT-AgentRisk into the text-only `DialogueEnvironment`. Add a runner abstraction under `src/benchmarks/runners/base.py`:

```python
class BenchmarkRunner(Protocol):
    def run(example, defender, run_context) -> EpisodeResult: ...
```

Implement:

- `DialogueBenchmarkRunner` for SafeDialBench and MultiBreak;
- `ToolShieldRunner` as a thin adapter around the official OpenHands/MCP harness for MT-AgentRisk.

The ToolShield runner must preserve workspace state, tool schemas, tool outputs, release conditions, refusal detection, step limits, and official result classification. It may import the completed trace into DCGS's common result schema after the official harness finishes.

DCGS currently emits text responses and has no general tool-action contract. Therefore MT-AgentRisk integration has two explicit stages:

1. run a compatible plain tool agent in the official harness and validate loading, tracing, and metrics;
2. add a DCGS LLM-call wrapper that injects selected intent guidance into the tool agent's next model call without changing its action schema or tool runtime.

Do not claim VDCGS/RDCGS compatibility until stage 2 passes tool-call fidelity tests.

## 5. Correct sequential-learning semantics first (On hold until clarification)

Before using any native or expanded multi-turn trajectory for TD updates, repair `collect_transitions_from_episodes()` and its callers in `src/main.py`.

Current failure mode:

1. a safe intermediate response in an adversarial episode sets `goal_achieved = 1` even though the environment continues;
2. replay records that transition as terminal and removes its bootstrap;
3. when the environment later produces the actual harmful-success or max-turn terminal step, collection skips it because `done_from_env` is already true.

Required behavior:

- append each completed action exactly once, including the decisive terminal action;
- use `terminated` as the TD terminal mask, never `goal_achieved`;
- record `truncated` separately and apply an explicit truncation-bootstrap policy;
- permit an episode to contribute its final transition before retirement;
- distinguish defender survival/safety from environment termination;
- add a transition sequence number or collected flag to prevent duplicates.

Recommended processing order:

```text
actor action
  -> environment step
  -> persist StepResult
  -> collect completed transition (including terminal)
  -> compute training update
  -> retire episode if terminated or truncated
```

Tests must cover:

- safe intermediate adversarial turn: `terminated=False`, bootstrap retained;
- harmful response: terminal transition included once, bootstrap removed;
- max-turn survival: final transition included once and marked according to the chosen truncation policy;
- benign goal completion: terminal transition included once;
- no duplicate transition after episode replacement.

This is a release blocker for trajectory training, but not for a read-only evaluation-only adapter that does not write replay.

## 6. Benchmark-specific implementation

### 6.1 SafeDialBench

#### Native adapter

Parse each complete history into ordered `(user, reference_assistant)` pairs. Generate one evaluation unit per assistant position:

```text
gold prefix through user turn t
  -> defended model produces model_response_t
  -> native/common evaluators score model_response_t
  -> record it, but restore reference_assistant_t in the prefix for turn t+1
```

Implementation requirements:

- Never put `model_response_t` into the next SafeDialBench native prompt.
- Store the exact gold prefix hash for every target turn.
- Score every assistant position independently with the official evaluator where available.
- Aggregate the official dialogue result using the benchmark's minimum-turn rule.
- Also run the common guard/helpfulness judges and report disagreement with the official score.
- Ensure DCGS candidate-intent generation receives the same allowed prefix as the plain actor and cannot see future turns, labels, or the reference response being predicted.

The normalized example may represent one source dialogue, but the runner should create explicit target-turn records so paired methods are evaluated on identical prefixes.

#### Simulator-expanded condition

Derive a simulator seed from only fields legally allowed for expansion: the dialogue's permitted initial context, category, and a sanitized goal representation. Do not feed the reference future transcript to the simulator unless that condition is explicitly named `transcript_conditioned_expansion`.

Use `AdaptiveLLMPolicy` and commit actual defended responses. Report this as a different interaction distribution, not as SafeDialBench's native score. Pair it to the native dialogue ID with `parent_example_id`.

#### SafeDialBench acceptance tests

- Changing the model response at turn 1 does not change the prompt at turn 2.
- Changing a gold reference response does change the later prefix hash.
- A future source turn cannot appear in current actor or critic prompts.
- The aggregate score equals the minimum of stored per-turn scores.
- Plain, VDCGS, and RDCGS receive byte-identical visible history for the same target turn before their own generated output.

### 6.2 MultiBreak

#### Native adapter

Normalize the harmful intent and the ordered 2-6 adversarial user prompts. Run one continuous dialogue:

```text
fixed user prompt 1
  -> actual defended response 1 committed
fixed user prompt 2
  -> actual defended response 2 committed
...
```

Implementation requirements:

- The next user prompt must not be regenerated or selected in response to the assistant.
- Actual defended responses remain in history.
- Preserve attack-family/construction metadata and source harmful-intent ID.
- Audit the exact official early-stopping behavior. Encode it in `stopping_rule`; do not assume that the first judged harmful response always stops if the official evaluator scores the complete sequence.
- Record per-turn harmfulness, first-break turn, survival through turn `t`, and final conversation success.
- Use the official attack-success evaluator plus the common judge.
- Treat the harmful intent as evaluator/simulator-private data. It must not be injected into DCGS intent candidates in native evaluation.

#### Simulator-expanded conditions

Create two optional paired conditions:

1. `hybrid_script_llm`: preserve each scripted prompt's objective and order, while an LLM may add a response-aware bridge or paraphrase under a semantic-equivalence validator;
2. `adaptive_llm`: give the simulator the hidden harmful intent and let it choose its own escalation based on actual responses.

The hybrid condition is the closer ablation because it changes realization while retaining script structure. The fully adaptive condition tests a different attacker. For both:

- publish simulator model, prompt hash, seed, and temperature;
- run multiple fixed seeds per source ID;
- validate target retention, reveal turn, repetition, drift, and semantic equivalence;
- report native, hybrid, and adaptive results separately.

#### Benign control

MultiBreak supplies harmful attacks, not a matched benign interaction set. Do not manufacture benign examples and call them MultiBreak. Compute benign GCR on a frozen external benign manifest, ideally one shared with the other safety experiments, and label it as the benign control set.

#### MultiBreak acceptance tests

- Replacing response 1 changes response history seen at turn 2 but never changes fixed user prompt 2.
- The number and order of source prompts are unchanged in native mode.
- Hidden harmful intent is absent from all defender prompts.
- First-break and ASR-by-turn calculations match hand-computed fixtures.
- The same seed ID maps deterministically to native and expanded condition IDs.

### 6.3 MT-AgentRisk / ToolShield

#### Artifact and capability audit

Before coding the runner, pin and inspect the official artifact:

- task directory format (`task-turn-*.md`, `turns.yml`, workspace files, dependency declarations, and attack-chain metadata);
- OpenHands and MCP versions;
- success-pattern and turn-release logic;
- refusal and generic-nudge behavior;
- maximum agent-step rules;
- official COMPLETE/REJECT/FAILED aggregation;
- supported models, tool-call format, and sandbox requirements.

Write the audit to `docs/benchmarks/mt-agentrisk-feasibility.md`. Unsupported cases must be counted, not silently dropped.

#### Native runner

Use the official environment as the source of truth. The adapter should:

- create a clean isolated workspace for every task and seed;
- copy or mount only task-declared files;
- submit the current fixed instruction;
- persist actual assistant messages, reasoning fields allowed by the harness, tool calls, tool results, and environment changes;
- release the next instruction only when the official success pattern is satisfied;
- stop on explicit refusal, official failure, or step cap;
- preserve the official final label and raw evidence;
- clean up safely and record whether the environment reset succeeded.

The common evaluator may add safety/helpfulness judgments, but it must not overwrite the official outcome.

#### DCGS integration

Wrap the LLM boundary used by the official agent rather than replacing the harness:

```text
official agent state + visible tool history
  -> DCGS high-level candidates
  -> VDCGS/RDCGS selects intent guidance
  -> original tool-agent model call with unchanged tool schema
  -> official parser executes action
```

The wrapper must preserve tool definitions, structured-output constraints, stop tokens, and retry semantics. Capture the selected intent as diagnostics but never place hidden attack-chain annotations, future instructions, or completion patterns in candidate generation.

#### Simulator-expanded condition (On hold)

Defer this until native execution is stable. The first expanded design should retain the official environment and task goals while allowing an adaptive LLM user to choose bridging language after each satisfied subtask. It must not grant new tools, modify success conditions, or see hidden defender state beyond the transcript and tool events intended for the user role.

#### MT-AgentRisk acceptance tests

- A task cannot advance until its current success pattern is satisfied.
- A refusal and a step-cap produce distinct final reasons.
- Environment files/state are reset between examples.
- Tool calls produced through the DCGS wrapper validate against the same schema as the plain agent.
- Future task turns and hidden attack-chain data never appear in defender prompts.
- Official and imported common result IDs and final labels match one-to-one.

## 7. Evaluation and output contract

Add `src/evaluation/benchmark_result.py` and a JSONL writer independent of training metrics. Each run record must include:

- dataset, source revision/hash, split, example ID, parent seed ID, and condition ID;
- native versus simulator-expanded flag;
- full visible turns, with separate model/reference/context assistant fields;
- tool calls, outputs, and relevant environment events;
- all DCGS intent candidates, probabilities, Q/regret scores, and selected intent;
- official raw score and outcome;
- common guard/helpfulness scores and judge disagreement;
- terminated/truncated status and reason;
- simulator model, prompt hash, seed, and validation results;
- actor, critic, simulator, guard, and judge tokens/calls;
- latency, peak memory, estimated cost, retries, and errors;
- code revision and frozen manifest hash.

Create evaluator adapters under `src/evaluation/official/` and never reduce an ordinal official score to a binary label without preserving the raw score and documented threshold.

Required aggregate outputs:

- DSR and benign GCR where supported;
- attack goal completion and harmful-assistance rate;
- refusal, over-refusal, safe-but-unhelpful, and unresolved rates;
- turn-indexed survival curves and first-failure distribution;
- benchmark-native official metrics;
- paired confidence intervals using shared IDs;
- native-versus-expanded paired deltas and method-rank correlation;
- complete cost and failure tables.

## 8. Manifests, leakage controls, and reproducibility

Add a manifest command, for example:

```powershell
python -m src.benchmarks.manifest build --benchmark safedialbench --split test --output manifests/safedialbench-test.json
```

The manifest must include source revision, source and normalized hashes, all example IDs before/after filtering, class and turn-count summaries, exclusions with reasons, and license/access notes.

Use separate manifests for:

- development/smoke examples;
- frozen native test examples;
- simulator-expanded children and seeds;
- permitted training data;
- external benign controls.

Add a preflight that rejects:

- duplicate IDs;
- duplicate normalized content across declared splits;
- overlap with critic-training manifests when metadata permits checking;
- missing artifact revision or prompt hash;
- a native condition with generated user turns;
- a benchmark test manifest passed to a training command;
- hidden fields appearing in serialized policy input.

## 9. Concrete work packages

### WP0 - Pin protocols and fixtures

**Files:** `docs/benchmarks/*.md`, `docs/benchmarks/artifacts.lock.json`, `scripts/benchmarks/preflight_artifacts.py`, `tests/fixtures/benchmarks/`

1. Run the Section 1.1 acquisition/preflight procedure and populate every available immutable revision and hash in `artifacts.lock.json`.
2. Verify the existing SafeDialBench checkout and raw English data rather than downloading another uncontrolled copy.
3. Record MultiBreak as pinned or `blocked_artifact_unreleased`; never replace it with the public generator checkpoint or reconstructed prompts.
4. Download and pin the gated MT-AgentRisk data separately from the ToolShield/OpenHands/MCPMark runner stack.
5. Record licenses, gated-access conditions, redistribution restrictions, schemas, official evaluator commands, and prohibited uses.
6. Save at least ten small raw fixtures per benchmark where redistribution is allowed; otherwise save hashes and local fixture-generation instructions.
7. Manually annotate expected turn order, history behavior, stopping, and outcome for three fixtures per available benchmark.
8. Make artifact preflight fail before model loading when a required path, file, revision, hash, access acknowledgment, or runner component is missing.

**Exit:** protocol and acquisition audit reviewed; available artifacts reproduce their lock entries, unavailable artifacts have explicit blockers, and no unresolved ambiguity is hidden in adapter defaults.

### WP1 - Core schema, protocol engine, and termination repair

**Files:** `src/benchmarks/`, `src/agents/user_policy.py`, `src/agents/defender_policy.py`, `src/environments/protocol_env.py`, `src/environments/history_controller.py`, `src/environments/dialogue_env.py`, `src/training/episode_state.py`, `src/main.py`, `src/configs/base_config.py`

1. Implement normalized schemas and capability registry, including `ProtectedBenchmarkFields`, `PolicyView`, `NativePolicyInit`, and `SimulatorInit`.
2. Implement policies and history controllers.
3. Add `ModelRequest`, `ModelResult`, `DefenderPolicy`, and `LegacyDCGSDefenderAdapter`, preserving the current model-loading and backend-generation behavior.
4. Add model-boundary serializers and leakage assertions that reject protected goals, labels, future turns, reference answers, and release conditions in defender prompts.
5. Add initialization tests proving that adaptive simulators receive their permitted private goal, fixed native policies do not instantiate an LLM simulator, and both paths produce the expected initial observation.
6. Add adapter-parity tests for standard, SmoothLLM, and TPO response normalization and token accounting.
7. Add new termination fields and migrate the episode loop.
8. Repair transition collection and add final-transition tests.
9. Add explicit protocol config validation and legacy translation.
10. Add dry-run normalized-record, `SimulatorInit`, `PolicyView`, `ModelRequest`, and protocol-trace commands that do not load large models.

**Exit:** scripted fixtures execute deterministically, and all terminal/visibility tests pass.

### WP2 - SafeDialBench native MVP

1. Implement loader and target-turn projection.
2. Implement gold-reference history substitution.
3. Integrate official and common evaluation.
4. Run ten examples each with plain actor, VDCGS, and RDCGS.
5. Compare generated prompts and outcomes manually to raw fixtures.

**Exit:** first end-to-end native benchmark satisfies I1's minimum deliverable.

### WP3 - MultiBreak native and paired simulator variants

**Entry gate:** `artifacts.lock.json` must contain an official accessible MultiBreak artifact, evaluator, license/terms, revision, and hashes. If its status remains `blocked_artifact_unreleased`, stop WP3 and report the blocker; do not implement against generated or reconstructed substitutes.

1. Implement fixed-script loader and continuous-dialogue runner.
2. Implement official ASR/turn metrics.
3. Add hybrid and adaptive simulator policies behind separate conditions.
4. Build paired manifests using the same source IDs and method seeds.
5. Run the frozen benign control alongside, without labeling it MultiBreak.

**Exit:** native/hybrid/adaptive results are separately reproducible and pairable.

### WP4 - MT-AgentRisk feasibility and native runner

1. Complete the artifact/tooling audit.
2. Implement official harness runner and common-result import.
3. Validate the plain compatible tool agent first.
4. Implement and test `ToolAgentModelCallWrapper` at the official agent's model-client boundary, including request passthrough, tool-schema fidelity, parser parity, retry parity, and protected-field leakage checks.
5. Only then run VDCGS/RDCGS and consider simulator expansion.

**Exit:** either fully paired results exist or I1 records a precise artifact/capability blocker plus a completed adapter audit.

### WP5 - Unified experiment commands and reporting

1. Add one evaluation entry point accepting benchmark, condition, defender, manifest, and seed.
2. Add capability preflight so incompatible defender/benchmark pairs are reported as unsupported.
3. Produce per-example JSONL, aggregates, failure tables, and native/expanded paired analysis.
4. Run adapter, smoke, pilot, and full stages from the TODO without changing the frozen protocol between methods.

**Exit:** all methods operate on shared IDs with complete provenance and cost accounting.

### WP6 - Optional training on eligible trajectories

1. Confirm dataset permissions and use only designated training data.
2. Materialize replay from completed transitions using the repaired terminal semantics.
3. Keep training-source and evaluation manifests disjoint.
4. Validate Q/V/regret targets on hand-computed two- and three-turn episodes.
5. Report training on native, hybrid, and adaptive trajectories as separate ablations.

**Exit:** terminal rewards propagate through the intended trajectory, and no benchmark test item enters training.

## 10. Test layout

Add at least:

```text
tests/benchmarks/test_schema.py
tests/benchmarks/test_manifest.py
tests/benchmarks/test_safedialbench_loader.py
tests/benchmarks/test_multibreak_loader.py
tests/benchmarks/test_mt_agentrisk_loader.py
tests/benchmarks/test_model_boundary.py
tests/agents/test_defender_policy.py
tests/environments/test_history_controller.py
tests/environments/test_protocol_env.py
tests/training/test_transition_terminal_semantics.py
tests/evaluation/test_safedialbench_metrics.py
tests/evaluation/test_multibreak_metrics.py
tests/evaluation/test_mt_agentrisk_result_import.py
tests/integration/test_native_benchmark_smoke.py
```

Essential invariants:

- Native user turns equal the official artifact byte-for-byte after documented normalization.
- Every actor action has exactly one evaluation record and, when training is enabled, at most one replay transition.
- Final actions are not dropped.
- `terminated`, `truncated`, task success, defender success, and attack success are independently representable.
- No hidden field enters defended-policy prompts.
- The same text-model adapter serves SafeDialBench and MultiBreak without owning either benchmark's advancement or evaluation logic.
- MT-AgentRisk plain and wrapped calls preserve tool schemas, structured-output settings, provider options, retry behavior, and parser-compatible results.
- Native and expanded condition IDs cannot collide.
- An unsupported capability is reported before model loading.

## 11. Rollout order and gates

1. **Protocol gate:** finish WP0 and resolve official artifact ambiguities.
2. **Correctness gate:** finish WP1, including terminal-transition tests.
3. **First native benchmark:** SafeDialBench, because it is text-only and its unusual history rule exercises the new history controller.
4. **Second native benchmark:** integrate MultiBreak only if its official artifact is available; otherwise preserve the explicit blocker and proceed to the tool-capability gate.
5. **Native-versus-expanded pilot:** use MultiBreak first if it is available because the seed harmful intent naturally supports controlled expansion; otherwise defer this pilot rather than substituting generated data.
6. **Tool capability gate:** execute MT-AgentRisk plain-agent feasibility before changing DCGS.
7. **DCGS tool integration:** wrap the official agent model boundary only after the plain native runner agrees with official outputs.
8. **Full runs:** only after ten-example adapter checks and balanced smoke/pilot runs produce complete records.

Do not start full simulator-expanded runs while native adapters or official metric mappings remain unverified.

## 12. Risks and mitigations


| Risk                                                                                         | Mitigation                                                                                  |
| -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Hidden target, label, reference response, or future-turn leakage into DCGS intent generation | Protected schema fields, `policy_view()`, serialized-prompt assertions, and fixture tests   |
| SafeDialBench is incorrectly treated as one live conversation                                | Target-turn projection plus gold-reference history controller                               |
| MultiBreak behavior is attributed to an adaptive attacker even though prompts are fixed      | Explicit fixed-script native condition and separately named adaptive variants               |
| MultiBreak lacks benign examples                                                             | Frozen external benign control; no synthetic relabeling as MultiBreak                       |
| MT-AgentRisk is called text-only or flattened into chat replay                               | Official harness runner boundary and capability preflight                                   |
| Current `goal_achieved` logic corrupts TD targets                                            | Termination migration and final-transition blocker tests before training                    |
| Simulator expansion changes the threat model                                                 | Paired IDs, separate result strata, diversity/drift measures, and rank-correlation analysis |
| Official artifact or evaluator version drifts                                                | Pin revisions and hashes in every manifest/output                                           |
| MultiBreak remains unpublished or access-restricted                                          | Keep WP3 blocked, preserve author-contact evidence, and never substitute generated prompts  |
| MT-AgentRisk gated data is confused with the public ToolShield package                        | Pin data and runner separately; require both in artifact preflight                          |
| Host OS cannot reproduce the OpenHands/MCP runner                                             | Use an isolated WSL2/Linux environment and validate a plain terminal/filesystem task first  |
| Dataset contamination or prohibited test training                                            | Separate immutable manifests and overlap preflight                                          |
| Judge disagreement hides benchmark-specific outcomes                                         | Preserve official raw result and common-judge result side by side                           |
| Tool tasks leave cross-example state                                                         | Isolated workspaces, reset verification, and failure-on-dirty-state checks                  |
| API/model variance obscures paired comparisons                                               | Fixed seeds, shared IDs, recorded revisions, multiple simulator rollouts, paired intervals  |


## 13. Definition of done

The integration is complete when:

- each benchmark has a pinned protocol audit, normalized loader, schema tests, and native runner or a documented external blocker;
- SafeDialBench uses gold reference histories while scoring the tested response at each turn;
- MultiBreak uses fixed user prompts and actual tested responses in history;
- MT-AgentRisk runs through the official progress-gated tool environment;
- plain actor, VDCGS, and RDCGS receive identical allowed inputs per shared example;
- hidden labels, goals, future turns, answers, and completion patterns never enter defender prompts;
- native and simulator-expanded records have distinct condition IDs and separate aggregate tables;
- terminal transitions are included exactly once and TD masks use true termination rather than `goal_achieved`;
- ten or more examples per dataset pass adapter inspection and smoke evaluation;
- outputs contain complete provenance, official/common outcomes, cost, errors, and frozen manifest hashes;
- a paired native-versus-expanded analysis answers whether simulator expansion preserves difficulty and method ranking.

## 14. Explicit non-goals

- Do not train on benchmark test prompts or trajectories.
- Do not call simulator-expanded interactions native benchmark evaluation.
- Do not average native and expanded outcomes into a single headline result.
- Do not replace MT-AgentRisk's official tool harness with a text-only approximation.
- Do not expose hidden benchmark intent merely because DCGS internally reasons over intent.
- Do not add dataset-specific branches to the main loop when the behavior belongs in a loader, policy, history controller, runner, or evaluator.
- Do not add more ambiguous online/offline booleans.
