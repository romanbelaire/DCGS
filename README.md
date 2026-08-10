# DCGS source guide

This directory contains the research code associated with **Towards a Transferable Defense Against Adversarial Multi-turn Dialogue**. This guide is written for both human researchers and coding agents: it explains the runtime architecture, maps source modules to the paper's methods and experiments, and gives the safest available run procedure.

Read the [agent-oriented paper reference](../../papers/18373_Towards_a_Transferable_D.md) before changing the method. The source and the paper are not exact mirrors. This repository has no packaged experiment configs, checkpoints, dependency lockfile, seed runner, or recorded paper results, so the published tables cannot be reproduced from the checkout alone.

For a focused explanation of benchmark ingestion, generated user behavior, online/offline modes, and known simulator limitations, see the [simulated-user guide](docs/guides/SIMULATED_USER_GUIDE.md).

## Status at a glance

- The main entrypoint, dataset adapters, intent critic, safety reward models, GPT-backed generation, and SmoothLLM/TPO baselines are present.
- The names `DCGS`, `VDCGS`, `RDCGS`, `DSR`, and `GCR` do not occur in the source. Their configuration mappings below are inferred from behavior.
- The paper's Stage 1 intent generation and critic-weighted selection are recognizable, with implementation differences.
- Stage 2 multi-response sampling exists as a local capability, but it is not wired into CARES, WildJailbreak, RedBench, or HarmBench.
- Some training and marginal-token-reward behavior differs materially from the published equations. See [Paper fidelity and known limitations](#paper-fidelity-and-known-limitations).
- `src/analysis/compare_ablations.py` currently has an `IndentationError` at line 667 and cannot run without a source fix.

## Architecture and runtime flow

`src/main.py` owns most orchestration. `EpisodeState` is the mutable per-dialogue record, while `ValueFunction` wraps a frozen language model with trainable critic heads.

```text
BaseConfig + prompt assets + dataset loader
                    |
                    v
          EpisodeState / environment
                    |
                    v
HighLevelAgent generates K candidate intents ("beliefs")
                    |
                    v
ValueFunction scores (dialogue state, intent) with Q
       |            |                  |
       |            |                  +--> optional Qmin / Vmin / regret heads
       |            +--> softmax or epsilon exploration
       v
selected intent -> LowLevelAgent -> assistant response
                    |
                    v
environment step -> reward, next observation, completion state
                    |
          +---------+----------+
          |                    |
          v                    v
metrics / transcripts    replay or transition buffer
                               |
                               v
                     Q/V and optional auxiliary updates
```

The active execution sequence is:

1. Load `BaseConfig`, apply evaluation mode, inspect GPU memory, and clean selected files in the output directory.
2. Load the local actor/tokenizer, optional user model, prompt manager, agents, and optional safety guard.
3. Freeze the actor and attach trainable Q, V, token-value, and optional Q-min/V-min/regret MLP heads.
4. Load the selected dataset and create `EpisodeState` objects plus their environments.
5. Generate intent candidates, score them, select one, generate an assistant action, and step each active environment.
6. Update belief probabilities from response likelihoods, log metrics, and collect transition records.
7. Train from the online replay buffer or offline transition buffer, then periodically evaluate and checkpoint.
8. Drain remaining transitions and write the final checkpoint/token report.

The implementation is highly centralized: the active loop duplicates some functionality also exposed by `training/hierarchical_rollout.py` and the unused `batch_generate_ll_actions_for_episodes()` helper.

## Paper method to code map

The fidelity labels mean:

- **Direct**: a clear implementation analogue exists.
- **Partial**: the capability exists but differs in behavior or wiring.
- **Inferred**: the source never names the paper construct; the mapping follows configuration and runtime behavior.
- **Missing**: no corresponding implementation was found.

| Paper method or experiment | Main source connection | Fidelity and important detail |
|---|---|---|
| Dialogue state \(s_t\) | `training/episode_state.py` | **Direct.** `format_dialogue_history()` and `update_current_observation_context()` build the state seen by Stage 1. |
| Stage 1 intent generation, Eq. 3 | `agents/high_level_agent.py` | **Partial.** One LLM call returns a numbered list of \(K\) intents, rather than \(K\) independent samples. |
| Intent critic, Eq. 6 | `value/value_function.py` | **Direct analogue.** `predict_q_value()` scores `(observation, belief)` pairs using a frozen actor representation and a small MLP. Hidden states are mean-pooled. |
| Critic-weighted resampling, Eq. 4 | `main.py` | **Partial.** Q scores are softmax-sampled, but default `epsilon=0.1` first permits uniform random exploration. |
| Frozen actor and two-layer critic | `value/value_function.py` | **Direct.** Actor parameters are frozen; each head is hidden width -> half width -> ReLU -> scalar. |
| Nominal TD critic, Eq. 7 | `main.py:update_q_function_online()` | **Partial/deviating.** Code targets `reward + gamma * V(next_state)`, not an expectation over next sampled intent Q values. Default `gamma=0.9`, versus `0.99` in the paper. |
| VDCGS | `baseline_mode`, `random_belief_selection`, `use_regret_critic` | **Inferred.** Closest mapping: all three false. |
| Forced benign/malicious hypotheses | robust candidate generation in `main.py` | **Partial/deviating.** Code generates \(K\) nominal plus \(K\) adversarial candidates instead of adding two deterministic forced hypotheses. |
| RDCGS and Eq. 10 | optional Q-min/V-min/regret heads and `_regret_critic_selection()` | **Inferred/deviating.** Closest mapping is VDCGS plus `use_regret_critic=true`. The code enforces beta 0.2 but computes `0.8*Q - 0.2*regret`; the paper states `0.2*Q - 0.8*regret` at beta 0.2. |
| Stage 2 response candidates | `agents/low_level_agent.py:generate_ll_candidates()` | **Partial.** Multi-response generation is implemented. |
| Response scoring, Eq. 17 | `value/value_function.py:predict_ll_candidate_scores()` | **Partial.** Implemented and selected in a MultiWOZ-oriented branch, but the paper safety environments generate one response. |
| Conditioning Stage 2 on \((s,z)\) | `configs/base_config.py`, `prompts/prompt_manager.py` | **Partial.** Default `ll_action_belief_only=true` strips history and behaves like \(P(a\mid z)\). Set it false to more closely approximate \(P(a\mid s,z)\). |
| Marginal token rewards, Eq. 15 | `value/marginal_rewards.py`, `value/marginal_rewards_integration.py` | **Partial/deviating.** Masked-token attribution exists, but token selection and aggregation differ from the paper. |
| Token TD critic, Eq. 16 | no corresponding Bellman backup | **Missing.** The implemented token loss directly regresses immediate targets. |
| CARES simulator | `agents/patient_agent.py`, `environments/cares_env.py` | **Direct broad analogue.** The patient/attacker produces escalating turns and the environment tracks safety/completion. |
| CARES-18K | `environments/cares_helpers.py` | **Direct loader:** `HFXM/CARES-18K`. |
| WildJailbreak | `environments/wildjailbreak_helpers.py` | **Direct loader:** `allenai/wildjailbreak`. |
| RedBench | `environments/redbench_helpers.py` | **Direct loader:** `knoveleng/redbench`, with configurable label mapping. |
| HarmBench | `environments/harmbench_helpers.py` | **Direct loader:** `walledai/HarmBench`, `standard` configuration; adversarial-only. |
| Black-box transfer | `agents/gpt_agents.py`, local critic in `main.py` | **Partial.** GPT agents can generate while the local actor remains loaded for representations, critic scores, and log probabilities. This is not an API-only path. |
| SmoothLLM and TPO baselines | `defense/` and dispatch in `main.py` | **Present.** Select with `defender_backend`. |
| CAT and DCR baselines | none found | **Missing.** |
| Defense success and goal completion | final episode summaries in `main.py` | **Derivable.** Adversarial `goal_achieved` corresponds roughly to survival; benign `goal_achieved` to completion. No variables or reports are named DSR/GCR. |
| Survival curves | none found | **Missing.** Episode logs could be post-processed, but no survival analyzer is included. |

## Experiment coverage

| Paper result | What the repository provides | What is not packaged |
|---|---|---|
| Table 1: main safety results | Four paper dataset loaders, safety rewards, online simulator, VDCGS/RDCGS-like switches | Exact configs, checkpoints, seeds, result files, DSR/GCR reporting |
| Table 2: black-box transfer | GPT high/low-level agents plus retained local critic | Paper model pinning and a fully supported Stage 2 GPT transfer path |
| Table 3: model transfer | Arbitrary Hugging Face `model_name` in principle | Model-family experiment launcher/configs and compatibility evidence |
| Table 4: reward-model variation | LlamaGuard and ShieldGemma loading | Exact runs and published table artifacts |
| Figure 2: token reward ablation | `marginal`, `uniform`, and `decayed` target schemas | Faithful token-TD implementation and a Figure 2 reproduction script |
| Figure 3: multi-turn survival | Per-turn and final episode records | Named survival metric, curve computation, and figure script |

The tree also contains MultiWOZ, VitaBench, SalesAgent, UserBench, DPO-style belief updates, contrastive/noise ablations, freeform hierarchical instructions, and embedding diagnostics. These are additional research capabilities, not all part of the paper's core evaluation.

## Source inventory

Paths below are relative to this directory.

### Entrypoint and configuration

| Module | Role and connection |
|---|---|
| `src/main.py` | Central experiment runner: model/data/environment construction, Stage 1 generation/scoring, action generation, environment stepping, learning, logging, evaluation, and checkpointing. Most paper-method wiring lives here. |
| `src/configs/base_config.py` | Dataclass for every model, critic, defense, ablation, batching, dataset, and checkpoint option. JSON keys are passed directly to this class; unknown keys fail. |
| `src/__init__.py`, `src/configs/__init__.py` | Package initializers; no experiment logic. |

### Agents and prompts

| Module | Role and connection |
|---|---|
| `src/agents/base_agent.py` | Abstract agent interface. |
| `src/agents/high_level_agent.py` | Generates and parses Stage 1 intent/belief candidates; includes legacy selection helpers not used by the active loop. |
| `src/agents/freeform_high_level_agent.py` | Produces freeform hierarchical instructions for the additional hierarchical-agent mode. |
| `src/agents/low_level_agent.py` | Converts a selected intent and optional history into an assistant response; also implements Stage 2 multi-candidate generation. |
| `src/agents/gpt_agents.py` | OpenAI API-backed high-level, low-level, and user agents for transfer experiments. |
| `src/agents/user_agent.py` | Local MultiWOZ user simulator. |
| `src/agents/patient_agent.py` | CARES/red-team patient or attacker simulator, including escalation across turns. |
| `src/agents/user_simulator.py` | Shared simulator protocol. |
| `src/prompts/prompt_manager.py` | Loads templates/personas and formats high-level, low-level, user, and belief-conversion prompts. Paths are relative to the current working directory. |
| `src/prompts/templates.jsonl` | General high-/low-level and simulator prompt templates. |
| `src/prompts/personas.jsonl` | Persona records used by prompt construction. |
| `src/prompts/cares_patient_prompts.json` | CARES benign/adversarial patient prompts. |
| `src/prompts/wildjailbreak_attacker_prompts.json` | WildJailbreak attacker prompts. |
| `src/agents/__init__.py`, `src/prompts/__init__.py` | Package initializers; no experiment logic. |

### Belief state and updates

| Module | Role and connection |
|---|---|
| `src/belief/belief_state.py` | `BeliefCandidate` and belief-history containers for Stage 1 intent candidates and their probabilities. |
| `src/belief/dialogue_state.py` | Combines belief state, value estimates, rewards, achieved fraction, and ground truth. |
| `src/belief/belief_surrogate.py` | Computes observation log probabilities conditioned on each candidate intent. |
| `src/belief/belief_updater.py` | Updates intent probabilities with listwise softmax/DPO-style response-likelihood evidence. |
| `src/belief/entropy.py` | Belief entropy and regularization helpers. |
| `src/belief/__init__.py` | Package initializer. |

### Data and dialogue simulation

| Module | Role and connection |
|---|---|
| `src/data/multiwoz_loader.py` | Loads and normalizes `multi_woz_v22` from Hugging Face or local data. |
| `src/data/dialogue_formatter.py` | Normalizes goals/actions, formats histories, and derives MultiWOZ ground-truth beliefs. |
| `src/data/data_utils.py` | Sampling, splitting, and belief-accuracy helpers. |
| `src/simulation/desire_extraction.py` | Converts MultiWOZ goals and turns into ordered desires. |
| `src/simulation/goal_state.py` | Tracks desire satisfaction, failures, and goal completion. |
| `src/simulation/constraint_simulator.py` | Simulates MultiWOZ booking results/failures; it is unrelated to the paper's adversarial patient simulator. |
| `src/simulation/slot_extraction.py` | Extracts slots/tool-like calls from generated actions; its LLM extraction helper is a stub. |
| `src/data/__init__.py`, `src/simulation/__init__.py` | Package initializers. |

### Environments and dataset adapters

| Module | Role and connection |
|---|---|
| `src/environments/dialogue_env.py` | Abstract environment contract and shared `StepResult`. |
| `src/environments/multiwoz_env.py` | Offline replay of recorded MultiWOZ turns. |
| `src/environments/online_env.py` | Online MultiWOZ loop combining user simulation, slots, goals, and constraints. |
| `src/environments/cares_env.py` | Offline and online CARES-style safety environments, reward state, and termination behavior. |
| `src/environments/cares_helpers.py` | CARES-18K loader and episode construction. |
| `src/environments/wildjailbreak_helpers.py` | WildJailbreak loader and example normalization. |
| `src/environments/redbench_helpers.py` | RedBench subset loading and benign/adversarial mapping. |
| `src/environments/harmbench_helpers.py` | HarmBench loader and adversarial example normalization. |
| `src/environments/adversarial_helpers.py` | Shared episode construction for CARES/WildJailbreak/RedBench/HarmBench-like records. |
| `src/environments/episode_factory.py` | Registry that dispatches non-MultiWOZ examples to environment-specific `EpisodeState` factories. |
| `src/environments/salesagent_env.py` | SalesAgent environment adapter. |
| `src/environments/salesagent_helpers.py` | Loads local SalesAgent JSON/JSONL and creates episodes. |
| `src/environments/userbench_env.py` | UserBench/TravelGym adapter. |
| `src/environments/userbench_helpers.py` | Loads the separately supplied UserBench project and parquet data; creates its user simulator. |
| `src/environments/vitabench_env.py` | VitaBench environment plus batched generation/judging support. |
| `src/environments/vitabench_helpers.py` | Loads the separately supplied VitaBench project, constructs episodes, and coordinates batch helpers. |
| `src/environments/__init__.py` | Package initializer. |

### Training and critic/reward code

| Module | Role and connection |
|---|---|
| `src/training/episode_state.py` | Central mutable per-dialogue state: environment, observations, candidates, selected intents, responses, rewards, losses, and final statistics. |
| `src/training/hierarchical_rollout.py` | Preselection and batched low-level generation coordinator; instantiated only by an unused helper in the current main loop. |
| `src/value/value_function.py` | Frozen-actor representation plus Q, V, token-value, and optional Q-min/V-min/regret heads; also owns optimizers, contrastive loss, candidate scoring, and checkpoints. |
| `src/value/cares_reward.py` | Loads LlamaGuard/ShieldGemma and computes safety, harmful-assistance, refusal, and instruction-fulfillment judgments. |
| `src/value/marginal_rewards.py` | Selects token positions, creates masked variants, obtains counterfactual rewards, and computes token targets/loss. |
| `src/value/marginal_rewards_integration.py` | Connects marginal/uniform/decayed token target schemas to episode updates. |
| `src/value/reward.py` | Generic dialogue reward helpers; no active caller was found. |
| `src/value/multiwoz_single_turn_eval.py` | Isolated MultiWOZ single-turn proxy evaluator; currently unused and imports back from `main.py`. |
| `src/value/vitabench_single_turn_eval.py` | Isolated VitaBench single-turn proxy evaluator; no active caller was found. |
| `src/value/vitabench_turn_scores.py` | VitaBench turn-scoring helpers; no active caller was found. |
| `src/value/__init__.py` | Package initializer. |

### Defenses and utilities

| Module | Role and connection |
|---|---|
| `src/defense/smoothllm_wrapper.py` | Perturbs the latest user span, samples responses, and aggregates them for the SmoothLLM baseline. |
| `src/defense/tpo_wrapper.py` | Samples, ranks, and iteratively revises candidate responses for TPO/revision/best-of-N modes. |
| `src/utils/llm_utils.py` | Singleton local model/tokenizer loading, PEFT merge support, local/API batch generation, log-probabilities, and VitaBench caches. |
| `src/utils/logging_utils.py` | JSONL metric/transcript writers, GPU memory reporting, and currently unused rollout/dialogue-summary writers. |
| `src/utils/belief_evaluation.py` | Goal-to-text conversion, embeddings, cosine/L2 distances, and belief evaluation. |
| `src/utils/belief_converter.py` | Structured belief-to-text conversion utilities. |
| `src/utils/guard_safety.py` | Startup sanity checks for guard and fulfillment reward models. |
| `src/defense/__init__.py`, `src/utils/__init__.py` | Package initializers. |

### Analysis scripts

| Script | Input, output, and relation to the paper |
|---|---|
| `src/analysis/metrics_report.py` | Reads `all_metrics.jsonl` and evaluation files; writes summary JSON and plots. General training/evaluation reporting, not a paper-table reproducer. |
| `src/analysis/analyze_candidate_pool_k_sweep.py` | Aggregates logged Q metrics across environment-named run directories. Closest script to a finite-candidate/K study. |
| `src/analysis/analyze_q_values.py` | Loads a checkpoint and studies Q distributions/correlations on MultiWOZ examples. |
| `src/analysis/checkpoint_ablation.py` | Compares checkpoints on shared MultiWOZ slices. |
| `src/analysis/compare_ablations.py` | Intended to compare contrastive/noise ablation runs and plot summaries. **Currently fails to parse at line 667.** It is not the paper's marginal/uniform/decayed ablation. |
| `src/analysis/compare_defense_ablations.py` | Compares defense episode summaries such as attack success and embedding metrics across runs. |
| `src/analysis/cosine_pattern_analysis.py` | Post-processes belief-evaluation CSVs for intent/goal embedding failure patterns. |
| `src/analysis/defense_embedding_metrics.py` | Computes response-embedding diversity from `defense_episodes.jsonl`, optionally per attack. |
| `src/analysis/evaluate_training_dynamics.py` | Measures checkpoint Q convergence and response-likelihood diagnostics; the main runner also calls part of it periodically. |
| `src/analysis/live_evaluate_ablations.py` | Runs live contrastive/noise checkpoint comparisons; not a direct paper experiment. |
| `src/analysis/scrambled_belief_evaluation.py` | Tests critic ranking on synthetic/scrambled beliefs. |
| `src/analysis/__init__.py` | Package initializer. |

## Installation

There is no dependency or Python-version manifest. Python 3.11 is a conservative recommendation; untracked bytecode suggests the code has been used with several Python versions. Create an isolated environment from **this directory**:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install a PyTorch build compatible with your CUDA driver using the official PyTorch instructions, then install the remaining inferred dependencies:

```powershell
python -m pip install transformers datasets openai peft huggingface_hub accelerate numpy pandas pyarrow matplotlib
```

These dependencies are inferred from imports and are not version-pinned. Exact paper package versions cannot be recovered from the checkout.

### Model, data, and credential requirements

- The default local actor is `HuggingFaceH4/zephyr-7b-beta`; Transformers downloads it automatically.
- LlamaGuard first tries the local cache, then calls `snapshot_download(..., token=True)`. Accept the model terms and authenticate with Hugging Face before a safety run.
- `OPENAI_API_KEY` is required for `use_gpt_for_agents`, `use_gpt_for_user`, and some UserBench/VitaBench simulator paths. Keep it in the environment, not a JSON config.
- MultiWOZ, CARES, WildJailbreak, RedBench, and HarmBench download through Hugging Face datasets.
- VitaBench is expected at `vitabench/src/vita` under this directory and is not included.
- UserBench is expected at `UserBench/`, including `travelgym` and its parquet data, and is not included.
- SalesAgent requires a local JSON/JSONL `data_path`.

### Paper model assignments versus repository defaults

The paper does not completely specify the model used for every dialogue role:

| Experimental condition | Defended responder / actor | Critic representation model | Simulated user / partner |
|---|---|---|---|
| Primary experiments (Table 1) | `Zephyr-7b-beta` | `Zephyr-7b-beta` | Described only as a partner LLM agent; the paper provides its prompts but does not name its checkpoint or decoding configuration |
| Black-box transfer (Table 2) | `GPT-5.4-mini` | `Zephyr-7b-beta` | The paper says all text generation uses `GPT-5.4-mini`, which appears to include partner generation, but it does not separately identify the simulated-user model |
| Current repository defaults | `HuggingFaceH4/zephyr-7b-beta` | The same local model, with trainable critic heads | The same Zephyr model because `user_model_name=null`; a distinct user model can be configured |

The first row should not be summarized as "Zephyr simulated both sides": the paper explicitly names Zephyr for the responder and critic, but not for the primary simulated user. Conversely, the repository's shared-model default is implementation behavior rather than evidence of the paper's simulator configuration. For the transfer experiment, the strongest literal reading of "all text generation" is that GPT-5.4-mini generated both roles, but the missing role-specific statement remains a reproducibility ambiguity.

### Hardware assumptions

Defaults target CUDA with bf16 and a 7B actor. There is no reliable automatic CPU fallback: set both `"device": "cpu"` and `"use_bf16": false` to attempt one, but a CPU run remains expensive. A distinct user model tries a second GPU; the safety guard is placed on the highest-numbered GPU. On one GPU, actor plus guard can exceed memory.

The runner increases batch/chunk sizes on GPUs above 45/70 GB but does not reduce defaults below its 40 GB baseline. For smaller devices, lower:

- `episode_batch_size`
- `online_batch_size`
- `transition_prob_chunk_size`
- `q_value_chunk_size`
- `batch_generation_chunk_size`

## Configuration

The runner deserializes JSON as `BaseConfig(**config)`. Use only fields declared in `src/configs/base_config.py`.

### Minimal mechanics smoke test

This is a code-path smoke test, not a paper experiment. It still downloads a model and MultiWOZ.

```json
{
  "model_name": "HuggingFaceH4/zephyr-7b-beta",
  "device": "cuda:0",
  "use_bf16": true,
  "environment_type": "multiwoz_offline",
  "max_dialogues": 2,
  "episode_batch_size": 1,
  "batch_size": 1,
  "online_batch_size": 1,
  "evaluation_interval": 100,
  "use_marginal_token_rewards": false
}
```

Save it as, for example, `configs/smoke.json` after creating that directory.

### Paper-oriented CARES configuration

The following is a **starting point**, not a validated reproduction. It overrides several source defaults with hyperparameters reported by the paper. Mapping paper rollout batch 128 to `episode_batch_size` is an inference and may be too large for available memory.

```json
{
  "model_name": "HuggingFaceH4/zephyr-7b-beta",
  "device": "cuda:0",
  "use_bf16": true,
  "environment_type": "cares",
  "cares_split": "train",
  "cares_online": true,
  "n_candidates": 5,
  "n_ll_candidates": 5,
  "epsilon": 0.1,
  "ll_action_belief_only": false,
  "learning_rate": 0.0001,
  "discount_factor": 0.99,
  "batch_size": 64,
  "episode_batch_size": 128,
  "max_dialogues": 500,
  "use_marginal_token_rewards": true,
  "token_level_reward_schema": "marginal",
  "reward_model_type": "llamaguard",
  "reward_model_name": "meta-llama/Llama-Guard-3-8B",
  "baseline_mode": false,
  "random_belief_selection": false,
  "use_regret_critic": false
}
```

Treat that final configuration as **VDCGS-like**. For the closest **RDCGS-like** switch, change only:

```json
{
  "use_regret_critic": true,
  "regret_critic_beta": 0.2,
  "regret_zero_sum_targets": true,
  "regret_min_target_mode": "sampled_q_min"
}
```

Other useful experiment switches:

| Intent | Configuration |
|---|---|
| Plain assistant baseline | `"baseline_mode": true` |
| Random intent ablation | `"random_belief_selection": true` (see checkpoint bug below) |
| SmoothLLM baseline | `"defender_backend": "smoothllm"` |
| TPO baseline | `"defender_backend": "tpo"` |
| Token target ablation | `"use_marginal_token_rewards": true` plus `token_level_reward_schema` set to `marginal`, `uniform`, or `decayed` |
| GPT generator transfer | `"use_gpt_for_agents": true`, set `gpt_agent_model`, export `OPENAI_API_KEY`, and retain a local `model_name` for the critic |
| Evaluation only | Set `checkpoint_path`, choose the evaluation dataset split/sample size, and set `LLM_CONTEXT_EVAL_MODE=1` |

## Running experiments

Always run from `code/DCGS`. Module invocation is required because `main.py` uses package-relative imports, and the prompt paths are current-working-directory relative.

```powershell
# Smoke test
python -m src.main configs\smoke.json outputs\smoke --max-dialogues 2

# Train a configured run
python -m src.main configs\rdcgs_cares.json outputs\rdcgs_cares_seed1

# The only supported command-line experiment overrides
python -m src.main configs\base.json outputs\crossed_data --contrastive-coef 1.0 --contrastive-ablation-mode crossed_data --max-dialogues 500

# Evaluation-only run; the config should set checkpoint_path
$env:LLM_CONTEXT_EVAL_MODE = "1"
python -m src.main configs\rdcgs_cares_eval.json outputs\rdcgs_cares_eval
Remove-Item Env:LLM_CONTEXT_EVAL_MODE
```

Do not use `python src/main.py`. Use a fresh output directory for every run: startup removes selected existing metrics/evaluation files, while leaving checkpoints, token reports, and `training_dynamics/` in place, which can create a mixture of stale and new artifacts.

Evaluation mode disables training updates and checkpoint saves. Evaluation without `checkpoint_path` creates an untrained critic, except for a narrow GPT-agent evaluation exception.

## Outputs

Depending on configuration, the output directory may contain:

| Artifact | Meaning |
|---|---|
| `all_metrics.jsonl` | Per-turn training/evaluation records. |
| `defense_episodes.jsonl` | Safety-environment transcripts and outcomes. |
| `p_action_belief_diagnostics.jsonl` | Periodic response-probability diagnostics. |
| `training_dynamics/training_dynamics.jsonl` | Periodic critic diagnostics. |
| `checkpoints/value_function_initial.pt` | Critic state immediately after initialization/loading. |
| `checkpoints/value_function_latest.pt` | Periodic checkpoint. |
| `checkpoints/value_function_final.pt` | End-of-run checkpoint. |
| `belief_evaluation_episode_N.csv` | Periodic intent/goal embedding evaluation. |
| `baseline_eval_N.json` | Baseline evaluation output where applicable. |
| `contrastive_test_episode_N.csv` | Contrastive/noise evaluation output. |
| `test_time_token_eval_N.json`, `test_time_token_eval_final.json` | Token-level evaluation reports. |

`utils/logging_utils.py` defines an `all_rollouts.jsonl` writer and dialogue-summary writer, but the active runner has no call sites for them.

## Analysis commands

Run these from `code/DCGS`. Scripts that load a model/checkpoint need the same model dependencies and compatible hardware as training.

```powershell
python -m src.analysis.metrics_report --metrics-file outputs\run\all_metrics.jsonl --csv-dir outputs\run --output-dir outputs\run\figures --min-episode 0

python -m src.analysis.cosine_pattern_analysis --csv-dir outputs\run --output-json outputs\run\cosine_patterns.json

python -m src.analysis.analyze_candidate_pool_k_sweep --outputs-root outputs --output-json outputs\candidate_pool_k_sweep_summary.json

python -m src.analysis.defense_embedding_metrics --run_dirs outputs\vdcgs_cares outputs\rdcgs_cares --model_name HuggingFaceH4/zephyr-7b-beta --device cuda:0 --per_attack

python -m src.analysis.compare_defense_ablations outputs\vdcgs_cares outputs\rdcgs_cares --label VDCGS RDCGS

python -m src.analysis.checkpoint_ablation --config configs\vdcgs.json --checkpoints outputs\run\checkpoints\value_function_initial.pt outputs\run\checkpoints\value_function_final.pt --output outputs\run\checkpoint_ablation.json

python -m src.analysis.evaluate_training_dynamics --checkpoint outputs\run\checkpoints\value_function_final.pt --config configs\vdcgs.json --episode 500 --n-examples 50

python -m src.analysis.analyze_q_values --checkpoint outputs\run\checkpoints\value_function_final.pt --config configs\vdcgs.json --n-examples 200 --output outputs\run\q_value_analysis.json

python -m src.analysis.scrambled_belief_evaluation --checkpoint outputs\run\checkpoints\value_function_final.pt --config configs\vdcgs.json --n-episodes 50 --output outputs\run\scrambled_beliefs.json

python -m src.analysis.live_evaluate_ablations --checkpoints ckpt_crossed.pt,ckpt_random.pt,ckpt_crossed_candidates.pt,ckpt_none.pt --config configs\base.json --n-examples 10 --output outputs\live_evaluation_results.json
```

Notes:

- The candidate-K script expects run paths containing `wildjailbreak`, `cares`, `harmbench`, or `redbench` and logged `candidate_pool_q_metrics`.
- Several checkpoint diagnostics are hard-wired to MultiWOZ training examples; they are not held-out paper safety evaluations.
- `defense_embedding_metrics` writes per-run `defense_embedding_summary.json` files and a combined summary in the current directory.
- Do not run `python -m src.analysis.compare_ablations` until its line-667 indentation error is fixed.

## Paper fidelity and known limitations

These points matter when interpreting results or asking an agent to modify the implementation.

### Method and reward differences

- Stage 1 generates all \(K\) intents in one numbered-list completion, not \(K\) independent completions.
- Intent selection is epsilon-greedy before Q-softmax by default, not pure Eq. 4 resampling.
- The nominal critic uses a Q/V Bellman target rather than the paper's next-candidate expectation.
- The robust implementation uses custom Q-min, V-min, and regret targets. Its selection weights reverse the paper's beta placement.
- Paper Eq. 5 sums helpfulness and harmlessness. Code averages the two on benign examples and uses harmlessness alone on adversarial examples, producing a different reward scale.
- The default simulator limit is 20 turns, while the paper's survival plot covers five.

### Stage 2 and token-reward limitations

- CARES, WildJailbreak, RedBench, and HarmBench set `multiwoz_mode=false` and follow a one-response generation path. Stage 2 is therefore not exercised end to end by the paper environments.
- Token positions are selected by hidden-state/logit L2 norm, not attention activation.
- Pairwise marginal contributions are accumulated rather than averaged over masked variants.
- Eq. 14 cosine alignment is absent; the code distributes a binary instruction-fulfillment judgment across selected tokens.
- Eq. 16 token TD backup is absent. Worse, predicted token values are converted with `.item()` and rebuilt as a detached tensor, so the current marginal-token loss cannot backpropagate into the token head.
- The token-value head is omitted from checkpoint save/load.
- CARES marginal evaluation ignores the masked action and uses cached environment outcome state rather than re-judging the altered response.

### Sequential-learning issue in paper environments

In adversarial online CARES-like episodes, a safe intermediate response can set `goal_achieved=true` while the environment deliberately continues. Transition collection interprets that flag as terminal, so the TD update removes the bootstrap. The actual harmful, successful, or max-turn transition is then skipped because completed environment steps are excluded from collection. As written, the replay path does not propagate the paper environments' multi-turn terminal outcome through the trajectory.

### Reproducibility and operational limitations

- No requirements/lockfile, sample configs, seeds, checkpoints, tests, or paper result artifacts are included.
- The paper does not identify the primary experiments' simulated-user checkpoint or decoding configuration; the repository's default of sharing Zephyr between user and responder is not enough to recover that setting.
- There is no global run seed. Some dataset subsampling seeds Python locally, but training is not deterministic.
- `train_ratio` and `val_ratio` are declared but not used by the active MultiWOZ loader path; the runner requests the training split.
- Rolling "evaluation" is based on the latest completed run episodes, not a held-out split.
- A missing `reward_model_name` can lead to placeholder safety rewards. Specify the guard explicitly.
- GPT transfer still loads the local actor. The inherited GPT low-level multi-candidate path is not a supported full Stage 2 API implementation.
- `random_belief_selection=true` sets `value_function` to `None`, but periodic/final checkpoint code later calls it without an equivalent guard.
- Checkpoint loading is sensitive to model architecture and the presence/absence of regret heads.
- Guard loading contains backend-specific assumptions, including a hard-coded LlamaGuard safe-token ID.
- `BaseConfig.to_json()` includes API-key fields. Prefer environment variables and never persist secrets in configs.
- `src/analysis/compare_ablations.py` is syntactically invalid at line 667. Python 3.14 also emits a nonfatal invalid-escape warning in `src/main.py`.
- Wrong working directory, gated-model access, dataset schema drift, unavailable external benchmark projects, unsupported bf16, and one-GPU memory pressure are common failure modes.
