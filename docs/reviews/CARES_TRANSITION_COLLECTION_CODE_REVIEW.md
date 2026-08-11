# CARES Transition Collection Code Review

**Status:** Source-verified review; no runtime fix implemented

**Scope:** CARES-like episode termination, transition collection, TD terminal masks, replay storage, and diagnostic logging

## Review outcome

The CARES online environment makes the correct rollout decision: a safe response in an adversarial episode does not end the conversation. The shared transition collector later mishandles that result in two distinct ways:

1. It marks a nonterminal safe adversarial transition as terminal because it derives the TD mask from `goal_achieved` instead of the environment's `done` result.
2. It skips a genuinely terminal transition because `done_from_env=True` is checked before the just-completed step is appended.

Together, replay can contain an intermediate safe response marked terminal while omitting the later harmful response that actually ended the episode.

## Source map

Some Markdown viewers do not open directory targets directly. Each directory below therefore links to a representative file inside it:

- `src/` — [open `main.py`](../../src/main.py)
- `src/environments/` — [open `dialogue_env.py`](../../src/environments/dialogue_env.py)
- `src/training/` — [open `episode_state.py`](../../src/training/episode_state.py)
- `src/value/` — [open `cares_reward.py`](../../src/value/cares_reward.py)
- `src/utils/` — [open `logging_utils.py`](../../src/utils/logging_utils.py)

The line-number links below work in GitHub-style renderers. A local Markdown preview may open the correct file without jumping to the referenced line.

The relevant files and entry points are:

| Responsibility | Source |
| --- | --- |
| Shared `StepResult.done` definition | [`src/environments/dialogue_env.py`](../../src/environments/dialogue_env.py#L9) |
| CARES online reward and termination | [`CARESOnlineEnvironment.step()`](../../src/environments/cares_env.py#L221) |
| CARES adversarial `goal_achieved` assignment | [`CARESOnlineEnvironment._compute_reward()`](../../src/environments/cares_env.py#L301) |
| Per-episode `goal_achieved` and `done_from_env` fields | [`EpisodeState`](../../src/training/episode_state.py#L13) |
| Environment-result copying and rollout continuation | [`process_turn_for_episode()`](../../src/main.py#L1806) |
| Transition collection | [`collect_transitions_from_episodes()`](../../src/main.py#L2419) |
| Online TD target construction | [`update_q_function_online()`](../../src/main.py#L3176) |
| In-memory buffer initialization | [`transition_buffer` and `replay_buffer`](../../src/main.py#L4536) |
| Collector call after stepping | [`main.py`](../../src/main.py#L5318) |
| Episode replacement after collection | [`main.py`](../../src/main.py#L5710) |
| Human-readable episode and turn logs | [`src/utils/logging_utils.py`](../../src/utils/logging_utils.py#L1) |

## Expected semantic contract

These values describe different facts and must remain separate:

| Field | Intended meaning |
| --- | --- |
| `goal_achieved` | Task success or, in adversarial CARES-like episodes, whether the current response was safe |
| `done` / `done_from_env` | Whether the environment says the rollout must stop after the action that just ran |
| `terminal` | Whether TD must remove the next-state bootstrap for the collected transition |
| `truncated` | Whether an external limit, such as `max_turns`, stopped the rollout without a natural terminal state |

The current code has no separate `truncated` field. It also overloads `goal_achieved` across task types, which makes it unsafe as a generic TD terminal mask.

An episode is the complete conversation; a transition is one completed model turn within that episode. A terminal transition must still be stored. Terminal means that its TD target has no future-value bootstrap, not that the final action, reward, and outcome should be discarded:

```text
nonterminal target = r_t + gamma * V(s_{t+1})
terminal target    = r_t
```

## Verified control flow

### 1. CARES computes per-turn safety

For an adversarial example, [`CARESOnlineEnvironment._compute_reward()`](../../src/environments/cares_env.py#L301) sets:

```python
goal_ok = r_harm > 0.5
```

Therefore, a harmless response, including a refusal, produces `goal_ok=True`. This means "safe on this turn" for an adversarial episode; it does not mean that the attacker has stopped.

### 2. CARES independently computes environment termination

[`CARESOnlineEnvironment.step()`](../../src/environments/cares_env.py#L221) assigns that safety result to `_goal_achieved`, then separately computes `should_end`:

```python
self._goal_achieved = goal_ok

should_end = False
if self._turn_idx >= self.max_turns:
    should_end = True
elif harmful_level == 0 and reward > 0.5:
    should_end = True
elif harmful_level > 0 and r_harm < 0.5:
    should_end = True
```

For a safe adversarial response, CARES returns the intentional combination:

```text
StepResult.done                 = False
StepResult.info.goal_achieved   = True
```

The simulated attacker should continue.

For a harmful adversarial response, CARES returns:

```text
StepResult.done                 = True
StepResult.info.goal_achieved   = False
```

The harmful response is the final action and must be collected once as a terminal transition.

### 3. `main.py` preserves both values

[`process_turn_for_episode()`](../../src/main.py#L1806) copies the two environment outputs into separate `EpisodeState` fields:

```python
episode.done_from_env = step_result.done
episode.env_info = step_result.info

# Later:
episode.goal_achieved = episode.env_info.get("goal_achieved", 0.0)
```

The same function correctly contains an adversarial exception that continues CARES-like episodes until the environment returns `done=True`:

```python
if harmful_level > 0:
    return True  # Adversarial: continue until env returns done
```

The CARES rollout logic is therefore not the source of the primary replay bug.

### 4. Collection happens after the environment step

The main loop calls [`collect_transitions_from_episodes()`](../../src/main.py#L5318) after `process_turn_for_episode()` has executed the action and copied `StepResult.done`.

At collection time, `done_from_env=True` can mean:

> The action that just ran produced the final transition, and that transition has not been recorded yet.

It does not necessarily mean that there is no new data to collect.

For a harmful adversarial CARES response, the verified order is:

1. [`CARESOnlineEnvironment.step()`](../../src/environments/cares_env.py#L228) judges the response, detects `r_harm < 0.5`, and returns `StepResult.done=True`.
2. [`process_turn_for_episode()`](../../src/main.py#L2240) immediately assigns `episode.done_from_env = step_result.done` while retaining the response and final reward on the episode.
3. The same function returns `False` because `done_from_env=True` at [`main.py`](../../src/main.py#L2397).
4. The main loop adds the still-present episode object to `episodes_to_replace` at [`main.py`](../../src/main.py#L5302).
5. The repository's only call to `collect_transitions_from_episodes()` runs afterward at [`main.py`](../../src/main.py#L5318).
6. The collector sees `done_from_env=True` and skips the episode at [`main.py`](../../src/main.py#L2441).
7. Episode replacement occurs later at [`main.py`](../../src/main.py#L5710), after the opportunity to collect the final transition has been lost.

Therefore, `done_from_env` is not set after collection, and there is no earlier or deferred collector call that records the harmful step.

## Finding 1: nonterminal safe transitions are marked terminal

**Severity:** High

**Location:** [`src/main.py`](../../src/main.py#L2454)

The collector currently writes:

```python
transition_buffer['terminals'].append(episode.goal_achieved >= 1.0)
```

For a safe adversarial turn:

```text
episode.done_from_env = False
episode.goal_achieved = True
stored terminal       = True
```

This is a false terminal. [`update_q_function_online()`](../../src/main.py#L3176) consumes the mask as follows:

```python
if terminal:
    v_next = 0.0
else:
    v_next = v_next_values[i]

target = reward_val + config.discount_factor * v_next
```

The false terminal changes the target from:

```text
r_t + gamma * V(s_{t+1})
```

to:

```text
r_t
```

This mask reaches the optimizer rather than remaining unused metadata:

```text
collector writes terminals                    main.py:2454
    -> online loop copies it to replay         main.py:5336
    -> replay sampling copies it to the batch  main.py:5377
    -> terminal rows skip predict_v_value      main.py:3290-3293
    -> v_next is forced to zero                main.py:3332-3338
    -> the target supervises both Q and V      main.py:3396-3398, 3713-3729
    -> backward() and optimizer.step() run     value_function.py:887-944
```

For example, if `r_t=1`, `gamma=0.9`, and `V(s_{t+1})=0.6`, the correct target is `1.54`; the false terminal trains toward `1.0`.

The default [`BaseConfig.use_hierarchical_agent=False`](../../src/configs/base_config.py#L92) follows this bootstrapped online Q/V path, so standard CARES training is affected whenever an erroneous transition is sampled. In the optional `use_hierarchical_agent=True` sequence-reward branch, the base Q/V target is already the immediate reward and this particular mask does not remove a base Q/V bootstrap; evaluation mode returns without training. If regret training is enabled, the terminal mask also suppresses next-state Q, Q-min, and regret calculations, so the regret target is affected. The offline update path likewise uses `terminal` to decide whether to include its next-state Q bootstrap.

The nominal critic therefore cannot propagate later outcomes through a falsely terminal transition. Replay sampling makes the timing stochastic—the record may not be selected immediately—but once selected it changes the target, loss, gradients, and learned critic parameters.

## Finding 2: the real terminal transition is skipped

**Severity:** High

**Location:** [`src/main.py`](../../src/main.py#L2443)

Before appending a transition, the collector checks:

```python
if episode.done_from_env or (
    not _is_online_mode(config)
    and episode.turn > 0
    and not episode.user_actions
):
    continue
```

Because collection happens after `env.step()`, a response that makes CARES return `done=True` is skipped before its transition is written.

Adding the episode to `episodes_to_replace` is not itself an error. That list only queues the completed conversation for later finalization and slot reuse; it does not immediately remove the episode. The error is that collection runs while the final response and reward are still available but rejects the queued episode before appending its just-completed transition. Earlier nonterminal turns that were already copied to replay remain there, but this final turn is never added to either `transition_buffer` or `replay_buffer`.

For an adversarial failure, the missing tuple is conceptually:

```text
(
    previous observation,
    selected high-level belief,
    harmful assistant response,
    final reward,
    terminal next observation,
    terminal=True,
)
```

Skipping an already-completed episode would be correct before taking another environment step, or after its final transition had already been consumed. It is not correct before recording the action that caused `done=True`.

## Combined replay corruption

The two findings interact as follows:

| Real step | Environment result | Current replay result |
| --- | --- | --- |
| Safe adversarial intermediate response | `done=False`, `goal_achieved=True` | Collected with `terminal=True` |
| Harmful adversarial final response | `done=True`, `goal_achieved=False` | Not collected |
| Benign successful final response | `done=True`, `goal_achieved=True` | Not collected |
| Max-turn final response | `done=True`; goal value depends on response | Not collected |

The resulting trajectory can look like:

```text
Real rollout:
s0 --safe--> s1 --safe--> s2 --harmful--> terminal

Replay:
s0 --safe--> terminal
```

This biases training toward immediate per-turn reward and prevents the critic from learning whether a defense remains safe under continued pressure.

## Where the data goes

[`main.py`](../../src/main.py#L4536) initializes two in-memory dictionaries:

- `transition_buffer`: batching buffer used by the training loop;
- `replay_buffer`: online replay storage populated from `transition_buffer`.

The collector appends to `transition_buffer`. Online mode copies those entries into `replay_buffer` at [`main.py`](../../src/main.py#L5329), and [`update_q_function_online()`](../../src/main.py#L3176) samples them for critic updates.

The raw transition and replay dictionaries are not currently persisted as a dataset. Separate diagnostic files are written under `config.output_dir` by [`logging_utils.py`](../../src/utils/logging_utils.py#L1):

- `all_metrics.jsonl` contains turn-level metrics;
- `defense_episodes.jsonl` contains CARES-like episode summaries and transcripts.

A harmful final response may remain visible in those logs even when its transition is absent from critic training data. The logs do not repair the replay omission, because they are not reloaded into the training buffer.

## Adjacent finding: gated episode-replacement condition

**Severity:** Requires configuration-specific confirmation

**Location:** [`_finalize_episodes_with_batched_dpo()`](../../src/main.py#L1310)

This helper contains:

```python
or episode.goal_achieved >= 1.0
```

and can append the episode to `episodes_to_replace`. For an adversarial safe turn, that condition can again interpret per-turn safety as episode completion. The call is gated off in baseline, evaluation, and critic-only modes, so this is separate from the always-relevant collector findings and must be tested under the exact training configuration before assigning runtime impact.

## Recommended repair

Do not fix only the terminal append expression. Changing it to `episode.done_from_env` while retaining the earlier `continue` would make every collected transition nonterminal and still discard all real terminal steps.

The repair should establish an explicit one-step ownership rule:

1. After each successful `env.step()`, create or mark exactly one pending transition.
2. Consume that transition once even if its step returned `done=True`.
3. Derive the TD terminal mask from true environment termination, not from `goal_achieved`.
4. Clear the pending marker after collection so a completed transition cannot be duplicated.
5. Prevent another environment step after termination.
6. Represent max-turn truncation separately from natural termination and choose the bootstrap policy explicitly.

A robust data model would expose:

```python
terminated: bool
truncated: bool
goal_achieved: float
transition_pending: bool
```

If a smaller initial patch is preferred, store the latest `StepResult` or a transition snapshot on `EpisodeState`, consume it exactly once, and retain `goal_achieved` only for metrics and benchmark success logic.

## Required regression tests

Add hand-computed tests before changing the training loop:

1. **Adversarial safe intermediate turn**
   - CARES returns `done=False`, `goal_achieved=True`.
   - One transition is collected.
   - Its terminal mask is `False`.
   - Its TD target includes the next-state bootstrap.

2. **Adversarial harmful final turn**
   - CARES returns `done=True`, `goal_achieved=False`.
   - The final harmful response is collected exactly once.
   - Its terminal mask is `True`.
   - Its TD target contains no next-state bootstrap.

3. **Benign successful final turn**
   - CARES returns `done=True`, `goal_achieved=True`.
   - The transition is collected exactly once and marked terminal.

4. **Max-turn truncation**
   - The final transition is collected exactly once.
   - `truncated=True` is distinguishable from `terminated=True`.
   - The chosen bootstrap convention is asserted explicitly.

5. **No duplicate final transition**
   - Calling collection again without a new step does not append another record.

6. **End-to-end replay trajectory**
   - A scripted safe-safe-harmful episode produces three ordered transitions.
   - Only the third transition is terminal.
   - The harmful response and its reward are present in the sampled replay batch.

## Acceptance criteria

The review can be closed when all of the following are true:

- every executed assistant action produces exactly one transition;
- safe adversarial intermediate turns retain their TD bootstrap;
- harmful, benign-success, and max-turn final actions are not dropped;
- terminal masks are derived from termination semantics, not benchmark success metrics;
- truncation behavior is explicit and tested;
- raw replay inspection can confirm the expected transition sequence;
- baseline, evaluation, critic-only, nominal-critic, and regret-critic configurations pass the relevant regression tests.
