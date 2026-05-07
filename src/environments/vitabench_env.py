"""VitaBench environment adapter for dialogue Q-learning.

Owns processing logic for user sim, system tools, and judge using local models
in synchronous batches (like multiwoz_env / main batch_user_model_interactions).
Uses VitaBench-provided data and helper APIs; no reliance on async/barriers.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

# Add vitabench directory to Python path if not already present
_project_root = Path(__file__).parent.parent.parent
_vitabench_path = _project_root / 'vitabench' / 'src'
if _vitabench_path.exists() and str(_vitabench_path) not in sys.path:
    sys.path.insert(0, str(_vitabench_path))

from .dialogue_env import DialogueEnvironment

if TYPE_CHECKING:
    from ..training.episode_state import EpisodeState


class PlaceholderAgent:
    """
    Minimal agent for the orchestrator when the driver injects actions via env.step(action).
    The training loop batches actions and passes them to env.step(); the orchestrator never
    calls generate_next_message in this flow (actions are injected in VitaBenchEnvironment.step).
    """

    def __init__(self, tools, domain_policy):
        self.tools = tools
        self.domain_policy = domain_policy

    def get_init_state(self, message_history=None):
        return None

    def generate_next_message(self, message, state):
        from vita.data_model.message import AssistantMessage
        return (AssistantMessage(role="assistant", content=""), None)

    @classmethod
    def is_stop(cls, message):
        return False


class VitaBenchEnvironment(DialogueEnvironment):
    """
    Adapter for VitaBench environments.
    
    Wraps VitaBench Environment and Orchestrator to provide
    DialogueEnvironment interface.
    """
    
    def __init__(
        self,
        domain_name: str,
        task,
        vitabench_env,
        orchestrator,
        max_steps: int = 300,
        debug: bool = False,
        llm_evaluator: Optional[str] = None,
        llm_args_evaluator: Optional[dict] = None,
        evaluator_model=None,
        evaluator_tokenizer=None,
        user_model=None,
        user_tokenizer=None,
        _slot_init: bool = False,
    ):
        self.domain_name = domain_name
        self.task = task
        self.vitabench_env = vitabench_env
        self.orchestrator = orchestrator
        self.max_steps = max_steps
        self.debug = debug
        self.llm_evaluator = llm_evaluator
        self.llm_args_evaluator = llm_args_evaluator
        self.evaluator_model = evaluator_model
        self.evaluator_tokenizer = evaluator_tokenizer
        self.user_model = user_model
        self.user_tokenizer = user_tokenizer
        self._message_history = []
        self._turn_count = 0
        self._evaluated_goal_fraction: Optional[float] = None
        self._evaluation_done: bool = False
        self._last_evaluated_turn: int = -1  # Track which turn was last evaluated
        self._previous_met_rubrics: set = set()  # Track which rubrics were met in the previous turn
        self._current_reward_info = None  # Store current reward info for rubric tracking
        self._slot_init = _slot_init

        if not _slot_init:
            print(f"VitaBenchEnvironment initialized with domain: {domain_name}")
    
    def reset(self) -> Tuple[str, Dict]:
        """Reset environment and return initial observation."""
        self.orchestrator.initialize()
        self._message_history = []
        self._turn_count = 0
        self._evaluated_goal_fraction = None
        self._evaluation_done = False
        self._last_evaluated_turn = -1  # Track which turn was last evaluated
        self._previous_met_rubrics = set()  # Reset previous met rubrics
        self._current_reward_info = None  # Reset current reward info

        if not self._slot_init:
            print(f"VitaBenchEnvironment reset")

        # After initialization, orchestrator may have a first agent message
        # We need to step to get the first user message if the first message is from agent
        from vita.data_model.message import AssistantMessage, UserMessage
        from vita.orchestrator.orchestrator import Role
        
        # If the first message is from agent and to user, step to get user response
        if (self.orchestrator.from_role == Role.AGENT and 
            self.orchestrator.to_role == Role.USER and 
            isinstance(self.orchestrator.message, AssistantMessage) and
            not self.orchestrator.message.is_tool_call()):
            # Step orchestrator to get user's first response
            if not self.orchestrator.done:
                self.orchestrator.step()
        
        # Get initial user message if available
        initial_obs = self._get_current_observation()
        info = self._get_info()
        
        return initial_obs, info
    
    def step(
        self,
        action: str,
        user_response: Optional[str] = None,
        skip_reward: bool = False,
    ) -> Tuple[str, float, bool, dict]:
        """
        Execute agent action and return observation, reward, done, info.
        This is the single place that advances env state (trajectory, turn count, reward, done).

        - user_response is None: get user reply by calling orchestrator (API or patched generate).
        - user_response is str: use this as the user message (batch path; no LLM call here).
        - skip_reward: if True, apply state update only and return reward=0.0, info without goal_achieved
          (used by batch_step_episodes so reward can be computed in one batched evaluator call).
        """
        agent_message = self._parse_action(action)
        from vita.orchestrator.orchestrator import Role

        self.orchestrator.trajectory.append(agent_message)
        self.orchestrator.message = agent_message
        self.orchestrator.from_role = Role.AGENT

        if user_response is not None:
            self.orchestrator.to_role = Role.USER
            from vita.data_model.message import UserMessage
            from vita.user.user_simulator import UserSimulator
            from vita.data_model.simulation import TerminationReason
            user_message = UserMessage(role="user", content=user_response, cost=0.0, usage={}, raw_data=None)
            self.orchestrator.user_state.messages.append(user_message)
            self.orchestrator.trajectory.append(user_message)
            self.orchestrator.message = user_message
            self.orchestrator.from_role = Role.USER
            self.orchestrator.to_role = Role.AGENT
            if UserSimulator.is_stop(user_message):
                self.orchestrator.done = True
                self.orchestrator.termination_reason = TerminationReason.USER_STOP
        else:
            if agent_message.is_tool_call():
                self.orchestrator.to_role = Role.ENV
            else:
                self.orchestrator.to_role = Role.USER
            if not self.orchestrator.done:
                self.orchestrator.step()

        self._turn_count += 1
        self._message_history = list(self.orchestrator.trajectory)

        if skip_reward:
            observation = self._get_current_observation()
            done = self.orchestrator.done or self._turn_count >= self.max_steps
            info = self._get_info(skip_goal=True)
            from .dialogue_env import StepResult
            return StepResult(observation, 0.0, done, info)
        return self._observation_reward_done_info()

    def apply_step_with_precomputed_user_response(self, agent_message, user_message) -> Tuple[str, float, bool, dict]:
        """Thin wrapper: advance state via step() with precomputed user content."""
        user_content = user_message.content if hasattr(user_message, "content") else str(user_message)
        return self.step(self._action_from_message(agent_message), user_response=user_content)

    def _action_from_message(self, agent_message) -> str:
        """Serialize agent message back to action string for step(action, ...)."""
        if agent_message.is_tool_call() and agent_message.tool_calls:
            tc = agent_message.tool_calls[0]
            return json.dumps({"name": tc.name, "arguments": tc.arguments})
        return agent_message.content or ""

    def _observation_reward_done_info(self):
        from .dialogue_env import StepResult
        observation = self._get_current_observation()
        reward = self._compute_reward()
        done = self._check_done()
        self._message_history = list(self.orchestrator.trajectory)
        info = self._get_info()
        return StepResult(observation, reward, done, info)
    
    def check_goal(self, observation: str) -> bool:
        """Check if task goal is satisfied."""
        return self._check_task_completion()

    def compute_reward_for_marginal(
        self, action: str, actual_reward: Optional[float] = None
    ) -> float:
        task_completed = self._check_task_completion()
        base_reward = 1.0 if task_completed else 0.0
        if actual_reward is not None:
            return actual_reward
        return base_reward * 0.5

    def _parse_action(self, action: str):
        """Parse text action into VitaBench AssistantMessage or ToolCall."""
        from vita.data_model.message import AssistantMessage, ToolCall
        
        # Try to parse as JSON tool call first
        try:
            tool_call_data = json.loads(action)
            if "name" in tool_call_data and "arguments" in tool_call_data:
                tool_call = ToolCall(
                    id=tool_call_data.get("id", ""),
                    name=tool_call_data["name"],
                    arguments=tool_call_data["arguments"],
                    requestor="assistant"
                )
                return AssistantMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[tool_call]
                )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        
        # Otherwise treat as text message
        return AssistantMessage(role="assistant", content=action)
    
    def _get_current_observation(self) -> str:
        """Extract current user observation from orchestrator."""
        from vita.data_model.message import UserMessage
        
        # Get last user message from trajectory
        for msg in reversed(self.orchestrator.trajectory):
            if isinstance(msg, UserMessage):
                return msg.content or ""
        return ""
    
    def _compute_reward(self) -> float:
        """Compute reward based on newly completed rubrics."""
        # Get current rubric states by running evaluation
        # This ensures we have the latest rubric information
        self._get_goal_achievement_fraction()
        
        # If we don't have reward info, return 0
        if self._current_reward_info is None or self._current_reward_info.nl_rubrics is None:
            return 0.0
        
        # Get current set of met rubrics
        current_met_rubrics = set()
        for idx, rubric_check in enumerate(self._current_reward_info.nl_rubrics):
            if rubric_check.met:
                # Use rubric text as identifier if available, otherwise use index
                if rubric_check.nl_rubric:
                    rubric_id = rubric_check.nl_rubric
                else:
                    rubric_id = f"rubric_{idx}"  # Fallback to index if no text
                current_met_rubrics.add(rubric_id)
        
        # Count newly completed rubrics (rubrics that are met now but weren't before)
        newly_completed = current_met_rubrics - self._previous_met_rubrics
        num_newly_completed = len(newly_completed)
        
        # Update previous met rubrics for next turn
        self._previous_met_rubrics = current_met_rubrics.copy()
        
        # Return 1 per newly completed rubric
        return float(num_newly_completed)
    
    def _check_done(self) -> bool:
        """Check if episode is done."""
        return (
            self.orchestrator.done or
            self._turn_count >= self.max_steps or
            self._check_task_completion()
        )
    
    def _get_goal_achievement_fraction(self) -> float:
        """Get goal achievement as fraction (0.0 to 1.0) via sliding window evaluation."""
        # Check if we need to re-evaluate (new turn or episode ended)
        episode_ended = self.orchestrator.done or self._turn_count >= self.max_steps
        needs_evaluation = (
            not self._evaluation_done or 
            self._turn_count != self._last_evaluated_turn or
            episode_ended
        )
        
        if needs_evaluation:
            # Run evaluation (can work mid-episode with partial trajectory)
            try:
                from vita.evaluator.evaluator_traj import TrajectoryEvaluator
                from vita.data_model.simulation import RewardType
                
                # Get current trajectory (may be partial if mid-episode)
                current_trajectory = list(self.orchestrator.trajectory)
                current_state = self._get_final_state()
                
                # Run sliding window evaluation with window_size=2
                # This can work with partial trajectories
                reward_info = TrajectoryEvaluator.calculate_reward(
                    task=self.task,
                    full_trajectory=current_trajectory,
                    final_state=current_state,
                    window_size=2,
                    overlap=1,  # Half of window_size
                    llm_evaluator=self.llm_evaluator,
                    llm_args_evaluator=self.llm_args_evaluator,
                    language=None
                )
                
                # Store reward info for rubric tracking
                self._current_reward_info = reward_info
                
                # Extract fraction from reward_breakdown
                self._evaluated_goal_fraction = reward_info.reward_breakdown.get(
                    RewardType.NL_ASSERTION, 0.0
                )
                self._last_evaluated_turn = self._turn_count
                
                # Mark as done only if episode has ended
                if episode_ended:
                    self._evaluation_done = True
            except Exception as e:
                # Fallback to binary check if evaluation fails
                if self.debug:
                    print(f"[VitaBench] Evaluation failed: {e}, falling back to binary check")
                from vita.data_model.simulation import TerminationReason, RewardInfo, RewardType
                if episode_ended and hasattr(self.orchestrator, 'termination_reason'):
                    if self.orchestrator.termination_reason == TerminationReason.USER_STOP:
                        self._evaluated_goal_fraction = 1.0
                        # Create a dummy reward info with all rubrics met
                        self._current_reward_info = RewardInfo(
                            reward=1.0,
                            nl_rubrics=[],
                            reward_breakdown={RewardType.NL_ASSERTION: 1.0},
                            info={"note": "Fallback: USER_STOP"}
                        )
                    else:
                        self._evaluated_goal_fraction = 0.0
                        self._current_reward_info = RewardInfo(
                            reward=0.0,
                            nl_rubrics=[],
                            reward_breakdown={RewardType.NL_ASSERTION: 0.0},
                            info={"note": "Fallback: evaluation failed"}
                        )
                else:
                    # Mid-episode: can't determine completion yet, return 0.0
                    self._evaluated_goal_fraction = 0.0
                    self._current_reward_info = RewardInfo(
                        reward=0.0,
                        nl_rubrics=[],
                        reward_breakdown={RewardType.NL_ASSERTION: 0.0},
                        info={"note": "Fallback: mid-episode"}
                    )
                self._last_evaluated_turn = self._turn_count
                if episode_ended:
                    self._evaluation_done = True
        
        return self._evaluated_goal_fraction if self._evaluated_goal_fraction is not None else 0.0
    
    def _get_final_state(self) -> Dict:
        """Get final state from orchestrator for evaluation."""
        try:
            if hasattr(self.orchestrator, 'environment') and hasattr(self.orchestrator.environment, 'tools'):
                db = self.orchestrator.environment.tools.db
                env_time = getattr(db, 'time', '') if hasattr(db, 'time') else ''
                if env_time:
                    return self.orchestrator.get_states(db, env_time)
        except Exception:
            pass
        # Fallback to empty state dict
        return {"old_states": [], "new_states": []}
    
    def _check_task_completion(self) -> bool:
        """Check if task is completed (binary, for backward compatibility)."""
        return self._get_goal_achievement_fraction() >= 1.0
    
    def _get_info(self, skip_goal: bool = False) -> Dict:
        """Get environment info dict. skip_goal=True avoids evaluator call (use cached or None)."""
        goal = None if skip_goal else self._get_goal_achievement_fraction()
        if skip_goal and self._evaluated_goal_fraction is not None:
            goal = self._evaluated_goal_fraction
        return {
            "domain": self.domain_name,
            "task_id": self.task.id if self.task else None,
            "turn": self._turn_count,
            "trajectory": [self._message_to_dict(msg) for msg in self._message_history],
            "goal_achieved": goal,
        }
    
    def _message_to_dict(self, msg) -> Dict:
        """Convert VitaBench message to dict for serialization."""
        if hasattr(msg, 'model_dump'):
            return msg.model_dump()
        elif hasattr(msg, 'dict'):
            return msg.dict()
        else:
            # Fallback: convert to string representation
            return {"role": getattr(msg, 'role', 'unknown'), "content": str(msg)}


def _is_tool_call_action(action: str) -> bool:
    """True if action is a JSON tool call."""
    try:
        data = json.loads(action)
        return isinstance(data, dict) and "name" in data and "arguments" in data
    except (json.JSONDecodeError, TypeError):
        return False


def batch_compute_vitabench_rewards(
    eligible: List["EpisodeState"],
    config,
) -> List[Tuple[float, dict]]:
    """
    Batch sliding-window evaluator across all eligible episodes (one LLM batch).
    Sets _current_reward_info, _evaluated_goal_fraction, etc. on each episode's env.
    Returns list of (reward, info) in same order as eligible.
    """
    from vita.data_model.simulation import RewardInfo, RewardType
    from ..utils.llm_utils import batch_generate
    from .vitabench_helpers import _vitabench_sliding_window_prompts, _vitabench_reward_from_window_results

    if not eligible:
        return []

    env0 = eligible[0].env
    evaluator_model = getattr(env0, "evaluator_model", None)
    evaluator_tokenizer = getattr(env0, "evaluator_tokenizer", None)
    if evaluator_model is None or evaluator_tokenizer is None:
        raise ValueError(
            "batch_compute_vitabench_rewards requires evaluator_model and evaluator_tokenizer on env"
        )
    llm_args = getattr(env0, "llm_args_evaluator", None) or {}
    max_new_tokens = llm_args.get("max_tokens", 256)
    temperature = llm_args.get("temperature", 0.7)
    do_sample = temperature > 0.0
    minibatch_size = getattr(config, "vitabench_minibatch_size", 8)

    def _slot(env):
        return getattr(env, "_slot", env)

    rows = []
    all_prompts = []
    for ep in eligible:
        env = ep.env
        slot = _slot(env)
        task = slot.task
        trajectory = list(slot.orchestrator.trajectory)
        prompts, num_windows = _vitabench_sliding_window_prompts(
            task, trajectory, window_size=2, overlap=1, language=None
        )
        rows.append((ep, env, slot, task, num_windows))
        all_prompts.extend(prompts)

    if not all_prompts:
        result = []
        for _, env, slot, _, _ in rows:
            slot._current_reward_info = RewardInfo(
                reward=0.0,
                nl_rubrics=[],
                reward_breakdown={RewardType.NL_ASSERTION: 0.0},
                info={"evaluation_method": "sliding_window_batched", "num_windows": 0},
            )
            slot._evaluated_goal_fraction = 0.0
            slot._last_evaluated_turn = slot._turn_count
            result.append((0.0, env._get_info()))
        return result

    generated_texts = batch_generate(
        model=evaluator_model,
        tokenizer=evaluator_tokenizer,
        prompts=all_prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        chunk_size=minibatch_size,
    )
    if len(generated_texts) != len(all_prompts):
        raise RuntimeError(
            f"batch_generate returned {len(generated_texts)} results for {len(all_prompts)} prompts"
        )

    result = []
    offset = 0
    for ep, env, slot, task, num_windows in rows:
        if num_windows == 0:
            reward_info = RewardInfo(
                reward=0.0,
                nl_rubrics=[],
                reward_breakdown={RewardType.NL_ASSERTION: 0.0},
                info={"evaluation_method": "sliding_window_batched", "num_windows": 0},
            )
        else:
            slice_texts = generated_texts[offset : offset + num_windows]
            offset += num_windows
            evaluation_criteria = task.evaluation_criteria
            reward_info = _vitabench_reward_from_window_results(evaluation_criteria, slice_texts)

        slot._current_reward_info = reward_info
        slot._evaluated_goal_fraction = reward_info.reward_breakdown.get(RewardType.NL_ASSERTION, 0.0)
        slot._last_evaluated_turn = slot._turn_count
        episode_ended = slot.orchestrator.done or slot._turn_count >= slot.max_steps
        if episode_ended:
            slot._evaluation_done = True

        info = env._get_info()
        result.append((reward_info.reward, info))

    return result


def batch_step_episodes(episodes: List["EpisodeState"], config) -> None:
    """
    Batch user-sim steps for VitaBench with local models (synchronous, like multiwoz batch_user_model_interactions).
    Batch-generates user responses, then calls env.step(action, user_response=...) for each episode
    so state flow always goes through step().
    Only includes episodes whose agent action is not a tool call (tool calls are handled by env.step).
    """
    from .vitabench_helpers import _prepare_user_sim_messages
    from ..utils.llm_utils import messages_to_prompt, batch_generate

    eligible = [
        ep for ep in episodes
        if ep.is_active and not ep.done_from_env and not ep.is_frozen
        and getattr(ep, "agent_response", None) is not None
        and not _is_tool_call_action(ep.agent_response)
    ]
    if not eligible:
        return

    print(f"[VitaBench] batch_step_episodes: {len(eligible)} eligible (calling env.step with precomputed user)")
    sys.stdout.flush()

    env0 = eligible[0].env
    user_model = getattr(env0, "user_model", None)
    user_tokenizer = getattr(env0, "user_tokenizer", None)
    if user_model is None or user_tokenizer is None:
        raise ValueError(
            "batch_step_episodes requires local user model; set user_model and user_tokenizer on VitaBenchEnvironment"
        )

    llm_args = env0.orchestrator.user.llm_args
    max_new_tokens = llm_args.get("max_tokens", 256)
    temperature = llm_args.get("temperature", 0.7)
    do_sample = temperature > 0.0
    minibatch_size = getattr(config, "vitabench_minibatch_size", 8)

    messages_per_episode = []
    agent_messages = []
    for ep in eligible:
        agent_message = ep.env._parse_action(ep.agent_response)
        agent_messages.append(agent_message)
        messages_per_episode.append(_prepare_user_sim_messages(ep.env, agent_message))

    prompts = [messages_to_prompt(msgs) for msgs in messages_per_episode]
    generated_texts = batch_generate(
        model=user_model,
        tokenizer=user_tokenizer,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        chunk_size=minibatch_size,
    )
    if len(generated_texts) != len(eligible):
        raise RuntimeError(
            f"batch_generate returned {len(generated_texts)} results for {len(eligible)} episodes"
        )

    # Apply state update only (no per-episode evaluator call)
    for ep, user_text in zip(eligible, generated_texts):
        step_result = ep.env.step(ep.agent_response, user_response=user_text, skip_reward=True)
        ep.observation = step_result.observation
        ep.reward_from_env = step_result.reward
        ep.done_from_env = step_result.done
        ep.env_info = step_result.info

    # Single batched evaluator call for all eligible episodes
    rewards_infos = batch_compute_vitabench_rewards(eligible, config)
    for ep, (reward, info) in zip(eligible, rewards_infos):
        ep.reward_from_env = reward
        ep.env_info = info
        if ep.env_info and ep.env_info.get("goal_state"):
            ep.last_known_goal_state = ep.env_info.get("goal_state")
        ep._step_done_by_batch = True


class BatchVitaBenchEnvironment:
    """
    Single batch environment for VitaBench when using local models.
    Holds N slots (each a VitaBenchEnvironment); shared user_model/user_tokenizer/evaluator.
    Replace a finished episode by replacing one row (replace_slot) instead of creating a new env.
    """

    def __init__(
        self,
        slots: List[VitaBenchEnvironment],
        user_model,
        user_tokenizer,
        evaluator_model=None,
        evaluator_tokenizer=None,
    ):
        self.slots = slots
        self.user_model = user_model
        self.user_tokenizer = user_tokenizer
        self.evaluator_model = evaluator_model
        self.evaluator_tokenizer = evaluator_tokenizer

    def step_single(
        self,
        idx: int,
        action: str,
        user_response: Optional[str] = None,
        skip_reward: bool = False,
    ) -> Tuple[str, float, bool, dict]:
        return self.slots[idx].step(action, user_response=user_response, skip_reward=skip_reward)

    def step_batch(
        self, actions: List[str], user_responses: Optional[List[str]] = None
    ) -> List[Tuple[str, float, bool, dict]]:
        if user_responses is None:
            user_responses = [None] * len(actions)
        if len(actions) != len(self.slots) or len(actions) != len(user_responses):
            raise ValueError("actions and user_responses length must match number of slots")
        return [
            self.slots[i].step(actions[i], user_response=user_responses[i])
            for i in range(len(actions))
        ]

    def reset_single(self, idx: int) -> Tuple[str, dict]:
        return self.slots[idx].reset()

    def reset_batch(self) -> List[Tuple[str, dict]]:
        return [self.slots[i].reset() for i in range(len(self.slots))]

    def replace_slot(
        self, idx: int, task_id: str, domain: str, config, language: str = "english", debug: bool = False
    ) -> Tuple[str, dict, object, str]:
        """
        Replace slot idx with a new task (same row of the tensor). Returns (initial_obs, initial_info, dialogue_data, ground_truth_goal).
        """
        from .vitabench_helpers import create_vitabench_episode_state
        preloaded = (
            self.user_model,
            self.user_tokenizer,
            self.evaluator_model,
            self.evaluator_tokenizer,
        )
        new_ep = create_vitabench_episode_state(
            task_id=task_id,
            domain=domain,
            config=config,
            debug=debug,
            language=language,
            _preloaded=preloaded,
        )
        self.slots[idx] = new_ep.env
        return (
            new_ep.initial_observation,
            new_ep.initial_env_info,
            new_ep.dialogue_data,
            new_ep.ground_truth_goal,
        )


class EpisodeVitaBenchEnvWrapper:
    """
    Thin wrapper so one episode sees a single-slot interface while sharing BatchVitaBenchEnvironment.
    Delegates attribute access to the slot; user_model/user_tokenizer come from the batch (shared).
    """

    def __init__(self, batch_env: BatchVitaBenchEnvironment, idx: int):
        self._batch_env = batch_env
        self._idx = idx

    @property
    def _slot(self) -> VitaBenchEnvironment:
        return self._batch_env.slots[self._idx]

    def step(
        self,
        action: str,
        user_response: Optional[str] = None,
        skip_reward: bool = False,
    ) -> Tuple[str, float, bool, dict]:
        return self._batch_env.step_single(
            self._idx, action, user_response=user_response, skip_reward=skip_reward
        )

    def reset(self) -> Tuple[str, dict]:
        return self._batch_env.reset_single(self._idx)

    @property
    def user_model(self):
        return self._batch_env.user_model

    @property
    def user_tokenizer(self):
        return self._batch_env.user_tokenizer

    def __getattr__(self, name: str):
        return getattr(self._slot, name)

