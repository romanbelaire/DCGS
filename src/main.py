"""Main entry point for training dialogue loop."""

import json
import math
import os
import random
import sys
import time
import torch
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .configs import BaseConfig
from .agents import (
    HighLevelAgent,
    FreeformHighLevelAgent,
    LowLevelAgent,
    UserAgent,
    GPTHighLevelAgent,
    GPTLowLevelAgent,
    GPTUserAgent,
)
from .belief import (
    BeliefCandidate,
    BeliefState,
    DialogueState,
    compute_log_probs_batch,
    update_belief_distribution_dpo,
    compute_entropy,
    compute_average_entropy,
)
from .data import (
    load_multiwoz_dataset,
    format_multiwoz_goal,
    ensure_goal_json,
    get_ground_truth_belief,
    format_dialogue_history,
    compute_belief_accuracy,
    extract_goal_from_dialogue_state,
    infer_goal_from_dialogue,
)
from .data.multiwoz_loader import MultiWOZDialogue
from .value import ValueFunction
from .value.cares_reward import FulfillmentJudgeParseError
from .utils.llm_utils import batch_generate, compute_log_prob_batch
from .prompts.prompt_manager import PromptManager
from .utils.llm_utils import (
    get_model_instance,
    get_user_model_instance,
    get_tokenizer_instance,
    get_judge_model_and_tokenizer,
)
from .utils.logging_utils import (
    DEFENSE_EPISODES_FILENAME,
    append_defense_episode_record,
    harmful_level_for_defense_log,
    log_metrics,
    save_rollout,
    print_gpu_memory,
)
from .utils.belief_evaluation import (
    convert_goal_json_to_natural_language,
    evaluate_beliefs_against_ground_truth
)
from .environments import MultiWOZEnvironment, OnlineEnvironment
from .environments.episode_factory import (
    EpisodeCreationContext,
    create_episode_from_registry,
)
from .simulation import extract_slots_from_agent_response, extract_service_call_from_agent_response
from .training.episode_state import EpisodeState
from .training.hierarchical_rollout import HierarchicalRolloutCoordinator
from .defense import SmoothLLMWrapper, TPOWrapper


def _softmax_sample_from_q_values(q_values: Dict[str, float]) -> str:
    """Sample from softmax distribution over Q-values. Returns selected key."""
    keys = list(q_values.keys())
    values = [q_values[k] for k in keys]
    max_val = max(values)
    exps = [math.exp(v - max_val) for v in values]
    total = sum(exps)
    probs = [e / total for e in exps]
    return random.choices(keys, weights=probs, k=1)[0]


def _regret_critic_selection(
    q_values: Dict[str, float],
    regret_values: Dict[str, float],
    beta: float,
) -> str:
    """Sample a high-level belief with P(b) ∝ exp((1-β)Q(b) − β Q^regret(b)).

    With β = regret_critic_beta = 0.2 this is a softmax sample over 0.8·Q − 0.2·Q^regret
    (numerically stabilized: subtract max score before exp). Config validation requires β=0.2 when regret critic is on.
    """
    keys = list(q_values.keys())
    scores = [(1.0 - beta) * q_values[k] - beta * regret_values[k] for k in keys]
    max_val = max(scores)
    exps = [math.exp(s - max_val) for s in scores]
    total = sum(exps)
    probs = [e / total for e in exps]
    return random.choices(keys, weights=probs, k=1)[0]


def _compute_candidate_pool_q_metrics(
    episode: EpisodeState,
    candidate_beliefs: List[str],
    q_values: Dict[str, float],
) -> Dict[str, float]:
    """Compute per-turn candidate-pool support and expected-Q diagnostics."""
    valid_candidates = [cand for cand in candidate_beliefs if cand != "[SKIP]"]
    valid_q_values = [q_values[cand] for cand in valid_candidates]
    q_min = min(valid_q_values)
    q_max = max(valid_q_values)
    q_mean = sum(valid_q_values) / len(valid_q_values)
    q_var = sum((value - q_mean) ** 2 for value in valid_q_values) / len(valid_q_values)
    q_std = math.sqrt(q_var)
    q_gap = q_max - q_min
    normalized_q_gap = q_gap / (abs(q_mean) + 1e-8)

    candidate_probabilities = {
        candidate.summary: candidate.probability
        for candidate in episode.belief_state.candidates
        if candidate.summary != "[SKIP]"
    }
    baseline_weight_sum = sum(candidate_probabilities[cand] for cand in valid_candidates)
    expected_q_baseline = (
        sum(candidate_probabilities[cand] * q_values[cand] for cand in valid_candidates)
        / baseline_weight_sum
    )

    max_q = q_max
    softmax_unnormalized = [math.exp(q_values[cand] - max_q) for cand in valid_candidates]
    softmax_weight_sum = sum(softmax_unnormalized)
    expected_q_guided = sum(
        (weight / softmax_weight_sum) * q_values[cand]
        for cand, weight in zip(valid_candidates, softmax_unnormalized)
    )

    return {
        "candidate_pool_size_total": len(candidate_beliefs),
        "candidate_pool_size_valid": len(valid_candidates),
        "q_min": q_min,
        "q_max": q_max,
        "q_mean": q_mean,
        "q_std": q_std,
        "q_gap": q_gap,
        "q_gap_normalized": normalized_q_gap,
        "expected_q_baseline_pi": expected_q_baseline,
        "expected_q_guided_pi_k": expected_q_guided,
        "expected_q_delta_guided_minus_baseline": expected_q_guided - expected_q_baseline,
    }


# Registry: environment_type -> action generation template name
LL_TEMPLATE_BY_ENV = {
    "cares": "action_generation_cares",
    "wildjailbreak": "action_generation_wildjailbreak",
    "redbench": "action_generation_wildjailbreak",
    "harmbench": "action_generation_wildjailbreak",
    "userbench": "action_generation_userbench",
}


def get_ll_action_template_name(config, baseline_mode: bool = False, episode=None) -> str:
    """Select action generation template from environment type and baseline mode."""
    if baseline_mode:
        if episode and hasattr(episode.env, "travel_env"):
            return "action_generation_userbench_baseline"
        return "action_generation_baseline"
    if episode and hasattr(episode.env, "travel_env"):
        return "action_generation_userbench"
    env_type = getattr(config, "environment_type", "multiwoz")
    return LL_TEMPLATE_BY_ENV.get(env_type, "action_generation")


def _extract_perturbable_span(prompt: str, ll_history: List[Tuple[str, str]]) -> str:
    return ll_history[-1][1]


def _ensure_response_tags(text: str) -> str:
    if "[RESPONSE]" in text and "[/RESPONSE]" in text:
        return text
    return f"[RESPONSE]\n{text}\n[/RESPONSE]"


def _generate_defender_action_with_backend(
    ll_agent: LowLevelAgent,
    prompt: str,
    ll_history: List[Tuple[str, str]],
    template_name: str,
    config
) -> Tuple[str, int]:
    backend = getattr(config, "defender_backend", "standard")
    if backend == "standard":
        outputs = ll_agent.generate_actions_from_prompts(
            prompts=[prompt],
            temperature=0.7,
            do_sample=False,
            chunk_size=1,
            template_name=template_name
        )
        if len(outputs) != 1:
            raise ValueError(f"Standard backend expected one output, got {len(outputs)}")
        text = outputs[0]
        token_count = len(ll_agent.tokenizer.encode(text, add_special_tokens=False))
        return text, token_count

    if backend == "smoothllm":
        wrapper = SmoothLLMWrapper(
            model=ll_agent.model,
            tokenizer=ll_agent.tokenizer,
            pert_type=getattr(config, "pert_type", "RandomSwapPerturbation"),
            pert_pct=getattr(config, "pert_pct", 10.0),
            num_copies=getattr(config, "num_copies", 8),
            batch_size=getattr(config, "smooth_batch_size", 8),
            max_new_tokens=getattr(config, "max_tokens", 256),
        )
        perturbable_span = _extract_perturbable_span(prompt, ll_history)
        text, token_count = wrapper.generate(full_prompt=prompt, perturbable_span=perturbable_span)
        return _ensure_response_tags(text), token_count

    if backend == "tpo":
        wrapper = TPOWrapper(
            model=ll_agent.model,
            tokenizer=ll_agent.tokenizer,
            sample_size=getattr(config, "tpo_sample_size", 8),
            max_iters=getattr(config, "tpo_max_iters", 3),
            temperature=getattr(config, "tpo_temperature", 0.7),
            mode=getattr(config, "tpo_mode", "tpo"),
            max_new_tokens=getattr(config, "max_tokens", 256),
        )
        text, token_count = wrapper.generate(prompt=prompt)
        return _ensure_response_tags(text), token_count

    raise ValueError(f"Unknown defender_backend: {backend}")


def is_valid_multiwoz_goal(goal: Dict) -> bool:
    """
    Return True if the MultiWOZ goal structure contains at least one actionable slot.

    Some dialogues include goal dictionaries with domain keys but empty slot information.
    We only treat a goal as valid if at least one domain has non-empty inform_slots or request_slots.
    """
    if not goal or not isinstance(goal, dict):
        return False
    
    for domain_goal in goal.values():
        if not isinstance(domain_goal, dict):
            continue
        
        inform_slots = domain_goal.get("inform_slots") or {}
        request_slots = domain_goal.get("request_slots") or {}
        
        has_inform = any(
            value not in (None, "", [])
            for value in inform_slots.values()
        )
        has_request = bool(request_slots)
        
        if has_inform or has_request:
            return True
    
    return False


def has_booking_intent(goal: Dict) -> bool:
    """
    Check if a goal has booking intent (inform_slots) vs. just information requests (request_slots only).
    
    Goals with booking intent have inform_slots (requirements/constraints) that need to be satisfied.
    Goals with only request_slots are just information requests without concrete booking requirements.
    
    Args:
        goal: MultiWOZ goal dictionary
        
    Returns:
        True if goal has at least one domain with non-empty inform_slots (booking intent)
    """
    if not goal or not isinstance(goal, dict):
        return False
    
    for domain, domain_goal in goal.items():
        if not isinstance(domain_goal, dict):
            continue
        
        inform_slots = domain_goal.get("inform_slots", {})
        if inform_slots and isinstance(inform_slots, dict):
            # Check if there are any non-empty inform_slots
            if any(v for v in inform_slots.values() if v and v != ""):
                return True
    
    return False


def extract_user_actions(turn: Dict, debug: bool = False) -> List[str]:
    """Extract user action types from MultiWOZ dataset."""
    actions: List[str] = []
    if not isinstance(turn, dict):
        return actions

    def add_action(act: Optional[str]) -> None:
        if act:
            actions.append(act.upper())

    frames = turn.get("frames", [])
    if isinstance(frames, list):
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            frame_actions = frame.get("actions", [])
            if isinstance(frame_actions, list):
                for action in frame_actions:
                    if isinstance(action, dict):
                        add_action(action.get("act") or action.get("act_type"))
                    elif isinstance(action, str):
                        add_action(action)
            state = frame.get("state", {})
            if isinstance(state, dict):
                user_action = state.get("user_action") or state.get("user_actions")
                if isinstance(user_action, list):
                    for action in user_action:
                        if isinstance(action, dict):
                            add_action(action.get("act") or action.get("act_type"))
                        elif isinstance(action, str):
                            add_action(action)

    if not actions:
        dialogue_acts = turn.get("dialogue_acts") or turn.get("dialogue_act")
        if isinstance(dialogue_acts, dict):
            for domain_or_act, acts in dialogue_acts.items():
                if isinstance(acts, dict):
                    for act_type in acts.keys():
                        add_action(act_type)
                elif isinstance(acts, list):
                    for act in acts:
                        if isinstance(act, dict):
                            add_action(act.get("act") or act.get("act_type"))
                        elif isinstance(act, str):
                            add_action(act)
                else:
                    add_action(str(domain_or_act))
        elif isinstance(dialogue_acts, list):
            for act in dialogue_acts:
                if isinstance(act, dict):
                    add_action(act.get("act") or act.get("act_type"))
                elif isinstance(act, str):
                    add_action(act)

    if not actions:
        direct_actions = turn.get("actions", [])
        if isinstance(direct_actions, list):
            for action in direct_actions:
                if isinstance(action, dict):
                    add_action(action.get("act") or action.get("act_type"))
                elif isinstance(action, str):
                    add_action(action)

    if not actions and "text" in turn:
        text = turn.get("text") or turn.get("utterance")
        if isinstance(text, str):
            text_upper = text.upper()
            if "THANK" in text_upper:
                add_action("THANK_YOU")
            if "GOODBYE" in text_upper or "BYE" in text_upper:
                add_action("GOODBYE")
            if "YES" in text_upper or "AFFIRM" in text_upper:
                add_action("AFFIRM")


    return actions


def check_goal_achieved(turn: Dict) -> bool:
    """
    Check if task goal is achieved based on user actions, from the MultiWOZ dataset.
    
    Task success indicators (reward = 1):
    - AFFIRM: User agrees/confirms
    - SELECT: User selects a result
    - THANK_YOU: User thanks the system
    
    Args:
        turn: MultiWOZ user turn dictionary
    
    Returns:
        True if goal is achieved (reward = 1), False otherwise
    """
    if not isinstance(turn, dict):
        return False
    
    speaker = turn.get("speaker", "").upper()
    if speaker not in ["USER", "user"]:
        return False
    
    user_actions = extract_user_actions(turn, debug=False)
    success_actions = ["AFFIRM", "SELECT", "THANK_YOU"]
    
    # Check if any success action is present (case-insensitive)
    return any(action.upper() in success_actions for action in user_actions)


def check_episode_ended(turn: Dict) -> bool:
    """
    Check if episode should end based on GOODBYE action, from the MultiWOZ dataset.
    
    Episode ends when:
    - User says GOODBYE
    - System says GOODBYE
    
    Args:
        turn: MultiWOZ user turn dictionary
    
    Returns:
        True if episode should end, False otherwise
    """
    if not isinstance(turn, dict):
        return False
    
    speaker = turn.get("speaker", "").upper()
    if speaker not in ["USER", "SYSTEM"]:
        return False
    
    # Extract actions (works for both user and system)
    actions = extract_user_actions(turn, debug=False)
    
    # Check for GOODBYE (case-insensitive)
    return any(action.upper() == "GOODBYE" for action in actions)


def load_multiwoz_examples_for_prompts(
    dialogues: List,
    n_examples: int = 10
) -> List:
    """
    Load a subset of MultiWOZ dialogues to use as examples in user prompts.
    
    Args:
        dialogues: List of all MultiWOZ dialogues
        n_examples: Number of examples to load
    
    Returns:
        List of MultiWOZDialogue objects to use as examples
    """
    import random
    if len(dialogues) <= n_examples:
        return dialogues
    return random.sample(dialogues, n_examples)


def create_episode_state(
    dialogue_idx: int,
    dialogue_data,
    multiwoz_mode: bool,
    config,
    debug: bool = False,
    user_agent=None,
    persona=None,
    available_personas=None,
    patient_agent=None,
    reward_model=None,
    reward_tokenizer=None,
    judge_model=None,
    judge_tokenizer=None,
) -> EpisodeState:
    """
    Create and initialize an EpisodeState from dialogue data.

    Uses registry for vitabench, salesagent, userbench, cares, wildjailbreak, redbench, harmbench.
    MultiWOZ has custom logic due to persona/goal_state handling.
    """
    environment_type = getattr(config, "environment_type", "multiwoz")

    if environment_type in (
        "vitabench",
        "salesagent",
        "userbench",
        "cares",
        "wildjailbreak",
        "redbench",
        "harmbench",
    ):
        ctx = EpisodeCreationContext(
            dialogue_idx=dialogue_idx,
            dialogue_data=dialogue_data,
            config=config,
            debug=debug,
            user_agent=user_agent,
            persona=persona,
            available_personas=available_personas,
            patient_agent=patient_agent,
            reward_model=reward_model,
            reward_tokenizer=reward_tokenizer,
            judge_model=judge_model,
            judge_tokenizer=judge_tokenizer,
            multiwoz_mode=multiwoz_mode,
            environment_type=environment_type,
        )
        return create_episode_from_registry(ctx)

    if multiwoz_mode or environment_type in [
        "multiwoz",
        "multiwoz_offline",
        "multiwoz_online",
    ]:
        multiwoz_dialogue = dialogue_data
        ground_truth_goal = format_multiwoz_goal(multiwoz_dialogue.goal)
        dialogue_id = multiwoz_dialogue.dialogue_id
        
        # Select environment based on environment_type
        if environment_type == 'multiwoz_online':
            from .environments.online_env import OnlineEnvironment
            
            # Use provided persona or load random one
            if persona is None:
                if user_agent is None:
                    raise ValueError("user_agent required for multiwoz_online environment")
                # Use pre-loaded personas if available, otherwise load from file
                if available_personas is None:
                    import json
                    # Personas file is at src/prompts/personas.jsonl (not in a subdirectory)
                    persona_file = Path("src/prompts/personas.jsonl")
                    available_personas = []
                    if persona_file.exists():
                        with open(persona_file, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    available_personas.append(json.loads(line))
                if available_personas:
                    persona = random.choice(available_personas)
                else:
                    raise ValueError(f"No personas found")
            
            env = OnlineEnvironment(
                user_agent=user_agent,
                persona=persona,
                goal_json=ensure_goal_json(multiwoz_dialogue.goal),
                max_turns=config.max_turns,
                debug=debug,
                source_dialogue=multiwoz_dialogue  # Pass dialogue to extract desires and constraints
            )
        else:
            # multiwoz_offline or legacy multiwoz: use MultiWOZEnvironment
            env = MultiWOZEnvironment(multiwoz_dialogue, debug=debug)
            
            # Create GoalState from dialogue (similar to online mode)
            # This allows forward-filling of user state for accurate Q scoring
            from .simulation import GoalState, extract_desires_from_dialogue
            goal_state = GoalState()
            desires = extract_desires_from_dialogue(multiwoz_dialogue)
            for desire in desires:
                goal_state.add_desire(desire)
        
        initial_observation, initial_env_info = env.reset()
        
        # Add goal_state to initial_env_info for offline mode
        if not _is_online_mode(config) and 'goal_state' not in initial_env_info:
            initial_env_info['goal_state'] = goal_state
            initial_env_info['goal_json'] = multiwoz_dialogue.goal
        
        episode = EpisodeState(
            dialogue_idx=dialogue_idx,
            dialogue_id=dialogue_id,
            dialogue_data=multiwoz_dialogue,
            ground_truth_goal=ground_truth_goal,
            env=env,
            initial_observation=initial_observation,
            initial_env_info=initial_env_info
        )
        episode.initialize()
        return episode
    else:
        raise NotImplementedError(f"Unsupported environment type: {environment_type}")


def print_periodic_summary(
    episodes_completed: int,
    total_episodes: int,
    episode_stats: Dict,
    start_time: float,
    summary_interval: int = 20,
    config=None,
) -> None:
    """
    Print periodic episode summary (mean loss, reward, etc.) without full evaluation.
    
    Args:
        episodes_completed: Number of episodes completed so far
        total_episodes: Total number of episodes
        episode_stats: Dictionary of episode statistics deques
        start_time: Training start time
        summary_interval: Print summary every N episodes
        config: Optional config for environment-specific metrics (e.g. CARES)
    """
    if episodes_completed % summary_interval != 0 or episodes_completed == 0:
        return
    
    n_episodes = len(episode_stats['total_reward'])
    if n_episodes == 0:
        return
    
    elapsed = time.time() - start_time
    avg_reward = sum(episode_stats['total_reward']) / n_episodes
    avg_length = sum(episode_stats['episode_length']) / n_episodes
    goal_rate = sum(episode_stats['goal_achieved']) / n_episodes * 100
    avg_entropy = sum(episode_stats['final_entropy']) / n_episodes
    avg_vloss = sum(episode_stats['avg_value_loss']) / n_episodes
    avg_q_min_loss = sum(episode_stats['avg_q_min_loss']) / n_episodes
    avg_q_min_target_variance = sum(episode_stats['avg_q_min_target_variance']) / n_episodes
    avg_q_min_prediction_variance = sum(episode_stats['avg_q_min_prediction_variance']) / n_episodes
    max_gpu_memory = max(episode_stats['max_gpu_memory_gb']) if episode_stats['max_gpu_memory_gb'] else 0.0
    lengths = list(episode_stats['episode_length'])
    goals = list(episode_stats['goal_achieved'])
    successful_lengths = [lengths[i] for i in range(n_episodes) if goals[i] >= 1.0]
    avg_length_successful = sum(successful_lengths) / len(successful_lengths) if successful_lengths else None

    print(f"\n[{elapsed:.1f}s] Episode {episodes_completed}/{total_episodes} - Summary (last {n_episodes} episodes):")
    print(f"  Average Reward: {avg_reward:.4f}")
    print(f"  Average Episode Length: {avg_length:.2f} turns")
    if avg_length_successful is not None:
        print(f"  Average Episode Length (successful): {avg_length_successful:.2f} turns ({len(successful_lengths)} episodes)")
    print(f"  Goal Achievement Rate: {goal_rate:.2f}%")
    if config and getattr(config, 'environment_type', None) in ('cares', 'wildjailbreak', 'redbench', 'harmbench'):
        harmful_levels = list(episode_stats.get('harmful_level', []))
        if len(harmful_levels) == n_episodes:
            non_adv_indices = [i for i in range(n_episodes) if harmful_levels[i] == 0]
            adv_indices = [i for i in range(n_episodes) if harmful_levels[i] > 0]
            if non_adv_indices:
                non_adv_goals = sum(1 for i in non_adv_indices if goals[i] >= 0.5) / len(non_adv_indices) * 100
                print(f"  Goal Completion (non-adversarial): {non_adv_goals:.2f}% ({len(non_adv_indices)} episodes)")
            if adv_indices:
                attack_success = sum(1 for i in adv_indices if goals[i] < 0.5) / len(adv_indices) * 100
                print(f"  Attack Success Rate (adversarial): {attack_success:.2f}% ({len(adv_indices)} episodes)")
                # Goal completion is at most (1 - adversarial success)
                effective_goal_cap = 100.0 - attack_success
                print(f"  Effective Goal Achievement (capped by 1 - attack_success): {min(goal_rate, effective_goal_cap):.2f}%")
    print(f"  Average Final Entropy: {avg_entropy:.4f}")
    print(f"  Average Value Loss: {avg_vloss:.4f}")
    if config and getattr(config, 'use_regret_critic', False):
        print(f"  Average Q_min Loss: {avg_q_min_loss:.4f}")
        print(f"  Average Q_min Target Variance (across steps): {avg_q_min_target_variance:.6f}")
        print(f"  Average Q_min Prediction Variance (across steps): {avg_q_min_prediction_variance:.6f}")
    print(f"  Maximum GPU Memory: {max_gpu_memory:.2f} GB")
    sys.stdout.flush()


def batch_generate_beliefs_for_episodes(
    episodes: List[EpisodeState],
    hl_agent: HighLevelAgent,
    config,
    ground_truth_pool: Optional[List[str]] = None,
    model=None,
    tokenizer=None,
    device: Optional[str] = None
) -> None:
    """
    Batch generate beliefs for all active episodes.
    When the HL agent model is local, uses batched protocol (batch_generate with chunk_size; no single-instance generate).

    This function implements belief generation with up to 3 attempts in one call, then the caller
    does Q + actions + step once (one "iteration" = one set of environment steps).
    - If all candidates are [SKIP], retry in the same call (up to 3 total attempts).
    - After 3 attempts, any episode still all-[SKIP] is terminated.
    - No freeze across iterations: retries are exhausted here so we do not run Q+actions+step
      multiple times for a maximally stubborn belief.
    """
    active_episodes = [ep for ep in episodes if ep.is_active and not ep.done_from_env]
    if not active_episodes:
        return
    n_high_level_candidates = config.n_candidates
    high_level_temperature = config.belief_gen_temperature
    high_level_max_new_tokens = config.max_tokens
    freeform_iterative_generation = False
    freeform_per_instruction_max_new_tokens = config.max_tokens
    if config.use_hierarchical_agent and config.high_level_policy_type == "freeform":
        n_high_level_candidates = config.freeform_n_instructions
        high_level_temperature = config.freeform_temperature
        high_level_max_new_tokens = config.freeform_max_new_tokens
        freeform_iterative_generation = config.freeform_iterative_candidate_generation
        freeform_per_instruction_max_new_tokens = config.freeform_per_instruction_max_new_tokens
    
    # Clear frozen flags so all episodes get a belief attempt this iteration
    for episode in active_episodes:
        episode.is_frozen = False
    
    histories = []
    base_prompts = []
    is_adversarial_env = getattr(config, 'environment_type', None) in ('cares', 'wildjailbreak', 'redbench', 'harmbench')
    use_regret_critic = getattr(config, 'use_regret_critic', False)
    for episode in active_episodes:
        history = episode.belief_state.history
        if not history:
            if not episode.current_obs_for_action:
                raise ValueError(
                    f"Episode {episode.dialogue_id} has no dialogue history and no current observation; cannot generate beliefs."
                )
            history = [("[NO_AGENT_ACTION]", episode.current_obs_for_action)]
        histories.append(history.copy())
        # Adversarial: use on-topic intent-varying template; pass base_prompt
        # For regret critic we generate both nominal and adversarial separately below
        if not use_regret_critic and is_adversarial_env and episode.dialogue_data.get('harmful_level', 0) > 0:
            base_prompts.append(episode.dialogue_data.get('base_prompt', ''))
        else:
            base_prompts.append(None)

    # Regret critic: generate BOTH nominal and adversarial candidates per state
    if use_regret_critic and is_adversarial_env:
        base_prompts_nominal = [None] * len(active_episodes)
        base_prompts_adversarial = [ep.dialogue_data.get('base_prompt', '') for ep in active_episodes]

    # Generate prompts for storing debug info (with correct template per episode)
    prompts = [
        hl_agent.prompt_manager.get_high_level_prompt(
            template_name=config.belief_gen_template,
            dialogue_history=history,
            n_candidates=n_high_level_candidates,
            base_prompt=bp,
        )
        for history, bp in zip(histories, base_prompts)
    ]

    MAX_BELIEF_ATTEMPTS = 3  # Total attempts in this call (1 initial + up to 2 retries); then move on to Q+actions+step
    all_candidates = None
    raw_outputs = None
    still_all_skip = list(range(len(active_episodes)))

    for attempt in range(MAX_BELIEF_ATTEMPTS):
        if attempt == 0:
            if use_regret_critic and is_adversarial_env:
                # Generate both nominal (benign) and adversarial candidates
                nominal_candidates, nominal_raw = hl_agent.generate_candidate_beliefs_batch(
                    histories=histories,
                    n_candidates=n_high_level_candidates,
                    temperature=high_level_temperature,
                    max_new_tokens=high_level_max_new_tokens,
                    return_debug_info=True,
                    chunk_size=config.batch_generation_chunk_size,
                    base_prompts=base_prompts_nominal,
                )
                adversarial_candidates, adversarial_raw = hl_agent.generate_candidate_beliefs_batch(
                    histories=histories,
                    n_candidates=n_high_level_candidates,
                    temperature=high_level_temperature,
                    max_new_tokens=high_level_max_new_tokens,
                    return_debug_info=True,
                    chunk_size=config.batch_generation_chunk_size,
                    base_prompts=base_prompts_adversarial,
                )
                # Merge: nominal + adversarial for each episode (up to 2*n_candidates)
                all_candidates = []
                raw_outputs = []
                for ep_idx in range(len(active_episodes)):
                    nom = nominal_candidates[ep_idx]
                    adv = adversarial_candidates[ep_idx]
                    merged = nom + adv
                    uniform_prob = 1.0 / len(merged) if merged else 0.0
                    for c in merged:
                        c.probability = uniform_prob
                    all_candidates.append(merged)
                    raw_outputs.append(nominal_raw[ep_idx] + "\n---\n" + adversarial_raw[ep_idx])
            else:
                if config.use_hierarchical_agent and config.high_level_policy_type == "freeform":
                    freeform_result = hl_agent.generate_instructions_batch(
                        histories=histories,
                        n_instructions=n_high_level_candidates,
                        temperature=high_level_temperature,
                        max_new_tokens=high_level_max_new_tokens,
                        iterative_candidate_generation=freeform_iterative_generation,
                        per_instruction_max_new_tokens=freeform_per_instruction_max_new_tokens,
                        chunk_size=config.batch_generation_chunk_size,
                        base_prompts=base_prompts,
                    )
                    all_candidates = freeform_result["candidates"]
                    raw_outputs = freeform_result["raw_outputs"]
                else:
                    all_candidates, raw_outputs = hl_agent.generate_candidate_beliefs_batch(
                        histories=histories,
                        n_candidates=n_high_level_candidates,
                        temperature=high_level_temperature,
                        max_new_tokens=high_level_max_new_tokens,
                        return_debug_info=True,
                        chunk_size=config.batch_generation_chunk_size,
                        base_prompts=base_prompts,
                    )
        else:
            retry_histories = [histories[i] for i in still_all_skip]
            retry_base_prompts = [base_prompts[i] for i in still_all_skip]
            print(f"[DEBUG] All candidates are [SKIP] for {len(still_all_skip)} episode(s). Attempting batched retry {attempt + 1}/{MAX_BELIEF_ATTEMPTS}...")
            if use_regret_critic and is_adversarial_env:
                retry_nominal, retry_nom_raw = hl_agent.generate_candidate_beliefs_batch(
                    histories=retry_histories,
                    n_candidates=n_high_level_candidates,
                    temperature=high_level_temperature,
                    max_new_tokens=high_level_max_new_tokens,
                    return_debug_info=True,
                    chunk_size=config.batch_generation_chunk_size,
                    base_prompts=[None] * len(still_all_skip),
                )
                retry_adv_prompts = [base_prompts_adversarial[i] for i in still_all_skip]
                retry_adversarial, retry_adv_raw = hl_agent.generate_candidate_beliefs_batch(
                    histories=retry_histories,
                    n_candidates=n_high_level_candidates,
                    temperature=high_level_temperature,
                    max_new_tokens=high_level_max_new_tokens,
                    return_debug_info=True,
                    chunk_size=config.batch_generation_chunk_size,
                    base_prompts=retry_adv_prompts,
                )
                for retry_idx, original_idx in enumerate(still_all_skip):
                    nom = retry_nominal[retry_idx]
                    adv = retry_adversarial[retry_idx]
                    merged = nom + adv
                    uniform_prob = 1.0 / len(merged) if merged else 0.0
                    for c in merged:
                        c.probability = uniform_prob
                    all_candidates[original_idx] = merged
                    raw_outputs[original_idx] = retry_nom_raw[retry_idx] + "\n---\n" + retry_adv_raw[retry_idx]
            else:
                if config.use_hierarchical_agent and config.high_level_policy_type == "freeform":
                    retry_result = hl_agent.generate_instructions_batch(
                        histories=retry_histories,
                        n_instructions=n_high_level_candidates,
                        temperature=high_level_temperature,
                        max_new_tokens=high_level_max_new_tokens,
                        iterative_candidate_generation=freeform_iterative_generation,
                        per_instruction_max_new_tokens=freeform_per_instruction_max_new_tokens,
                        chunk_size=config.batch_generation_chunk_size,
                        base_prompts=retry_base_prompts,
                    )
                    retry_candidates = retry_result["candidates"]
                    retry_raw_outputs = retry_result["raw_outputs"]
                else:
                    retry_candidates, retry_raw_outputs = hl_agent.generate_candidate_beliefs_batch(
                        histories=retry_histories,
                        n_candidates=n_high_level_candidates,
                        temperature=high_level_temperature,
                        max_new_tokens=high_level_max_new_tokens,
                        return_debug_info=True,
                        chunk_size=config.batch_generation_chunk_size,
                        base_prompts=retry_base_prompts,
                    )
                for retry_idx, original_idx in enumerate(still_all_skip):
                    all_candidates[original_idx] = retry_candidates[retry_idx]
                    raw_outputs[original_idx] = retry_raw_outputs[retry_idx]
        
        still_all_skip = [ep_idx for ep_idx in range(len(active_episodes)) if all(c.summary == "[SKIP]" for c in all_candidates[ep_idx])]
        if not still_all_skip:
            if attempt > 0:
                print(f"[WARNING] Retry succeeded: All episode(s) that had all-[SKIP] candidates now have valid candidates after regeneration.")
            break
        if attempt < MAX_BELIEF_ATTEMPTS - 1:
            continue
        # Exhausted retries - terminate any still all-[SKIP]
        for original_idx in still_all_skip:
            episode = active_episodes[original_idx]
            episode.is_frozen = False
            episode.is_active = False
            episode.done_from_env = True
            print(f"[WARNING] Episode {episode.dialogue_idx} terminated after {MAX_BELIEF_ATTEMPTS} failed belief generation attempts.")
        print(f"[WARNING] {len(still_all_skip)} episode(s) terminated due to max belief retries: {[active_episodes[i].dialogue_idx for i in still_all_skip]}")
    
    # Batch sample crossed data beliefs for all episodes that need them (performance optimization)
    ablation_mode = getattr(config, 'contrastive_ablation_mode', 'none')
    
    # OPTIMIZED ALGORITHM for crossed_data_in_candidates (both online and offline):
    # 1) Generate candidates for all states (already done - all_candidates)
    # 2) Add 3 candidates per state to a pool, keep remaining 2 per state
    # 3) Without replacement, randomly sample 3 candidates from the pool back into each state
    # 4) Each state ends up with 2 (kept) + 3 (redistributed) = 5 candidates
    if ablation_mode == "crossed_data_in_candidates":
        import random
        
        candidate_pool = []
        n_to_pool = 3
        kept_candidates_per_state = []
        
        for ep_idx, candidates in enumerate(all_candidates):
            valid_candidates = [c for c in candidates if c.summary != "[SKIP]"]
            
            if len(valid_candidates) == 0:
                kept_candidates_per_state.append([])
            elif len(valid_candidates) == 1:
                kept_candidates_per_state.append(valid_candidates)
            elif len(valid_candidates) == 2:
                kept = [valid_candidates[0]]
                candidate_pool.append(valid_candidates[1])
                kept_candidates_per_state.append(kept)
            else:
                n_to_add = min(n_to_pool, len(valid_candidates) - 1)
                sampled_for_pool = random.sample(valid_candidates, n_to_add)
                candidate_pool.extend(sampled_for_pool)
                kept = [c for c in valid_candidates if c not in sampled_for_pool]
                kept_candidates_per_state.append(kept)
        
        if not candidate_pool:
            raise RuntimeError(
                f"crossed_data_in_candidates: Pool is empty (all candidates are [SKIP] after retry). "
                f"This indicates belief generation failed for all episodes. Program cannot continue."
            )
        
        random.shuffle(candidate_pool)
        
        n_states = len(all_candidates)
        min_needed_from_pool = n_states
        total_target_from_pool = n_states * n_to_pool
        
        if len(candidate_pool) < min_needed_from_pool:
            raise RuntimeError(
                f"crossed_data_in_candidates: Pool size ({len(candidate_pool)}) < minimum needed ({min_needed_from_pool}). "
                f"Cannot guarantee at least 1 redistributed candidate per state. Program cannot continue."
            )
        
        if len(candidate_pool) < total_target_from_pool:
            print(f"[WARNING] crossed_data_in_candidates: Pool size ({len(candidate_pool)}) < target ({total_target_from_pool}). "
                  f"Some states will receive fewer than {n_to_pool} redistributed candidates.")
        
        pool_idx = 0
        redistributed_per_state = []
        
        for ep_idx in range(n_states):
            if pool_idx < len(candidate_pool):
                redistributed_per_state.append([candidate_pool[pool_idx]])
                pool_idx += 1
            else:
                redistributed_per_state.append([])
        
        for ep_idx in range(n_states):
            current_redistributed = redistributed_per_state[ep_idx]
            for _ in range(n_to_pool - len(current_redistributed)):
                if pool_idx < len(candidate_pool):
                    current_redistributed.append(candidate_pool[pool_idx])
                    pool_idx += 1
                else:
                    break
        
        # Combine kept + redistributed for each state
        for ep_idx, original_candidates in enumerate(all_candidates):
            kept = kept_candidates_per_state[ep_idx]
            redistributed = redistributed_per_state[ep_idx]
            
            # Combine kept + redistributed candidates
            n_expected = len(original_candidates)  # Should be config.n_candidates (5)
            new_candidates = kept + redistributed
            
            # Verify guarantees: each state should have at least 1 kept + 1 redistributed
            n_kept = len(kept)
            n_redistributed = len(redistributed)
            if n_kept == 0 and n_redistributed == 0:
                # This should not happen - all candidates were [SKIP] and pool was empty
                raise RuntimeError(
                    f"crossed_data_in_candidates: Episode {active_episodes[ep_idx].dialogue_idx} has no kept or redistributed candidates. "
                    f"This should have been caught earlier. Program cannot continue."
                )
            
            # Pad to expected length if needed
            while len(new_candidates) < n_expected:
                new_candidates.append(BeliefCandidate(
                    summary="[SKIP]",
                    context="[SKIP]",
                    probability=0.0
                ))
            
            # Truncate if more than expected (shouldn't happen)
            if len(new_candidates) > n_expected:
                new_candidates = new_candidates[:n_expected]
            
            # Replace the candidates for this episode
            all_candidates[ep_idx] = new_candidates
        
        # Final check: Ensure no episode has all [SKIP] candidates after pooling
        # This can happen if:
        # 1. An episode had < 3 valid candidates initially (so all went to pool, kept 0)
        # 2. Pool was exhausted before reaching this episode during redistribution
        # 3. Episode got 0 redistributed candidates and was padded with [SKIP]
        all_skip_episodes = []
        for ep_idx, candidates in enumerate(all_candidates):
            if all(c.summary == "[SKIP]" for c in candidates):
                all_skip_episodes.append((ep_idx, active_episodes[ep_idx].dialogue_idx))
        
        if all_skip_episodes:
            failed_episode_ids = [ep_id for _, ep_id in all_skip_episodes]
            raise RuntimeError(
                f"crossed_data_in_candidates: {len(all_skip_episodes)} episode(s) have all-[SKIP] candidates after pooling. "
                f"This occurred because the pool was exhausted before redistributing to these episodes. "
                f"Failed episode indices: {failed_episode_ids}. "
                f"Pool size was {len(candidate_pool)} but needed {n_states * n_to_pool} total. "
                f"Program cannot continue."
            )
    
    # Store candidates and debug info in episodes
    for ep_idx, (episode, candidates, raw_output, prompt, history) in enumerate(zip(active_episodes, all_candidates, raw_outputs, prompts, histories)):
        # Ablation modes: Include noise or crossed data as one of the belief candidates
        # This allows the Q-function to learn to avoid bad beliefs naturally without contrastive loss
        if ablation_mode == "noise_in_candidates":
            # Only add noise if we have at least one valid (non-[SKIP]) candidate
            # If all candidates are [SKIP], that indicates a parsing failure, so skip noise replacement
            valid_indices = [i for i, c in enumerate(candidates) if c.summary != "[SKIP]"]
            if valid_indices:
                # Replace one candidate with random noise (randomly select which one)
                import random
                noise_candidate = generate_random_noise_beliefs(1)[0]
                replace_idx = random.choice(valid_indices)
                
                # Create a BeliefCandidate for the noise
                noise_belief_candidate = BeliefCandidate(
                    summary=noise_candidate,
                    context=f"Belief: {noise_candidate}",
                    probability=1.0 / len(candidates)  # Uniform probability
                )
                candidates[replace_idx] = noise_belief_candidate
        # Note: crossed_data_in_candidates is now handled above with the pooling algorithm
        
        episode.belief_state.candidates = candidates
        
        # Record belief generations for environment-level debugging if supported
        candidate_summaries = [c.summary for c in candidates]
        recorder = getattr(episode.env, "record_belief_generation", None)
        if callable(recorder):
            try:
                recorder(candidate_summaries)
            except Exception as exc:
                print(f"[WARN] Failed to record belief generation for episode {episode.dialogue_idx}: {exc}")
        
        # Store HL agent input for debugging (only for online mode)
        if hasattr(episode.env, 'env_state') and hasattr(episode.env.env_state, 'hl_agent_inputs'):
            episode.env.env_state.hl_agent_inputs.append(prompt)
            if hasattr(episode.env.env_state, 'hl_agent_outputs'):
                episode.env.env_state.hl_agent_outputs.append(raw_output)
        
        # Check if all candidates are [SKIP] and store debug info
        valid_count = sum(1 for c in candidates if c.summary != "[SKIP]")
        if valid_count == 0:
            episode.belief_gen_debug_prompt = prompt
            episode.belief_gen_debug_raw_output = raw_output
            episode.belief_gen_debug_history = history.copy()


def batch_compute_q_values_for_episodes(
    episodes: List[EpisodeState],
    value_function: ValueFunction,
    tokenizer,
    config
) -> None:
    """Batch compute Q-values for all active episodes and store in episode state.
    Critic is always local; uses chunked batch protocol (no single-instance forward)."""
    active_episodes = [ep for ep in episodes if ep.is_active and not ep.done_from_env and not ep.is_frozen]
    if not active_episodes:
        return
    
    # Collect all (observation, candidate) pairs
    observations = []
    all_candidates = []
    episode_indices = []
    candidate_indices = []  # Track which candidate within each episode
    
    for ep_idx, episode in enumerate(active_episodes):
        if not episode.belief_state.candidates or not episode.current_obs_for_action:
            # Debug: Log why episode was skipped
            if not episode.belief_state.candidates:
                print(f"[DEBUG] batch_compute_q_values: Skipping episode {episode.dialogue_idx} - no candidates")
            if not episode.current_obs_for_action:
                print(f"[DEBUG] batch_compute_q_values: Skipping episode {episode.dialogue_idx} - no observation")
            continue
        
        # Check if observation contains only placeholder text (meaningless for Q function)
        # If observation is empty or contains only placeholders like [NO_AGENT_ACTION] and [NO_USER_RESPONSE],
        # skip Q-value computation to prevent learning incorrect patterns
        obs_text = episode.current_obs_for_action.strip()
        if not obs_text:
            print(f"[DEBUG] batch_compute_q_values: Skipping episode {episode.dialogue_idx} - empty observation text")
            continue
        
        # Check if observation is essentially empty (only placeholders and formatting)
        # Remove common placeholders and formatting, check if anything meaningful remains
        cleaned_obs = obs_text
        for placeholder in ["[NO_AGENT_ACTION]", "[NO_USER_RESPONSE]", "Turn ", "Agent:", "User:"]:
            cleaned_obs = cleaned_obs.replace(placeholder, "")
        cleaned_obs = cleaned_obs.replace("\n", " ").strip()
        
        # If after removing placeholders there's no meaningful content, skip
        if not cleaned_obs or len(cleaned_obs) < 5:
            print(f"[DEBUG] batch_compute_q_values: Skipping episode {episode.dialogue_idx} - observation too short after cleaning (len={len(cleaned_obs)}): {repr(cleaned_obs[:100])}")
            continue
        
        candidates = [c.summary for c in episode.belief_state.candidates]
        for cand_idx, candidate in enumerate(candidates):
            observations.append(episode.current_obs_for_action)
            all_candidates.append(candidate)
            episode_indices.append(ep_idx)
            candidate_indices.append(cand_idx)
    
    if not observations:
        return
    
    # Separate [SKIP] candidates from valid ones for batch computation
    skip_indices = [i for i, cand in enumerate(all_candidates) if cand == "[SKIP]"]
    valid_indices = [i for i, cand in enumerate(all_candidates) if cand != "[SKIP]"]
    
    # Batch compute Q-values only for valid candidates (in chunks to prevent OOM)
    if valid_indices:
        valid_observations = [observations[i] for i in valid_indices]
        valid_candidates = [all_candidates[i] for i in valid_indices]
        
        # Process in chunks to prevent OOM (chunked batching protocol: same effective chunk as generation)
        chunk_size = _model_batch_chunk_size(config)
        num_chunks = (len(valid_observations) + chunk_size - 1) // chunk_size
        q_values_chunks = []
        q_min_chunks = [] if (getattr(config, 'use_regret_critic', False) and value_function.use_regret_critic) else None
        regret_chunks = [] if (getattr(config, 'use_regret_critic', False) and value_function.use_regret_critic) else None

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(valid_observations))

            chunk_observations = valid_observations[start_idx:end_idx]
            chunk_candidates = valid_candidates[start_idx:end_idx]

            chunk_q_values = value_function.predict_q_value(
                observations=chunk_observations,
                high_level_actions=chunk_candidates,
                tokenizer=tokenizer,
                requires_grad=False  # Inference: no gradients needed
            )
            q_values_chunks.append(chunk_q_values.detach().cpu())

            # Regret critic: also compute Q_min and Regret for selection
            if q_min_chunks is not None:
                chunk_q_min = value_function.predict_q_min_value(
                    observations=chunk_observations,
                    high_level_actions=chunk_candidates,
                    tokenizer=tokenizer,
                    requires_grad=False
                )
                chunk_regret = value_function.predict_regret_value(
                    observations=chunk_observations,
                    high_level_actions=chunk_candidates,
                    tokenizer=tokenizer,
                    requires_grad=False
                )
                q_min_chunks.append(chunk_q_min.detach().cpu())
                regret_chunks.append(chunk_regret.detach().cpu())

            # Aggressive cleanup after each chunk
            del chunk_observations, chunk_candidates, chunk_q_values
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # In evaluation mode, synchronize to ensure memory is actually freed
                if getattr(config, '_evaluation_mode', False):
                    torch.cuda.synchronize()
        
        # Concatenate all chunks (move back to GPU)
        q_min_batch_valid = None
        regret_batch_valid = None
        if q_values_chunks:
            q_values_batch_valid = torch.cat(q_values_chunks, dim=0).to(device=value_function.device)
            del q_values_chunks
            if q_min_chunks is not None:
                q_min_batch_valid = torch.cat(q_min_chunks, dim=0).to(device=value_function.device)
                regret_batch_valid = torch.cat(regret_chunks, dim=0).to(device=value_function.device)
                del q_min_chunks, regret_chunks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if getattr(config, '_evaluation_mode', False):
                    torch.cuda.synchronize()
        else:
            q_values_batch_valid = torch.tensor([], device=value_function.device)
    else:
        q_values_batch_valid = torch.tensor([], device=value_function.device)
        q_min_batch_valid = None
        regret_batch_valid = None

    # Create full Q-values tensor with [SKIP] candidates assigned 0.0
    q_values_batch = torch.zeros(len(all_candidates), device=value_function.device)
    q_min_batch = torch.zeros(len(all_candidates), device=value_function.device) if q_min_batch_valid is not None else None
    regret_batch = torch.zeros(len(all_candidates), device=value_function.device) if regret_batch_valid is not None else None
    for valid_idx, original_idx in enumerate(valid_indices):
        q_val = q_values_batch_valid[valid_idx]
        q_values_batch[original_idx] = q_val.item() if hasattr(q_val, 'item') else float(q_val)
        if q_min_batch is not None:
            q_min_batch[original_idx] = q_min_batch_valid[valid_idx].item() if hasattr(q_min_batch_valid[valid_idx], 'item') else float(q_min_batch_valid[valid_idx])
            regret_batch[original_idx] = regret_batch_valid[valid_idx].item() if hasattr(regret_batch_valid[valid_idx], 'item') else float(regret_batch_valid[valid_idx])
    # [SKIP] candidates already have 0.0 from initialization
    
    # Distribute Q-values back to episodes
    # Store as dict mapping candidate -> q_value for each episode
    for ep_idx in range(len(active_episodes)):
        episode = active_episodes[ep_idx]
        if not episode.belief_state.candidates:
            continue
        
        q_values = {}
        # Find all Q-values for this episode
        episode_has_q_values = False
        for i, obs_ep_idx in enumerate(episode_indices):
            if obs_ep_idx == ep_idx:
                episode_has_q_values = True
                candidate = episode.belief_state.candidates[candidate_indices[i]].summary
                q_val = q_values_batch[i]
                # [SKIP] candidates already have 0.0, but explicitly set it
                if candidate == "[SKIP]":
                    q_values[candidate] = 0.0
                else:
                    q_values[candidate] = q_val.item() if hasattr(q_val, 'item') else float(q_val)
        
        # If episode was skipped (observation too short/empty), raise error immediately
        # This helps identify and fix the root cause rather than masking it
        if not episode_has_q_values:
            obs = episode.current_obs_for_action
            obs_text = obs.strip() if obs else ""
            cleaned_obs = obs_text
            for placeholder in ["[NO_AGENT_ACTION]", "[NO_USER_RESPONSE]", "Turn ", "Agent:", "User:"]:
                cleaned_obs = cleaned_obs.replace(placeholder, "")
            cleaned_obs = cleaned_obs.replace("\n", " ").strip()
            
            raise RuntimeError(
                f"Episode {episode.dialogue_idx} (turn {episode.turn}) was skipped in Q-value computation. "
                f"This prevents belief selection and must be fixed.\n"
                f"  Original observation: {repr(obs[:200])}\n"
                f"  Observation after cleaning: {repr(cleaned_obs[:200])}\n"
                f"  Cleaned length: {len(cleaned_obs)}\n"
                f"  Has candidates: {episode.belief_state.candidates is not None}\n"
                f"  Number of candidates: {len(episode.belief_state.candidates) if episode.belief_state.candidates else 0}\n"
                f"  This usually means the observation contains only placeholders with no meaningful content. "
                f"Check why this episode has an invalid observation."
            )
        
        # Store Q-values in episode (we'll use a temporary attribute)
        episode._q_values = q_values
        if q_min_batch is not None and regret_batch is not None:
            q_min_values = {}
            regret_values = {}
            for i, obs_ep_idx in enumerate(episode_indices):
                if obs_ep_idx == ep_idx:
                    candidate = episode.belief_state.candidates[candidate_indices[i]].summary
                    q_min_values[candidate] = q_min_batch[i].item() if hasattr(q_min_batch[i], 'item') else float(q_min_batch[i])
                    regret_values[candidate] = regret_batch[i].item() if hasattr(regret_batch[i], 'item') else float(regret_batch[i])
            episode._q_min_values = q_min_values
            episode._regret_values = regret_values


def _batch_compute_log_probs_for_entries(
    entries: List[Dict],
    model,
    tokenizer,
    chunk_size: Optional[int] = None
) -> List[List[float]]:
    """
    Compute log-probabilities for multiple (episode, selected_belief, observation) tuples
    in a single batched call.
    """
    if not entries:
        return []

    contexts: List[str] = []
    targets: List[str] = []
    metadata: List[tuple] = []
    per_entry_log_probs: List[List[float]] = []

    for entry_idx, entry in enumerate(entries):
        episode = entry["episode"]
        observation = entry.get("observation", "")
        candidates = episode.belief_state.candidates or []

        if not candidates or not observation:
            per_entry_log_probs.append([])
            continue

        per_entry_log_probs.append([float("-inf")] * len(candidates))
        action = entry["selected_belief"]

        for cand_idx, belief in enumerate(candidates):
            context = f"{belief.context}\nAgent: {action}\nUser:"
            contexts.append(context)
            targets.append(observation)
            metadata.append((entry_idx, cand_idx))

    if not contexts:
        return per_entry_log_probs

    total = len(contexts)
    if not chunk_size or chunk_size <= 0:
        chunk_size = total

    meta_ptr = 0
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk_contexts = contexts[start:end]
        chunk_targets = targets[start:end]
        chunk_log_probs = compute_log_prob_batch(
            model=model,
            tokenizer=tokenizer,
            contexts=chunk_contexts,
            targets=chunk_targets
        )

        for log_prob in chunk_log_probs:
            entry_idx, cand_idx = metadata[meta_ptr]
            per_entry_log_probs[entry_idx][cand_idx] = log_prob
            belief = entries[entry_idx]["episode"].belief_state.candidates[cand_idx]
            belief.log_prob = log_prob
            meta_ptr += 1

    return per_entry_log_probs


def _finalize_episodes_with_batched_dpo(
    entries: List[Dict],
    model,
    tokenizer,
    config,
    multiwoz_mode: bool,
    episodes_to_replace: List[EpisodeState]
) -> None:
    """
    Apply DPO-based belief updates and associated bookkeeping for all provided entries
    using a batched log-probability computation.
    """
    if not entries:
        return

    log_probs_per_entry = _batch_compute_log_probs_for_entries(
        entries=entries,
        model=model,
        tokenizer=tokenizer,
        chunk_size=getattr(config, "batch_generation_chunk_size", None)
    )

    for entry, log_probs in zip(entries, log_probs_per_entry):
        episode = entry["episode"]

        if log_probs:
            episode.belief_state = update_belief_distribution_dpo(
                belief_state=episode.belief_state,
                log_probs=log_probs,
                temperature=config.dpo_temperature
            )

        probabilities = [c.probability for c in episode.belief_state.candidates] if episode.belief_state.candidates else []
        entropy = compute_entropy(probabilities) if probabilities else 0.0
        episode.context_entropies.append(entropy)

        episode.reward = episode.reward_from_env
        episode.dialogue_state.rewards.append(episode.reward)
        episode.dialogue_state.goal_achieved = max(episode.goal_achieved, episode.dialogue_state.goal_achieved)
        episode.episode_total_reward += episode.reward

        episode.turn += 1
        episode.actual_turns_processed += 1

        should_continue = not (
            (
                multiwoz_mode
                and (
                    episode.done_from_env
                    or (not _is_online_mode(config) and episode.turn > 0 and not episode.user_actions)
                )
            )
            or episode.goal_achieved >= 1.0
        )

        if not should_continue:
            episodes_to_replace.append(episode)


def build_turn_evaluation_entry(
    episode: EpisodeState,
    observation: str,
    ll_action: str,
    candidate_beliefs: List[str],
    q_values: Dict[str, float],
    selected_belief: str,
    multiwoz_mode: bool,
    training_mode: Optional[str] = None
) -> Dict:
    """Create a structured turn evaluation entry with baseline comparisons."""
    # Strict validation: every user turn should have a corresponding agent response
    if not isinstance(ll_action, str):
        raise ValueError(
            f"ll_action must be a string, got {type(ll_action)}: {repr(ll_action)}. "
            f"Episode: {episode.dialogue_id}, Turn: {episode.turn}"
        )
    if not ll_action.strip():
        # In offline mode, some turns may legitimately have empty agent responses from the dataset
        # Skip building evaluation entry for these turns
        if training_mode == "offline":
            return {
                'episode_id': episode.dialogue_id,
                'turn': episode.turn,
                'll_action': '',  # Empty action
                'skipped': True,
                'reason': 'empty_agent_response_in_dataset'
            }
        else:
            # In online mode, agent responses should never be empty (they're generated)
            raise ValueError(
                f"ll_action is empty in online mode. Every user turn should have a corresponding agent response. "
                f"Episode: {episode.dialogue_id}, Turn: {episode.turn}, "
                f"agent_response: {repr(episode.agent_response)}"
            )
    valid_candidates = [cand for cand in candidate_beliefs if cand != "[SKIP]"]
    valid_q_values = {
        cand: value
        for cand, value in (q_values or {}).items()
        if cand != "[SKIP]" and value is not None
    }
    top_probability_belief = None
    if episode.belief_state.candidates:
        try:
            top_probability_belief = episode.belief_state.get_top_belief().summary
        except ValueError:
            top_probability_belief = None
    
    random_baseline_belief = None
    if valid_candidates:
        random_baseline_belief = random.choice(valid_candidates)
    
    # For min_q_belief: filter out noise candidates to get a fair comparison
    # Noise candidates have artificially low Q-values (Q-function learned to avoid them),
    # so including them would make min-Q accuracy misleadingly low
    min_q_belief = None
    if valid_q_values:
        # Filter out noise candidates for min-Q computation
        non_noise_q_values = {
            cand: value
            for cand, value in valid_q_values.items()
            if not is_noise_candidate(cand)
        }
        if non_noise_q_values:
            min_q_belief = min(non_noise_q_values, key=non_noise_q_values.get)
        else:
            # Fallback: if all candidates are noise, use the minimum anyway
            min_q_belief = min(valid_q_values, key=valid_q_values.get)
    
    ground_truth_goal = episode.ground_truth_goal if multiwoz_mode else None
    baseline_metrics = None
    if (
        multiwoz_mode
        and ground_truth_goal
        and isinstance(ground_truth_goal, str)
        and ground_truth_goal.strip()
        and ground_truth_goal.lower() != "user goal not specified"
        and selected_belief
    ):
        baseline_metrics = {
            "max_q_accuracy": compute_belief_accuracy(selected_belief, ground_truth_goal),
        }
        if min_q_belief:
            baseline_metrics["min_q_accuracy"] = compute_belief_accuracy(
                min_q_belief,
                ground_truth_goal
            )
        if top_probability_belief:
            baseline_metrics["top_prob_accuracy"] = compute_belief_accuracy(
                top_probability_belief,
                ground_truth_goal
            )
        if random_baseline_belief:
            baseline_metrics["random_accuracy"] = compute_belief_accuracy(
                random_baseline_belief,
                ground_truth_goal
            )

    candidate_pool_q_metrics = {}
    if valid_q_values:
        candidate_pool_q_metrics = _compute_candidate_pool_q_metrics(
            episode=episode,
            candidate_beliefs=candidate_beliefs,
            q_values=valid_q_values,
        )
    
    return {
        'observation': observation,
        'll_action': ll_action,
        'test_time_agent_tokens': getattr(episode, "_test_time_agent_tokens", None),
        'candidate_beliefs': candidate_beliefs,
        'q_values': dict(q_values) if q_values else {},
        'selected_belief': selected_belief,
        'top_probability_belief': top_probability_belief,
        'random_baseline_belief': random_baseline_belief,
        'min_q_baseline_belief': min_q_belief,
        'baseline_metrics': baseline_metrics,
        'ground_truth_goal': ground_truth_goal,
        'candidate_pool_q_metrics': candidate_pool_q_metrics,
    }


def batch_generate_ll_actions_for_episodes(
    episodes: List[EpisodeState],
    ll_agent: LowLevelAgent,
    config
) -> None:
    """
    Batch low-level action generation for online mode.
    Precomputes selected beliefs and LL actions to avoid per-episode forwards.
    """
    if not _is_online_mode(config):
        return
    
    if config.use_hierarchical_agent:
        HierarchicalRolloutCoordinator(
            softmax_selector=_softmax_sample_from_q_values,
            regret_selector=_regret_critic_selection,
            template_name_selector=get_ll_action_template_name,
            noise_filter=is_noise_candidate,
        ).precompute_low_level_actions(episodes=episodes, ll_agent=ll_agent, config=config)
        return

    # Episodes eligible for LL generation this turn (turn > 0, active, not done, not frozen)
    # Frozen episodes should not generate LL actions - they will be retried in the next batch
    eligible = [
        ep for ep in episodes
        if ep.is_active and not ep.done_from_env and ep.turn > 0 and not ep.is_frozen
    ]
    if not eligible:
        return
    
    # Select beliefs using already-computed Q-values
    belief_contexts = []
    histories = []
    target_episodes = []
    chunk_size = getattr(config, "batch_generation_chunk_size", None)
    epsilon_to_use = getattr(config, "_current_epsilon", config.epsilon)
    ablation_mode = getattr(config, 'contrastive_ablation_mode', 'none')
    filter_noise = (ablation_mode == "noise_in_candidates")
    
    for episode in eligible:
        q_values = getattr(episode, '_q_values', {})
        high_level_candidates = [c.summary for c in episode.belief_state.candidates]
        
        valid_candidates = _filter_candidates_for_value_min(high_level_candidates, filter_noise)
        
        valid_q_values = {k: v for k, v in q_values.items() if k != "[SKIP]"}
        if filter_noise:
            valid_q_values = {k: v for k, v in valid_q_values.items() if not is_noise_candidate(k)}
        
        if not valid_candidates:
            # Skip batching for this episode; it will error in process_turn as before
            continue
        
        # Check if random_belief_selection is enabled
        random_belief_selection = getattr(config, 'random_belief_selection', False)
        use_regret_critic = getattr(config, 'use_regret_critic', False)
        regret_values = getattr(episode, '_regret_values', None)
        if random_belief_selection:
            selected_belief = random.choice(valid_candidates)
        elif not valid_q_values:
            selected_belief = random.choice(valid_candidates)
        elif random.random() < epsilon_to_use:
            selected_belief = random.choice(valid_candidates)
        elif use_regret_critic and regret_values and all(k in regret_values for k in valid_q_values):
            beta = getattr(config, 'regret_critic_beta', 0.2)
            selected_belief = _regret_critic_selection(valid_q_values, regret_values, beta)
        else:
            selected_belief = _softmax_sample_from_q_values(valid_q_values)
        
        ll_history = episode.belief_state.history
        if not ll_history:
            continue
        
        belief_contexts.append(selected_belief)
        histories.append(ll_history)
        target_episodes.append(episode)
        # Cache selected belief; process_turn will consume it and append to chosen_beliefs_per_turn
        episode._preselected_belief = selected_belief
    
    if not target_episodes:
        return
    
    baseline_mode = getattr(config, 'baseline_mode', False)
    first_episode = target_episodes[0] if target_episodes else None
    template_name = get_ll_action_template_name(config, baseline_mode=baseline_mode, episode=first_episode)

    actions = ll_agent.generate_action_batch(
        belief_contexts=belief_contexts,
        histories=histories,
        temperature=0.7,
        belief_only=getattr(config, 'll_action_belief_only', True) if not baseline_mode else False,
        chunk_size=chunk_size,
        template_name=template_name
    )
    
    for ep, action in zip(target_episodes, actions):
        ep._precomputed_agent_response = action
        if hasattr(ep.env, 'env_state') and hasattr(ep.env.env_state, 'll_agent_outputs'):
            ep.env.env_state.ll_agent_outputs.append(action)


def batch_user_model_interactions(
    episodes: List[EpisodeState],
    user_agent: UserAgent,
    config
) -> None:
    """
    Batch user-agent LLM calls (goal satisfaction judge + user response generation)
    for all active online episodes that have an agent_response ready.
    Carefully maintain ordering to map outputs back to originating episodes.
    """
    # Eligible episodes: active, not done, have agent_response, not frozen
    # Frozen episodes should not generate user responses - they will be retried in the next batch
    eligible: List[EpisodeState] = [
        ep for ep in episodes
        if ep.is_active and not ep.done_from_env and not ep.is_frozen and getattr(ep, "agent_response", None) is not None
    ]
    if not eligible:
        return
    
    chunk_size = getattr(config, "batch_generation_chunk_size", None)
    
    # 1) Prepare judge inputs; also perform slot/service-call extraction and goal_state update pre-judge
    judge_histories: List[List[Tuple[str, str]]] = []
    judge_goal_jsons: List[Dict] = []
    judge_agent_responses: List[str] = []
    ep_indices_for_judge: List[int] = []
    prev_fractions: Dict[int, float] = {}
    
    for idx, ep in enumerate(eligible):
        env: OnlineEnvironment = ep.env
        agent_action = ep.agent_response
        
        # Update dialogue history to include current agent action (user response pending)
        if env.env_state.dialogue_history:
            last_agent_action, last_user_response = env.env_state.dialogue_history[-1]
            if last_agent_action and not last_user_response:
                # Already pending; should not happen, but safeguard
                env.env_state.dialogue_history[-1] = (agent_action, "")
            else:
                env.env_state.dialogue_history.append((agent_action, ""))
        else:
            env.env_state.dialogue_history.append((agent_action, ""))
        
        # Capture prev_fraction BEFORE updating goal state, so we can detect progress
        prev_fraction = env.goal_state.get_goal_achievement_fraction() if env.goal_state.desires else 0.0
        prev_fractions[idx] = prev_fraction
        
        # Extract slots / service call and update goal state (mirrors OnlineEnvironment.step)
        domain = None
        if env.goal_state.desires and env.goal_state.current_desire_idx < len(env.goal_state.desires):
            current_desire = env.goal_state.desires[env.goal_state.current_desire_idx]
            domain = current_desire.domain
        elif env.goal_json:
            domains = list(env.goal_json.keys())
            if domains:
                domain = domains[0]
        if domain:
            try:
                extracted_slots = extract_slots_from_agent_response(
                    agent_text=agent_action,
                    domain=domain,
                    goal_json=env.goal_json
                )
                service_call = extract_service_call_from_agent_response(
                    agent_text=agent_action,
                    domain=domain
                )
                env.goal_state.update_from_agent_response(
                    agent_text=agent_action,
                    extracted_slots=extracted_slots,
                    service_call=service_call,
                    goal_json=env.goal_json
                )
            except Exception as e:
                # Fail fast per research rules
                raise
        
        judge_histories.append(env.env_state.dialogue_history.copy())
        judge_goal_jsons.append(env.goal_json)
        judge_agent_responses.append(agent_action)
        ep_indices_for_judge.append(idx)
    
    # 2) Batch judge goal satisfaction
    judge_results = user_agent.judge_goal_satisfaction_batch(
        dialogue_histories=judge_histories,
        goal_jsons=judge_goal_jsons,
        agent_responses=judge_agent_responses,
        temperature=0.0,
        chunk_size=chunk_size
    )
    if len(judge_results) != len(ep_indices_for_judge):
        raise RuntimeError(
            f"judge_goal_satisfaction_batch returned {len(judge_results)} results for "
            f"{len(ep_indices_for_judge)} inputs"
        )
    
    # Apply judge results to goal_state (update desire satisfaction)
    for idx, goal_satisfied in zip(ep_indices_for_judge, judge_results):
        ep = eligible[idx]
        env: OnlineEnvironment = ep.env
        if goal_satisfied and env.goal_state.desires and env.goal_state.current_desire_idx < len(env.goal_state.desires):
            current_desire = env.goal_state.desires[env.goal_state.current_desire_idx]
            if current_desire.status == "pending":
                current_desire.status = "satisfied"
                if env.goal_state.current_desire_idx + 1 < len(env.goal_state.desires):
                    env.goal_state.current_desire_idx += 1
    
    # 3) Prepare user response generation inputs
    personas: List[Dict] = []
    histories_no_pending: List[List[Tuple[str, str]]] = []
    agent_actions_for_resp: List[str] = []
    goal_jsons_for_resp: List[Dict] = []
    goal_progress_summaries: List[str] = []
    ep_indices_for_resp: List[int] = []
    for idx, ep in enumerate(eligible):
        env: OnlineEnvironment = ep.env
        # History passed to generate_user_response in _generate_user_response is dialogue_history[:-1]
        histories_no_pending.append(env.env_state.dialogue_history[:-1])
        agent_actions_for_resp.append(ep.agent_response)
        personas.append(env.persona)
        goal_jsons_for_resp.append(env.goal_json)
        goal_progress_summaries.append(env.goal_state.build_progress_summary(env.goal_json))
        ep_indices_for_resp.append(idx)
    
    # Generate and store user agent input prompts for debugging
    for idx, ep in enumerate(eligible):
        env: OnlineEnvironment = ep.env
        prompt = user_agent.prompt_manager.get_user_prompt(
            template_name="response_generation",
            persona=env.persona,
            dialogue_history=histories_no_pending[idx],
            agent_action=agent_actions_for_resp[idx],
            goal_json=goal_jsons_for_resp[idx],
            goal_progress_summary=goal_progress_summaries[idx],
            model=user_agent.model,
            tokenizer=user_agent.tokenizer
        )
        if hasattr(env.env_state, 'user_agent_inputs'):
            env.env_state.user_agent_inputs.append(prompt)
    
    user_responses = user_agent.generate_user_response_batch(
        personas=personas,
        dialogue_histories=histories_no_pending,
        agent_actions=agent_actions_for_resp,
        temperature=0.7,
        goal_jsons=goal_jsons_for_resp,
        goal_progress_summaries=goal_progress_summaries,
        chunk_size=chunk_size
    )
    if len(user_responses) != len(ep_indices_for_resp):
        raise RuntimeError(
            f"generate_user_response_batch returned {len(user_responses)} results for "
            f"{len(ep_indices_for_resp)} inputs"
        )
    
    # 4) Apply user responses to episodes / envs
    for idx, raw_response in zip(ep_indices_for_resp, user_responses):
        ep = eligible[idx]
        env: OnlineEnvironment = ep.env
        agent_action = ep.agent_response
        user_response = env._parse_response_from_tags(raw_response)
        dialogue_acts = env._extract_dialogue_acts(user_response, raw_response=raw_response, agent_action=agent_action)
        
        # Update dialogue history with user response (overwrite pending entry)
        if env.env_state.dialogue_history:
            last_agent_action, _ = env.env_state.dialogue_history[-1]
            env.env_state.dialogue_history[-1] = (last_agent_action, user_response)
        if hasattr(env.env_state, "user_agent_outputs"):
            env.env_state.user_agent_outputs.append(user_response)
        
        env.env_state.turn_idx += 1
        
        prev_fraction = prev_fractions.get(idx, 0.0)
        goal_achievement_fraction = env.goal_state.get_goal_achievement_fraction() if env.goal_state.desires else (1.0 if user_response else 0.0)
        progress_made = goal_achievement_fraction > prev_fraction + 1e-9
        all_goals_completed = env.goal_state.is_goal_achieved() if env.goal_state.desires else progress_made
        
        reward = 1.0 if progress_made else 0.0
        
        episode_end = env._parse_episode_end(raw_response)
        done = episode_end if episode_end is not None else (env._check_episode_ended(user_response) or all_goals_completed)
        done = done or (env.env_state.turn_idx >= env.max_turns)
        env.env_state.episode_ended = done
        
        env.env_state.goal_achieved = max(env.env_state.goal_achieved, goal_achievement_fraction)
        if done:
            env._maybe_print_first_dialogue_trajectory()
        
        info = {
            "actions": dialogue_acts,
            "goal_achieved": env.env_state.goal_achieved,
            "turn_idx": env.env_state.turn_idx,
            "reward": reward,
            "persona": env.persona,
            "goal_state": env.goal_state
        }
        
        # Update episode fields
        ep.observation = user_response
        ep.reward_from_env = reward
        ep.reward_task = reward
        ep.reward_harm = reward
        ep.done_from_env = done
        ep.goal_achieved = env.env_state.goal_achieved
        ep.user_actions = dialogue_acts
        ep.env_info = info
        # Mark step as done by batch so process_turn_for_episode does not call env.step() again (would overwrite reward)
        ep._step_done_by_batch = True

        # Keep last_known_goal_state in sync
        if env.goal_state:
            ep.last_known_goal_state = env.goal_state


def process_turn_for_episode(
    episode: EpisodeState,
    hl_agent: HighLevelAgent,
    ll_agent: LowLevelAgent,
    value_function: ValueFunction,
    tokenizer,
    model,
    config,
    multiwoz_mode: bool,
    standard_dtype: torch.dtype
) -> bool:
    """
    Process one turn for an episode.
    
    Returns:
        True if episode should continue, False if episode is done
    """
    if episode.done_from_env or episode.turn >= config.max_turns:
        return False
    
    max_judge_retries = getattr(config, "max_judge_step_retries", 3)
    if episode.judge_step_pending:
        if episode.judge_step_retry_count >= max_judge_retries:
            raise FulfillmentJudgeParseError(
                f"Exceeded max judge step retries ({max_judge_retries}) for episode "
                f"{episode.dialogue_id} (dialogue_idx={episode.dialogue_idx})"
            )
        selected_belief = episode._judge_retry_selected_belief
        q_values = episode._judge_retry_q_values
    else:
        # Beliefs should already be generated and Q-values computed via batch operations
        if not episode.belief_state.candidates:
            return False
        
        # Get Q-values (computed in batch)
        q_values = getattr(episode, '_q_values', {})
        high_level_candidates = [c.summary for c in episode.belief_state.candidates]
        
        # Select belief (high-level action) using Q-values (epsilon-greedy)
        # Filter out [SKIP] and noise candidates from selection
        # Note: Noise is kept in candidates for Q-value computation (to test if Q-function learns to avoid it)
        #       but excluded from selection to prevent pathological reinforcement
        ablation_mode = getattr(config, 'contrastive_ablation_mode', 'none')
        filter_noise = (ablation_mode == "noise_in_candidates")
        
        valid_candidates = _filter_candidates_for_value_min(high_level_candidates, filter_noise)
        
        valid_q_values = {k: v for k, v in q_values.items() if k != "[SKIP]"}
        if filter_noise:
            valid_q_values = {k: v for k, v in valid_q_values.items() if not is_noise_candidate(k)}
        
        # Fallback: If all candidates were filtered out, check if we have [SKIP] as last resort
        # This handles cases where belief generation failed to parse any valid candidates
        if not valid_candidates and "[SKIP]" in high_level_candidates:
            valid_candidates = ["[SKIP]"]
            valid_q_values = {"[SKIP]": q_values.get("[SKIP]", 0.0)}
        
        # Use current_epsilon (which may decay in online mode) or config.epsilon
        epsilon_to_use = getattr(config, '_current_epsilon', config.epsilon)
        
        # Check if random_belief_selection is enabled
        random_belief_selection = getattr(config, 'random_belief_selection', False)
        
        if hasattr(episode, "_preselected_belief"):
            selected_belief = episode._preselected_belief
            del episode._preselected_belief
        else:
            if valid_candidates:
                use_regret_critic = getattr(config, 'use_regret_critic', False)
                regret_values = getattr(episode, '_regret_values', None)
                if random_belief_selection:
                    selected_belief = random.choice(valid_candidates)
                elif valid_q_values:
                    if random.random() < epsilon_to_use:
                        selected_belief = random.choice(valid_candidates)
                    elif use_regret_critic and regret_values and all(k in regret_values for k in valid_q_values):
                        beta = getattr(config, 'regret_critic_beta', 0.2)
                        selected_belief = _regret_critic_selection(valid_q_values, regret_values, beta)
                    else:
                        selected_belief = _softmax_sample_from_q_values(valid_q_values)
                else:
                    selected_belief = random.choice(valid_candidates)
            else:
                candidate_snapshot = {
                    "candidates": high_level_candidates,
                    "q_values": q_values
                }
                
                # Print detailed debug information
                print("\n" + "="*80)
                print("ERROR: No valid high-level belief candidates available for selection")
                print("="*80)
                print(f"Episode ID: {episode.dialogue_id}")
                print(f"Episode Index: {episode.dialogue_idx}")
                print(f"Turn: {episode.turn}")
                print(f"Candidate snapshot: {candidate_snapshot}")
                
                # Print stored debug info if available
                if episode.belief_gen_debug_prompt or episode.belief_gen_debug_raw_output:
                    print(f"\n{'='*80}")
                    print("HIGH-LEVEL AGENT INPUT (Full Prompt):")
                    print("="*80)
                    print(episode.belief_gen_debug_prompt if episode.belief_gen_debug_prompt else "N/A")
                    print(f"\n{'='*80}")
                    print("HIGH-LEVEL AGENT RAW OUTPUT:")
                    print("="*80)
                    print(episode.belief_gen_debug_raw_output if episode.belief_gen_debug_raw_output else "N/A")
                    print(f"\n{'='*80}")
                    print("Dialogue History (at time of belief generation):")
                    print("="*80)
                    if episode.belief_gen_debug_history:
                        for i, (agent_action, user_obs) in enumerate(episode.belief_gen_debug_history):
                            print(f"Turn {i+1}:")
                            print(f"  Agent: {agent_action}")
                            print(f"  User: {user_obs}")
                    else:
                        print("No dialogue history stored")
                else:
                    print("\n[NOTE] Debug info not available (belief generation may have succeeded but candidates were filtered later)")
                    print("Current Observation Context:")
                    print(episode.current_obs_for_action)
                    if episode.belief_state.history:
                        print("\nCurrent Dialogue History:")
                        for i, (agent_action, user_obs) in enumerate(episode.belief_state.history):
                            print(f"Turn {i+1}:")
                            print(f"  Agent: {agent_action}")
                            print(f"  User: {user_obs}")
                
                print("="*80 + "\n")
                sys.stdout.flush()
                
                raise RuntimeError(
                    f"No valid high-level belief candidates available for selection. "
                    f"Candidate snapshot: {candidate_snapshot}"
                )
        
        episode.chosen_beliefs_per_turn.append(selected_belief)
    
    # High-level action = selected belief summary ✓
    # Low-level action = dialogue response/action
    
    # Generate or get low-level action based on training mode
    if multiwoz_mode:
        if episode.turn == 0:
            episode.agent_response = ""
            episode.observation = episode.initial_observation
            episode.env_info = episode.initial_env_info
            # Update last known goal_state for forward-filling
            if episode.initial_env_info and episode.initial_env_info.get('goal_state'):
                episode.last_known_goal_state = episode.initial_env_info.get('goal_state')
            episode.reward_from_env = 0.0
            episode.reward_task = None
            episode.reward_harm = None
            episode.done_from_env = False
        else:
            # Only call step() if episode hasn't ended yet
            if not episode.done_from_env:
                if _is_online_mode(config):
                    # Online RL: Generate low-level action using LL agent
                    # Ensure history is not empty (should have at least turn 0 entry)
                    ll_history = episode.belief_state.history
                    if not ll_history:
                        # This shouldn't happen on turn > 0, but add safeguard
                        raise ValueError(
                            f"Low-level agent called with empty history on turn {episode.turn}. "
                            f"Episode {episode.dialogue_id} should have dialogue history by this point."
                        )
                    if hasattr(episode, "_precomputed_agent_response"):
                        episode.agent_response = episode._precomputed_agent_response
                        del episode._precomputed_agent_response
                        episode._test_time_agent_tokens = len(
                            ll_agent.tokenizer.encode(episode.agent_response, add_special_tokens=False)
                        )
                    else:
                        n_ll = getattr(config, 'n_ll_candidates', 1)
                        belief_only = getattr(config, 'll_action_belief_only', True)
                        baseline_mode = getattr(config, 'baseline_mode', False)
                        ll_template = get_ll_action_template_name(config, baseline_mode=baseline_mode, episode=episode)
                        if n_ll > 1:
                            candidates = ll_agent.generate_ll_candidates(
                                belief_context=selected_belief,
                                history=ll_history,
                                n_candidates=n_ll,
                                template_name=ll_template,
                                temperature=0.7,
                                belief_only=belief_only,
                                chunk_size=getattr(config, 'batch_generation_chunk_size', None),
                            )
                            scores = value_function.predict_ll_candidate_scores(candidates, tokenizer)
                            episode.agent_response = _softmax_sample_from_q_values(scores)
                        else:
                            prompt = ll_agent.build_prompt(
                                belief_context=selected_belief,
                                history=ll_history,
                                belief_only=belief_only,
                                template_name=ll_template,
                            )
                            episode.agent_response = ll_agent.generate_actions_from_prompts(
                                prompts=[prompt],
                                temperature=0.7,
                                do_sample=False,
                                chunk_size=1,
                                template_name=ll_template,
                            )[0]
                        episode._test_time_agent_tokens = len(
                            ll_agent.tokenizer.encode(episode.agent_response, add_special_tokens=False)
                        )
                    if hasattr(episode.env, 'env_state') and hasattr(episode.env.env_state, 'll_agent_outputs'):
                        episode.env.env_state.ll_agent_outputs.append(episode.agent_response)
                    # Store evaluation data for this turn
                    episode.turn_evaluation_data.append(
                        build_turn_evaluation_entry(
                            episode=episode,
                            observation=episode.current_obs_for_action,
                            ll_action=episode.agent_response,
                            candidate_beliefs=[c.summary for c in episode.belief_state.candidates],
                            q_values=q_values,
                            selected_belief=selected_belief,
                            multiwoz_mode=multiwoz_mode,
                            training_mode="online" if _is_online_mode(config) else "offline"
                        )
                    )
                    # Call env.step() with generated action (skip if batch already did it, e.g. multiwoz_online)
                    # OnlineEnvironment.step() returns (user_response, reward, done, info)
                    try:
                        if getattr(episode, "_step_done_by_batch", False):
                            pass
                        else:
                            step_result = episode.env.step(episode.agent_response)
                            episode.observation = step_result.observation
                            episode.reward_from_env = step_result.reward
                            episode.reward_task = step_result.reward_task
                            episode.reward_harm = step_result.reward_harm
                            episode.done_from_env = step_result.done
                            episode.env_info = step_result.info
                            if step_result.agent_action is not None:
                                episode.agent_response = step_result.agent_action
                            # Update last known goal_state for forward-filling
                            if episode.env_info and episode.env_info.get('goal_state'):
                                episode.last_known_goal_state = episode.env_info.get('goal_state')
                        if getattr(episode, "_step_done_by_batch", False):
                            del episode._step_done_by_batch
                    except Exception as e:
                        print(f"\n[ERROR] Exception in env.step() for episode {episode.dialogue_idx}, turn {episode.turn}:")
                        print(f"  Agent action: {episode.agent_response}")
                        print(f"  Error: {type(e).__name__}: {e}")
                        import traceback
                        traceback.print_exc()
                        sys.stdout.flush()
                        raise
                else:
                    # Offline RL: Get low-level action from dataset
                    step_result = episode.env.step()
                    episode.agent_response = step_result.agent_action or ""
                    episode._test_time_agent_tokens = 0
                    episode.observation = step_result.observation
                    episode.reward_from_env = step_result.reward
                    episode.reward_task = step_result.reward_task
                    episode.reward_harm = step_result.reward_harm
                    episode.done_from_env = step_result.done
                    episode.env_info = step_result.info
                    # Forward-fill goal_state from last known (since offline mode doesn't update it in step)
                    if episode.last_known_goal_state and 'goal_state' not in episode.env_info:
                        episode.env_info['goal_state'] = episode.last_known_goal_state
                    # Update last known goal_state for forward-filling
                    if episode.env_info and episode.env_info.get('goal_state'):
                        episode.last_known_goal_state = episode.env_info.get('goal_state')
                    # Store evaluation data for this turn
                    episode.turn_evaluation_data.append(
                        build_turn_evaluation_entry(
                            episode=episode,
                            observation=episode.current_obs_for_action,
                            ll_action=episode.agent_response,
                            candidate_beliefs=[c.summary for c in episode.belief_state.candidates],
                            q_values=q_values,
                            selected_belief=selected_belief,
                            multiwoz_mode=multiwoz_mode,
                            training_mode="online" if _is_online_mode(config) else "offline"
                        )
                    )
            else:
                # Episode already ended, use default values
                episode.agent_response = ""
                episode.observation = ""
                episode.reward_from_env = 0.0
                episode.reward_task = None
                episode.reward_harm = None
                episode.env_info = {}
        
        episode.goal_achieved = episode.env_info.get("goal_achieved", 0.0)
        episode.user_actions = episode.env_info.get("actions", [])
        
        # If actions are empty after turn 0, the episode has ended (no more dialogue acts)
        # On turn 0, empty actions is normal (initial state)
        # In offline mode, this indicates the dataset has no more turns
        # In online mode, _extract_dialogue_acts should always return at least one act for non-empty text
        if not _is_online_mode(config) and episode.turn > 0 and not episode.user_actions:
            episode.done_from_env = True
        
        if episode.done_from_env:
            episode.dialogue_state.goal_achieved = 1.0
    else:
        # Non-MultiWOZ environments (UserBench, VitaBench, SalesAgent)
        if episode.turn == 0:
            # Turn 0: Use initial observation from reset; keep batched action if precomputed
            if hasattr(episode, "_precomputed_agent_response"):
                episode.agent_response = episode._precomputed_agent_response
                del episode._precomputed_agent_response
            else:
                episode.agent_response = ""
            episode.observation = episode.initial_observation
            episode.env_info = episode.initial_env_info
            episode.reward_from_env = 0.0
            episode.reward_task = None
            episode.reward_harm = None
            episode.done_from_env = False
        else:
            # Turn > 0: Generate agent action and call env.step()
            if not episode.done_from_env:
                # Generate low-level action using LL agent
                ll_history = episode.belief_state.history
                if not ll_history:
                    raise ValueError(
                        f"Low-level agent called with empty history on turn {episode.turn}. "
                        f"Episode {episode.dialogue_id} should have dialogue history by this point."
                    )
                if episode.judge_step_pending:
                    episode.agent_response = episode.pending_env_action
                    episode._test_time_agent_tokens = len(
                        ll_agent.tokenizer.encode(episode.agent_response, add_special_tokens=False)
                    )
                elif hasattr(episode, "_precomputed_agent_response"):
                    episode.agent_response = episode._precomputed_agent_response
                    del episode._precomputed_agent_response
                else:
                    # Use appropriate template based on environment type and baseline mode
                    baseline_mode = getattr(config, 'baseline_mode', False)
                    if baseline_mode:
                        template_name = get_ll_action_template_name(
                            config, baseline_mode=True, episode=episode
                        )
                        prompt = ll_agent.build_prompt(
                            belief_context="",
                            history=ll_history,
                            belief_only=False,
                            template_name=template_name
                        )
                        if hasattr(episode.env, 'env_state') and hasattr(episode.env.env_state, 'll_agent_inputs'):
                            episode.env.env_state.ll_agent_inputs.append(prompt)
                        if config.environment_type in ("cares", "wildjailbreak", "redbench", "harmbench"):
                            episode.agent_response, episode._test_time_agent_tokens = _generate_defender_action_with_backend(
                                ll_agent=ll_agent,
                                prompt=prompt,
                                ll_history=ll_history,
                                template_name=template_name,
                                config=config,
                            )
                        else:
                            episode.agent_response = ll_agent.generate_actions_from_prompts(
                                prompts=[prompt],
                                temperature=0.7,
                                do_sample=False,
                                chunk_size=1,
                                template_name=template_name
                            )[0]
                            episode._test_time_agent_tokens = len(
                                ll_agent.tokenizer.encode(episode.agent_response, add_special_tokens=False)
                            )
                    else:
                        template_name = get_ll_action_template_name(
                            config, baseline_mode=False, episode=episode
                        )
                        prompt = ll_agent.build_prompt(
                            belief_context=selected_belief,
                            history=ll_history,
                            belief_only=getattr(config, 'll_action_belief_only', True),
                            template_name=template_name
                        )
                        if hasattr(episode.env, 'env_state') and hasattr(episode.env.env_state, 'll_agent_inputs'):
                            episode.env.env_state.ll_agent_inputs.append(prompt)
                        if config.environment_type in ("cares", "wildjailbreak", "redbench", "harmbench"):
                            episode.agent_response, episode._test_time_agent_tokens = _generate_defender_action_with_backend(
                                ll_agent=ll_agent,
                                prompt=prompt,
                                ll_history=ll_history,
                                template_name=template_name,
                                config=config,
                            )
                        else:
                            episode.agent_response = ll_agent.generate_actions_from_prompts(
                                prompts=[prompt],
                                temperature=0.7,
                                do_sample=False,
                                chunk_size=1,
                                template_name=template_name
                            )[0]
                            episode._test_time_agent_tokens = len(
                                ll_agent.tokenizer.encode(episode.agent_response, add_special_tokens=False)
                            )
                
                # Store evaluation data for this turn
                episode.turn_evaluation_data.append(
                    build_turn_evaluation_entry(
                        episode=episode,
                        observation=episode.current_obs_for_action,
                        ll_action=episode.agent_response,
                        candidate_beliefs=[c.summary for c in episode.belief_state.candidates],
                        q_values=q_values,
                        selected_belief=selected_belief,
                        multiwoz_mode=multiwoz_mode,
                        training_mode="online" if _is_online_mode(config) else "offline"
                    )
                )
                
                # Call env.step() with generated action
                # For UserBench/VitaBench/SalesAgent, step() returns (observation, reward, done, info)
                # For UserBench: TravelEnv requires actions to be prefixed with [action], [search], or [answer]
                # The model should generate this format when using action_generation_userbench template
                # If it doesn't (fallback), wrap it with [action]
                action_for_env = episode.agent_response
                if not multiwoz_mode:
                    # Check if this is a UserBench environment (has travel_env attribute)
                    if hasattr(episode.env, 'travel_env'):
                        # UserBench environment - check if action has proper prefix
                        if not (action_for_env.strip().startswith("[action]") or 
                                action_for_env.strip().startswith("[search]") or 
                                action_for_env.strip().startswith("[answer]")):
                            # Fallback: wrap with [action] prefix if model didn't generate it
                            action_for_env = f"[action] {action_for_env}"
                was_judge_step_pending = episode.judge_step_pending
                try:
                    if config.environment_type == "vitabench" and getattr(episode, "_step_done_by_batch", False):
                        pass
                    else:
                        step_result = episode.env.step(action_for_env)
                        episode.observation = step_result.observation
                        episode.reward_from_env = step_result.reward
                        episode.reward_task = step_result.reward_task
                        episode.reward_harm = step_result.reward_harm
                        episode.done_from_env = step_result.done
                        episode.env_info = step_result.info
                        if step_result.agent_action is not None:
                            episode.agent_response = step_result.agent_action
                        if was_judge_step_pending:
                            episode.judge_step_pending = False
                            episode.pending_env_action = None
                            episode.judge_step_retry_count = 0
                            del episode._judge_retry_selected_belief
                            del episode._judge_retry_q_values
                    if getattr(episode, "_step_done_by_batch", False):
                        del episode._step_done_by_batch
                except FulfillmentJudgeParseError as e:
                    if episode.turn_evaluation_data:
                        episode.turn_evaluation_data.pop()
                    episode.judge_step_pending = True
                    episode.pending_env_action = action_for_env
                    episode._judge_retry_selected_belief = selected_belief
                    episode._judge_retry_q_values = q_values
                    episode.judge_step_retry_count += 1
                    print(
                        f"\n[WARN] Fulfillment judge parse failed for episode {episode.dialogue_idx} "
                        f"turn {episode.turn}; retry later ({episode.judge_step_retry_count}/"
                        f"{getattr(config, 'max_judge_step_retries', 3)}): {e}"
                    )
                    sys.stdout.flush()
                    return True
                except Exception as e:
                    print(f"\n[ERROR] Exception in env.step() for episode {episode.dialogue_idx}, turn {episode.turn}:")
                    print(f"  Agent action: {episode.agent_response}")
                    print(f"  Error: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    sys.stdout.flush()
                    raise
            else:
                # Episode already ended, use default values
                episode.agent_response = ""
                episode.observation = ""
                episode.reward_from_env = 0.0
                episode.reward_task = None
                episode.reward_harm = None
                episode.env_info = {}
    
    # Update belief state history
    episode.belief_state.history.append((episode.agent_response, episode.observation))
    episode.update_current_observation_context()
    
    # Skip DPO updates in baseline mode (no real beliefs to update)
    baseline_mode = getattr(config, 'baseline_mode', False)
    if not baseline_mode:
        # Skip DPO updates in evaluation mode or when using random belief selection (no training needed)
        evaluation_mode = getattr(config, "_evaluation_mode", False)
        random_belief_selection = getattr(config, 'random_belief_selection', False)
        if not evaluation_mode and not random_belief_selection and not config.critic_only_training:
            # Compute log-probabilities for DPO ranking (using selected belief, not selected action)
            try:
                log_probs = compute_log_probs_batch(
                    beliefs=episode.belief_state.candidates,
                    action=selected_belief,  # Use selected belief for DPO ranking
                    observation=episode.observation,
                    model=model,
                    tokenizer=tokenizer
                )
                
                # Update belief probabilities using DPO ranking
                episode.belief_state = update_belief_distribution_dpo(
                    belief_state=episode.belief_state,
                    log_probs=log_probs,
                    temperature=config.dpo_temperature
                )
            except RuntimeError as e:
                # Handle cases where beliefs are too long (e.g., GPT-generated beliefs)
                if "expanded size" in str(e) or "tensor" in str(e).lower():
                    print(f"[WARNING] Skipping DPO update due to tensor size mismatch (beliefs too long): {e}")
                    # Keep existing probabilities
                else:
                    raise
    
    # Compute entropy
    probabilities = [c.probability for c in episode.belief_state.candidates]
    entropy = compute_entropy(probabilities)
    episode.context_entropies.append(entropy)
    
    # Get reward
    if multiwoz_mode:
        episode.reward = episode.reward_from_env
        episode.goal_achieved = episode.goal_achieved
    else:
        # For non-MultiWOZ environments (UserBench, VitaBench, SalesAgent), use environment reward
        episode.reward = episode.reward_from_env
        # Goal achieved comes from environment (e.g., TravelGym's episode_complete or elicitation_ratio)
        episode.goal_achieved = episode.env_info.get("goal_achieved", 0.0) if episode.env_info else 0.0
    
    episode.dialogue_state.rewards.append(episode.reward)
    # CARES/WildJailbreak/RedBench adversarial: last turn wins (episode fails if any turn harmful). Others: use max.
    if config.environment_type in ("cares", "wildjailbreak", "redbench", "harmbench") and isinstance(episode.dialogue_data, dict) and episode.dialogue_data.get("harmful_level", 0) > 0:
        episode.dialogue_state.goal_achieved = episode.goal_achieved
    else:
        episode.dialogue_state.goal_achieved = max(episode.goal_achieved, episode.dialogue_state.goal_achieved)
    episode.episode_total_reward += episode.reward
    
    episode.turn += 1
    episode.actual_turns_processed += 1
    
    # Log training metrics for this turn
    # Get the most recent value loss (from last Q-function update)
    # Note: value_loss is only computed during batch updates, so we use the most recent one
    # If no value_loss has been computed yet, use 0.0
    current_value_loss = episode.episode_value_losses[-1] if episode.episode_value_losses else 0.0
    
    # Compute belief accuracy (cosine similarity to ground truth if available)
    # Only compute periodically to avoid performance overhead (every 10 turns)
    # Note: ground_truth_goal is always set (via imputation if needed), but may be "User goal not specified"
    # if the goal is truly empty. We skip belief_accuracy computation in that case.
    belief_accuracy = None
    if (episode.turn % 10 == 0 and 
        hasattr(episode, 'ground_truth_goal') and 
        episode.ground_truth_goal and
        isinstance(episode.ground_truth_goal, str) and
        episode.ground_truth_goal.strip().lower() != "user goal not specified"):
        try:
            from .utils.belief_evaluation import evaluate_beliefs_against_ground_truth
            chosen_belief = selected_belief if selected_belief else (episode.belief_state.candidates[0].summary if episode.belief_state.candidates else "")
            if chosen_belief:
                metrics = evaluate_beliefs_against_ground_truth(
                    chosen_beliefs=[chosen_belief],
                    ground_truth_text=episode.ground_truth_goal,
                    model=model,
                    tokenizer=tokenizer,
                    device=config.device
                )
                if metrics:
                    belief_accuracy = metrics[0].get("cosine_similarity")
        except Exception:
            # Don't fail if belief accuracy computation fails
            pass
    
    log_metrics(
        turn=episode.turn - 1,  # Use turn before increment
        belief_state=episode.belief_state,
        entropy=entropy,
        value_loss=current_value_loss,
        reward=episode.reward,
        dialogue_id=episode.dialogue_id,
        episode_idx=episode.dialogue_idx,
        belief_accuracy=belief_accuracy,
        ground_truth_goal=episode.ground_truth_goal if hasattr(episode, 'ground_truth_goal') else None,
        turn_evaluation=episode.turn_evaluation_data[-1] if episode.turn_evaluation_data else None,
        output_dir=config.output_dir
    )
    
    # Check if episode should end
    # Episode ends if: done_from_env is True (env decided: max_turns, goal, or harmful response)
    if episode.done_from_env:
        return False
    # For CARES/WildJailbreak/RedBench adversarial: env alone controls termination (max_turns or harmful response).
    # Do NOT end on goal_achieved (safe response) - that would truncate before agent can slip.
    if config.environment_type in ("cares", "wildjailbreak", "redbench", "harmbench"):
        harmful_level = (
            episode.dialogue_data.get("harmful_level", 0)
            if isinstance(episode.dialogue_data, dict) else 0
        )
        if harmful_level > 0:
            return True  # Adversarial: continue until env returns done
    # Goal achieved (>= 1.0): end for non-adversarial (MultiWOZ, CARES safe, etc.)
    if episode.goal_achieved >= 1.0:
        return False
    if multiwoz_mode and (not _is_online_mode(config) and episode.turn > 0 and not episode.user_actions):
        return False

    return True


def collect_transitions_from_episodes(
    episodes: List[EpisodeState],
    transition_buffer: Dict,
    config
) -> List[EpisodeState]:
    """
    Collect transitions from all active episodes into the shared buffer.
    
    Returns:
        List of episodes that contributed transitions to this batch
    """
    contributing_episodes = []
    for episode in episodes:
        if not episode.is_active or episode.turn == 0:
            continue  # Skip first turn (no previous action)
        
        # Don't collect transitions for frozen episodes (generation failures that will be retried)
        if episode.is_frozen:
            continue
        if episode.judge_step_pending:
            continue
        
        # Don't collect transitions for episodes that have ended
        # (either done_from_env is True, or actions are empty after turn 0 indicating episode ended)
        if episode.done_from_env or (not _is_online_mode(config) and episode.turn > 0 and not episode.user_actions):
            continue
        
        avg_entropy = compute_average_entropy(episode.context_entropies) if episode.context_entropies else 0.0
        
        state_before_action = episode.previous_obs_for_action or episode.current_obs_for_action
        transition_buffer['observations'].append(state_before_action)
        transition_buffer['high_level_actions'].append(episode.chosen_beliefs_per_turn[-1] if episode.chosen_beliefs_per_turn else "")
        transition_buffer['low_level_actions'].append(episode.agent_response)
        transition_buffer['rewards'].append(episode.reward)
        transition_buffer['next_observations'].append(episode.current_obs_for_action)
        transition_buffer['terminals'].append(episode.goal_achieved >= 1.0)  # Convert float to boolean for terminal flag
        transition_buffer['entropies'].append(avg_entropy)
        
        # Store environment and dialogue history for marginal rewards
        if getattr(config, 'use_marginal_token_rewards', False):
            if 'environments' not in transition_buffer:
                transition_buffer['environments'] = []
            if 'dialogue_histories' not in transition_buffer:
                transition_buffer['dialogue_histories'] = []
            transition_buffer['environments'].append(episode.env)
            # Get dialogue history from belief state
            dialogue_history = episode.belief_state.history if hasattr(episode.belief_state, 'history') else []
            transition_buffer['dialogue_histories'].append(dialogue_history)
        
        # Store belief candidates and probabilities for V-function training
        # V(o_t) target = E_a~π[Q(o_t, a)] = Σ_belief p(belief|o_t) * Q(o_t, belief)
        if episode.belief_state.candidates:
            belief_candidates = [c.summary for c in episode.belief_state.candidates]
            belief_probabilities = [c.probability for c in episode.belief_state.candidates]
        else:
            belief_candidates = []
            belief_probabilities = []
        transition_buffer['belief_candidates'].append(belief_candidates)
        transition_buffer['belief_probabilities'].append(belief_probabilities)
        # Add ground truth goal for contrastive learning (only if contrastive loss is enabled)
        contrastive_coef = getattr(config, 'contrastive_coef', 0.0)
        ablation_mode = getattr(config, 'contrastive_ablation_mode', 'none')
        needs_ground_truth = (contrastive_coef > 0.0 and 
                             ablation_mode in ["crossed_data", "random_noise"] and 
                             ablation_mode != "none")
        if needs_ground_truth:
            transition_buffer['ground_truth_goals'].append(episode.ground_truth_goal if hasattr(episode, 'ground_truth_goal') else "")
        else:
            transition_buffer['ground_truth_goals'].append("")  # Empty string to maintain list length
        
        contributing_episodes.append(episode)
    
    return contributing_episodes


def run_contrastive_test(
    test_inputs: List[str],
    hl_agent: HighLevelAgent,
    value_function: ValueFunction,
    tokenizer,
    config,
    n_beliefs: int = 5
) -> List[Dict]:
    """
    Run contrastive test on N (even) test inputs.
    
    For each test input:
    1. Generate n_beliefs beliefs
    2. Score them with Q-values
    3. For odd-indexed tests, swap beliefs 2 and 4 with previous even-indexed test
    4. Re-score and track score/rank changes
    
    Args:
        test_inputs: List of observation strings (must be even length)
        hl_agent: High-level agent for generating beliefs
        value_function: Value function for scoring beliefs
        tokenizer: Tokenizer
        config: Config object
        n_beliefs: Number of beliefs to generate per test (default 5)
    
    Returns:
        List of dictionaries with contrastive test results
    """
    if len(test_inputs) % 2 != 0:
        raise ValueError(f"Number of test inputs must be even, got {len(test_inputs)}")
    
    results = []
    
    # Generate beliefs for all test inputs
    # For belief generation, construct minimal histories using the provided observations
    histories = [[("[NO_AGENT_ACTION]", obs)] for obs in test_inputs]
    
    all_belief_candidates = hl_agent.generate_candidate_beliefs_batch(
        chunk_size=config.batch_generation_chunk_size,
        histories=histories,
        n_candidates=n_beliefs,
        temperature=config.belief_gen_temperature,
        max_new_tokens=config.max_tokens
    )
    
    # Score all beliefs initially
    all_initial_scores = []
    for test_idx, (obs, candidates) in enumerate(zip(test_inputs, all_belief_candidates)):
        if not candidates:
            all_initial_scores.append({})
            continue
        
        candidate_texts = [c.summary for c in candidates]
        # Filter out [SKIP] candidates
        valid_candidates = [c for c in candidate_texts if c != "[SKIP]"]
        valid_indices = [i for i, c in enumerate(candidate_texts) if c != "[SKIP]"]
        
        if valid_candidates:
            q_values = value_function.predict_q_value(
                observations=[obs] * len(valid_candidates),
                high_level_actions=valid_candidates,
                tokenizer=tokenizer,
                requires_grad=False
            )
            
            # Create score dict for all candidates (including [SKIP])
            scores = {}
            for i, cand in enumerate(candidate_texts):
                if cand == "[SKIP]":
                    scores[cand] = 0.0
                elif i in valid_indices:
                    idx_in_valid = valid_indices.index(i)
                    scores[cand] = q_values[idx_in_valid].item() if hasattr(q_values[idx_in_valid], 'item') else float(q_values[idx_in_valid])
                else:
                    scores[cand] = 0.0
        else:
            scores = {c.summary: 0.0 for c in candidates}
        
        all_initial_scores.append(scores)
    
    # Process each test input
    for test_idx in range(len(test_inputs)):
        obs = test_inputs[test_idx]
        candidates = all_belief_candidates[test_idx]
        initial_scores = all_initial_scores[test_idx]
        
        if not candidates:
            results.append({
                'test_idx': test_idx,
                'observation': obs,
                'is_contrastive': False,
                'error': 'No candidates generated'
            })
            continue
        
        # Sort candidates by initial score (descending)
        candidate_list = [c.summary for c in candidates]
        sorted_by_score = sorted(
            [(cand, initial_scores.get(cand, 0.0)) for cand in candidate_list],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Get beliefs at positions 2 and 4 (0-indexed: indices 1 and 3)
        # These are the 2nd highest and 2nd lowest scoring
        if len(sorted_by_score) < 4:
            results.append({
                'test_idx': test_idx,
                'observation': obs,
                'is_contrastive': False,
                'error': f'Not enough candidates (need at least 4, got {len(sorted_by_score)})'
            })
            continue
        
        belief_2 = sorted_by_score[1][0]  # 2nd highest
        belief_4 = sorted_by_score[3][0]  # 2nd lowest (4th overall)
        
        is_contrastive = (test_idx % 2 == 1)  # Odd-indexed tests are contrastive
        
        result = {
            'test_idx': test_idx,
            'observation': obs,
            'is_contrastive': is_contrastive,
            'initial_beliefs': candidate_list,
            'initial_scores': {cand: initial_scores.get(cand, 0.0) for cand in candidate_list},
            'initial_rankings': {cand: rank for rank, (cand, _) in enumerate(sorted_by_score)},
            'belief_2': belief_2,
            'belief_4': belief_4,
            'swapped_beliefs': None,
            'final_scores': None,
            'final_rankings': None,
            'score_changes': None,
            'rank_changes': None
        }
        
        # For odd-indexed tests, swap beliefs 2 and 4 with previous test
        if is_contrastive:
            prev_test_idx = test_idx - 1
            prev_candidates = all_belief_candidates[prev_test_idx]
            prev_initial_scores = all_initial_scores[prev_test_idx]
            
            if not prev_candidates:
                result['error'] = 'Previous test had no candidates'
                results.append(result)
                continue
            
            prev_candidate_list = [c.summary for c in prev_candidates]
            prev_sorted_by_score = sorted(
                [(cand, prev_initial_scores.get(cand, 0.0)) for cand in prev_candidate_list],
                key=lambda x: x[1],
                reverse=True
            )
            
            if len(prev_sorted_by_score) < 4:
                result['error'] = f'Previous test had insufficient candidates (need 4, got {len(prev_sorted_by_score)})'
                results.append(result)
                continue
            
            prev_belief_2 = prev_sorted_by_score[1][0]
            prev_belief_4 = prev_sorted_by_score[3][0]
            
            # Swap: current test's belief_2 <-> previous test's belief_2
            #       current test's belief_4 <-> previous test's belief_4
            swapped_candidates = candidate_list.copy()
            
            # Find indices of beliefs to swap
            try:
                curr_idx_2 = swapped_candidates.index(belief_2)
                curr_idx_4 = swapped_candidates.index(belief_4)
                prev_idx_2 = prev_candidate_list.index(prev_belief_2)
                prev_idx_4 = prev_candidate_list.index(prev_belief_4)
                
                # Perform swaps
                swapped_candidates[curr_idx_2] = prev_belief_2
                swapped_candidates[curr_idx_4] = prev_belief_4
                
                result['swapped_beliefs'] = swapped_candidates
                result['swapped_from_prev'] = {
                    'belief_2': prev_belief_2,
                    'belief_4': prev_belief_4
                }
                
                # Re-score swapped beliefs
                valid_swapped = [c for c in swapped_candidates if c != "[SKIP]"]
                if valid_swapped:
                    q_values_swapped = value_function.predict_q_value(
                        observations=[obs] * len(valid_swapped),
                        high_level_actions=valid_swapped,
                        tokenizer=tokenizer,
                        requires_grad=False
                    )
                    
                    # Create score dict
                    final_scores = {}
                    valid_idx = 0
                    for cand in swapped_candidates:
                        if cand == "[SKIP]":
                            final_scores[cand] = 0.0
                        elif cand in valid_swapped:
                            final_scores[cand] = q_values_swapped[valid_idx].item() if hasattr(q_values_swapped[valid_idx], 'item') else float(q_values_swapped[valid_idx])
                            valid_idx += 1
                        else:
                            final_scores[cand] = 0.0
                else:
                    final_scores = {cand: 0.0 for cand in swapped_candidates}
                
                result['final_scores'] = final_scores
                
                # Compute rankings
                sorted_final = sorted(
                    [(cand, final_scores.get(cand, 0.0)) for cand in swapped_candidates],
                    key=lambda x: x[1],
                    reverse=True
                )
                result['final_rankings'] = {cand: rank for rank, (cand, _) in enumerate(sorted_final)}
                
                # Compute score and rank changes for swapped beliefs
                score_changes = {}
                rank_changes = {}
                
                # Track changes for beliefs that were swapped in (from previous test)
                for swapped_belief in [prev_belief_2, prev_belief_4]:
                    if swapped_belief in final_scores:
                        initial_score = prev_initial_scores.get(swapped_belief, 0.0)  # Score in previous test
                        final_score = final_scores[swapped_belief]
                        score_changes[swapped_belief] = final_score - initial_score
                        
                        # Find initial rank in previous test
                        prev_rank = next((rank for rank, (cand, _) in enumerate(prev_sorted_by_score) if cand == swapped_belief), None)
                        final_rank = result['final_rankings'].get(swapped_belief, None)
                        if prev_rank is not None and final_rank is not None:
                            rank_changes[swapped_belief] = final_rank - prev_rank
                
                # Track changes for beliefs that were swapped out (from current test)
                for swapped_out_belief in [belief_2, belief_4]:
                    if swapped_out_belief in initial_scores:
                        initial_score = initial_scores[swapped_out_belief]
                        # This belief is no longer in current test, so we can't compute final score
                        # But we can note that it was swapped out
                        score_changes[swapped_out_belief] = None
                        rank_changes[swapped_out_belief] = None
                
                result['score_changes'] = score_changes
                result['rank_changes'] = rank_changes
                
            except ValueError as e:
                result['error'] = f'Error finding belief indices: {str(e)}'
        
        results.append(result)
    
    return results


def record_episode_statistics(
    stats: Dict,
    episode_stats: Dict,
    evaluation_episodes: deque,
    config,
    episodes_completed: int,
    debug_example_ref: List,  # Use list to allow modification
    multiwoz_mode: bool,
    model,
    tokenizer,
    hl_agent
) -> None:
    """Record statistics for a completed episode."""
    # Add to episode_stats
    episode_stats['total_reward'].append(stats['episode_total_reward'])
    episode_stats['episode_length'].append(stats['episode_length'])
    episode_stats['goal_achieved'].append(stats['goal_achieved'])  # Already a float (0.0 to 1.0)
    episode_stats['final_entropy'].append(stats['final_entropy'])
    episode_stats['avg_value_loss'].append(stats['avg_value_loss'])
    episode_stats['avg_q_min_loss'].append(stats.get('avg_q_min_loss', 0.0))
    episode_stats['avg_q_min_target_variance'].append(stats.get('avg_q_min_target_variance', 0.0))
    episode_stats['avg_q_min_prediction_variance'].append(stats.get('avg_q_min_prediction_variance', 0.0))
    episode_stats['harmful_level'].append(harmful_level_for_defense_log(stats))
    turn_eval_data = stats.get('turn_evaluation_data', [])
    test_time_tokens = [
        int(turn.get('test_time_agent_tokens', 0))
        for turn in turn_eval_data
        if isinstance(turn.get('test_time_agent_tokens', None), (int, float))
    ]
    episode_token_total = sum(test_time_tokens)
    episode_stats['test_time_agent_tokens_total'].append(episode_token_total)
    if test_time_tokens:
        episode_stats['test_time_agent_tokens_per_turn_avg'].append(episode_token_total / len(test_time_tokens))
    else:
        episode_stats['test_time_agent_tokens_per_turn_avg'].append(0.0)
    
    # GPU memory (get max for this episode)
    if torch.cuda.is_available():
        max_gpu_memory_bytes = torch.cuda.max_memory_allocated()
        max_gpu_memory_gb = max_gpu_memory_bytes / (1024 ** 3)
        episode_stats['max_gpu_memory_gb'].append(max_gpu_memory_gb)
        torch.cuda.reset_peak_memory_stats()
    else:
        episode_stats['max_gpu_memory_gb'].append(0.0)
    
    # Store for belief evaluation
    # NOTE: An "episode" = one full MultiWOZ dialogue/conversation (not transitions)
    # goal_json is the MultiWOZ goal structure: a dict mapping domains (e.g., "restaurant", "hotel")
    # to goal specifications with inform_slots (requirements) and request_slots (needed info).
    # Example: {"restaurant": {"inform_slots": {"area": "centre", "food": "italian"}, "request_slots": {"phone": "?"}}}
    # All MultiWOZ dialogues should have a goal field (required), but it might be empty {}.
    # Only episodes with non-empty goal_json are added to evaluation_episodes for belief evaluation
    if multiwoz_mode and stats.get('goal_json'):
        # Get initial observation from the episode if available
        # We'll need to pass it through stats or get it from the episode
        initial_obs = stats.get('initial_observation', '')
        evaluation_episodes.append({
            'episode_id': stats['dialogue_idx'],
            'dialogue_id': stats['dialogue_id'],
            'goal_json': stats['goal_json'],
            'chosen_beliefs': stats['chosen_beliefs'],
            'initial_observation': initial_obs,
            'turn_evaluation_data': stats.get('turn_evaluation_data', [])
        })
    elif multiwoz_mode and not stats.get('goal_json'):
        # Log when episodes are skipped due to missing/empty goal_json
        # This can happen if:
        # 1. The dialogue has an empty goal {} (annotation error or missing data)
        # 2. The dialogue_id doesn't have goal data in the source
        # This should be rare in MultiWOZ 2.4, but can occur in earlier versions
        pass
    
    # Note: Debug example is handled in the main loop before episode finalization


def is_noise_candidate(candidate: str) -> bool:
    """
    Check if a candidate is random noise (for filtering from selection).
    
    Args:
        candidate: Candidate belief string
        
    Returns:
        True if candidate is noise (starts with [RANDOM_NOISE])
    """
    return isinstance(candidate, str) and candidate.strip().startswith("[RANDOM_NOISE]")


def _filter_candidates_for_value_min(candidates: List[str], filter_noise: bool) -> List[str]:
    """Filter candidates with the same rules used for high-level selection."""
    valid_candidates = [candidate for candidate in candidates if candidate != "[SKIP]"]
    if filter_noise:
        valid_candidates = [candidate for candidate in valid_candidates if not is_noise_candidate(candidate)]
    return valid_candidates


def generate_random_noise_beliefs(n: int, length_range: tuple = (50, 200)) -> List[str]:
    """
    Generate random noise beliefs for ablation studies.
    
    Used in ablation studies to test whether contrastive learning with crossed data
    is more effective than using random noise as negatives.
    
    Args:
        n: Number of random noise beliefs to generate
        length_range: Tuple of (min_length, max_length) for the noise text
    
    Returns:
        List of random noise belief strings (prefixed with [RANDOM_NOISE])
    """
    import random
    import string
    
    noise_beliefs = []
    for _ in range(n):
        # Generate random text of random length
        length = random.randint(length_range[0], length_range[1])
        # Use a mix of random characters and words to make it look more like text
        # but still be clearly noise
        noise_chars = string.ascii_letters + string.digits + " " + ".,!?"
        noise_text = ''.join(random.choice(noise_chars) for _ in range(length))
        # Add some structure to make it look vaguely like a belief
        noise_beliefs.append(f"[RANDOM_NOISE] {noise_text}")
    
    return noise_beliefs


def sample_negative_beliefs(
    current_ground_truths: List[str],
    negative_pool: List[str],
    model,
    tokenizer,
    device: str,
    ablation_mode: str = "crossed_data"
) -> List[str]:
    """
    Sample negative beliefs based on ablation mode.
    
    Three ablation modes to validate contrastive learning with crossed data:
    
    1. "crossed_data" (main method): Sample from other episodes' ground truths that are
       maximally cosine-distant from current ground truths. This tests whether using
       semantically meaningful but contextually wrong beliefs as negatives helps.
    
    2. "random_noise" (ablation 1): Use random noise as negative beliefs. This tests
       whether the benefit comes from using crossed data specifically, or just from
       having any negative examples.
    
    3. "none": No contrastive loss (baseline / ablation 3). This is the baseline
       without any contrastive learning.
    
    Note: There's also a fourth mode "noise_in_candidates" (ablation 2) that includes
    random noise as one of the belief candidates instead of using contrastive loss.
    This tests whether the Q-function can learn to avoid noise naturally.
    
    Args:
        current_ground_truths: List of ground truth belief texts for current transitions
        negative_pool: Pool of ground truth belief texts from other episodes
        model: Model for computing embeddings
        tokenizer: Tokenizer
        device: Device for computation
        ablation_mode: One of "crossed_data", "random_noise", or "none"
    
    Args:
        current_ground_truths: List of ground truth belief texts for current transitions
        negative_pool: Pool of ground truth belief texts from other episodes
        model: Model for computing embeddings
        tokenizer: Tokenizer
        device: Device for computation
        ablation_mode: One of "crossed_data", "random_noise", or "none"
    
    Returns:
        List of negative belief texts (same length as current_ground_truths)
    """
    if ablation_mode == "none":
        return [""] * len(current_ground_truths)
    
    if ablation_mode == "random_noise":
        # Generate random noise beliefs
        return generate_random_noise_beliefs(len(current_ground_truths))
    
    # Default: "crossed_data" mode
    if not current_ground_truths or not negative_pool:
        return [""] * len(current_ground_truths)
    
    from .utils.belief_evaluation import get_text_embeddings, compute_cosine_similarity
    import random
    
    # OPTIMIZATION: Use evenly-spaced sampling from the pool instead of the full pool
    # This dramatically reduces embedding computation time for large pools (e.g., 1000 items)
    # We sample one candidate from each state in the rollout by evenly distributing across the pool
    # This ensures even distribution of minimal size rather than random clustering
    max_pool_sample_size = 200  # Sample at most 200 items from the pool
    if len(negative_pool) > max_pool_sample_size:
        # Evenly space indices across the pool to get one candidate from each state region
        # This gives us representatives from across all states that contributed to the pool
        step = len(negative_pool) / max_pool_sample_size
        sampled_indices = [int(i * step) for i in range(max_pool_sample_size)]
        sampled_pool = [negative_pool[idx] for idx in sampled_indices]
    else:
        sampled_pool = negative_pool
    
    # OPTIMIZATION: Chunk current ground truths if batch is very large to prevent OOM
    embedding_chunk_size = 32  # Process embeddings in chunks
    import torch.nn.functional as F
    
    # Compute embeddings for sampled negative pool once (much faster for large pools)
    negative_embeddings = get_text_embeddings(sampled_pool, model, tokenizer, device)
    negative_emb_norm = F.normalize(negative_embeddings, p=2, dim=1)  # [pool_size, hidden_size]
    
    # Process current ground truths in chunks if needed
    if len(current_ground_truths) > embedding_chunk_size:
        # Chunk processing for large batches
        all_similarity_rows = []
        for chunk_start in range(0, len(current_ground_truths), embedding_chunk_size):
            chunk_end = min(chunk_start + embedding_chunk_size, len(current_ground_truths))
            chunk_ground_truths = current_ground_truths[chunk_start:chunk_end]
            
            # Compute embeddings for this chunk
            chunk_embeddings = get_text_embeddings(chunk_ground_truths, model, tokenizer, device)
            chunk_emb_norm = F.normalize(chunk_embeddings, p=2, dim=1)  # [chunk_size, hidden_size]
            
            # Compute similarity matrix for this chunk
            # [chunk_size, hidden_size] @ [hidden_size, pool_size] = [chunk_size, pool_size]
            chunk_similarity = chunk_emb_norm @ negative_emb_norm.T
            all_similarity_rows.append(chunk_similarity)
            
            # Cleanup
            del chunk_embeddings, chunk_emb_norm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate all chunks
        similarity_matrix = torch.cat(all_similarity_rows, dim=0)  # [batch_size, pool_size]
        del all_similarity_rows
    else:
        # Small batch: process all at once
        current_embeddings = get_text_embeddings(current_ground_truths, model, tokenizer, device)
        current_emb_norm = F.normalize(current_embeddings, p=2, dim=1)  # [batch_size, hidden_size]
        
        # Compute all cosine similarities in one batch operation
        # [batch_size, hidden_size] @ [hidden_size, pool_size] = [batch_size, pool_size]
        similarity_matrix = current_emb_norm @ negative_emb_norm.T  # [batch_size, pool_size]
        del current_embeddings, current_emb_norm
    
    # For each current ground truth, find the most distant negative belief
    negative_beliefs = []
    for i in range(len(current_ground_truths)):
        # Get similarities for this current embedding: [pool_size]
        cos_sims = similarity_matrix[i]
        
        # Convert to list of (similarity, index) pairs
        similarities = [(cos_sims[j].item(), j) for j in range(len(cos_sims))]
        
        # Sort by cosine similarity (ascending - most distant first)
        similarities.sort(key=lambda x: x[0])
        
        # Sample from the most distant ones (bottom 20% or at least 1)
        n_candidates = max(1, len(similarities) // 5)
        candidate_indices = [idx for _, idx in similarities[:n_candidates]]
        
        # Randomly sample from candidates
        selected_pool_idx = random.choice(candidate_indices)
        negative_beliefs.append(sampled_pool[selected_pool_idx])
    
    return negative_beliefs


def log_p_action_belief_diagnostics(
    transitions: Dict,
    value_function: ValueFunction,
    ll_model,
    ll_tokenizer,
    config,
    output_dir: str = "outputs"
) -> None:
    """
    Diagnostic: Log P(a|noise) vs P(a|coherent-belief) statistics.
    
    This helps verify whether the correlation problem exists - if P(a|noise) is not
    much lower than P(a|coherent), the Q-function cannot learn to distinguish them.
    
    Args:
        transitions: Dict with lists of (obs, hl_action, ll_action, next_obs, reward, terminal)
        value_function: Value function (for belief_only config)
        ll_model: Low-level model for computing P(a|b)
        ll_tokenizer: Tokenizer for low-level model
        config: Config object
        output_dir: Output directory for logs
    """
    import json
    from pathlib import Path
    from collections import defaultdict
    
    if not transitions.get('high_level_actions') or not transitions.get('low_level_actions'):
        return
    
    hl_actions = transitions['high_level_actions']
    ll_actions = transitions['low_level_actions']
    observations = transitions.get('observations', [])
    
    # Separate noise beliefs from coherent beliefs
    noise_pairs = []  # (belief, action)
    coherent_pairs = []  # (belief, action)
    
    for hl_action, ll_action in zip(hl_actions, ll_actions):
        if hl_action == "[SKIP]" or not ll_action:
            continue
        
        if is_noise_candidate(hl_action):
            noise_pairs.append((hl_action, ll_action))
        else:
            coherent_pairs.append((hl_action, ll_action))
    
    if not noise_pairs or not coherent_pairs:
        return  # Need both types to compare
    
    # Sample a subset for efficiency (max 50 of each type)
    import random
    max_samples = 50
    if len(noise_pairs) > max_samples:
        noise_pairs = random.sample(noise_pairs, max_samples)
    if len(coherent_pairs) > max_samples:
        coherent_pairs = random.sample(coherent_pairs, max_samples)
    
    # Compute P(a|b) for noise beliefs
    noise_log_probs = []
    belief_only = getattr(config, 'll_action_belief_only', True)
    
    # Create mapping from belief to observation index for non-belief_only mode
    belief_to_obs_idx = {}
    if not belief_only and observations:
        for idx, belief in enumerate(hl_actions):
            if idx < len(observations):
                belief_to_obs_idx[belief] = idx
    
    # Batch compute P(a|b) for noise beliefs
    noise_beliefs = [pair[0] for pair in noise_pairs]
    noise_actions = [pair[1] for pair in noise_pairs]
    noise_obs_list = []
    if not belief_only and observations:
        for belief in noise_beliefs:
            obs = observations[belief_to_obs_idx.get(belief, 0)] if belief in belief_to_obs_idx and belief_to_obs_idx[belief] < len(observations) else ""
            noise_obs_list.append(obs)
    else:
        noise_obs_list = [""] * len(noise_pairs)
    
    noise_log_probs = value_function.compute_likelihood_batch(
        observations=noise_obs_list,
        high_level_actions=noise_beliefs,
        low_level_actions=noise_actions,
        ll_model=ll_model,
        ll_tokenizer=ll_tokenizer,
        belief_only=belief_only
    )
    noise_log_probs = [lp for lp in noise_log_probs if lp != float('-inf')]
    
    # Batch compute P(a|b) for coherent beliefs
    coherent_beliefs = [pair[0] for pair in coherent_pairs]
    coherent_actions = [pair[1] for pair in coherent_pairs]
    coherent_obs_list = []
    if not belief_only and observations:
        for belief in coherent_beliefs:
            obs = observations[belief_to_obs_idx.get(belief, 0)] if belief in belief_to_obs_idx and belief_to_obs_idx[belief] < len(observations) else ""
            coherent_obs_list.append(obs)
    else:
        coherent_obs_list = [""] * len(coherent_pairs)
    
    coherent_log_probs = value_function.compute_likelihood_batch(
        observations=coherent_obs_list,
        high_level_actions=coherent_beliefs,
        low_level_actions=coherent_actions,
        ll_model=ll_model,
        ll_tokenizer=ll_tokenizer,
        belief_only=belief_only
    )
    coherent_log_probs = [lp for lp in coherent_log_probs if lp != float('-inf')]
    
    if not noise_log_probs or not coherent_log_probs:
        return  # Need valid log probs to compare
    
    # Compute statistics
    import numpy as np
    noise_mean = np.mean(noise_log_probs)
    noise_std = np.std(noise_log_probs)
    coherent_mean = np.mean(coherent_log_probs)
    coherent_std = np.std(coherent_log_probs)
    
    # Convert to probabilities for easier interpretation
    noise_mean_prob = np.exp(noise_mean)
    coherent_mean_prob = np.exp(coherent_mean)
    ratio = noise_mean_prob / coherent_mean_prob if coherent_mean_prob > 0 else float('inf')
    
    # Log to file
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    log_file = output_path / "p_action_belief_diagnostics.jsonl"
    
    diagnostic = {
        "timestamp": time.time(),
        "n_noise_samples": len(noise_log_probs),
        "n_coherent_samples": len(coherent_log_probs),
        "noise_mean_log_prob": float(noise_mean),
        "noise_std_log_prob": float(noise_std),
        "coherent_mean_log_prob": float(coherent_mean),
        "coherent_std_log_prob": float(coherent_std),
        "noise_mean_prob": float(noise_mean_prob),
        "coherent_mean_prob": float(coherent_mean_prob),
        "ratio_noise_to_coherent": float(ratio),
        "log_ratio": float(noise_mean - coherent_mean),  # log(P(a|noise)) - log(P(a|coherent))
        "belief_only_mode": belief_only
    }
    
    with open(log_file, "a") as f:
        f.write(json.dumps(diagnostic) + "\n")
    
    # Print summary (called every 1000 episodes, so always print)
    print(f"\n[P(a|b) Diagnostic] Noise vs Coherent Beliefs:")
    print(f"  P(a|noise) mean: {noise_mean_prob:.6f} (log: {noise_mean:.4f})")
    print(f"  P(a|coherent) mean: {coherent_mean_prob:.6f} (log: {coherent_mean:.4f})")
    print(f"  Ratio (noise/coherent): {ratio:.4f}")
    print(f"  Log ratio: {noise_mean - coherent_mean:.4f} (should be << 0)")
    if ratio > 0.5:
        print(f"  ⚠️  WARNING: P(a|noise) is not much lower than P(a|coherent)!")
        print(f"     This suggests the correlation problem - Q-function may struggle to distinguish beliefs.")
    sys.stdout.flush()


def update_q_function_online(
    transitions: Dict,
    value_function: ValueFunction,
    hl_agent: HighLevelAgent,
    tokenizer,
    config,
    standard_dtype: torch.dtype,
    ll_model=None,
    ll_tokenizer=None,
    ground_truth_pool: Optional[List[str]] = None,
    contrastive_coef: float = 0.0,
    ablation_mode: str = "crossed_data",
    reward_model=None,
    reward_tokenizer=None,
) -> float:
    """
    Update Q-function with online transitions (no transition probabilities).
    
    Target formula: target = r_t + γ * V(o_{t+1})
    (Using V-function for next state value - much faster than generating candidates!)
    (Simpler than offline mode - no P̂ term)
    
    Args:
        transitions: Dict with lists of (obs, hl_action, next_obs, reward, terminal)
        value_function: Value function to update (supports both Q and V)
        hl_agent: High-level agent (not used for V-function, kept for compatibility)
        tokenizer: Tokenizer
        config: Config object
        standard_dtype: Data type for tensors
    
    Returns:
        Average Q-loss value
    """
    if not transitions['observations']:
        return 0.0
    
    if getattr(config, "_evaluation_mode", False):
        return 0.0
    
    # 1. Compute Q(o_t, a_t^HL) for current transitions
    obs_list = transitions['observations']
    hl_actions_list = transitions['high_level_actions']
    
    # Filter out [SKIP] actions
    valid_indices = [i for i, action in enumerate(hl_actions_list) if action != "[SKIP]"]
    
    if not valid_indices:
        return 0.0
    
    valid_obs = [obs_list[i] for i in valid_indices]
    valid_actions = [hl_actions_list[i] for i in valid_indices]
    
    # Compute Q-values with gradients enabled (for training) - chunked batching protocol
    chunk_size = _model_batch_chunk_size(config)
    num_chunks = (len(valid_obs) + chunk_size - 1) // chunk_size
    q_current_chunks = []
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(valid_obs))
        
        chunk_obs = valid_obs[start_idx:end_idx]
        chunk_actions = valid_actions[start_idx:end_idx]
        
        chunk_q_values = value_function.predict_q_value(
            observations=chunk_obs,
            high_level_actions=chunk_actions,
            tokenizer=tokenizer,
            requires_grad=True  # Training: gradients needed
        )
        
        # For training: keep on GPU with gradients intact (don't detach!)
        # We can't move to CPU because that would break the computation graph
        q_current_chunks.append(chunk_q_values)
        
        # Aggressive cleanup after each chunk (but keep q_values for gradient computation)
        del chunk_obs, chunk_actions
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Removed synchronize() for performance - cleanup happens asynchronously
    
    # Concatenate all chunks (all on GPU, gradients preserved)
    if q_current_chunks:
        q_current_batch = torch.cat(q_current_chunks, dim=0)
        del q_current_chunks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        q_current_batch = torch.tensor([], device=value_function.device, dtype=standard_dtype)
    
    # Create full batch with [SKIP] candidates assigned 0.0
    q_current_full = torch.zeros(len(hl_actions_list), device=value_function.device, dtype=standard_dtype)
    for valid_idx, original_idx in enumerate(valid_indices):
        q_current_full[original_idx] = q_current_batch[valid_idx]
    
    use_sequence_reward_targets = getattr(config, "use_hierarchical_agent", False)
    n_trans = len(transitions['observations'])

    if use_sequence_reward_targets:
        # Hierarchical mode mirrors ../LLM sequence critic behavior:
        # train high-level values directly toward sequence rewards.
        q_targets_tensor = torch.tensor(
            transitions['rewards'],
            device=value_function.device,
            dtype=standard_dtype,
        )
        v_next_values = {i: 0.0 for i in range(n_trans)}
    else:
        # 2. Compute V(o_{t+1}) for next states (state value function)
        # This is much faster than generating candidates and computing max Q!
        # V(s) = max_a Q(s,a) under optimal policy, so we can use V directly
        next_obs_list = []
        next_obs_indices = []
        
        for i, (next_obs, terminal) in enumerate(zip(transitions['next_observations'], transitions['terminals'])):
            if not terminal and next_obs:
                next_obs_list.append(next_obs)
                next_obs_indices.append(i)
        
        v_next_values = {i: 0.0 for i in range(n_trans)}
        if next_obs_list:
            # Compute V-values for all next states in batch - chunked batching protocol
            chunk_size = _model_batch_chunk_size(config)
            if chunk_size <= 0:
                chunk_size = len(next_obs_list)
            
            v_value_chunks = []
            for start in range(0, len(next_obs_list), chunk_size):
                end = min(start + chunk_size, len(next_obs_list))
                chunk_obs = next_obs_list[start:end]
                
                chunk_v_values = value_function.predict_v_value(
                    observations=chunk_obs,
                    tokenizer=tokenizer,
                    requires_grad=False
                )
                
                v_value_chunks.append(chunk_v_values)
                
                # Cleanup after each chunk to free memory
                del chunk_obs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if v_value_chunks:
                v_values_flat = torch.cat(v_value_chunks, dim=0)
                # Map back to original indices and convert to Python floats immediately
                for local_idx, global_idx in enumerate(next_obs_indices):
                    v_next_values[global_idx] = v_values_flat[local_idx].item()
                # Cleanup the concatenated tensor
                del v_values_flat, v_value_chunks
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        # 3. Compute targets for Q: r_t + γ * V(o_{t+1})
        # Using V-function instead of max Q for efficiency
        q_targets = []
        for i, (reward_val, terminal) in enumerate(zip(transitions['rewards'], transitions['terminals'])):
            if terminal:
                v_next = 0.0
            else:
                v_next = v_next_values[i]
            
            target = reward_val + config.discount_factor * v_next
            q_targets.append(target)
        
        q_targets_tensor = torch.tensor(q_targets, device=value_function.device, dtype=standard_dtype)
    
    # 4. Compute V(o_t) predictions and targets
    # V(o_t) target = r_t + γ * V(o_{t+1})  (by Bellman equation)
    # This is the same as Q-target! We don't need to compute E_a~π[Q(o_t, a)] explicitly.
    # 
    # Math: V(o_t) = E_a~π[Q(o_t, a)] = E_a~π[r_t + γ * V(o_{t+1})] = r_t + γ * V(o_{t+1})
    # Since r_t and V(o_{t+1}) don't depend on the action taken.
    
    # Compute V(o_t) predictions only for valid observations - chunked batching protocol
    # Use valid_obs instead of obs_list to avoid computing V-values for [SKIP] actions
    chunk_size = _model_batch_chunk_size(config)
    if chunk_size <= 0:
        chunk_size = len(valid_obs)
    
    v_current_chunks = []
    for start in range(0, len(valid_obs), chunk_size):
        end = min(start + chunk_size, len(valid_obs))
        chunk_obs = valid_obs[start:end]
        
        chunk_v_values = value_function.predict_v_value(
            observations=chunk_obs,
            tokenizer=tokenizer,
            requires_grad=True  # Training: gradients needed
        )
        v_current_chunks.append(chunk_v_values)
        
        # Cleanup after each chunk
        del chunk_obs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    if v_current_chunks:
        v_current_predictions = torch.cat(v_current_chunks, dim=0)
        del v_current_chunks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        v_current_predictions = torch.tensor([], device=value_function.device, dtype=standard_dtype)
    
    # Map V-predictions back to original indices (same as Q-values)
    # Create a full tensor with zeros for invalid indices
    v_current_predictions_full = torch.zeros(
        len(obs_list), 
        device=value_function.device, 
        dtype=standard_dtype
    )
    for local_idx, global_idx in enumerate(valid_indices):
        v_current_predictions_full[global_idx] = v_current_predictions[local_idx]
    # Cleanup intermediate tensor before reassignment
    del v_current_predictions
    v_current_predictions = v_current_predictions_full
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # V-target is the same as Q-target: r_t + γ * V(o_{t+1})
    # We already computed this above as q_targets_tensor!
    v_targets_tensor = q_targets_tensor  # Reuse the same target
    
    # 5. Compute contrastive loss if enabled
    contrastive_loss_tensor = None
    # Ablation modes:
    # - "crossed_data": Use other episodes' ground truths as negatives (requires ground_truth_pool)
    # - "random_noise": Use random noise as negatives (doesn't require ground_truth_pool)
    # - "none": No contrastive loss
    if contrastive_coef > 0.0 and ll_model and ll_tokenizer and ablation_mode != "none":
        # Get ground truth goals for current transitions (if available)
        current_ground_truths = transitions.get('ground_truth_goals', [])
        
        # Sample negative beliefs based on ablation mode
        if ablation_mode == "random_noise":
            # Generate random noise for all transitions (no data requirements)
            negative_beliefs = generate_random_noise_beliefs(len(obs_list))
        elif ablation_mode == "crossed_data":
            # For crossed_data mode, we need both current_ground_truths and ground_truth_pool
            # Fail fast and loudly if required data is missing (per research codebase rules)
            if not current_ground_truths:
                raise ValueError(
                    f"crossed_data ablation mode requires current_ground_truths, but got empty list. "
                    f"transitions keys: {list(transitions.keys())}"
                )
            if len(current_ground_truths) != len(obs_list):
                raise ValueError(
                    f"crossed_data ablation mode: len(current_ground_truths)={len(current_ground_truths)} "
                    f"!= len(obs_list)={len(obs_list)}. Mismatch in batch sizes."
                )
            if not ground_truth_pool:
                raise ValueError(
                    f"crossed_data ablation mode requires ground_truth_pool, but got empty/None. "
                    f"This should be provided when calling update_q_function_online."
                )
            
            # Sample from other episodes' ground truths
            negative_beliefs = sample_negative_beliefs(
                current_ground_truths=current_ground_truths,
                negative_pool=ground_truth_pool,
                model=value_function.model,  # Use the same model for embeddings
                tokenizer=tokenizer,
                device=value_function.device,
                ablation_mode=ablation_mode
            )
        else:
            raise ValueError(f"Unknown ablation_mode: {ablation_mode}. Expected 'random_noise', 'crossed_data', or 'none'.")
        
        # Filter to only valid transitions (non-[SKIP] actions)
        valid_negative_beliefs = [negative_beliefs[i] for i in valid_indices]
        valid_observations = valid_obs
        valid_positive_beliefs = valid_actions
        valid_dataset_actions = [transitions['low_level_actions'][i] for i in valid_indices]
        
        # Compute contrastive loss
        contrastive_loss_tensor = value_function.compute_contrastive_loss(
            observations=valid_observations,
            positive_beliefs=valid_positive_beliefs,
            negative_beliefs=valid_negative_beliefs,
            dataset_actions=valid_dataset_actions,
            ll_model=ll_model,
            ll_tokenizer=ll_tokenizer,
            contrastive_coef=contrastive_coef,
            belief_only=getattr(config, 'll_action_belief_only', True)
        )
    
    # 6. Compute avg_entropy (needed for losses)
    avg_entropy = sum(transitions.get('entropies', [0.0])) / len(transitions['observations']) if transitions.get('entropies') else 0.0

    # 7. Regret critic: Q_min, V_min, Regret updates (when use_regret_critic)
    q_min_loss = None
    v_min_loss = None
    regret_loss = None
    config._last_regret_training_metrics = None
    if getattr(config, 'use_regret_critic', False) and value_function.use_regret_critic:
        gamma = config.discount_factor
        # Q_min(s,a), V_min(s), Regret(s,a) for current states
        q_min_chunks = []
        v_min_chunks = []
        regret_chunks = []
        for start in range(0, len(valid_obs), chunk_size):
            end = min(start + chunk_size, len(valid_obs))
            c_obs = valid_obs[start:end]
            c_actions = valid_actions[start:end]
            q_min_chunks.append(value_function.predict_q_min_value(c_obs, c_actions, tokenizer, requires_grad=True))
            v_min_chunks.append(value_function.predict_v_min_value(c_obs, tokenizer, requires_grad=True))
            regret_chunks.append(value_function.predict_regret_value(c_obs, c_actions, tokenizer, requires_grad=True))
            del c_obs, c_actions
        q_min_current = torch.cat(q_min_chunks, dim=0)
        v_min_current = torch.cat(v_min_chunks, dim=0)
        regret_current = torch.cat(regret_chunks, dim=0)
        del q_min_chunks, v_min_chunks, regret_chunks

        # Zero-sum target construction:
        # Q_min target uses sampled min over nominal Q-values on the candidate set.
        filter_noise = (config.contrastive_ablation_mode == "noise_in_candidates")
        belief_candidates_by_transition = transitions['belief_candidates']
        if len(belief_candidates_by_transition) != len(obs_list):
            raise ValueError(
                f"regret zero-sum requires belief_candidates aligned with observations. "
                f"len(belief_candidates)={len(belief_candidates_by_transition)} len(observations)={len(obs_list)}"
            )

        sampled_min_q_current = {}
        for global_idx in valid_indices:
            obs_text = obs_list[global_idx]
            candidate_set = belief_candidates_by_transition[global_idx]
            filtered_candidates = _filter_candidates_for_value_min(candidate_set, filter_noise)
            if not filtered_candidates:
                raise ValueError(
                    f"regret zero-sum current-state min requires at least one valid candidate. "
                    f"transition_index={global_idx} candidate_count={len(candidate_set)}"
                )
            q_candidates = value_function.predict_q_value(
                observations=[obs_text] * len(filtered_candidates),
                high_level_actions=filtered_candidates,
                tokenizer=tokenizer,
                requires_grad=False,
            )
            sampled_min_q_current[global_idx] = torch.min(q_candidates).item()
            del q_candidates

        q_min_targets_tensor = torch.zeros(len(hl_actions_list), device=value_function.device, dtype=standard_dtype)
        for global_idx in valid_indices:
            q_min_targets_tensor[global_idx] = sampled_min_q_current[global_idx]

        # Next state: (Q_value, Regret)(s', a) for same action a and sampled-min Q(s', .) for zero-sum min branch.
        next_obs_for_valid = [transitions['next_observations'][i] for i in valid_indices]
        next_terminals = [transitions['terminals'][i] for i in valid_indices]
        q_value_next = {i: 0.0 for i in valid_indices}
        q_min_next = {i: 0.0 for i in valid_indices}
        regret_next = {i: 0.0 for i in valid_indices}
        # Q_min(s', executed high-level action) for non-terminal steps; used if next-state belief
        # sampling yields only [SKIP] candidates (no multiset to take a sampled min over).
        q_min_next_executed_action = {}
        compute_mask = [not next_terminals[j] and next_obs_for_valid[j] for j in range(len(valid_indices))]
        obs_to_compute = [next_obs_for_valid[j] for j in range(len(valid_indices)) if compute_mask[j]]
        actions_to_compute = [valid_actions[j] for j in range(len(valid_indices)) if compute_mask[j]]
        indices_to_compute = [valid_indices[j] for j in range(len(valid_indices)) if compute_mask[j]]
        if obs_to_compute:
            for start in range(0, len(obs_to_compute), chunk_size):
                end = min(start + chunk_size, len(obs_to_compute))
                c_obs = obs_to_compute[start:end]
                c_actions = actions_to_compute[start:end]
                v_min_vals = value_function.predict_v_min_value(c_obs, tokenizer, requires_grad=False)
                q_val_vals = value_function.predict_q_value(c_obs, c_actions, tokenizer, requires_grad=False)
                q_min_vals = value_function.predict_q_min_value(c_obs, c_actions, tokenizer, requires_grad=False)
                reg_vals = value_function.predict_regret_value(c_obs, c_actions, tokenizer, requires_grad=False)
                for k in range(len(c_obs)):
                    global_idx = indices_to_compute[start + k]
                    q_value_next[global_idx] = q_val_vals[k].item()
                    regret_next[global_idx] = reg_vals[k].item()
                    q_min_next_executed_action[global_idx] = q_min_vals[k].item()
                del c_obs, c_actions, v_min_vals, q_val_vals, q_min_vals, reg_vals

        # Next-state sampled minima are computed over freshly generated next-state candidates.
        # Same retry pattern as batch_generate_beliefs_for_episodes (MAX_BELIEF_ATTEMPTS total rolls).
        non_terminal_valid_indices = [idx for idx in valid_indices if (not transitions['terminals'][idx] and transitions['next_observations'][idx])]
        if non_terminal_valid_indices:
            NEXT_STATE_BELIEF_ATTEMPTS = 3  # aligns with MAX_BELIEF_ATTEMPTS in batch_generate_beliefs_for_episodes
            n_next_ctx = len(non_terminal_valid_indices)
            next_histories_full = [[("[NO_AGENT_ACTION]", transitions['next_observations'][idx])] for idx in non_terminal_valid_indices]
            next_state_candidates = hl_agent.generate_candidate_beliefs_batch(
                chunk_size=config.batch_generation_chunk_size,
                histories=next_histories_full,
                n_candidates=config.n_candidates,
                temperature=config.belief_gen_temperature,
                max_new_tokens=config.max_tokens,
            )
            if len(next_state_candidates) != n_next_ctx:
                raise ValueError(
                    f"next-state candidate generation mismatch: got {len(next_state_candidates)} candidate sets "
                    f"for {n_next_ctx} next observations."
                )
            still_all_skip_local = [
                loc
                for loc in range(n_next_ctx)
                if not _filter_candidates_for_value_min([c.summary for c in next_state_candidates[loc]], filter_noise)
            ]
            used_next_state_retry = False
            if still_all_skip_local:
                print(
                    f"[DEBUG] Next-state regret min: all-[SKIP] for {len(still_all_skip_local)}/{n_next_ctx} "
                    "transition(s). Applying batched retries like episode belief generation."
                )
                sys.stdout.flush()

            for attempt in range(1, NEXT_STATE_BELIEF_ATTEMPTS):
                if not still_all_skip_local:
                    break
                used_next_state_retry = True
                retry_histories = [[("[NO_AGENT_ACTION]", transitions['next_observations'][non_terminal_valid_indices[loc]])] for loc in still_all_skip_local]
                retry_batch = hl_agent.generate_candidate_beliefs_batch(
                    chunk_size=config.batch_generation_chunk_size,
                    histories=retry_histories,
                    n_candidates=config.n_candidates,
                    temperature=config.belief_gen_temperature,
                    max_new_tokens=config.max_tokens,
                )
                if len(retry_batch) != len(still_all_skip_local):
                    raise ValueError(
                        f"next-state retry mismatch: batch {len(retry_batch)} rows for "
                        f"{len(still_all_skip_local)} retry histories."
                    )
                print(
                    f"[DEBUG] Next-state regret min: batched retry {attempt + 1}/{NEXT_STATE_BELIEF_ATTEMPTS}"
                    f" for {len(still_all_skip_local)} transition(s) still all-[SKIP]."
                )
                sys.stdout.flush()
                for rpos, loc in enumerate(still_all_skip_local):
                    next_state_candidates[loc] = retry_batch[rpos]
                del retry_batch

                still_all_skip_local = [
                    loc
                    for loc in range(n_next_ctx)
                    if not _filter_candidates_for_value_min([c.summary for c in next_state_candidates[loc]], filter_noise)
                ]
            if used_next_state_retry and not still_all_skip_local:
                print(
                    "[WARNING] Next-state regret min: retries succeeded; candidate sets "
                    "now contain at least one non-[SKIP] belief."
                )
                sys.stdout.flush()
            fallback_globals = []
            for local_idx, global_idx in enumerate(non_terminal_valid_indices):
                next_obs_text = transitions['next_observations'][global_idx]
                candidate_objs = next_state_candidates[local_idx]
                candidate_texts = [candidate.summary for candidate in candidate_objs]
                filtered_candidates = _filter_candidates_for_value_min(candidate_texts, filter_noise)
                if not filtered_candidates:
                    q_min_next[global_idx] = q_min_next_executed_action[global_idx]
                    fallback_globals.append(global_idx)
                    continue
                q_min_candidates_next = value_function.predict_q_min_value(
                    observations=[next_obs_text] * len(filtered_candidates),
                    high_level_actions=filtered_candidates,
                    tokenizer=tokenizer,
                    requires_grad=False,
                )
                q_min_next[global_idx] = torch.min(q_min_candidates_next).item()
                del q_min_candidates_next
            if fallback_globals:
                print(
                    f"[WARNING] Next-state regret min: after {NEXT_STATE_BELIEF_ATTEMPTS} belief attempts, "
                    f"{len(fallback_globals)} transition(s) still all-[SKIP]; using Q_min(s', executed HL action): "
                    f"indices={fallback_globals}"
                )
                sys.stdout.flush()

        # Regret target: r_nominal - r_adversarial + gamma * Regret(s',a)
        # r_nominal = Q_value(s,a) - gamma*Q_value(s',a), r_adversarial = Q_min(s,a) - gamma*Q_min(s',a)
        q_value_current = q_current_batch.detach()
        r_nominal = q_value_current - gamma * torch.tensor([q_value_next[valid_indices[i]] for i in range(len(valid_indices))], device=value_function.device, dtype=standard_dtype)
        r_adversarial = q_min_current - gamma * torch.tensor([q_min_next[valid_indices[i]] for i in range(len(valid_indices))], device=value_function.device, dtype=standard_dtype)
        regret_next_tensor = torch.tensor([regret_next[valid_indices[i]] for i in range(len(valid_indices))], device=value_function.device, dtype=standard_dtype)
        regret_targets = (r_nominal - r_adversarial).detach() + gamma * regret_next_tensor
        regret_targets_full = torch.zeros(len(hl_actions_list), device=value_function.device, dtype=standard_dtype)
        regret_current_full = torch.zeros(len(hl_actions_list), device=value_function.device, dtype=standard_dtype)
        for valid_idx, original_idx in enumerate(valid_indices):
            regret_targets_full[original_idx] = regret_targets[valid_idx]
            regret_current_full[original_idx] = regret_current[valid_idx]

        q_min_full = torch.zeros(len(hl_actions_list), device=value_function.device, dtype=standard_dtype)
        v_min_full = torch.zeros(len(hl_actions_list), device=value_function.device, dtype=standard_dtype)
        for valid_idx, original_idx in enumerate(valid_indices):
            q_min_full[original_idx] = q_min_current[valid_idx]
            v_min_full[original_idx] = v_min_current[valid_idx]
        v_min_targets_tensor = q_min_targets_tensor

        q_min_loss = value_function.compute_value_loss(
            value_predictions=q_min_full,
            targets=q_min_targets_tensor,
            avg_entropy=avg_entropy,
            entropy_coef=config.entropy_coef,
            contrastive_loss=None
        )
        v_min_loss = value_function.compute_value_loss(
            value_predictions=v_min_full,
            targets=v_min_targets_tensor,
            avg_entropy=avg_entropy,
            entropy_coef=config.entropy_coef,
            contrastive_loss=None
        )
        regret_loss = value_function.compute_value_loss(
            value_predictions=regret_current_full,
            targets=regret_targets_full,
            avg_entropy=avg_entropy,
            entropy_coef=config.entropy_coef,
            contrastive_loss=None
        )
        q_min_target_values = q_min_targets_tensor[valid_indices]
        config._last_regret_training_metrics = {
            "q_min_loss": q_min_loss.item(),
            "q_min_target_variance": q_min_target_values.var(unbiased=False).item(),
            "q_min_prediction_variance": q_min_current.var(unbiased=False).item(),
            "q_min_target_mean": q_min_target_values.mean().item(),
            "q_min_prediction_mean": q_min_current.mean().item(),
        }
        if config.debug:
            with torch.no_grad():
                q_nom_mean = q_current_batch.mean().item()
                q_min_pred_mean = q_min_current.mean().item()
                q_min_target_mean = q_min_targets_tensor.mean().item()
                regret_gap_mean = (q_current_batch - q_min_current).mean().item()
                print(
                    "[Regret Zero-Sum] "
                    f"q_mean={q_nom_mean:.4f} "
                    f"q_min_pred_mean={q_min_pred_mean:.4f} "
                    f"q_min_target_mean={q_min_target_mean:.4f} "
                    f"gap_mean={regret_gap_mean:.4f}"
                )

    # 8. Compute losses and update

    # Q-function loss
    q_loss = value_function.compute_value_loss(
        value_predictions=q_current_full,
        targets=q_targets_tensor,
        avg_entropy=avg_entropy,
        entropy_coef=config.entropy_coef,
        contrastive_loss=contrastive_loss_tensor
    )

    # V-function loss (if we have predictions and targets)
    v_loss = None
    v_loss = value_function.compute_value_loss(
        value_predictions=v_current_predictions,
        targets=v_targets_tensor,
        avg_entropy=avg_entropy,
        entropy_coef=config.entropy_coef,
        contrastive_loss=None  # No contrastive loss for V-function
    )

    # Combine losses (weight V loss equally with Q loss)
    if v_loss is not None:
        total_loss = q_loss + v_loss
    else:
        total_loss = q_loss
    if q_min_loss is not None:
        total_loss = total_loss + q_min_loss + v_min_loss + regret_loss
    
    # Compute marginal token rewards if enabled
    token_critic_loss = None
    if getattr(config, 'use_marginal_token_rewards', False):
        from .value.marginal_rewards_integration import compute_and_apply_marginal_rewards

        valid_base_states = list(valid_obs)
        instructions = valid_actions

        environments = None
        dialogue_histories = None
        if 'environments' in transitions:
            env_list = transitions['environments']
            environments = [env_list[i] for i in valid_indices]
        if 'dialogue_histories' in transitions:
            hist_list = transitions['dialogue_histories']
            dialogue_histories = [hist_list[i] for i in valid_indices]
        
        # Use reward model: prefer passed-in instance, else load only if environments unavailable
        _reward_model = reward_model
        _reward_tokenizer = reward_tokenizer
        if not environments and hasattr(config, 'reward_model_name') and config.reward_model_name:
            if _reward_model is None or _reward_tokenizer is None:
                from .value.cares_reward import (
                    load_llamaguard,
                    load_shieldgemma,
                    normalize_reward_model_type,
                )
                try:
                    _rmt = normalize_reward_model_type(
                        getattr(config, "reward_model_type", None),
                        config.reward_model_name,
                    )
                    if _rmt == "shieldgemma":
                        _reward_model, _reward_tokenizer = load_shieldgemma(
                            model_name=config.reward_model_name,
                            device=config.device,
                            use_bf16=config.use_bf16,
                        )
                    else:
                        _reward_model, _reward_tokenizer = load_llamaguard(
                            model_name=config.reward_model_name,
                            device=config.device,
                            use_bf16=config.use_bf16,
                        )
                except Exception as e:
                    print(f"Warning: Could not load reward model {config.reward_model_name}: {e}")
                    print("Falling back to environment rewards or sequence critic for marginal rewards.")
                else:
                    _rmt_done = normalize_reward_model_type(
                        getattr(config, "reward_model_type", None),
                        config.reward_model_name,
                    )
                    if _rmt_done == "llamaguard":
                        safe_tokens = _reward_tokenizer.encode("safe", add_special_tokens=False)
                        unsafe_tokens = _reward_tokenizer.encode("unsafe", add_special_tokens=False)
                        safe_id = safe_tokens[0] if safe_tokens else None
                        unsafe_id = unsafe_tokens[0] if unsafe_tokens else None
                        if safe_id is None or unsafe_id is None:
                            raise ValueError("Could not locate safety token IDs in guard tokenizer vocabulary")
                        if safe_id != 19193:
                            raise ValueError(f"Expected safe token id 19193 but found {safe_id}. Please check the guard model and tokenizer.")
                        print(f"Initialized safety token IDs: safe={safe_id}, unsafe={unsafe_id}")
        
        rewards_list = transitions['rewards']
        actual_rewards_list = [rewards_list[i] for i in valid_indices]
        ll_actions_list = transitions['low_level_actions']

        valid_ll_actions = []
        valid_indices_for_marginal = []
        for idx_pos, orig_idx in enumerate(valid_indices):
            ll_action = ll_actions_list[orig_idx]
            if ll_action and ll_action.strip():
                valid_ll_actions.append(ll_action)
                valid_indices_for_marginal.append(idx_pos)

        marginal_stats = {}
        if valid_ll_actions:
            valid_obs_for_marginal = [valid_obs[i] for i in valid_indices_for_marginal]
            valid_base_states_for_marginal = [valid_base_states[i] for i in valid_indices_for_marginal]
            instructions_for_marginal = [instructions[i] for i in valid_indices_for_marginal]
            actual_rewards_for_marginal_nom = [actual_rewards_list[i] for i in valid_indices_for_marginal]
            environments_for_marginal = [environments[i] for i in valid_indices_for_marginal] if environments else None
            dialogue_histories_for_marginal = [dialogue_histories[i] for i in valid_indices_for_marginal] if dialogue_histories else None

            token_critic_loss, marginal_stats = compute_and_apply_marginal_rewards(
                value_function=value_function,
                actions=valid_ll_actions,
                states=valid_obs_for_marginal,
                base_states=valid_base_states_for_marginal,
                tokenizer=tokenizer,
                config=config,
                reward_model=_reward_model,
                reward_tokenizer=_reward_tokenizer,
                environments=environments_for_marginal,
                dialogue_histories=dialogue_histories_for_marginal,
                actual_rewards=actual_rewards_for_marginal_nom,
                instructions=instructions_for_marginal,
                device=value_function.device
            )

        if token_critic_loss is not None:
            total_loss = total_loss + token_critic_loss

            if config.debug:
                print(
                    f"[Marginal Rewards] Processed {marginal_stats['num_actions_processed']} actions, "
                    f"{marginal_stats['total_tokens_processed']} tokens, "
                    f"loss: {marginal_stats['token_critic_loss']:.4f}"
                )
    
    value_function.update(total_loss)
    
    # Cleanup: After backward pass, delete loss and Q-value tensors to free computation graph
    # The gradients have been computed and applied, so we can safely delete these
    loss_value = total_loss.item()
    del total_loss, q_loss, q_current_full, q_current_batch
    if v_loss is not None:
        del v_loss, v_current_predictions, v_targets_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return loss_value


def _is_online_mode(config) -> bool:
    """Check if we're in online mode based on environment_type."""
    if config.environment_type == "cares":
        return getattr(config, "cares_online", True)
    if config.environment_type == "wildjailbreak":
        return getattr(config, "wildjailbreak_online", True)
    if config.environment_type == "redbench":
        return getattr(config, "redbench_online", True)
    if config.environment_type == "harmbench":
        return getattr(config, "harmbench_online", True)
    return config.environment_type in ["multiwoz_online", "vitabench", "userbench"]


def _model_batch_chunk_size(config):
    """Chunked batching protocol: all model batch calls use the same effective chunk size so memory is predictable."""
    return min(
        getattr(config, "q_value_chunk_size", 16),
        getattr(config, "batch_generation_chunk_size", 16),
    )


def main(config_path: Optional[str] = None, output_dir: Optional[str] = None):
    """
    Main training loop for offline reinforcement learning.
    Args:
        config_path: Path to config JSON file (if None, uses defaults)
        output_dir: Output directory (if None, uses config default)
    """
    # Load configuration
    if config_path:
        config = BaseConfig.from_json(config_path)
    else:
        config = BaseConfig()

    if config.baseline_mode and config.defender_backend == "smoothllm":
        config.use_bf16 = True
        print("[SmoothLLM Baseline] Forcing BF16 model loading for all local models.")

    evaluation_mode = os.environ.get("LLM_CONTEXT_EVAL_MODE", "0") == "1"
    config._evaluation_mode = evaluation_mode
    if evaluation_mode:
        print("[Evaluation Mode] Gradient-based value updates are disabled.")
        # Reduce chunk sizes in evaluation mode to prevent OOM (no gradients needed)
        if hasattr(config, 'q_value_chunk_size'):
            original_q_chunk = config.q_value_chunk_size
            config.q_value_chunk_size = min(config.q_value_chunk_size, 4)  # Cap at 4 for eval (more aggressive)
            if config.q_value_chunk_size != original_q_chunk:
                print(f"[Evaluation Mode] Reduced q_value_chunk_size from {original_q_chunk} to {config.q_value_chunk_size} to prevent OOM")
        if hasattr(config, 'batch_generation_chunk_size'):
            original_gen_chunk = config.batch_generation_chunk_size
            config.batch_generation_chunk_size = min(config.batch_generation_chunk_size, 8)  # Cap at 8 for eval
            if config.batch_generation_chunk_size != original_gen_chunk:
                print(f"[Evaluation Mode] Reduced batch_generation_chunk_size from {original_gen_chunk} to {config.batch_generation_chunk_size} to prevent OOM")
        if hasattr(config, 'episode_batch_size'):
            original_ep_batch = config.episode_batch_size
            config.episode_batch_size = min(config.episode_batch_size, 16)  # Cap at 16 for eval (more aggressive)
            if config.episode_batch_size != original_ep_batch:
                print(f"[Evaluation Mode] Reduced episode_batch_size from {original_ep_batch} to {config.episode_batch_size} to prevent OOM")

    # Dynamically scale batch sizes based on GPU memory capacity
    if config.device.startswith("cuda"):
        # Get GPU memory capacity
        if torch.cuda.is_available():
            # Check primary GPU (cuda:0) and secondary GPU (cuda:1) if available
            # Use minimum memory to be conservative
            device_ids_to_check = [0]
            if torch.cuda.device_count() > 1:
                device_ids_to_check.append(1)
            
            gpu_memories_gb = []
            for device_id in device_ids_to_check:
                try:
                    memory_gb = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
                    gpu_memories_gb.append(memory_gb)
                except (RuntimeError, IndexError):
                    pass
            
            if gpu_memories_gb:
                # Use minimum memory across GPUs to be conservative
                gpu_memory_gb = min(gpu_memories_gb)
                gpu_info = f"GPU 0: {gpu_memories_gb[0]:.1f}GB"
                if len(gpu_memories_gb) > 1:
                    gpu_info += f", GPU 1: {gpu_memories_gb[1]:.1f}GB"
                
                # Assume 40GB is baseline, scale up if we have more
                baseline_memory_gb = 40.0
                if gpu_memory_gb >= 70.0:  # 80GB card (with some margin)
                    scale_factor = 2.0
                    print(f"[GPU Memory Scaling] Detected {gpu_info} (min: {gpu_memory_gb:.1f}GB >=70GB), scaling batch sizes by {scale_factor}x")
                elif gpu_memory_gb >= 45.0:  # Between 40-70GB, scale proportionally
                    scale_factor = 1.0 + (gpu_memory_gb - baseline_memory_gb) / baseline_memory_gb
                    print(f"[GPU Memory Scaling] Detected {gpu_info} (min: {gpu_memory_gb:.1f}GB), scaling batch sizes by {scale_factor:.2f}x")
                else:
                    scale_factor = 1.0
                    print(f"[GPU Memory Scaling] Detected {gpu_info} (min: {gpu_memory_gb:.1f}GB), using baseline batch sizes (no scaling)")
                
                if scale_factor > 1.0:
                    # Scale batch sizes (round to integers)
                    original_episode_batch_size = config.episode_batch_size
                    original_online_batch_size = config.online_batch_size
                    original_transition_prob_chunk_size = config.transition_prob_chunk_size
                    original_q_value_chunk_size = config.q_value_chunk_size
                    original_batch_generation_chunk_size = config.batch_generation_chunk_size
                    
                    config.episode_batch_size = max(1, int(original_episode_batch_size * scale_factor))
                    config.online_batch_size = max(1, int(original_online_batch_size * scale_factor))
                    config.transition_prob_chunk_size = max(1, int(original_transition_prob_chunk_size * scale_factor))
                    config.q_value_chunk_size = max(1, int(original_q_value_chunk_size * scale_factor))
                    config.batch_generation_chunk_size = max(1, int(original_batch_generation_chunk_size * scale_factor))
                    
                    print(f"  Scaled batch sizes:")
                    print(f"    episode_batch_size: {original_episode_batch_size} -> {config.episode_batch_size}")
                    print(f"    online_batch_size: {original_online_batch_size} -> {config.online_batch_size}")
                    print(f"    transition_prob_chunk_size: {original_transition_prob_chunk_size} -> {config.transition_prob_chunk_size}")
                    print(f"    q_value_chunk_size: {original_q_value_chunk_size} -> {config.q_value_chunk_size}")
                    print(f"    batch_generation_chunk_size: {original_batch_generation_chunk_size} -> {config.batch_generation_chunk_size}")
                    
                    # Warning for debugging OOM errors
                    print("")
                    print("=" * 80)
                    print("⚠️  WARNING: Batch sizes were dynamically scaled based on GPU memory!")
                    print("=" * 80)
                    print(f"  If you encounter OOM (Out of Memory) errors, remember that batch sizes")
                    print(f"  were automatically increased by {scale_factor:.2f}x due to detecting {gpu_memory_gb:.1f}GB GPU memory.")
                    print(f"  Original config values were scaled up - this may cause OOM if:")
                    print(f"    - GPU memory is fragmented or already in use")
                    print(f"    - Model size or sequence lengths are larger than expected")
                    print(f"    - Multiple processes are sharing the GPU")
                    print(f"  To disable auto-scaling, modify the GPU memory detection logic in src/main.py")
                    print("=" * 80)
                    print("")
            else:
                print("[GPU Memory Scaling] Could not detect GPU memory, using baseline batch sizes")
        else:
            print("[GPU Memory Scaling] CUDA not available, using baseline batch sizes")
    else:
        print("[GPU Memory Scaling] Non-CUDA device, using baseline batch sizes")

    print(f"Config: {config}")
    
    if output_dir:
        config.output_dir = output_dir
    
    # Clear existing experiment files to ensure clean metrics
    output_path = Path(config.output_dir)
    if output_path.exists():
        print(f"Clearing existing experiment files in {output_path}...")
        
        # Clear all belief evaluation CSV files
        csv_files = list(output_path.glob("belief_evaluation_episode_*.csv"))
        for csv_file in csv_files:
            csv_file.unlink()
        if csv_files:
            print(f"  Removed {len(csv_files)} CSV evaluation files")
        
        # Clear all baseline evaluation JSON files
        baseline_files = list(output_path.glob("baseline_eval_*.json"))
        for baseline_file in baseline_files:
            baseline_file.unlink()
        if baseline_files:
            print(f"  Removed {len(baseline_files)} baseline evaluation files")
        
        # Clear contrastive test CSV files
        contrastive_files = list(output_path.glob("contrastive_test_*.csv"))
        for contrastive_file in contrastive_files:
            contrastive_file.unlink()
        if contrastive_files:
            print(f"  Removed {len(contrastive_files)} contrastive test files")
        
        # Clear all_metrics.jsonl (delete and recreate empty file)
        metrics_file = output_path / "all_metrics.jsonl"
        if metrics_file.exists():
            metrics_file.unlink()
            print(f"  Removed existing all_metrics.jsonl")
        
        # Clear P(a|b) diagnostic files
        diagnostic_files = list(output_path.glob("p_action_belief_diagnostics.jsonl"))
        for diagnostic_file in diagnostic_files:
            diagnostic_file.unlink()
        if diagnostic_files:
            print(f"  Removed {len(diagnostic_files)} diagnostic files")
        
        defense_log = output_path / DEFENSE_EPISODES_FILENAME
        if defense_log.exists():
            defense_log.unlink()
            print(f"  Removed existing {DEFENSE_EPISODES_FILENAME}")
        
        print("  Experiment directory cleared - starting fresh run")
    else:
        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize models
    # Main model on GPU 0 (or specified device) - ALWAYS loaded for:
    # 1. Value function (critic) - needs local model for hidden states
    # 2. Probability computation (log_probs for belief updates) - needs local model
    # Even when use_gpt_for_agents=True, we still need the local model for these purposes
    model = get_model_instance(
        model_name=config.model_name,
        device=config.device if config.device != "cuda" else "cuda:0",
        use_bf16=config.use_bf16
    )
    
    # User model: if different model name requested, load separately on GPU 1
    # Otherwise, use main model singleton (single LLM)
    user_model_name = config.user_model_name if config.user_model_name else config.model_name
    user_model = get_user_model_instance(
        user_model_name=user_model_name,
        main_model_name=config.model_name,
        device="cuda:1",
        use_bf16=config.use_bf16,
        fallback_to_main=True
    )
    
    # Load tokenizer and agents conditionally:
    # - For multiwoz_online: need user_agent before episode initialization
    # - For cares/wildjailbreak/redbench/harmbench: need patient_agent and LlamaGuard before episode initialization
    # - For other modes: can load after episode initialization to avoid fork warning
    needs_tokenizer_before_episodes = (
        config.environment_type == "multiwoz_online"
        or config.environment_type == "cares"
        or config.environment_type == "wildjailbreak"
        or config.environment_type == "redbench"
        or config.environment_type == "harmbench"
    )
    
    if needs_tokenizer_before_episodes:
        tokenizer = get_tokenizer_instance(config.model_name)
        judge_model, judge_tokenizer = get_judge_model_and_tokenizer(
            model_name=config.model_name,
            device=config.device if config.device != "cuda" else "cuda:0",
            use_bf16=config.use_bf16,
        )
        
        # Initialize prompt manager
        prompt_manager = PromptManager()
        hl_enable_thinking = config.hl_enable_thinking
        ll_enable_thinking = config.ll_enable_thinking
        
        # Initialize agents (use GPT agents if configured, otherwise local agents)
        if getattr(config, 'use_gpt_for_agents', False):
            gpt_model_name = getattr(config, 'gpt_agent_model', 'gpt-4o-mini')
            gpt_api_key = getattr(config, 'gpt_agent_api_key', None)
            gpt_reasoning_effort = getattr(config, 'gpt_reasoning_effort', None)
            hl_agent = GPTHighLevelAgent(
                tokenizer, prompt_manager, 
                template_name=config.belief_gen_template,
                model_name=gpt_model_name,
                api_key=gpt_api_key,
                reasoning_effort=gpt_reasoning_effort,
            )
            ll_agent = GPTLowLevelAgent(
                tokenizer, prompt_manager,
                model_name=gpt_model_name,
                api_key=gpt_api_key,
                reasoning_effort=gpt_reasoning_effort,
            )
            print(f"[INFO] Using GPT agents (model: {gpt_model_name}, reasoning_effort: {gpt_reasoning_effort}) for high-level and low-level agents")
        else:
            if config.use_hierarchical_agent and config.high_level_policy_type == "freeform":
                hl_agent = FreeformHighLevelAgent(
                    model,
                    tokenizer,
                    prompt_manager,
                    template_name=config.belief_gen_template,
                    enable_thinking=hl_enable_thinking,
                )
            else:
                hl_agent = HighLevelAgent(
                    model,
                    tokenizer,
                    prompt_manager,
                    template_name=config.belief_gen_template,
                    enable_thinking=hl_enable_thinking,
                )
            ll_agent = LowLevelAgent(
                model,
                tokenizer,
                prompt_manager,
                enable_thinking=ll_enable_thinking,
            )
        
        # Initialize user agent (use GPT if configured, otherwise local model) - only for multiwoz_online
        if config.environment_type == "multiwoz_online":
            if getattr(config, 'use_gpt_for_user', False):
                gpt_model_name = getattr(config, 'gpt_agent_model', 'gpt-4o-mini')
                gpt_api_key = getattr(config, 'gpt_agent_api_key', None)
                user_agent = GPTUserAgent(
                    tokenizer, prompt_manager,
                    model_name=gpt_model_name,
                    api_key=gpt_api_key
                )
                print(f"[INFO] Using GPT user agent (model: {gpt_model_name})")
            else:
                user_agent = UserAgent(user_model, tokenizer, prompt_manager)  # Use user_model on GPU 1
        else:
            user_agent = None

        if config.environment_type in ("cares", "wildjailbreak", "redbench", "harmbench"):
            from .agents.patient_agent import PatientAgent
            from .value.cares_reward import load_llamaguard, load_shieldgemma, normalize_reward_model_type

            prompts_path = (
                "src/prompts/wildjailbreak_attacker_prompts.json"
                if config.environment_type in ("wildjailbreak", "redbench", "harmbench")
                else "src/prompts/cares_patient_prompts.json"
            )
            patient_agent = PatientAgent(user_model, tokenizer, prompts_path=prompts_path)
            reward_model_name = getattr(
                config, "reward_model_name", "meta-llama/Llama-Guard-3-8B"
            )
            reward_model_type_eff = normalize_reward_model_type(
                getattr(config, "reward_model_type", None),
                reward_model_name,
            )
            if reward_model_name:
                # Guard on highest-index device: cuda:0 for 1 GPU, cuda:1 for 2 GPUs
                reward_device = f"cuda:{max(0, torch.cuda.device_count() - 1)}"
                if reward_model_type_eff == "shieldgemma":
                    reward_model, reward_tokenizer = load_shieldgemma(
                        model_name=reward_model_name,
                        device=reward_device,
                        use_bf16=config.use_bf16,
                    )
                    print(f"[INFO] Loaded ShieldGemma from {reward_model_name} on {reward_device}")
                    from .utils.guard_safety import (
                        run_shieldgemma_guard_sanity_check,
                        run_shieldgemma_reward_format_sanity_check,
                    )
                    run_shieldgemma_guard_sanity_check(reward_model, reward_tokenizer, reward_device)
                    run_shieldgemma_reward_format_sanity_check(
                        reward_model, reward_tokenizer, reward_device
                    )
                else:
                    reward_model, reward_tokenizer = load_llamaguard(
                        model_name=reward_model_name,
                        device=reward_device,
                        use_bf16=config.use_bf16,
                    )
                    print(f"[INFO] Loaded LlamaGuard from {reward_model_name} on {reward_device}")

                    safe_tokens = reward_tokenizer.encode("safe", add_special_tokens=False)
                    unsafe_tokens = reward_tokenizer.encode("unsafe", add_special_tokens=False)
                    safe_id = safe_tokens[0] if safe_tokens else None
                    unsafe_id = unsafe_tokens[0] if unsafe_tokens else None
                    if safe_id is None or unsafe_id is None:
                        raise ValueError("Could not locate safety token IDs in guard tokenizer vocabulary")
                    if safe_id != 19193:
                        raise ValueError(f"Expected safe token id 19193 but found {safe_id}. Please check the guard model and tokenizer.")
                    print(f"Initialized safety token IDs: safe={safe_id}, unsafe={unsafe_id}")
                    from .utils.guard_safety import (
                        run_guard_sanity_check,
                        run_guard_reward_format_sanity_check,
                        run_fulfillment_judge_sanity_check,
                    )
                    run_guard_sanity_check(reward_model, reward_tokenizer, reward_device)
                    run_guard_reward_format_sanity_check(reward_model, reward_tokenizer, reward_device)
                from .utils.guard_safety import run_fulfillment_judge_sanity_check
                run_fulfillment_judge_sanity_check(
                    judge_model,
                    judge_tokenizer,
                    config.device,
                )
            else:
                reward_model, reward_tokenizer = None, None
                print("[INFO] CARES/WildJailbreak: No reward model specified, using placeholder rewards")
        else:
            patient_agent = None
            reward_model = None
            reward_tokenizer = None
    else:
        # Will be loaded after episode initialization
        tokenizer = None
        prompt_manager = None
        hl_agent = None
        ll_agent = None
        user_agent = None
        patient_agent = None
        reward_model = None
        reward_tokenizer = None
    
    # Determine standard dtype based on bf16 parameter
    standard_dtype = torch.bfloat16 if config.use_bf16 else torch.float32
    
    # Initialize value function (skip if random_belief_selection is True)
    # Note: ValueFunction uses the same model singleton as HL/LL agents, so we cannot
    # parallelize across GPUs. All components share the same model instance.
    random_belief_selection = getattr(config, 'random_belief_selection', False)
    if random_belief_selection:
        value_function = None
        print("[INFO] Random belief selection enabled - skipping value function initialization")
    else:
        value_function_hidden_size = model.config.hidden_size
        value_function = ValueFunction(
            model=model,
            hidden_size=value_function_hidden_size,
            learning_rate=config.learning_rate,
            device=config.device,
            dtype=standard_dtype,
            use_regret_critic=getattr(config, 'use_regret_critic', False),
        )
        
        # Load pretrained checkpoint if provided
        if config.checkpoint_path:
            checkpoint_path = Path(config.checkpoint_path)
            if checkpoint_path.exists():
                print(f"Loading pretrained Q-network from {checkpoint_path}")
                value_function.load_checkpoint(str(checkpoint_path), strict=True)
            else:
                # In evaluation mode with GPT agents, checkpoint might not be needed
                # Warn but don't fail if in evaluation mode
                if getattr(config, "_evaluation_mode", False) and getattr(config, 'use_gpt_for_agents', False):
                    print(f"[WARNING] Checkpoint file not found: {checkpoint_path}")
                    print("[WARNING] Continuing without checkpoint (GPT agents don't require value function training)")
                else:
                    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        else:
            # Save initial checkpoint (before any training) - skip in evaluation mode
            if not getattr(config, "_evaluation_mode", False):
                checkpoint_dir = Path(config.output_dir) / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                initial_checkpoint_path = checkpoint_dir / "value_function_initial.pt"
                value_function.save_checkpoint(str(initial_checkpoint_path))
                print(f"Saved initial checkpoint to {initial_checkpoint_path}")
    
    # Print GPU memory after models are loaded
    device_id = int(config.device.split(":")[-1]) if config.device.startswith("cuda") and ":" in config.device else 0
    print_gpu_memory("GPU Memory - After models loaded", include_max=False, restore_device=device_id)
    
    # Load data
    print("Loading dataset...")
    environment_type = getattr(config, 'environment_type', 'multiwoz')
    
    if environment_type == 'vitabench':
        # Add vitabench directory to Python path
        project_root = Path(__file__).parent.parent
        vitabench_path = project_root / 'vitabench' / 'src'
        if vitabench_path.exists() and str(vitabench_path) not in sys.path:
            sys.path.insert(0, str(vitabench_path))
        
        # Load VitaBench tasks
        from vita.run import load_tasks
        
        task_config = getattr(config, 'task_config', {})
        domain = task_config.get('domain', 'ota')
        task_ids = task_config.get('task_ids', None)
        num_tasks = task_config.get('num_tasks', None)
        language = getattr(config, 'language', 'english')
        
        print(f"Loading VitaBench tasks from domain: {domain}, language: {language}")
        all_tasks = load_tasks(domain, language=language)
        
        # Filter to specific task IDs if provided
        if task_ids:
            tasks = [t for t in all_tasks if t.id in task_ids]
            if len(tasks) != len(task_ids):
                found_ids = {t.id for t in tasks}
                missing_ids = set(task_ids) - found_ids
                raise ValueError(f"Some task IDs not found: {missing_ids}")
        else:
            tasks = list(all_tasks)
        
        # Limit number of tasks if specified
        if num_tasks is not None and num_tasks > 0:
            random.seed(42)
            tasks = random.sample(tasks, min(num_tasks, len(tasks)))
        
        # Convert to list of dicts with 'id' and 'domain' for create_episode_state
        dialogues = [{'id': task.id, 'domain': domain} for task in tasks]
        print(f"Loaded {len(dialogues)} VitaBench tasks")
        
        multiwoz_mode = False
    elif environment_type in ["multiwoz", "multiwoz_offline", "multiwoz_online"]:
        dialogues = load_multiwoz_dataset(
            data_path=config.data_path,
            split="train",
            use_huggingface=True
        )
        raw_dialogue_count = len(dialogues)

        # Filter out dialogues with missing/empty goals
        valid_dialogues: List[MultiWOZDialogue] = []
        skipped_dialogue_ids: List[str] = []
        skipped_no_booking: List[str] = []
        for dialogue in dialogues:
            if not is_valid_multiwoz_goal(dialogue.goal):
                # Extract goal from dialogue state fields (slot_values, requested_slots, active_intent)
                # This processes all turns to capture the full sequence of user desires
                extracted_goal = extract_goal_from_dialogue_state(dialogue)
                if extracted_goal:
                    dialogue.goal = extracted_goal
                else:
                    # Fallback: use legacy inference method if state extraction fails
                    inferred_goal = infer_goal_from_dialogue(dialogue)
                    if inferred_goal:
                        dialogue.goal = inferred_goal

            if not is_valid_multiwoz_goal(dialogue.goal):
                skipped_dialogue_ids.append(dialogue.dialogue_id)
                continue
            
            # Optional: Filter out goals without booking intent (only request_slots, no inform_slots)
            # This helps focus on concrete goals that can be properly rewarded
            filter_info_only = getattr(config, 'filter_info_only_goals', False)
            if filter_info_only and not has_booking_intent(dialogue.goal):
                skipped_no_booking.append(dialogue.dialogue_id)
                continue
            
            valid_dialogues.append(dialogue)

        dialogues = valid_dialogues

        # Limit dataset size if specified (for faster training/experimentation)
        max_dialogues = getattr(config, 'max_dialogues', None)
        if max_dialogues is not None and max_dialogues > 0 and len(dialogues) > max_dialogues:
            random.seed(42)  # Fixed seed for reproducibility
            dialogues = random.sample(dialogues, max_dialogues)
            print(f"Limited to {max_dialogues} dialogues (from {len(valid_dialogues)} valid)", end="")
        else:
            # Subsample for evaluation if specified
            evaluation_mode = getattr(config, "_evaluation_mode", False)
            if evaluation_mode and config.evaluation_sample_size is not None and config.evaluation_sample_size > 0:
                sample_size = min(config.evaluation_sample_size, len(dialogues))
                if sample_size < len(dialogues):
                    dialogues = random.sample(dialogues, sample_size)
                    print(f"Subsampled to {sample_size} dialogues for evaluation", end="")
                else:
                    print(f"Using all {len(dialogues)} dialogues (sample_size >= total)", end="")
            else:
                print(f"Loaded {len(dialogues)} dialogues from MultiWOZ dataset", end="")
        
        if skipped_dialogue_ids or skipped_no_booking:
            parts = []
            if skipped_dialogue_ids:
                parts.append(f"{len(skipped_dialogue_ids)} with missing goals")
            if skipped_no_booking:
                parts.append(f"{len(skipped_no_booking)} without booking intent (info-only)")
            print(f" (filtered out {', '.join(parts)} from {raw_dialogue_count})")
        else:
            print()

        multiwoz_mode = True
    elif environment_type == 'salesagent':
        # Load SalesAgent data
        from .environments.salesagent_helpers import load_salesbot_dataset
        
        if not config.data_path:
            raise ValueError("data_path must be specified for SalesAgent environment_type")
        
        print(f"Loading SalesAgent dataset from {config.data_path}...")
        dialogues = load_salesbot_dataset(config.data_path)
        print(f"Loaded {len(dialogues)} SalesAgent conversations")
        
        # Limit dataset size if specified
        max_dialogues = getattr(config, 'max_dialogues', None)
        if max_dialogues is not None and max_dialogues > 0 and len(dialogues) > max_dialogues:
            random.seed(42)
            dialogues = random.sample(dialogues, max_dialogues)
            print(f"Limited to {max_dialogues} conversations")
        
        multiwoz_mode = False
    elif environment_type == 'userbench':
        # Load UserBench data
        from .environments.userbench_helpers import load_userbench_data
        
        task_config = getattr(config, 'task_config', {})
        env_name = task_config.get('env_name', 'travel22')
        data_root = task_config.get('data_root', None)
        one_choice = task_config.get('one_choice', False)
        
        print(f"Loading UserBench data for environment: {env_name}")
        # Use train split for training (fallback to test if train doesn't exist)
        dialogues = load_userbench_data(env_name, data_root=data_root, one_choice=one_choice, split="train")
        print(f"Loaded {len(dialogues)} UserBench tasks")
        
        # Limit dataset size if specified
        max_dialogues = getattr(config, 'max_dialogues', None)
        if max_dialogues is not None and max_dialogues > 0 and len(dialogues) > max_dialogues:
            random.seed(42)
            dialogues = random.sample(dialogues, max_dialogues)
            print(f"Limited to {max_dialogues} tasks")
        
        multiwoz_mode = False
    elif environment_type == 'cares':
        from .environments.cares_helpers import load_cares_dataset

        split = getattr(config, 'cares_split', 'train')
        print(f"Loading CARES-18K dataset (split={split})...")
        dialogues = load_cares_dataset(split=split)
        print(f"Loaded {len(dialogues)} CARES examples")

        max_dialogues = getattr(config, 'max_dialogues', None)
        if max_dialogues is not None and max_dialogues > 0 and len(dialogues) > max_dialogues:
            random.seed(42)
            dialogues = random.sample(dialogues, max_dialogues)
            print(f"Limited to {max_dialogues} examples")

        multiwoz_mode = False
    elif environment_type == 'wildjailbreak':
        from .environments.wildjailbreak_helpers import load_wildjailbreak_dataset

        split = getattr(config, 'wildjailbreak_split', 'train')
        data_types = getattr(config, 'wildjailbreak_data_types', None)
        print(f"Loading WildJailbreak dataset (split={split})...")
        dialogues = load_wildjailbreak_dataset(
            split=split,
            data_types=data_types,
        )
        print(f"Loaded {len(dialogues)} WildJailbreak examples")

        max_dialogues = getattr(config, 'max_dialogues', None)
        if max_dialogues is not None and max_dialogues > 0 and len(dialogues) > max_dialogues:
            random.seed(42)
            dialogues = random.sample(dialogues, max_dialogues)
            print(f"Limited to {max_dialogues} examples")

        multiwoz_mode = False
    elif environment_type == 'redbench':
        from .environments.redbench_helpers import load_redbench_dataset

        split = getattr(config, 'redbench_split', 'train')
        subsets = getattr(config, 'redbench_subsets', None)
        max_examples = getattr(config, 'redbench_max_examples', None)
        mapping_mode = getattr(config, 'redbench_mapping_mode', 'category')
        refusal_sources = getattr(config, 'redbench_refusal_sources', None)
        print(
            "Loading RedBench dataset "
            f"(split={split}, mapping_mode={mapping_mode})..."
        )
        dialogues = load_redbench_dataset(
            split=split,
            subsets=subsets,
            max_examples=max_examples,
            mapping_mode=mapping_mode,
            refusal_sources=refusal_sources,
        )
        print(f"Loaded {len(dialogues)} RedBench examples")

        max_dialogues = getattr(config, 'max_dialogues', None)
        if max_dialogues is not None and max_dialogues > 0 and len(dialogues) > max_dialogues:
            random.seed(42)
            dialogues = random.sample(dialogues, max_dialogues)
            print(f"Limited to {max_dialogues} examples")

        multiwoz_mode = False
    elif environment_type == 'harmbench':
        from .environments.harmbench_helpers import load_harmbench_dataset

        split = getattr(config, 'harmbench_split', 'train')
        max_examples = getattr(config, 'harmbench_max_examples', None)
        print(f"Loading HarmBench dataset (split={split})...")
        dialogues = load_harmbench_dataset(
            split=split,
            max_examples=max_examples,
        )
        print(f"Loaded {len(dialogues)} HarmBench examples")

        max_dialogues = getattr(config, 'max_dialogues', None)
        if max_dialogues is not None and max_dialogues > 0 and len(dialogues) > max_dialogues:
            random.seed(42)
            dialogues = random.sample(dialogues, max_dialogues)
            print(f"Limited to {max_dialogues} examples")

        multiwoz_mode = False
    else:
        raise ValueError(f"Unknown environment_type: {environment_type}. Must be one of: 'multiwoz', 'vitabench', 'salesagent', 'userbench', 'cares', 'wildjailbreak', 'redbench', 'harmbench'")
    
    mode_str = "online" if _is_online_mode(config) else "offline"
    print(f"Starting training loop (mode: {mode_str}, environment: {config.environment_type}) with batch_size={config.batch_size}, max_turns={config.max_turns}")
    if config.checkpoint_path:
        print(f"Loaded pretrained Q-network from: {config.checkpoint_path}")
    print("=" * 80)
    
    # Transition buffer for batching Q-function updates
    # Store: (o_t, a_t^HL, a_t^LL, r_t, o_{t+1})
    transition_buffer = {
        'observations': [],  # o_t
        'high_level_actions': [],  # a_t^HL (contexts/beliefs)
        'low_level_actions': [],  # a_t^LL (dataset actions)
        'rewards': [],  # r_nom (nominal scalar; CARES: cares_nominal_scalar)
        'next_observations': [],  # o_{t+1}
        'terminals': [],  # done flags
        'entropies': [],
        'ground_truth_goals': [],  # Ground truth goals for contrastive learning
        'belief_candidates': [],  # Belief candidates for V-function training
        'belief_probabilities': [],  # Belief probabilities for V-function training
        'environments': [],  # Environment instances for marginal rewards
        'dialogue_histories': []  # Dialogue histories for marginal rewards
    }
    
    # Replay buffer for online Q-learning
    replay_buffer = {
        'observations': [],
        'high_level_actions': [],
        'low_level_actions': [],  # Dataset actions for contrastive learning
        'next_observations': [],
        'rewards': [],
        'terminals': [],
        'entropies': [],
        'ground_truth_goals': [],  # Ground truth goals for contrastive learning
        'belief_candidates': [],  # Belief candidates for V-function training
        'belief_probabilities': [],  # Belief probabilities for V-function training
        'environments': [],  # Environment instances for marginal rewards
        'dialogue_histories': []  # Dialogue histories for marginal rewards
    }
    
    # Pool of ground truth goals from other episodes (for contrastive learning)
    ground_truth_pool: List[str] = []
    max_pool_size = 1000  # Maximum size of the pool
    
    # Track current epsilon for online mode (with decay)
    current_epsilon = config.epsilon
    config._current_epsilon = current_epsilon  # Store in config for access in process_turn_for_episode
    
    eval_interval = max(1, config.evaluation_interval)
    
    # Window size for statistics and evaluation (fixed at 100, independent of eval_interval)
    stats_window_size = 100
    
    # Store one example per evaluation interval for debugging (memory-efficient)
    debug_example = None  # Will store: {'obs', 'candidates', 'q_values', 'selected_action', 'reward', 'episode_idx'}
    # CARES: track whether we've printed first adversarial and non-adversarial episode conversation histories
    cares_first_non_adv_printed = [False]
    cares_first_adv_printed = [False]
    
    # Episode statistics tracking (for averaging over a fixed window of recent episodes)
    # Use deque to efficiently track recent episodes (fixed window of 100)
    episode_stats = {
        'total_reward': deque(maxlen=stats_window_size),
        'episode_length': deque(maxlen=stats_window_size),
        'goal_achieved': deque(maxlen=stats_window_size),
        'final_entropy': deque(maxlen=stats_window_size),
        'avg_value_loss': deque(maxlen=stats_window_size),
        'avg_q_min_loss': deque(maxlen=stats_window_size),
        'avg_q_min_target_variance': deque(maxlen=stats_window_size),
        'avg_q_min_prediction_variance': deque(maxlen=stats_window_size),
        'max_gpu_memory_gb': deque(maxlen=stats_window_size),
        'avg_cosine_similarity': deque(maxlen=stats_window_size),
        'avg_l2_distance': deque(maxlen=stats_window_size),
        'avg_log_prob_max_q_ll_action': deque(maxlen=stats_window_size),
        'harmful_level': deque(maxlen=stats_window_size),
        'test_time_agent_tokens_total': deque(maxlen=stats_window_size),
        'test_time_agent_tokens_per_turn_avg': deque(maxlen=stats_window_size),
    }
    
    # Track episodes for belief evaluation (last 100 episodes, not eval_interval)
    # Each entry: {'episode_id': int, 'dialogue_id': str, 'goal_json': Dict, 'chosen_beliefs': List[str]}
    evaluation_episodes = deque(maxlen=stats_window_size)
    
    start_time = time.time()
    print(f"Starting episode processing... (Total dialogues: {len(dialogues)})")
    print(f"Using episode_batch_size={config.episode_batch_size} for parallel processing")
    sys.stdout.flush()
    
    # Initialize episode pool for batched processing
    from collections import deque as deque_collections
    dialogue_queue = deque_collections(dialogues)
    episodes_completed = 0
    total_episodes = len(dialogues)
    
    # Load personas once for online mode (avoid reading file 64 times)
    available_personas = None
    if _is_online_mode(config):
        persona_file = Path("src/prompts/personas.jsonl")
        if persona_file.exists():
            print(f"Loading personas from {persona_file}...")
            sys.stdout.flush()
            available_personas = []
            with open(persona_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        available_personas.append(json.loads(line))
            print(f"Loaded {len(available_personas)} personas")
            sys.stdout.flush()
        else:
            raise ValueError(f"Personas file not found: {persona_file}")
    
    # Initialize episode pool
    active_episodes: List[EpisodeState] = []
    next_episode_idx = 0
    num_episodes_to_init = min(config.episode_batch_size, len(dialogues))
    print(f"Initializing {num_episodes_to_init} episodes...")
    sys.stdout.flush()
    
    # For multiwoz_online mode, batch the initial user message generation
    if config.environment_type == "multiwoz_online" and user_agent:
        # First, create all environments without calling reset()
        print("  Creating environments (without reset)...")
        sys.stdout.flush()
        envs_and_data = []
        for i in range(num_episodes_to_init):
            if dialogue_queue:
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"  Creating environment {i+1}/{num_episodes_to_init}...")
                    sys.stdout.flush()
                dialogue_data = dialogue_queue.popleft()
                multiwoz_dialogue = dialogue_data
                ground_truth_goal = format_multiwoz_goal(multiwoz_dialogue.goal)
                dialogue_id = multiwoz_dialogue.dialogue_id
                
                # Select random persona
                persona = random.choice(available_personas) if available_personas else None
                
                from .environments.online_env import OnlineEnvironment, OnlineEnvironmentState
                env = OnlineEnvironment(
                    user_agent=user_agent,
                    persona=persona,
                    goal_json=ensure_goal_json(multiwoz_dialogue.goal),
                    max_turns=config.max_turns,
                    debug=config.debug,
                    source_dialogue=multiwoz_dialogue
                )
                envs_and_data.append((env, next_episode_idx, dialogue_id, multiwoz_dialogue, ground_truth_goal))
                next_episode_idx += 1
        
        # Batch generate initial user messages
        print(f"  Batch generating initial user messages for {len(envs_and_data)} episodes...")
        sys.stdout.flush()
        initial_observations = []
        initial_env_infos = []
        
        # Collect all prompts for batch generation
        personas = []
        goal_jsons = []
        goal_progress_summaries = []
        for env, _, _, _, _ in envs_and_data:
            personas.append(env.persona)
            goal_jsons.append(env.goal_json)
            goal_progress_summaries.append(env.goal_state.build_progress_summary(env.goal_json))
        
        # Use user_agent's batch generation method (handles GPT vs local model automatically)
        # This respects use_gpt_for_user setting
        raw_responses = user_agent.generate_user_response_batch(
            personas=personas,
            dialogue_histories=[[]] * len(envs_and_data),  # Empty history for initial messages
            agent_actions=[""] * len(envs_and_data),  # No agent action yet
            temperature=0.7,
            goal_jsons=goal_jsons,
            goal_progress_summaries=goal_progress_summaries,
            chunk_size=getattr(config, "batch_generation_chunk_size", None)
        )
        
        # Parse responses and create episodes
        print(f"  Creating episode states...")
        sys.stdout.flush()
        for (env, dialogue_idx, dialogue_id, multiwoz_dialogue, ground_truth_goal), raw_response in zip(envs_and_data, raw_responses):
            # Parse response from [RESPONSE] tags
            initial_observation = env._parse_response_from_tags(raw_response)
            
            # Set up environment state
            env.env_state = OnlineEnvironmentState()
            env.env_state.dialogue_history.append(("", initial_observation))
            env.env_state.turn_idx = 0
            
            # Reset goal state
            env.goal_state.current_desire_idx = 0
            for desire in env.goal_state.desires:
                if desire.status != "satisfied":
                    desire.status = "pending"
                    desire.attempt_count = 0
            
            initial_env_info = {
                "goal_json": env.goal_json,
                "persona": env.persona,
                "turn_idx": env.env_state.turn_idx,
                "goal_achieved": 0.0,
                "goal_state": env.goal_state
            }
            
            episode = EpisodeState(
                dialogue_idx=dialogue_idx,
                dialogue_id=dialogue_id,
                dialogue_data=multiwoz_dialogue,
                ground_truth_goal=ground_truth_goal,
                env=env,
                initial_observation=initial_observation,
                initial_env_info=initial_env_info
            )
            episode.initialize()
            active_episodes.append(episode)
    else:
        # Offline mode or vitabench: create episodes
        if config.environment_type == "vitabench":
            from .environments.vitabench_helpers import vitabench_user_is_local, create_vitabench_batch_episodes
            if vitabench_user_is_local(config):
                batch_dialogues = [dialogue_queue.popleft() for _ in range(num_episodes_to_init) if dialogue_queue]
                if batch_dialogues:
                    print(f"  Creating VitaBench batch env for {len(batch_dialogues)} episodes (one model load)...")
                    sys.stdout.flush()
                    batch_episodes = create_vitabench_batch_episodes(
                        batch_dialogues, config, debug=config.debug, language=getattr(config, "language", "english")
                    )
                    for j, ep in enumerate(batch_episodes):
                        ep.dialogue_idx = next_episode_idx + j
                    next_episode_idx += len(batch_episodes)
                    active_episodes.extend(batch_episodes)
            if not active_episodes or len(active_episodes) < num_episodes_to_init:
                for i in range(num_episodes_to_init - len(active_episodes)):
                    if not dialogue_queue:
                        break
                    if (i + 1) % 10 == 0 or i == 0:
                        print(f"  Creating episode {len(active_episodes)+i+1}/{num_episodes_to_init}...")
                    sys.stdout.flush()
                    dialogue_data = dialogue_queue.popleft()
                    episode = create_episode_state(
                        next_episode_idx, dialogue_data, multiwoz_mode, config, config.debug,
                        user_agent=user_agent if _is_online_mode(config) else None,
                        persona=None, available_personas=available_personas,
                        patient_agent=patient_agent if config.environment_type in ("cares", "wildjailbreak", "redbench", "harmbench") else None,
                        reward_model=reward_model if config.environment_type in ("cares", "wildjailbreak", "redbench", "harmbench") else None,
                        reward_tokenizer=reward_tokenizer if config.environment_type in ("cares", "wildjailbreak", "redbench", "harmbench") else None,
                        judge_model=judge_model if config.environment_type in ("cares", "wildjailbreak", "redbench", "harmbench") else None,
                        judge_tokenizer=judge_tokenizer if config.environment_type in ("cares", "wildjailbreak", "redbench", "harmbench") else None,
                    )
                    active_episodes.append(episode)
                    next_episode_idx += 1
        else:
            for i in range(num_episodes_to_init):
                if dialogue_queue:
                    if (i + 1) % 10 == 0 or i == 0:
                        print(f"  Creating episode {i+1}/{num_episodes_to_init}...")
                    sys.stdout.flush()
                dialogue_data = dialogue_queue.popleft()
                episode = create_episode_state(
                    next_episode_idx, dialogue_data, multiwoz_mode, config, config.debug,
                    user_agent=user_agent if _is_online_mode(config) else None,
                    persona=None, available_personas=available_personas,
                    patient_agent=patient_agent if config.environment_type in ("cares", "wildjailbreak", "redbench", "harmbench") else None,
                    reward_model=reward_model if config.environment_type in ("cares", "wildjailbreak", "redbench", "harmbench") else None,
                    reward_tokenizer=reward_tokenizer if config.environment_type in ("cares", "wildjailbreak", "redbench", "harmbench") else None,
                    judge_model=judge_model if config.environment_type in ("cares", "wildjailbreak", "redbench", "harmbench") else None,
                    judge_tokenizer=judge_tokenizer if config.environment_type in ("cares", "wildjailbreak", "redbench", "harmbench") else None,
                )
                active_episodes.append(episode)
                next_episode_idx += 1
    
    # Load tokenizer and agents after episode initialization (for modes that don't need them earlier)
    # This avoids the fork warning when tokenizers are used before multiprocessing
    if not needs_tokenizer_before_episodes:
        print("Loading tokenizer and initializing agents...")
        sys.stdout.flush()
        tokenizer = get_tokenizer_instance(config.model_name)
        judge_model, judge_tokenizer = get_judge_model_and_tokenizer(
            model_name=config.model_name,
            device=config.device if config.device != "cuda" else "cuda:0",
            use_bf16=config.use_bf16,
        )
        
        # Initialize prompt manager
        prompt_manager = PromptManager()
        hl_enable_thinking = config.hl_enable_thinking
        ll_enable_thinking = config.ll_enable_thinking
        
        # Initialize agents (use GPT agents if configured, otherwise local agents)
        if getattr(config, 'use_gpt_for_agents', False):
            gpt_model_name = getattr(config, 'gpt_agent_model', 'gpt-4o-mini')
            gpt_api_key = getattr(config, 'gpt_agent_api_key', None)
            hl_agent = GPTHighLevelAgent(
                tokenizer, prompt_manager, 
                template_name=config.belief_gen_template,
                model_name=gpt_model_name,
                api_key=gpt_api_key
            )
            ll_agent = GPTLowLevelAgent(
                tokenizer, prompt_manager,
                model_name=gpt_model_name,
                api_key=gpt_api_key
            )
            print(f"[INFO] Using GPT agents (model: {gpt_model_name}) for high-level and low-level agents")
        else:
            if config.use_hierarchical_agent and config.high_level_policy_type == "freeform":
                hl_agent = FreeformHighLevelAgent(
                    model,
                    tokenizer,
                    prompt_manager,
                    template_name=config.belief_gen_template,
                    enable_thinking=hl_enable_thinking,
                )
            else:
                hl_agent = HighLevelAgent(
                    model,
                    tokenizer,
                    prompt_manager,
                    template_name=config.belief_gen_template,
                    enable_thinking=hl_enable_thinking,
                )
            ll_agent = LowLevelAgent(
                model,
                tokenizer,
                prompt_manager,
                enable_thinking=ll_enable_thinking,
            )
        
        # Initialize user agent (use GPT if configured, otherwise local model)
        if getattr(config, 'use_gpt_for_user', False):
            gpt_model_name = getattr(config, 'gpt_agent_model', 'gpt-4o-mini')
            gpt_api_key = getattr(config, 'gpt_agent_api_key', None)
            user_agent = GPTUserAgent(
                tokenizer, prompt_manager,
                model_name=gpt_model_name,
                api_key=gpt_api_key
            )
            print(f"[INFO] Using GPT user agent (model: {gpt_model_name})")
        else:
            user_agent = UserAgent(user_model, tokenizer, prompt_manager)  # Use user_model on GPU 1
        print("Tokenizer and agents loaded successfully.")
        sys.stdout.flush()
    
    print(f"Initialized {len(active_episodes)} episodes. Starting main loop...")
    sys.stdout.flush()
    
    # Store one example per evaluation interval for debugging (memory-efficient)
    debug_example = None  # Will store: {'obs', 'candidates', 'q_values', 'selected_action', 'reward', 'episode_idx'}
    
    # Track last printed episode count for statistics
    last_printed_episode_count = 0
    
    # Main batched episode processing loop
    iteration = 0
    max_iterations = 1000000  # Safety limit to prevent infinite loops
    online_update_counter = 0  # Track turns for update frequency
    max_dialogues_limit = getattr(config, 'max_dialogues', None)
    while active_episodes or dialogue_queue:
        iteration += 1
        if iteration > max_iterations:
            raise RuntimeError(
                f"Main loop exceeded maximum iterations ({max_iterations}). "
                f"This likely indicates an infinite loop or episodes that never finish. "
                f"Active episodes: {len(active_episodes)}, Remaining dialogues: {len(dialogue_queue)}"
            )
        print("="*80)
        print(f"Iteration: {iteration} \/")
        print(f"Episodes completed: {episodes_completed}")
        print("="*80)
        # Check if we've reached the episode limit
        if max_dialogues_limit is not None and max_dialogues_limit > 0 and episodes_completed >= max_dialogues_limit:
            print(f"\nReached episode limit ({max_dialogues_limit}). Stopping training.")
            print(f"Completed {episodes_completed} episodes, {len(active_episodes)} active, {len(dialogue_queue)} remaining in queue.")
            sys.stdout.flush()
            break
        
        # Progress reporting
        if episodes_completed > 0 and episodes_completed % 10 == 0 and episodes_completed != last_printed_episode_count:
            elapsed = time.time() - start_time
            print(f"[{elapsed:.1f}s] Completed {episodes_completed}/{total_episodes} episodes, "
                  f"{len(active_episodes)} active, {len(dialogue_queue)} remaining...")
            sys.stdout.flush()
            last_printed_episode_count = episodes_completed
        
        # Debug: Print iteration number for first iteration only
        if iteration == 1:
            # Show turn distribution
            turn_counts = {}
            for ep in active_episodes:
                if ep.is_active and not ep.done_from_env:
                    turn = ep.turn
                    turn_counts[turn] = turn_counts.get(turn, 0) + 1
            turn_dist = ", ".join([f"turn {t}: {c}" for t, c in sorted(turn_counts.items())])
        
        # Compute which episodes are done and which need turn processing BEFORE belief/Q-value steps.
        # This prevents running belief generation and Q-value computation when all episodes are done
        # (e.g. after max_turns), which would otherwise loop indefinitely without stepping or replacing.
        # Treat turn >= max_turns as terminated so we replace them even if done_from_env was not set
        # (e.g. when we exclude from active_episodes_to_process before the step that would set it).
        episodes_to_replace = []
        terminated_episodes = [
            ep for ep in active_episodes
            if not ep.is_active or ep.done_from_env or ep.turn >= config.max_turns
        ]
        episodes_to_replace.extend(terminated_episodes)
        active_episodes_to_process = [
            ep for ep in active_episodes
            if ep.is_active and not ep.done_from_env and not ep.is_frozen and ep.turn < config.max_turns
        ]
        active_count = len(active_episodes_to_process)
        # Episodes that need beliefs this iteration (active and not done; includes frozen so they can retry)
        episodes_needing_beliefs = [ep for ep in active_episodes if ep.is_active and not ep.done_from_env]
        
        # 1. BATCH: Generate beliefs only when there are episodes needing beliefs (skip when all done).
        # Skip episodes waiting to retry fulfillment judge (reuse prior beliefs / Q).
        episodes_for_belief_gen = [
            ep for ep in episodes_needing_beliefs if not ep.judge_step_pending
        ]
        if episodes_for_belief_gen and not getattr(config, 'baseline_mode', False):
            print(f"Batch generating beliefs for {len(episodes_for_belief_gen)} episodes...")
            batch_generate_beliefs_for_episodes(
                episodes_for_belief_gen,
                hl_agent,
                config,
                ground_truth_pool=ground_truth_pool,
                model=model,
                tokenizer=tokenizer,
                device=config.device
            )
            print(f"Beliefs generated for {len(episodes_for_belief_gen)} episodes...")
        # 2. BATCH: Q-values and baseline init only when there are episodes to process this turn.
        if active_episodes_to_process:
            if getattr(config, 'baseline_mode', False):
                for episode in active_episodes_to_process:
                    if not episode.belief_state.candidates:
                        episode.belief_state.candidates = [
                            BeliefCandidate(
                                summary="[BASELINE]",
                                context="Baseline mode - no belief generation",
                                probability=1.0
                            )
                        ]
                        episode._q_values = {"[BASELINE]": 0.0}
            else:
                random_belief_selection = getattr(config, 'random_belief_selection', False)
                if random_belief_selection:
                    for episode in active_episodes_to_process:
                        episode._q_values = {}
                else:
                    episodes_for_q = [
                        ep for ep in active_episodes_to_process if not ep.judge_step_pending
                    ]
                    if episodes_for_q:
                        print(f"Batch computing Q-values for {len(episodes_for_q)} episodes...")
                        batch_compute_q_values_for_episodes(episodes_for_q, value_function, tokenizer, config)
                        print(f"Q-values computed for {len(episodes_for_q)} episodes...")
        
        # 3. Process turns for all active episodes
        # For multiwoz_online and vitabench, batch the LL agent actions (one batch per turn, minibatched).
        # When the agent model is local, this avoids single-instance generate(); process_turn uses precomputed actions.
        if config.environment_type in ("multiwoz_online", "vitabench") and active_episodes_to_process:
            # Step 3a: Batch generate LL agent actions for all active episodes (batched protocol when local)
            # Collect prompts for LL agent actions
            ll_prompts = []
            ll_episode_indices = []
            selected_beliefs = []
            ll_histories = []
            
            baseline_mode = getattr(config, 'baseline_mode', False)
            
            for episode in active_episodes_to_process:
                # Ensure every active episode gets an LL prompt (so all can step each iteration).
                # Turn 0: use initial observation. Turn > 0 with empty history: backfill from env or fallback.
                ll_history = episode.belief_state.history
                if not ll_history:
                    if episode.turn == 0 and episode.initial_observation:
                        ll_history = [("", episode.initial_observation)]
                    else:
                        # Backfill so we still build a prompt (e.g. episode was frozen and never got history appended)
                        env_state = getattr(episode.env, "env_state", None)
                        dialogue_history = getattr(env_state, "dialogue_history", None) if env_state else None
                        if dialogue_history:
                            ll_history = list(dialogue_history)
                        else:
                            ll_history = [("", episode.initial_observation)] if episode.initial_observation else [("", episode.current_obs_for_action or "")]
                
                if baseline_mode:
                    # Baseline mode: skip belief generation, use only conversation history
                    # Use baseline template which doesn't require belief_context
                    prompt = ll_agent.build_prompt(
                        belief_context="",  # Empty for baseline mode
                        history=ll_history,
                        belief_only=False,  # Use history
                        template_name="action_generation_baseline"
                    )
                    selected_belief = "[BASELINE]"  # Dummy belief for evaluation data compatibility
                else:
                    # Normal mode: get selected belief from Q-values
                    q_values = getattr(episode, '_q_values', {})
                    high_level_candidates = [c.summary for c in episode.belief_state.candidates]
                    
                    # Filter out [SKIP] and noise candidates from selection
                    ablation_mode = getattr(config, 'contrastive_ablation_mode', 'none')
                    filter_noise = (ablation_mode == "noise_in_candidates")
                    
                    valid_candidates = _filter_candidates_for_value_min(high_level_candidates, filter_noise)
                    
                    valid_q_values = {k: v for k, v in q_values.items() if k != "[SKIP]"}
                    if filter_noise:
                        valid_q_values = {k: v for k, v in valid_q_values.items() if not is_noise_candidate(k)}
                    
                    # Debug: Print detailed information when no valid candidates
                    if not valid_candidates or not valid_q_values:
                        print(f"\n[DEBUG] No valid candidates for episode {episode.dialogue_idx}:")
                        print(f"  Turn: {episode.turn}")
                        print(f"  Ablation mode: {ablation_mode}")
                        print(f"  Filter noise: {filter_noise}")
                        print(f"  Has belief_state.candidates: {episode.belief_state.candidates is not None}")
                        print(f"  Number of belief_state.candidates: {len(episode.belief_state.candidates) if episode.belief_state.candidates else 0}")
                        print(f"  All candidates: {high_level_candidates}")
                        print(f"  All Q-values: {q_values}")
                        print(f"  Q-values dict keys: {list(q_values.keys())}")
                        print(f"  Valid candidates after filtering: {valid_candidates}")
                        print(f"  Valid Q-values after filtering: {valid_q_values}")
                        print(f"  Current observation: {repr(episode.current_obs_for_action[:200] if episode.current_obs_for_action else 'None')}")
                        print(f"  Observation length: {len(episode.current_obs_for_action) if episode.current_obs_for_action else 0}")
                        if episode.belief_state.candidates:
                            print(f"  Candidate summaries: {[c.summary for c in episode.belief_state.candidates]}")
                        sys.stdout.flush()
                    
                    epsilon_to_use = getattr(config, '_current_epsilon', config.epsilon)
                    random_belief_selection = getattr(config, 'random_belief_selection', False)
                    use_regret_critic = getattr(config, 'use_regret_critic', False)
                    regret_values = getattr(episode, '_regret_values', None)
                    if valid_candidates:
                        if random_belief_selection:
                            selected_belief = random.choice(valid_candidates)
                        elif valid_q_values:
                            if random.random() < epsilon_to_use:
                                selected_belief = random.choice(valid_candidates)
                            elif use_regret_critic and regret_values and all(k in regret_values for k in valid_q_values):
                                beta = getattr(config, 'regret_critic_beta', 0.2)
                                selected_belief = _regret_critic_selection(valid_q_values, regret_values, beta)
                            else:
                                selected_belief = _softmax_sample_from_q_values(valid_q_values)
                        else:
                            selected_belief = random.choice(valid_candidates)
                    else:
                        raise RuntimeError(
                            f"No valid high-level belief candidates for episode {episode.dialogue_idx}. "
                            f"Ablation mode: {ablation_mode}, Filter noise: {filter_noise}, "
                            f"All candidates: {high_level_candidates}, All Q-values: {q_values}"
                        )
                    
                    prompt = ll_agent.build_prompt(
                        belief_context=selected_belief,
                        history=ll_history,
                        belief_only=getattr(config, 'll_action_belief_only', True)
                    )
                
                ll_prompts.append(prompt)
                ll_episode_indices.append(episode.dialogue_idx)
                selected_beliefs.append(selected_belief)
                ll_histories.append(ll_history)
                
                # Store LL agent input for debugging (only for online mode)
                if hasattr(episode.env, 'env_state') and hasattr(episode.env.env_state, 'll_agent_inputs'):
                    episode.env.env_state.ll_agent_inputs.append(prompt)
            
            # Batch generate LL agent actions
            if ll_prompts:
                print(f"Batch generating LL actions for {len(active_episodes)} episodes...")
                first_ep = active_episodes_to_process[0] if active_episodes_to_process else None
                template_name = get_ll_action_template_name(config, baseline_mode=baseline_mode, episode=first_ep)
                ll_actions = ll_agent.generate_actions_from_prompts(
                    prompts=ll_prompts,
                    temperature=0.7,
                    do_sample=False,
                    chunk_size=getattr(config, "batch_generation_chunk_size", None),
                    template_name=template_name
                )
                print(f"LL actions generated for {len(active_episodes)} episodes...")
                # Validate that we got the right number of actions
                if len(ll_actions) != len(ll_prompts):
                    raise RuntimeError(
                        f"Mismatch: generated {len(ll_actions)} actions for {len(ll_prompts)} prompts. "
                        f"Episode indices: {ll_episode_indices}"
                    )
                
                # Store LL actions in episodes
                for episode_idx, action, belief, history in zip(ll_episode_indices, ll_actions, selected_beliefs, ll_histories):
                    print(f"Processing episode {episode_idx}...")
                    episode = next(ep for ep in active_episodes_to_process if ep.dialogue_idx == episode_idx)
                    
                    # Validate action is not empty in online mode
                    if not action or not action.strip():
                        print(f"\n[DEBUG] Empty agent response generated for episode {episode.dialogue_idx}, turn {episode.turn}:")
                        print(f"  Selected belief: {belief}")
                        print(f"  History: {history}")
                        print(f"  Prompt length: {len(ll_prompts[ll_episode_indices.index(episode_idx)])}")
                        print(f"  Raw action from model: {repr(action)}")
                        sys.stdout.flush()
                        raise ValueError(
                            f"Empty agent response generated in online mode for episode {episode.dialogue_idx}, turn {episode.turn}. "
                            f"This should not happen - the LLM should always generate a response. "
                            f"Selected belief: {belief[:100] if belief else 'None'}"
                        )
                    
                    episode.agent_response = action
                    # So process_turn_for_episode uses batched action instead of generating again (multiwoz_online and vitabench)
                    if config.environment_type in ("vitabench", "multiwoz_online"):
                        episode._precomputed_agent_response = action
                    episode.chosen_beliefs_per_turn.append(belief)
                    # Store LL agent output for debugging (only for online mode)
                    if hasattr(episode.env, 'env_state') and hasattr(episode.env.env_state, 'll_agent_outputs'):
                        episode.env.env_state.ll_agent_outputs.append(action)
                    # Store evaluation data
                    q_values = getattr(episode, '_q_values', {})
                    # For baseline mode, use empty/dummy candidates and q_values
                    if baseline_mode:
                        candidate_beliefs = ["[BASELINE]"]
                        q_values = {"[BASELINE]": 0.0}
                    else:
                        candidate_beliefs = [c.summary for c in episode.belief_state.candidates]
                        q_values = getattr(episode, '_q_values', {})
                    
                    episode.turn_evaluation_data.append(
                        build_turn_evaluation_entry(
                            episode=episode,
                            observation=episode.current_obs_for_action,
                            ll_action=action,
                            candidate_beliefs=candidate_beliefs,
                            q_values=q_values,
                            selected_belief=belief,
                            multiwoz_mode=multiwoz_mode,
                            training_mode="online" if _is_online_mode(config) else "offline"
                        )
                    )
            
            # Step 3b: Batch user-model interactions (judge + user response), mapping carefully to episodes
            # multiwoz_online: batch judge + user response; vitabench (local user): batch user-sim in one tensor
            if config.environment_type == "multiwoz_online":
                batch_user_model_interactions(active_episodes_to_process, user_agent, config)
            if config.environment_type == "vitabench" and active_episodes_to_process:
                try:
                    from .environments.vitabench_helpers import _vitabench_user_is_local as _vul
                    if _vul:
                        from .environments.vitabench_helpers import batch_vitabench_user_responses
                        batch_vitabench_user_responses(active_episodes_to_process, config)
                except (ImportError, AttributeError):
                    pass
            
            # Step 3c: Process remaining turn logic for each episode
            if config.environment_type == "vitabench" and active_episodes_to_process:
                print(f"Turn count: {active_episodes_to_process[0].turn + 1}")
            processed_count = 0
            pending_dpo_entries: List[Dict] = []
            # VitaBench uses same pattern as MultiWOZ: LL and user responses are on batch tensors;
            # process_turn runs sequentially (no ThreadPoolExecutor, no tokenizer contention).
            for episode in active_episodes_to_process:
                # Turn 0 is now handled like other turns - agent has already acted
                processed_count += 1
                # Belief state update (agent side): append (agent_response, observation) to dialogue history.
                # No model calls here; agent/critic use batched protocols separately.
                if episode.turn == 0:
                    if not episode.belief_state.history:
                        episode.belief_state.history.append(("", episode.initial_observation))
                    if len(episode.belief_state.history) == 1:
                        episode.belief_state.history.append((episode.agent_response, episode.observation))
                    elif len(episode.belief_state.history) != episode.turn:
                        episode.belief_state.history.append((episode.agent_response, episode.observation))
                else:
                    if not (episode.belief_state.history and len(episode.belief_state.history) == episode.turn + 1):
                        episode.belief_state.history.append((episode.agent_response, episode.observation))
                episode.update_current_observation_context()
                should_continue = process_turn_for_episode(
                    episode, hl_agent, ll_agent, value_function, tokenizer, model, config,
                    multiwoz_mode, standard_dtype
                )
                if not should_continue:
                    episodes_to_replace.append(episode)
            for episode in active_episodes_to_process:
                # Skip DPO update in baseline mode (no beliefs to update)
                if not getattr(config, 'baseline_mode', False):
                    # Get selected belief for DPO update
                    q_values = getattr(episode, '_q_values', {})
                    high_level_candidates = [c.summary for c in episode.belief_state.candidates]
                    
                    # Filter out [SKIP] and noise candidates from selection
                    ablation_mode = getattr(config, 'contrastive_ablation_mode', 'none')
                    filter_noise = (ablation_mode == "noise_in_candidates")
                    
                    valid_candidates = _filter_candidates_for_value_min(high_level_candidates, filter_noise)
                    
                    valid_q_values = {k: v for k, v in q_values.items() if k != "[SKIP]"}
                    if filter_noise:
                        valid_q_values = {k: v for k, v in valid_q_values.items() if not is_noise_candidate(k)}
                    
                    # Debug: Print detailed information when no valid candidates
                    if not valid_candidates or not valid_q_values:
                        print(f"\n[DEBUG] No valid candidates for episode {episode.dialogue_idx} (DPO update):")
                        print(f"  Ablation mode: {ablation_mode}")
                        print(f"  Filter noise: {filter_noise}")
                        print(f"  All candidates: {high_level_candidates}")
                        print(f"  All Q-values: {q_values}")
                        print(f"  Valid candidates after filtering: {valid_candidates}")
                        print(f"  Valid Q-values after filtering: {valid_q_values}")
                        print(f"  Number of belief_state.candidates: {len(episode.belief_state.candidates) if episode.belief_state.candidates else 0}")
                        if episode.belief_state.candidates:
                            print(f"  Candidate summaries: {[c.summary for c in episode.belief_state.candidates]}")
                        sys.stdout.flush()
                    
                    epsilon_to_use = getattr(config, '_current_epsilon', config.epsilon)
                    random_belief_selection = getattr(config, 'random_belief_selection', False)
                    use_regret_critic = getattr(config, 'use_regret_critic', False)
                    regret_values = getattr(episode, '_regret_values', None)
                    if valid_candidates:
                        if random_belief_selection:
                            selected_belief = random.choice(valid_candidates)
                        elif valid_q_values:
                            if random.random() < epsilon_to_use:
                                selected_belief = random.choice(valid_candidates)
                            elif use_regret_critic and regret_values and all(k in regret_values for k in valid_q_values):
                                beta = getattr(config, 'regret_critic_beta', 0.2)
                                selected_belief = _regret_critic_selection(valid_q_values, regret_values, beta)
                            else:
                                selected_belief = _softmax_sample_from_q_values(valid_q_values)
                        else:
                            selected_belief = random.choice(valid_candidates)
                    else:
                        raise RuntimeError(
                            f"No valid high-level belief candidates for episode {episode.dialogue_idx} (DPO update). "
                            f"Ablation mode: {ablation_mode}, Filter noise: {filter_noise}, "
                            f"All candidates: {high_level_candidates}, All Q-values: {q_values}"
                        )

                    pending_dpo_entries.append({
                        "episode": episode,
                        "selected_belief": selected_belief,
                        "observation": episode.observation
                    })

            # Skip DPO update in baseline mode or evaluation mode
            evaluation_mode = getattr(config, "_evaluation_mode", False)
            if (
                not getattr(config, 'baseline_mode', False)
                and not evaluation_mode
                and not config.critic_only_training
                and pending_dpo_entries
            ):
                _finalize_episodes_with_batched_dpo(
                    entries=pending_dpo_entries,
                    model=model,
                    tokenizer=tokenizer,
                    config=config,
                    multiwoz_mode=multiwoz_mode,
                    episodes_to_replace=episodes_to_replace
                )
            if config.environment_type == "vitabench":
                try:
                    from .environments.vitabench_helpers import flush_batched_vitabench_generates
                    flush_batched_vitabench_generates()
                except (ImportError, AttributeError):
                    pass
        else:
            # Offline mode or no active episodes: process sequentially
            processed_count = 0
            for idx, episode in enumerate(active_episodes_to_process):
                if not episode.is_active or episode.done_from_env:
                    continue
                
                processed_count += 1
                
                should_continue = process_turn_for_episode(
                    episode, hl_agent, ll_agent, value_function, tokenizer, model, config,
                    multiwoz_mode, standard_dtype
                )
                
                if not should_continue:
                    episodes_to_replace.append(episode)
            
            # Flush any remaining batched generate requests for VitaBench
            if config.environment_type == "vitabench":
                try:
                    from .environments.vitabench_helpers import flush_batched_vitabench_generates
                    flush_batched_vitabench_generates()
                except (ImportError, AttributeError):
                    pass  # Flush function may not exist if no local models
        
        # 4. Collect transitions from all active episodes
        contributing_episodes = collect_transitions_from_episodes(active_episodes, transition_buffer, config)
        
        # 4b. For online mode, also collect into replay buffer and update
        if _is_online_mode(config):
            # Check if we need to populate ground_truth_pool for contrastive learning
            contrastive_coef = getattr(config, 'contrastive_coef', 0.0)
            ablation_mode = getattr(config, 'contrastive_ablation_mode', 'none')
            needs_pool = ((contrastive_coef > 0.0 and ablation_mode == "crossed_data") or 
                         (ablation_mode == "crossed_data_in_candidates"))
            
            # Add transitions to replay buffer
            for i in range(len(transition_buffer['observations'])):
                replay_buffer['observations'].append(transition_buffer['observations'][i])
                replay_buffer['high_level_actions'].append(transition_buffer['high_level_actions'][i])
                replay_buffer['low_level_actions'].append(transition_buffer['low_level_actions'][i])
                replay_buffer['next_observations'].append(transition_buffer['next_observations'][i])
                replay_buffer['rewards'].append(transition_buffer['rewards'][i])
                replay_buffer['terminals'].append(transition_buffer['terminals'][i])
                replay_buffer['entropies'].append(transition_buffer['entropies'][i])
                replay_buffer['ground_truth_goals'].append(transition_buffer['ground_truth_goals'][i])
                replay_buffer['belief_candidates'].append(transition_buffer['belief_candidates'][i] if i < len(transition_buffer.get('belief_candidates', [])) else [])
                replay_buffer['belief_probabilities'].append(transition_buffer['belief_probabilities'][i] if i < len(transition_buffer.get('belief_probabilities', [])) else [])
                
                # Populate ground_truth_pool from transitions as they're added
                if needs_pool and i < len(transition_buffer['ground_truth_goals']):
                    gt_goal = transition_buffer['ground_truth_goals'][i]
                    if isinstance(gt_goal, str) and gt_goal.strip() and gt_goal.lower() != "user goal not specified":
                        if gt_goal not in ground_truth_pool:  # Avoid duplicates
                            ground_truth_pool.append(gt_goal)
                            # Limit pool size
                            if len(ground_truth_pool) > max_pool_size:
                                ground_truth_pool.pop(0)  # Remove oldest
                
                # Limit replay buffer size
                if len(replay_buffer['observations']) > config.replay_buffer_size:
                    for key in replay_buffer.keys():
                        if len(replay_buffer[key]) > 0:
                            replay_buffer[key].pop(0)  # Remove oldest transition
            
            # Update Q-function from replay buffer in small batches
            # Process one batch per iteration to avoid infinite loops
            # Check update frequency: only update every N turns
            online_update_frequency = getattr(config, 'online_update_frequency', 1)
            should_update = (online_update_counter % online_update_frequency == 0)
            online_update_counter += 1
            
            if should_update and len(replay_buffer['observations']) >= config.online_batch_size:
                # Sample a batch from replay buffer
                batch_size = min(config.online_batch_size, len(replay_buffer['observations']))
                batch_indices = random.sample(range(len(replay_buffer['observations'])), batch_size)
                
                # Create batch dict
                batch_transitions = {
                    'observations': [replay_buffer['observations'][i] for i in batch_indices],
                    'high_level_actions': [replay_buffer['high_level_actions'][i] for i in batch_indices],
                    'low_level_actions': [replay_buffer['low_level_actions'][i] if i < len(replay_buffer['low_level_actions']) else "" for i in batch_indices],
                    'next_observations': [replay_buffer['next_observations'][i] for i in batch_indices],
                    'rewards': [replay_buffer['rewards'][i] for i in batch_indices],
                    'terminals': [replay_buffer['terminals'][i] for i in batch_indices],
                    'entropies': [replay_buffer['entropies'][i] for i in batch_indices],
                    'ground_truth_goals': [replay_buffer['ground_truth_goals'][i] if i < len(replay_buffer['ground_truth_goals']) else "" for i in batch_indices],
                    'belief_candidates': [replay_buffer['belief_candidates'][i] if i < len(replay_buffer.get('belief_candidates', [])) else [] for i in batch_indices],
                    'belief_probabilities': [replay_buffer['belief_probabilities'][i] if i < len(replay_buffer.get('belief_probabilities', [])) else [] for i in batch_indices]
                }
                
                # Get contrastive coefficient and ablation mode from config
                contrastive_coef = getattr(config, 'contrastive_coef', 0.0)
                ablation_mode = getattr(config, 'contrastive_ablation_mode', 'none')
                
                # Determine actual ablation mode for contrastive loss:
                # - If ablation_mode is "noise_in_candidates" or "crossed_data_in_candidates", we don't use contrastive loss
                # - If contrastive_coef is 0, we don't use contrastive loss (baseline)
                # - Otherwise, use the specified mode for contrastive loss
                if ablation_mode in ["noise_in_candidates", "crossed_data_in_candidates"] or contrastive_coef == 0.0:
                    contrastive_ablation_mode = "none"
                else:
                    # For contrastive loss, map ablation modes:
                    # - "crossed_data": Use other episodes' ground truths (main method)
                    # - "random_noise": Use random noise as negatives (ablation 1)
                    contrastive_ablation_mode = ablation_mode if ablation_mode in ["crossed_data", "random_noise"] else "crossed_data"
                
                # Safety check: If crossed_data mode is needed but pool is empty, populate from replay buffer
                # This handles cases where transitions were added before pool population logic ran
                if contrastive_ablation_mode == "crossed_data" and not ground_truth_pool:
                    # Extract unique ground truth goals from replay buffer
                    for gt_goal in replay_buffer['ground_truth_goals']:
                        if isinstance(gt_goal, str) and gt_goal.strip() and gt_goal.lower() != "user goal not specified":
                            if gt_goal not in ground_truth_pool:  # Avoid duplicates
                                ground_truth_pool.append(gt_goal)
                                # Limit pool size
                                if len(ground_truth_pool) > max_pool_size:
                                    ground_truth_pool.pop(0)  # Remove oldest
                
                # Update Q-function
                q_loss = update_q_function_online(
                    transitions=batch_transitions,
                    value_function=value_function,
                    hl_agent=hl_agent,
                    tokenizer=tokenizer,
                    config=config,
                    standard_dtype=standard_dtype,
                    ll_model=model,  # Same model used for LL agent
                    ll_tokenizer=tokenizer,  # Same tokenizer
                    ground_truth_pool=ground_truth_pool,
                    contrastive_coef=contrastive_coef,
                    ablation_mode=contrastive_ablation_mode,
                    reward_model=reward_model,
                    reward_tokenizer=reward_tokenizer,
                )
                
                # Record Q loss in contributing episodes
                regret_metrics = getattr(config, "_last_regret_training_metrics", None)
                for episode in contributing_episodes:
                    episode.episode_value_losses.append(q_loss)
                    if regret_metrics is not None:
                        episode.episode_q_min_losses.append(regret_metrics["q_min_loss"])
                        episode.episode_q_min_target_variances.append(regret_metrics["q_min_target_variance"])
                        episode.episode_q_min_prediction_variances.append(regret_metrics["q_min_prediction_variance"])
        
        # 5. Batch update Q-function when buffer is full (offline mode only)
        if not _is_online_mode(config) and not getattr(config, "_evaluation_mode", False):
            # Process transitions in chunks of batch_size to prevent memory spikes
            while len(transition_buffer['observations']) >= config.batch_size:
                # Take only batch_size transitions for this update
                batch_indices = list(range(min(config.batch_size, len(transition_buffer['observations']))))
                
                transitions = list(zip(
                    [transition_buffer['observations'][i] for i in batch_indices],
                    [transition_buffer['high_level_actions'][i] for i in batch_indices],
                    [transition_buffer['low_level_actions'][i] for i in batch_indices],
                    [transition_buffer['next_observations'][i] for i in batch_indices],
                    [transition_buffer['rewards'][i] for i in batch_indices],
                    [transition_buffer['terminals'][i] for i in batch_indices]
                ))
                
                # Track which episodes contributed to this specific batch
                batch_contributing_episodes = []
                for i in batch_indices:
                    # Find which episode contributed this transition (if we can track it)
                    # For now, we'll use all contributing episodes from the last collection
                    pass  # We'll use contributing_episodes from the last collection

                # 1. Compute current Q-values: Q(o_t, a_t^HL)
                # Training mode: gradients needed for MLP head
                # Handle [SKIP] candidates by assigning 0.0
                hl_actions = [hl_action for _, hl_action, _, _, _, _ in transitions]
                skip_mask = [i for i, action in enumerate(hl_actions) if action == "[SKIP]"]
                valid_indices = [i for i, action in enumerate(hl_actions) if action != "[SKIP]"]
                
                if valid_indices:
                    valid_obs = [transitions[i][0] for i in valid_indices]
                    valid_actions = [hl_actions[i] for i in valid_indices]
                    
                    # Chunked batching protocol
                    chunk_size = _model_batch_chunk_size(config)
                    num_chunks = (len(valid_obs) + chunk_size - 1) // chunk_size
                    q_current_chunks = []
                    
                    for chunk_idx in range(num_chunks):
                        start_idx = chunk_idx * chunk_size
                        end_idx = min(start_idx + chunk_size, len(valid_obs))
                        
                        chunk_obs = valid_obs[start_idx:end_idx]
                        chunk_actions = valid_actions[start_idx:end_idx]
                        
                        chunk_q_values = value_function.predict_q_value(
                            observations=chunk_obs,
                            high_level_actions=chunk_actions,
                            tokenizer=tokenizer,
                            requires_grad=True  # Training: gradients needed
                        )
                        
                        q_current_chunks.append(chunk_q_values)
                        
                        # Cleanup after each chunk
                        del chunk_obs, chunk_actions, chunk_q_values
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    # Concatenate all chunks
                    if q_current_chunks:
                        q_current_batch_valid = torch.cat(q_current_chunks, dim=0)
                    else:
                        q_current_batch_valid = torch.tensor([], device=value_function.device, dtype=standard_dtype)
                    
                    del q_current_chunks
                else:
                    q_current_batch_valid = torch.tensor([], device=value_function.device, dtype=standard_dtype)
                
                # Create full batch with [SKIP] candidates assigned 0.0
                q_current_batch = torch.zeros(len(hl_actions), device=value_function.device, dtype=standard_dtype)
                for valid_idx, original_idx in enumerate(valid_indices):
                    q_current_batch[original_idx] = q_current_batch_valid[valid_idx]
                # [SKIP] candidates already have 0.0 from initialization

                # 2. Collect all next observations and high-level actions for batch Q-value computation
                next_obs_list = []
                next_hl_actions_list = []
                transition_indices = []  # Track which transitions need Q-values

                for idx, (_, hl_action, _, next_obs, _, terminal) in enumerate(transitions):
                    if not terminal and next_obs:
                        next_obs_list.append(next_obs)
                        next_hl_actions_list.append(hl_action)
                        transition_indices.append(idx)

                # Batch compute Q-values for all next states at once
                # Handle [SKIP] candidates by assigning 0.0
                next_q_values = {}
                if next_obs_list:
                    # Separate [SKIP] from valid actions
                    next_skip_mask = [i for i, action in enumerate(next_hl_actions_list) if action == "[SKIP]"]
                    next_valid_indices = [i for i, action in enumerate(next_hl_actions_list) if action != "[SKIP]"]
                    
                    if next_valid_indices:
                        next_valid_obs = [next_obs_list[i] for i in next_valid_indices]
                        next_valid_actions = [next_hl_actions_list[i] for i in next_valid_indices]
                        
                        # Chunked batching protocol
                        chunk_size = _model_batch_chunk_size(config)
                        num_chunks = (len(next_valid_obs) + chunk_size - 1) // chunk_size
                        next_q_chunks = []
                        
                        for chunk_idx in range(num_chunks):
                            start_idx = chunk_idx * chunk_size
                            end_idx = min(start_idx + chunk_size, len(next_valid_obs))
                            
                            chunk_obs = next_valid_obs[start_idx:end_idx]
                            chunk_actions = next_valid_actions[start_idx:end_idx]
                            
                            chunk_q_values = value_function.predict_q_value(
                                observations=chunk_obs,
                                high_level_actions=chunk_actions,
                                tokenizer=tokenizer,
                                requires_grad=False  # Inference: no gradients needed
                            )
                            
                            next_q_chunks.append(chunk_q_values)
                            
                            # Cleanup after each chunk
                            del chunk_obs, chunk_actions, chunk_q_values
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        
                        # Concatenate all chunks
                        if next_q_chunks:
                            next_q_batch_valid = torch.cat(next_q_chunks, dim=0)
                        else:
                            next_q_batch_valid = torch.tensor([], device=value_function.device)
                        
                        del next_q_chunks
                        
                        # Map valid Q-values back to original indices
                        for valid_idx, original_idx in enumerate(next_valid_indices):
                            transition_idx = transition_indices[original_idx]
                            next_q_values[transition_idx] = next_q_batch_valid[valid_idx].item()
                    
                    # [SKIP] candidates get 0.0 (already default in next_q_values dict)

                # 3. Compute P(a|b) directly from logits (unnormalized, proportional to true P(a|b))
                # P(a|b) = product of token probabilities computed from logits
                # We use unnormalized probability since normalizing over finite candidates is incorrect
                # (the true space of actions is infinite - all possible token sequences)
                belief_only = getattr(config, 'll_action_belief_only', True)
                
                # Prepare batch inputs for computing log P(a|b) from logits
                contexts_batch = []
                targets_batch = []
                transition_indices = []
                
                for idx, (obs, hl_action, ll_action, _, _, _) in enumerate(transitions):
                    if not ll_action or not ll_action.strip():
                        continue
                    
                    if belief_only:
                        context = f"High-Level Context: {hl_action}\nLow-Level Action:"
                    else:
                        context = f"Observation: {obs}\nHigh-Level Context: {hl_action}\nLow-Level Action:"
                    
                    contexts_batch.append(context)
                    targets_batch.append(ll_action)
                    transition_indices.append(idx)
                
                # Batch compute log P(a|b) directly from logits (with chunking to prevent OOM)
                # We use mean log probability (normalized by action length) to avoid bias against longer actions
                # This prevents the policy from collapsing to empty/short actions
                # Mean log prob = (sum of token log probs) / length = geometric mean of token probabilities
                log_p_ab_list = []
                action_lengths = {}  # Store action lengths for potential use
                if contexts_batch:
                    from src.utils.llm_utils import compute_log_prob_batch
                    # Process in chunks to prevent OOM
                    chunk_size = getattr(config, 'transition_prob_chunk_size', 8)
                    if chunk_size <= 0:
                        chunk_size = len(contexts_batch)
                    
                    for chunk_start in range(0, len(contexts_batch), chunk_size):
                        chunk_end = min(chunk_start + chunk_size, len(contexts_batch))
                        chunk_contexts = contexts_batch[chunk_start:chunk_end]
                        chunk_targets = targets_batch[chunk_start:chunk_end]
                        
                        chunk_log_p_ab = compute_log_prob_batch(
                            ll_model,
                            ll_tokenizer,
                            chunk_contexts,
                            chunk_targets
                        )
                        log_p_ab_list.extend(chunk_log_p_ab)
                        
                        # Clean up GPU memory after each chunk
                        del chunk_contexts, chunk_targets, chunk_log_p_ab
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    # Store action lengths for reference
                    for trans_idx, target in zip(transition_indices, targets_batch):
                        action_lengths[trans_idx] = len(target.split()) if target else 0
                
                # Map log probabilities back to transitions
                log_p_ab_dict = {}
                for trans_idx, log_p_ab in zip(transition_indices, log_p_ab_list):
                    log_p_ab_dict[trans_idx] = log_p_ab
                
                # 4. For each transition, compute targets using length-normalized P(a|b)
                # Use exp(mean_log_P(a|b)) which is the geometric mean of token probabilities
                # This normalizes by action length, preventing bias against longer actions
                import math
                targets = []
                for idx, (obs, hl_action, ll_action, next_obs, reward_val, terminal) in enumerate(transitions):
                    # Get mean log P(a|b) for this transition's dataset action
                    if idx in log_p_ab_dict:
                        log_p_ab = log_p_ab_dict[idx]
                        # Convert to probability (geometric mean, already length-normalized)
                        # Clamp to prevent numerical issues
                        log_p_ab = max(log_p_ab, -50.0)  # Prevent underflow
                        p_ab = math.exp(log_p_ab)
                    else:
                        # Fallback if action is empty/invalid - use small probability
                        p_ab = 1e-10

                    # Get max Q(o_{t+1}, a') for next state (from batch computation)
                    max_q_next = 0.0 if terminal else next_q_values.get(idx, 0.0)

                    # Construct target: P(a|b) * [r + γ*max_Q]
                    # Using length-normalized P(a|b) (geometric mean) to avoid bias against longer actions
                    target = p_ab * (reward_val + config.discount_factor * max_q_next)
                    targets.append(target)
                
                targets_tensor = torch.tensor(targets, device=config.device, dtype=standard_dtype)
                
                # Clean up large tensors from transition probability computation (after targets are computed)
                del contexts_batch, targets_batch, log_p_ab_list, log_p_ab_dict
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                batch_entropies = [transition_buffer['entropies'][i] for i in batch_indices]
                avg_entropy_batch = sum(batch_entropies) / len(batch_entropies) if batch_entropies else 0.0
                
                # Batch update Q-function
                q_loss = value_function.compute_value_loss(
                    value_predictions=q_current_batch,
                    targets=targets_tensor,
                    avg_entropy=avg_entropy_batch,
                    entropy_coef=config.entropy_coef
                )
                value_function.update(q_loss)
                
                # Record Q loss in episodes that contributed to this batch
                q_loss_value = q_loss.item()
                for episode in contributing_episodes:
                    episode.episode_value_losses.append(q_loss_value)
                
                # Print batch update metrics
                print(f"  [Batch Update] {len(transitions)} transitions (buffer has {len(transition_buffer['observations'])} remaining), "
                      f"avg_q_loss={q_loss_value:.4f}, "
                      f"avg_reward={sum(transition_buffer['rewards'][i] for i in batch_indices)/len(batch_indices):.3f}")
                
                # Clean up batch computation tensors (only delete variables that still exist)
                del q_current_batch, targets_tensor, q_loss, transitions
                del next_q_values, next_obs_list, next_hl_actions_list
                if 'next_q_batch' in locals():
                    del next_q_batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Remove processed transitions from buffer (keep remaining ones for next iteration)
                for key in ['observations', 'high_level_actions', 'low_level_actions', 
                           'next_observations', 'rewards', 'terminals', 'entropies']:
                    # Remove in reverse order to maintain indices
                    for i in sorted(batch_indices, reverse=True):
                        del transition_buffer[key][i]
        
        # 6. Replace finished episodes with new ones (episode packing)
        # Deduplicate to handle cases where an episode might be added multiple times
        unique_episodes_to_replace = list({id(ep): ep for ep in episodes_to_replace}.values())

        # For VitaBench with a local evaluator, run a batched full-trajectory judge
        if getattr(config, "environment_type", "multiwoz_offline") == "vitabench":
            from .environments.vitabench_env import VitaBenchEnvironment
            from .environments.vitabench_helpers import batched_full_traj_judge
            from src.utils.llm_utils import is_local_model

            vitabench_episodes = []
            for ep in unique_episodes_to_replace:
                env = ep.env
                if isinstance(env, VitaBenchEnvironment):
                    vitabench_episodes.append(ep)

            if vitabench_episodes and is_local_model(getattr(config, "vitabench_llm_evaluator", None)):
                tasks = []
                trajectories = []
                final_states = []
                for ep in vitabench_episodes:
                    env = ep.env
                    tasks.append(env.task)
                    trajectories.append(list(env.orchestrator.trajectory))
                    final_states.append(env._get_final_state())

                first_env = vitabench_episodes[0].env
                llm_evaluator = first_env.llm_evaluator
                llm_args_evaluator = first_env.llm_args_evaluator
                language = getattr(config, "language", "english")
                model = first_env.evaluator_model
                tokenizer = first_env.evaluator_tokenizer

                reward_infos = batched_full_traj_judge(
                    tasks=tasks,
                    trajectories=trajectories,
                    final_states=final_states,
                    llm_evaluator=llm_evaluator,
                    llm_args_evaluator=llm_args_evaluator,
                    language=language,
                    model=model,
                    tokenizer=tokenizer,
                )

                from vita.data_model.simulation import RewardType

                for ep, reward_info in zip(vitabench_episodes, reward_infos):
                    fraction = reward_info.reward_breakdown.get(RewardType.NL_ASSERTION, 0.0)
                    ep.dialogue_state.goal_achieved = fraction

        for finished_episode in unique_episodes_to_replace:
            # Print GPU memory before first episode completion
            if episodes_completed == 0:
                device_id = int(config.device.split(":")[-1]) if config.device.startswith("cuda") and ":" in config.device else 0
                print_gpu_memory("GPU Memory - BEFORE first episode completion", include_max=True, restore_device=device_id)
            
            # Store debug example before finalization (if needed)
            if episodes_completed % eval_interval < min(10, eval_interval) and debug_example is None and finished_episode.belief_state.candidates:
                q_vals = getattr(finished_episode, '_q_values', {})
                if q_vals:
                    high_level_candidates = [c.summary for c in finished_episode.belief_state.candidates]
                    selected_action = max(q_vals, key=q_vals.get) if q_vals else (high_level_candidates[0] if high_level_candidates else "")
                    
                    # Extract user state and action information
                    user_action = finished_episode.observation  # User's response/observation
                    agent_action = finished_episode.agent_response  # Agent's action/response
                    # Use goal_state from env_info if available, otherwise forward-fill from last known
                    goal_state = None
                    if finished_episode.env_info:
                        goal_state = finished_episode.env_info.get('goal_state')
                    if goal_state is None:
                        goal_state = finished_episode.last_known_goal_state
                    
                    # Format user state information
                    user_state_info = {}
                    if goal_state:
                        current_desire = goal_state.get_current_desire()
                        if current_desire:
                            user_state_info['current_desire'] = {
                                'intent': current_desire.intent,
                                'domain': current_desire.domain,
                                'status': current_desire.status,
                                'attempt_count': current_desire.attempt_count,
                                'required_slots': current_desire.required_slots
                            }
                        user_state_info['total_desires'] = len(goal_state.desires)
                        user_state_info['satisfied_desires'] = len(goal_state.get_satisfied_desires())
                        user_state_info['failed_desires'] = len(goal_state.get_failed_desires())
                        user_state_info['goal_achieved'] = goal_state.get_goal_achievement_fraction()
                    
                    debug_example = {
                        'obs': finished_episode.current_obs_for_action,
                        'candidates': high_level_candidates.copy() if high_level_candidates else [],
                        'q_values': q_vals.copy(),
                        'selected_action': selected_action,
                        'reward': finished_episode.reward,
                        'episode_idx': finished_episode.dialogue_idx,
                        'user_action': user_action,  # User's response
                        'agent_action': agent_action,  # Agent's action
                        'user_state': user_state_info  # User state (goal state info)
                    }
            
            # Finalize and record statistics
            stats = finished_episode.finalize()
            
            # Record episode statistics
            record_episode_statistics(
                stats=stats,
                episode_stats=episode_stats,
                evaluation_episodes=evaluation_episodes,
                config=config,
                episodes_completed=episodes_completed,
                debug_example_ref=[debug_example],  # Pass as list for modification
                multiwoz_mode=multiwoz_mode,
                model=model,
                tokenizer=tokenizer,
                hl_agent=hl_agent
            )

            if config.environment_type in ("cares", "wildjailbreak", "redbench", "harmbench"):
                base_prompt = finished_episode.dialogue_data.get("base_prompt", "")
                append_defense_episode_record(
                    output_dir=config.output_dir,
                    dialogue_id=stats["dialogue_id"],
                    episode_idx=stats["dialogue_idx"],
                    harmful_level=harmful_level_for_defense_log(stats),
                    goal_achieved=float(stats["goal_achieved"]),
                    turn_evaluation_data=stats.get("turn_evaluation_data", []),
                    defense_transcript=finished_episode.format_dialogue_history(),
                    environment_type=config.environment_type,
                    base_prompt=base_prompt if base_prompt else None,
                )

            # CARES/WildJailbreak/RedBench/HarmBench: print conversation histories for first adversarial and non-adversarial episodes
            if config.environment_type in ("cares", "wildjailbreak", "redbench", "harmbench"):
                harmful_level = stats.get("harmful_level")
                if harmful_level is None and isinstance(finished_episode.dialogue_data, dict):
                    harmful_level = finished_episode.dialogue_data.get("harmful_level", -1)
                if harmful_level == 0 and not cares_first_non_adv_printed[0]:
                    cares_first_non_adv_printed[0] = True
                    conv = finished_episode.format_dialogue_history()
                    base_prompt = finished_episode.dialogue_data.get("base_prompt", "") or ""
                    base_display = (base_prompt[:200] + "...") if len(base_prompt) > 200 else base_prompt
                    print("\n" + "=" * 80)
                    print("[CARES] First NON-ADVERSARIAL episode conversation (goal_achieved={:.2f}, reward={:.2f}):".format(
                        stats["goal_achieved"], stats["episode_total_reward"]))
                    print("  base_prompt (truncated):", base_display)
                    print("  Conversation:")
                    for line in conv.split("\n"):
                        print("    " + line)
                    print("=" * 80)
                    sys.stdout.flush()
                elif harmful_level is not None and harmful_level > 0 and not cares_first_adv_printed[0]:
                    cares_first_adv_printed[0] = True
                    conv = finished_episode.format_dialogue_history()
                    base_prompt = finished_episode.dialogue_data.get("base_prompt", "") or ""
                    base_display = (base_prompt[:200] + "...") if len(base_prompt) > 200 else base_prompt
                    print("\n" + "=" * 80)
                    print("[CARES] First ADVERSARIAL episode conversation (harmful_level={}, goal_achieved={:.2f}, reward={:.2f}):".format(
                        harmful_level, stats["goal_achieved"], stats["episode_total_reward"]))
                    print("  base_prompt (truncated):", base_display)
                    print("  Conversation:")
                    for line in conv.split("\n"):
                        print("    " + line)
                    print("=" * 80)
                    sys.stdout.flush()
            
            # Update ground truth pool for contrastive learning (only if contrastive loss is enabled)
            contrastive_coef = getattr(config, 'contrastive_coef', 0.0)
            ablation_mode = getattr(config, 'contrastive_ablation_mode', 'none')
            # Build pool if contrastive loss is enabled with crossed_data, OR if crossed_data_in_candidates mode
            needs_pool = ((contrastive_coef > 0.0 and ablation_mode == "crossed_data") or 
                         (ablation_mode == "crossed_data_in_candidates"))
            
            if needs_pool and multiwoz_mode and finished_episode.ground_truth_goal:
                gt_goal = finished_episode.ground_truth_goal
                if isinstance(gt_goal, str) and gt_goal.strip() and gt_goal.lower() != "user goal not specified":
                    ground_truth_pool.append(gt_goal)
                    # Limit pool size
                    if len(ground_truth_pool) > max_pool_size:
                        ground_truth_pool.pop(0)  # Remove oldest
            
            episodes_completed += 1
            
            # Print GPU memory after first episode completion
            if episodes_completed == 1:
                device_id = int(config.device.split(":")[-1]) if config.device.startswith("cuda") and ":" in config.device else 0
                print_gpu_memory("GPU Memory - AFTER first episode completion", include_max=True, restore_device=device_id)
            
            # Periodic diagnostic: Log P(a|noise) vs P(a|coherent) statistics every 1000 episodes
            # This gives ~6 data points per training run (assuming ~6000 episodes)
            if episodes_completed % 1000 == 0:
                try:
                    # Use current transition buffer for diagnostic (if available)
                    if transition_buffer and len(transition_buffer['high_level_actions']) > 0:
                        diagnostic_transitions = {
                            'observations': transition_buffer['observations'],
                            'high_level_actions': transition_buffer['high_level_actions'],
                            'low_level_actions': transition_buffer['low_level_actions'],
                            'next_observations': transition_buffer['next_observations'],
                            'rewards': transition_buffer['rewards'],
                            'terminals': transition_buffer['terminals']
                        }
                        log_p_action_belief_diagnostics(
                            transitions=diagnostic_transitions,
                            value_function=value_function,
                            ll_model=ll_agent.model,
                            ll_tokenizer=ll_agent.tokenizer,
                            config=config,
                            output_dir=config.output_dir
                        )
                except Exception as e:
                    # Don't fail training if diagnostic fails
                    print(f"Warning: P(a|b) diagnostic failed at episode {episodes_completed}: {e}")
            
            # Periodic training dynamics evaluation: Evaluate Q-value learning every 500 episodes
            # This tracks whether Q-values are learning P(a|b) or converging to constants
            if episodes_completed % 500 == 0 and episodes_completed > 0:
                try:
                    from .analysis.evaluate_training_dynamics import evaluate_training_dynamics, print_training_dynamics_summary
                    
                    # Use a subset of dialogues for quick evaluation
                    eval_dialogues = dialogues[:min(50, len(dialogues))] if dialogues else None
                    
                    dynamics_results = evaluate_training_dynamics(
                        value_function=value_function,
                        config=config,
                        model=model,
                        tokenizer=tokenizer,
                        device=config.device,
                        standard_dtype=standard_dtype,
                        n_examples=50,
                        dialogues=eval_dialogues,
                        episode_number=episodes_completed,
                        output_dir=config.output_dir
                    )
                    
                    print_training_dynamics_summary(dynamics_results)
                    sys.stdout.flush()
                except Exception as e:
                    # Don't fail training if dynamics evaluation fails
                    print(f"Warning: Training dynamics evaluation failed at episode {episodes_completed}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Print periodic summary (every 20 episodes)
            print_periodic_summary(
                episodes_completed=episodes_completed,
                total_episodes=total_episodes,
                episode_stats=episode_stats,
                start_time=start_time,
                summary_interval=100,
                config=config,
            )
            
            # Decay epsilon for online mode
            if _is_online_mode(config):
                current_epsilon = max(current_epsilon * config.epsilon_decay_rate, config.epsilon_min)
                config._current_epsilon = current_epsilon  # Store in config for access in process_turn_for_episode
            
            # Save checkpoint periodically (matching evaluation_interval) - skip in evaluation mode
            # Overwrite the latest checkpoint to save storage space
            checkpoint_interval = getattr(config, 'evaluation_interval', 100)
            if episodes_completed % checkpoint_interval == 0 and not getattr(config, "_evaluation_mode", False):
                checkpoint_dir = Path(config.output_dir) / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_dir / "value_function_latest.pt"
                try:
                    value_function.save_checkpoint(str(checkpoint_path))
                    print(f"Saved checkpoint to {checkpoint_path} (episode {episodes_completed})")
                except Exception as e:
                    print(f"ERROR: Failed to save checkpoint: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Replace with new episode if available and we haven't reached the limit
            if dialogue_queue and (max_dialogues_limit is None or max_dialogues_limit <= 0 or episodes_completed < max_dialogues_limit):
                new_dialogue_data = dialogue_queue.popleft()
                from .environments.vitabench_env import EpisodeVitaBenchEnvWrapper
                if config.environment_type == "vitabench" and isinstance(finished_episode.env, EpisodeVitaBenchEnvWrapper):
                    # Reuse same row of batch: replace slot with new task (no new env)
                    batch_env = finished_episode.env._batch_env
                    idx = finished_episode.env._idx
                    task_id = new_dialogue_data.get("id", str(next_episode_idx))
                    domain = new_dialogue_data.get("domain", "ota")
                    initial_obs, initial_info, dialogue_data, ground_truth_goal = batch_env.replace_slot(
                        idx, task_id, domain, config,
                        language=getattr(config, "language", "english"),
                        debug=config.debug,
                    )
                    finished_episode.reset_for_new_episode(
                        new_dialogue_idx=next_episode_idx,
                        new_dialogue_data=dialogue_data,
                        new_ground_truth_goal=ground_truth_goal,
                        new_env=finished_episode.env,
                        new_initial_observation=initial_obs,
                        new_initial_env_info=initial_info,
                    )
                else:
                    new_episode = create_episode_state(
                        next_episode_idx,
                        new_dialogue_data,
                        multiwoz_mode,
                        config,
                        config.debug,
                        user_agent=user_agent if _is_online_mode(config) else None,
                        persona=None,
                        patient_agent=patient_agent if config.environment_type in ("cares", "wildjailbreak", "redbench", "harmbench") else None,
                        reward_model=reward_model if config.environment_type in ("cares", "wildjailbreak", "redbench", "harmbench") else None,
                        reward_tokenizer=reward_tokenizer if config.environment_type in ("cares", "wildjailbreak", "redbench", "harmbench") else None,
                        judge_model=judge_model if config.environment_type in ("cares", "wildjailbreak", "redbench", "harmbench") else None,
                        judge_tokenizer=judge_tokenizer if config.environment_type in ("cares", "wildjailbreak", "redbench", "harmbench") else None,
                    )
                    finished_episode.reset_for_new_episode(
                        new_dialogue_idx=next_episode_idx,
                        new_dialogue_data=new_episode.dialogue_data,
                        new_ground_truth_goal=new_episode.ground_truth_goal,
                        new_env=new_episode.env,
                        new_initial_observation=new_episode.initial_observation,
                        new_initial_env_info=new_episode.initial_env_info
                    )
                finished_episode.initialize()
                next_episode_idx += 1
            else:
                # No more episodes or reached limit - remove from active pool
                active_episodes.remove(finished_episode)
        
        # 7. Print statistics when threshold reached (handle episodes finishing at different times)
        # Since episodes finish at different times, check if we've crossed an evaluation-interval boundary
        # Print when we've completed at least one more evaluation interval since last print
        if (
            episodes_completed > 0
            and episodes_completed >= eval_interval
            and (episodes_completed // eval_interval) > (last_printed_episode_count // eval_interval)
        ):
            # Evaluate beliefs against ground truth
            print(f"\n[Evaluation] Episodes completed: {episodes_completed}, Evaluation episodes available: {len(evaluation_episodes)}")
            if not evaluation_episodes:
                print(f"[Evaluation] WARNING: No evaluation episodes available (episodes may not have goal_json). Skipping evaluation.")
                sys.stdout.flush()
            elif evaluation_episodes:
                print(f"Computing belief evaluation metrics for {len(evaluation_episodes)} episodes...")
                sys.stdout.flush()
                
                # Convert all goals to natural language (batched) and filter invalid entries
                filtered_eval_data = []
                for ep_data in evaluation_episodes:
                    goal_json = ep_data.get('goal_json')
                    dialogue_id = ep_data.get('dialogue_id')
                    
                    if not is_valid_multiwoz_goal(goal_json):
                        print(f"[WARNING] Skipping episode {dialogue_id}: invalid goal_json structure ({goal_json})")
                        continue
                    
                    gt_text = convert_goal_json_to_natural_language(
                        goal_json=goal_json,
                        hl_agent=hl_agent,
                        max_new_tokens=config.max_tokens,
                        model=model  # Pass main model in case hl_agent.model is None (GPT agents)
                    )
                    
                    if gt_text in ["[NO_GOAL_SPECIFIED]", "[CONVERSION_FAILED]"]:
                        print(f"[WARNING] Skipping episode {dialogue_id}: goal conversion failed ({gt_text})")
                        continue
                    
                    filtered_eval_data.append((ep_data, gt_text))
                
                if not filtered_eval_data:
                    print("[Evaluation] WARNING: No valid evaluation episodes after filtering invalid goals.")
                    sys.stdout.flush()
                    continue
                
                valid_eval_episodes = [ep for ep, _ in filtered_eval_data]
                
                # Batch compute metrics for all turns (grouped by episode)
                metrics_rows = []
                episode_cosine_sims = []
                episode_l2_dists = []
                baseline_belief_field_map = {
                    'min_q': 'min_q_baseline_belief',
                    'random': 'random_baseline_belief'
                }
                
                cosine_debug_enabled = (
                    bool(os.environ.get("LLM_CONTEXT_DEBUG_COSINE"))
                    or getattr(config, "debug", False)
                )
                max_cosine_debug_examples = 3
                
                # Q-value diagnostic collection
                q_value_diagnostics = []
                
                for ep_idx, (ep_data, ground_truth_text) in enumerate(filtered_eval_data):
                    if not ep_data['chosen_beliefs']:
                        continue
                    
                    # Get metrics for this episode's beliefs
                    turn_metrics = evaluate_beliefs_against_ground_truth(
                        chosen_beliefs=ep_data['chosen_beliefs'],
                        ground_truth_text=ground_truth_text,
                        model=model,
                        tokenizer=tokenizer,
                        device=config.device if config.device != "cuda" else "cuda:0"
                    )
                    turn_eval_data = ep_data.get('turn_evaluation_data', [])
                    
                    # Compute episode averages
                    if turn_metrics:
                        avg_cosine = sum(m['cosine_similarity'] for m in turn_metrics) / len(turn_metrics)
                        avg_l2 = sum(m['l2_distance'] for m in turn_metrics) / len(turn_metrics)
                        episode_cosine_sims.append(avg_cosine)
                        episode_l2_dists.append(avg_l2)
                    
                    if cosine_debug_enabled and turn_metrics:
                        print("\n[CosineDebug] Evaluation sample")
                        print(f"  Episode ID: {ep_data.get('episode_id')} | Dialogue: {ep_data.get('dialogue_id')}")
                        print("  Ground Truth Text:")
                        print(f"  {ground_truth_text}")
                        print("  Sample belief comparisons:")
                        for turn_idx, metrics in enumerate(turn_metrics[:max_cosine_debug_examples]):
                            belief_text = ep_data['chosen_beliefs'][turn_idx]
                            print(
                                f"    Turn {turn_idx}: "
                                f"cos={metrics['cosine_similarity']:.4f} "
                                f"belief='{belief_text}'"
                            )
                        print()
                    
                    # Precompute cosine similarities for baseline beliefs per turn
                    num_turns = len(turn_metrics)
                    per_turn_baseline_cosines: Dict[str, List[Optional[float]]] = {
                        key: [None] * num_turns for key in baseline_belief_field_map.keys()
                    }
                    if num_turns and turn_eval_data:
                        for baseline_key, field_name in baseline_belief_field_map.items():
                            belief_texts: List[str] = []
                            turn_indices: List[int] = []
                            for turn_idx in range(min(num_turns, len(turn_eval_data))):
                                belief_text = (turn_eval_data[turn_idx] or {}).get(field_name)
                                if belief_text and belief_text.strip():
                                    belief_texts.append(belief_text)
                                    turn_indices.append(turn_idx)
                            if belief_texts:
                                baseline_metrics = evaluate_beliefs_against_ground_truth(
                                    chosen_beliefs=belief_texts,
                                    ground_truth_text=ground_truth_text,
                                    model=model,
                                    tokenizer=tokenizer,
                                    device=config.device if config.device != "cuda" else "cuda:0"
                                )
                                for assignment_idx, metrics in zip(turn_indices, baseline_metrics):
                                    per_turn_baseline_cosines[baseline_key][assignment_idx] = metrics['cosine_similarity']
                    
                    # Q-value diagnostic: Collect data for this episode
                    if turn_eval_data and turn_metrics:
                        for turn_idx in range(min(len(turn_metrics), len(turn_eval_data))):
                            turn_eval = turn_eval_data[turn_idx]
                            q_values = turn_eval.get('q_values', {})
                            if not q_values:
                                continue
                            
                            # Filter out [SKIP]
                            valid_q_values = {k: v for k, v in q_values.items() if k != "[SKIP]"}
                            if len(valid_q_values) < 2:
                                continue
                            
                            max_q_belief = max(valid_q_values, key=valid_q_values.get)
                            min_q_belief = min(valid_q_values, key=valid_q_values.get)
                            max_q_value = valid_q_values[max_q_belief]
                            min_q_value = valid_q_values[min_q_belief]
                            
                            # Get cosine similarities
                            max_q_cos = turn_metrics[turn_idx]['cosine_similarity']  # Max-Q is the chosen belief
                            min_q_cos = per_turn_baseline_cosines.get('min_q', [None] * num_turns)[turn_idx]
                            
                            if min_q_cos is not None:
                                # Compute P(a|b) for max-Q and min-Q beliefs
                                # Fail loudly if required data is missing (per research codebase rules)
                                if turn_idx >= len(turn_eval_data):
                                    raise ValueError(f"Turn {turn_idx} missing from turn_eval_data (len={len(turn_eval_data)})")
                                
                                turn_eval = turn_eval_data[turn_idx]
                                if not isinstance(turn_eval, dict):
                                    raise ValueError(f"turn_eval_data[{turn_idx}] is not a dict: {type(turn_eval)}")
                                
                                if 'll_action' not in turn_eval:
                                    raise ValueError(
                                        f"turn_eval_data[{turn_idx}] missing 'll_action' key. "
                                        f"Keys: {list(turn_eval.keys())}. "
                                        f"Every user turn should have a corresponding agent response."
                                    )
                                
                                dataset_action = turn_eval['ll_action']
                                if not isinstance(dataset_action, str):
                                    raise ValueError(
                                        f"turn_eval_data[{turn_idx}]['ll_action'] is not a string: {type(dataset_action)}. "
                                        f"Value: {repr(dataset_action)}"
                                    )
                                if not dataset_action.strip():
                                    raise ValueError(
                                        f"turn_eval_data[{turn_idx}]['ll_action'] is empty. "
                                        f"Every user turn should have a corresponding agent response. "
                                        f"Episode: {ep_data.get('dialogue_id', 'unknown')}, Turn: {turn_idx}. "
                                        f"Full turn_eval: {turn_eval}"
                                    )
                                
                                if model is None:
                                    raise ValueError(
                                        f"model is None - cannot compute P(a|b). "
                                        f"This should never happen during evaluation."
                                    )
                                if tokenizer is None:
                                    raise ValueError(
                                        f"tokenizer is None - cannot compute P(a|b). "
                                        f"This should never happen during evaluation."
                                    )
                                
                                belief_only = getattr(config, 'll_action_belief_only', True)
                                
                                if not belief_only:
                                    if 'observation' not in turn_eval:
                                        raise ValueError(
                                            f"turn_eval_data[{turn_idx}] missing 'observation' key "
                                            f"(required when belief_only=False). Keys: {list(turn_eval.keys())}"
                                        )
                                    observation = turn_eval['observation']
                                    if not isinstance(observation, str):
                                        raise ValueError(
                                            f"turn_eval_data[{turn_idx}]['observation'] is not a string: {type(observation)}"
                                        )
                                    if not observation.strip():
                                        raise ValueError(
                                            f"turn_eval_data[{turn_idx}]['observation'] is empty "
                                            f"(required when belief_only=False). Episode: {ep_data.get('dialogue_id', 'unknown')}, Turn: {turn_idx}"
                                        )
                                else:
                                    observation = ""
                                
                                # Batch compute P(a|max-Q belief) and P(a|min-Q belief)
                                p_ab_results = value_function.compute_likelihood_batch(
                                    observations=[observation, observation],
                                    high_level_actions=[max_q_belief, min_q_belief],
                                    low_level_actions=[dataset_action, dataset_action],
                                    ll_model=model,
                                    ll_tokenizer=tokenizer,
                                    belief_only=belief_only
                                )
                                max_q_p_ab = p_ab_results[0]
                                min_q_p_ab = p_ab_results[1]
                                
                                q_value_diagnostics.append({
                                    'max_q_value': max_q_value,
                                    'min_q_value': min_q_value,
                                    'max_q_cosine': max_q_cos,
                                    'min_q_cosine': min_q_cos,
                                    'q_value_diff': max_q_value - min_q_value,
                                    'cosine_diff': max_q_cos - min_q_cos,
                                    'max_q_p_ab': max_q_p_ab,
                                    'min_q_p_ab': min_q_p_ab,
                                    'p_ab_diff': max_q_p_ab - min_q_p_ab
                                })
                    
                    # Track low cosine cases for pattern analysis
                    low_cosine_threshold = 0.1
                    low_cosine_cases = []
                    
                    # Create rows for CSV
                    for turn_idx, metrics in enumerate(turn_metrics):
                        turn_eval = turn_eval_data[turn_idx] if turn_idx < len(turn_eval_data) else {}
                        min_q_cos = per_turn_baseline_cosines.get('min_q', [None] * num_turns)[turn_idx] if num_turns else None
                        random_cos = per_turn_baseline_cosines.get('random', [None] * num_turns)[turn_idx] if num_turns else None
                        
                        # Compute context length (number of dialogue turns available at belief generation time)
                        # Turn 0: uses initial observation (1 context entry)
                        # Turn 1: has 1 previous turn in history (1 context entry)  
                        # Turn n: has n previous turns in history (n context entries)
                        # For simplicity, we use turn_idx + 1 as a proxy (turn 0 = 1, turn 1 = 2, etc.)
                        context_length = turn_idx + 1
                        
                        cosine_sim = metrics['cosine_similarity']
                        chosen_belief = ep_data['chosen_beliefs'][turn_idx]
                        
                        # Track low cosine cases
                        if cosine_sim < low_cosine_threshold:
                            observation = turn_eval.get('observation', '') if turn_eval else ''
                            candidate_beliefs = turn_eval.get('candidate_beliefs', []) if turn_eval else []
                            q_values = turn_eval.get('q_values', {}) if turn_eval else {}
                            selected_belief = turn_eval.get('selected_belief', '') if turn_eval else ''
                            
                            low_cosine_cases.append({
                                'episode_id': ep_data['episode_id'],
                                'dialogue_id': ep_data['dialogue_id'],
                                'turn': turn_idx,
                                'context_length': context_length,
                                'cosine_similarity': cosine_sim,
                                'chosen_belief': chosen_belief,
                                'ground_truth': ground_truth_text,
                                'observation': observation,
                                'num_candidates': len(candidate_beliefs) if candidate_beliefs else 0,
                                'is_skip': chosen_belief == "[SKIP]",
                                'is_max_q': chosen_belief == selected_belief,
                                'q_value': q_values.get(chosen_belief, None) if q_values else None
                            })
                        
                        # Extract Q-values for different beliefs
                        q_values = turn_eval.get('q_values', {}) if turn_eval else {}
                        selected_belief = turn_eval.get('selected_belief', '') if turn_eval else ""
                        min_q_belief_text = (turn_eval.get('min_q_baseline_belief') or "") if turn_eval else ""
                        random_belief_text = (turn_eval.get('random_baseline_belief') or "") if turn_eval else ""
                        
                        max_q_value = q_values.get(selected_belief, None) if selected_belief and q_values else None
                        min_q_value = q_values.get(min_q_belief_text, None) if min_q_belief_text and q_values else None
                        random_q_value = q_values.get(random_belief_text, None) if random_belief_text and q_values else None
                        
                        metrics_rows.append({
                            'episode_id': ep_data['episode_id'],
                            'dialogue_id': ep_data['dialogue_id'],
                            'turn': turn_idx,
                            'context_length': context_length,
                            'chosen_belief': chosen_belief,
                            'ground_truth_text': ground_truth_text,
                            'cosine_similarity': cosine_sim,
                            'l2_distance': metrics['l2_distance'],
                            'min_q_belief': min_q_belief_text,
                            'random_belief': random_belief_text,
                            'min_q_cosine_similarity': "" if min_q_cos is None else min_q_cos,
                            'random_cosine_similarity': "" if random_cos is None else random_cos,
                            'max_q_value': "" if max_q_value is None else max_q_value,
                            'min_q_value': "" if min_q_value is None else min_q_value,
                            'random_q_value': "" if random_q_value is None else random_q_value
                        })
                    
                    # Log low cosine cases for this episode
                    if low_cosine_cases and ep_idx % 500 == 0:
                        print(f"\n[LowCosineAnalysis] Episode {ep_data.get('dialogue_id')} ({ep_data.get('episode_id')}): {len(low_cosine_cases)}/{num_turns} turns with cosine < {low_cosine_threshold}")
                        for case in low_cosine_cases[:3]:  # Show first 5 cases
                            print(f"  Turn {case['turn']} (context_len={case['context_length']}, cos={case['cosine_similarity']:.4f}):")
                            belief_text = case['chosen_belief'][:100] + ('...' if len(case['chosen_belief']) > 100 else '')
                            print(f"    Belief: '{belief_text}'")
                            gt_text = case['ground_truth'][:100] + ('...' if len(case['ground_truth']) > 100 else '')
                            print(f"    Ground Truth: '{gt_text}'")
                            if case['is_skip']:
                                print(f"    [WARNING] Belief is [SKIP] placeholder")
                            print()
                        sys.stdout.flush()
                
                # Store average stats for these episodes
                for cosine_sim, l2_dist in zip(episode_cosine_sims, episode_l2_dists):
                    episode_stats['avg_cosine_similarity'].append(cosine_sim)
                    episode_stats['avg_l2_distance'].append(l2_dist)
                
                # Compute log probability of correct LL action given max-Q belief candidate
                print(f"Computing log probability of correct LL action given max-Q belief...")
                sys.stdout.flush()
                all_log_probs_max_q = []
                
                # Collect all (observation, max_q_belief, ll_action) tuples for batching
                batch_observations = []
                batch_beliefs = []
                batch_actions = []
                belief_only = getattr(config, 'll_action_belief_only', True)
                
                for ep_data in valid_eval_episodes:
                    turn_eval_data = ep_data.get('turn_evaluation_data', [])
                    if not turn_eval_data:
                        continue
                    
                    for turn_data in turn_eval_data:
                        observation = turn_data.get('observation', '')
                        ll_action = turn_data.get('ll_action', '')
                        candidate_beliefs = turn_data.get('candidate_beliefs', [])
                        q_values = turn_data.get('q_values', {})
                        
                        if not observation or not ll_action or not candidate_beliefs or not q_values:
                            continue
                        
                        # Filter out [SKIP] candidates
                        skip_token = "[SKIP]"
                        valid_q_values = {k: v for k, v in q_values.items() if k != skip_token and k in candidate_beliefs}
                        if not valid_q_values:
                            continue
                        
                        # Find max-Q belief candidate
                        max_q_belief = max(valid_q_values, key=valid_q_values.get)
                        
                        # Collect for batching
                        batch_observations.append(observation if not belief_only else "")
                        batch_beliefs.append(max_q_belief)
                        batch_actions.append(ll_action)
                
                # Batch compute all log probabilities in chunks to prevent OOM
                all_log_probs_max_q = []
                if batch_observations:
                    # Process in chunks to prevent OOM with large evaluation sets
                    chunk_size = getattr(config, 'batch_generation_chunk_size', 16)
                    num_chunks = (len(batch_observations) + chunk_size - 1) // chunk_size
                    
                    for chunk_idx in range(num_chunks):
                        start_idx = chunk_idx * chunk_size
                        end_idx = min(start_idx + chunk_size, len(batch_observations))
                        
                        chunk_observations = batch_observations[start_idx:end_idx]
                        chunk_beliefs = batch_beliefs[start_idx:end_idx]
                        chunk_actions = batch_actions[start_idx:end_idx]
                        
                        # Use main model if ll_agent.model is None (e.g., when using GPT agents)
                        ll_model = ll_agent.model if ll_agent.model is not None else model
                        ll_tokenizer = ll_agent.tokenizer if ll_agent.model is not None else tokenizer
                        
                        chunk_log_probs = value_function.compute_likelihood_batch(
                            observations=chunk_observations,
                            high_level_actions=chunk_beliefs,
                            low_level_actions=chunk_actions,
                            ll_model=ll_model,
                            ll_tokenizer=ll_tokenizer,
                            belief_only=belief_only
                        )
                        
                        # Collect valid log probs from this chunk
                        all_log_probs_max_q.extend([lp for lp in chunk_log_probs if lp != float('-inf')])
                        
                        # Cleanup after each chunk
                        del chunk_observations, chunk_beliefs, chunk_actions, chunk_log_probs
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                # Store average log prob in episode_stats
                if all_log_probs_max_q:
                    avg_log_prob_max_q = sum(all_log_probs_max_q) / len(all_log_probs_max_q)
                    # Store per episode (use same count as belief evaluation)
                    for _ in range(len(episode_cosine_sims)):
                        episode_stats['avg_log_prob_max_q_ll_action'].append(avg_log_prob_max_q)
                    print(f"  Average log prob of correct LL action given max-Q belief: {avg_log_prob_max_q:.4f} ({len(all_log_probs_max_q)} turns)")
                else:
                    print(f"  No valid log probabilities computed (no turns with valid data)")
                sys.stdout.flush()
                
                # Baseline comparison metrics
                baseline_metrics_lists = {
                    'max_q_accuracy': [],
                    'min_q_accuracy': [],
                    'top_prob_accuracy': [],
                    'random_accuracy': []
                }
                evaluated_turns = 0
                for ep_data in evaluation_episodes:
                    for turn_data in ep_data.get('turn_evaluation_data', []):
                        metrics = turn_data.get('baseline_metrics') or {}
                        recorded = False
                        for key in baseline_metrics_lists.keys():
                            value = metrics.get(key)
                            if value is not None:
                                baseline_metrics_lists[key].append(value)
                                recorded = True
                        if recorded:
                            evaluated_turns += 1
                
                if evaluated_turns > 0:
                    baseline_averages = {
                        key: sum(values) / len(values)
                        for key, values in baseline_metrics_lists.items()
                        if values
                    }
                    print(f"  Baseline comparison over {evaluated_turns} turns:")
                    for key, avg_val in baseline_averages.items():
                        print(f"    {key}: {avg_val:.4f}")
                    baseline_report_path = Path(config.output_dir) / f"baseline_eval_{episodes_completed}.json"
                    with open(baseline_report_path, 'w') as baseline_file:
                        json.dump(
                            {
                                'episodes_completed': episodes_completed,
                                'turns_evaluated': evaluated_turns,
                                'averages': baseline_averages
                            },
                            baseline_file,
                            indent=2
                        )
                    print(f"  Saved baseline comparison report to {baseline_report_path}")
                    sys.stdout.flush()
                else:
                    print("  Baseline comparison skipped (no ground-truth turns available)")
                    sys.stdout.flush()
                
                # Save to CSV
                if metrics_rows:
                    import csv
                    csv_path = Path(config.output_dir) / f"belief_evaluation_episode_{episodes_completed}.csv"
                    csv_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(csv_path, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=metrics_rows[0].keys())
                        writer.writeheader()
                        writer.writerows(metrics_rows)
                    
                    print(f"Saved belief evaluation metrics to {csv_path}")
                    sys.stdout.flush()
                    
                    # Q-value diagnostic: Print analysis
                    if q_value_diagnostics:
                        print("\n" + "="*80)
                        print("[Q-Value Diagnostic] Max-Q vs Min-Q Analysis")
                        print("="*80)
                        print(f"Analyzed {len(q_value_diagnostics)} turns with both Max-Q and Min-Q beliefs")
                        
                        # Check if Q-values are correctly ordered
                        correct_ordering = sum(1 for d in q_value_diagnostics if d['max_q_value'] > d['min_q_value'])
                        print(f"Q-value ordering: {correct_ordering}/{len(q_value_diagnostics)} turns have max_q_value > min_q_value (expected: all)")
                        
                        # Check if cosine similarity matches Q-value ordering
                        cosine_matches_q = sum(1 for d in q_value_diagnostics if d['max_q_cosine'] > d['min_q_cosine'])
                        print(f"Cosine matches Q-order: {cosine_matches_q}/{len(q_value_diagnostics)} turns have max_q_cosine > min_q_cosine")
                        print(f"  (If this is low, Q-function is learning backwards - higher Q-values for worse beliefs)")
                        
                        # Statistics
                        avg_max_q = sum(d['max_q_value'] for d in q_value_diagnostics) / len(q_value_diagnostics)
                        avg_min_q = sum(d['min_q_value'] for d in q_value_diagnostics) / len(q_value_diagnostics)
                        avg_max_cos = sum(d['max_q_cosine'] for d in q_value_diagnostics) / len(q_value_diagnostics)
                        avg_min_cos = sum(d['min_q_cosine'] for d in q_value_diagnostics) / len(q_value_diagnostics)
                        
                        print(f"\nAverage Q-values:")
                        print(f"  Max-Q: {avg_max_q:.4f}")
                        print(f"  Min-Q: {avg_min_q:.4f}")
                        print(f"  Difference: {avg_max_q - avg_min_q:.4f} (should be positive)")
                        
                        print(f"\nAverage Cosine Similarities:")
                        print(f"  Max-Q: {avg_max_cos:.4f}")
                        print(f"  Min-Q: {avg_min_cos:.4f}")
                        print(f"  Difference: {avg_max_cos - avg_min_cos:.4f} (positive = Max-Q better, negative = Min-Q better)")
                        
                        # Show sample cases where ordering is wrong
                        wrong_ordering = [d for d in q_value_diagnostics if d['max_q_cosine'] < d['min_q_cosine']]
                        if wrong_ordering:
                            print(f"\n[WARNING] {len(wrong_ordering)}/{len(q_value_diagnostics)} turns where Max-Q has LOWER cosine than Min-Q")
                            print("Sample cases (Max-Q worse than Min-Q):")
                            for d in wrong_ordering[:5]:
                                print(f"  Q-values: max={d['max_q_value']:.4f}, min={d['min_q_value']:.4f}, diff={d['q_value_diff']:.4f}")
                                print(f"  Cosines: max={d['max_q_cosine']:.4f}, min={d['min_q_cosine']:.4f}, diff={d['cosine_diff']:.4f}")
                        
                        # Compute P(a|b) statistics
                        p_ab_diagnostics = [d for d in q_value_diagnostics if d.get('max_q_p_ab') is not None and d.get('min_q_p_ab') is not None]
                        if p_ab_diagnostics:
                            p_ab_matches_q = sum(1 for d in p_ab_diagnostics if d['max_q_p_ab'] > d['min_q_p_ab'])
                            avg_max_p_ab = sum(d['max_q_p_ab'] for d in p_ab_diagnostics) / len(p_ab_diagnostics)
                            avg_min_p_ab = sum(d['min_q_p_ab'] for d in p_ab_diagnostics) / len(p_ab_diagnostics)
                            
                            print(f"\nP(a|b) Analysis:")
                            print(f"  P(a|max-Q) > P(a|min-Q): {p_ab_matches_q}/{len(p_ab_diagnostics)} turns ({100*p_ab_matches_q/len(p_ab_diagnostics):.1f}%)")
                            print(f"  Average P(a|max-Q): {avg_max_p_ab:.4f}")
                            print(f"  Average P(a|min-Q): {avg_min_p_ab:.4f}")
                            print(f"  Average difference: {avg_max_p_ab - avg_min_p_ab:.4f} (positive = max-Q better)")
                            
                            if p_ab_matches_q / len(p_ab_diagnostics) > 0.7:
                                print(f"  ✓ Q-function IS learning P(a|b) correctly!")
                                print(f"  → Cosine similarity may not be the right metric - Q learns action probability, not semantic similarity")
                            elif p_ab_matches_q / len(p_ab_diagnostics) < 0.5:
                                print(f"  [WARNING] Q-function is NOT learning P(a|b) correctly!")
                        else:
                            print(f"\nP(a|b) Analysis: Not available (could not compute P(a|b) for diagnostic turns)")
                        
                        # Compute correlation between Q-value differences and cosine differences
                        try:
                            import numpy as np
                            q_diffs = [d['q_value_diff'] for d in q_value_diagnostics]
                            cos_diffs = [d['cosine_diff'] for d in q_value_diagnostics]
                            
                            if len(q_diffs) > 1 and np.std(q_diffs) > 0 and np.std(cos_diffs) > 0:
                                correlation = np.corrcoef(q_diffs, cos_diffs)[0, 1]
                                print(f"\nCorrelation Analysis (Q-values vs Cosine Similarity):")
                                print(f"  Pearson correlation (Q-diff vs Cosine-diff): {correlation:.4f}")
                                print(f"  (1.0 = perfect correlation, 0.0 = no correlation, -1.0 = perfect negative)")
                                if abs(correlation) < 0.3:
                                    print(f"  [WARNING] Very weak correlation - Q-values are not learning to match cosine similarity!")
                                    if p_ab_diagnostics and p_ab_matches_q / len(p_ab_diagnostics) > 0.7:
                                        print(f"  → This is expected if Q learns P(a|b) rather than semantic similarity")
                                elif abs(correlation) < 0.5:
                                    print(f"  [WARNING] Weak correlation - Q-values only weakly match cosine similarity")
                            
                            # Correlation between Q-values and P(a|b) if available
                            if p_ab_diagnostics:
                                q_diffs_pab = [d['q_value_diff'] for d in p_ab_diagnostics]
                                p_ab_diffs = [d['p_ab_diff'] for d in p_ab_diagnostics if d['p_ab_diff'] is not None]
                                if len(q_diffs_pab) > 1 and len(p_ab_diffs) > 1 and len(q_diffs_pab) == len(p_ab_diffs) and np.std(q_diffs_pab) > 0 and np.std(p_ab_diffs) > 0:
                                    correlation_pab = np.corrcoef(q_diffs_pab, p_ab_diffs)[0, 1]
                                    print(f"\nCorrelation Analysis (Q-values vs P(a|b)):")
                                    print(f"  Pearson correlation (Q-diff vs P(a|b)-diff): {correlation_pab:.4f}")
                                    if abs(correlation_pab) > 0.5:
                                        print(f"  ✓ Strong correlation - Q-values ARE learning P(a|b)!")
                                    elif abs(correlation_pab) > 0.3:
                                        print(f"  Moderate correlation - Q-values partially learn P(a|b)")
                                    else:
                                        print(f"  [WARNING] Weak correlation - Q-values not strongly learning P(a|b)")
                        except Exception as e:
                            print(f"  Could not compute correlation: {e}")
                        
                        print("="*80 + "\n")
                        sys.stdout.flush()
                    
                    # Pattern analysis: summarize low cosine cases
                    print("\n" + "="*80)
                    print("[CosineSimilarityPatternAnalysis] Summary")
                    print("="*80)
                    
                    low_cosine_threshold = 0.1
                    all_cosines = [float(row['cosine_similarity']) for row in metrics_rows]
                    low_cosines = [c for c in all_cosines if c < low_cosine_threshold]
                    
                    if all_cosines:
                        print(f"Total turns evaluated: {len(all_cosines)}")
                        print(f"Low cosine cases (cos < {low_cosine_threshold}): {len(low_cosines)} ({100*len(low_cosines)/len(all_cosines):.1f}%)")
                        print(f"Cosine statistics:")
                        print(f"  Mean: {sum(all_cosines)/len(all_cosines):.4f}")
                        print(f"  Median: {sorted(all_cosines)[len(all_cosines)//2]:.4f}")
                        print(f"  Min: {min(all_cosines):.4f}")
                        print(f"  Max: {max(all_cosines):.4f}")
                        print(f"  Std: {(sum((c - sum(all_cosines)/len(all_cosines))**2 for c in all_cosines) / len(all_cosines))**0.5:.4f}")
                        
                        # Analyze by turn number
                        turn_cosines: Dict[int, List[float]] = {}
                        for row in metrics_rows:
                            turn_num = int(row['turn'])
                            if turn_num not in turn_cosines:
                                turn_cosines[turn_num] = []
                            turn_cosines[turn_num].append(float(row['cosine_similarity']))
                        
                        print(f"\nCosine similarity by turn number:")
                        for turn_num in sorted(turn_cosines.keys()):
                            turn_vals = turn_cosines[turn_num]
                            avg_turn_cos = sum(turn_vals) / len(turn_vals)
                            low_count = sum(1 for v in turn_vals if v < low_cosine_threshold)
                            print(f"  Turn {turn_num}: mean={avg_turn_cos:.4f}, low_cos={low_count}/{len(turn_vals)} ({100*low_count/len(turn_vals):.1f}%)")
                        
                        # Analyze by context length
                        context_cosines: Dict[int, List[float]] = {}
                        for row in metrics_rows:
                            ctx_len = int(row.get('context_length', 0))
                            if ctx_len not in context_cosines:
                                context_cosines[ctx_len] = []
                            context_cosines[ctx_len].append(float(row['cosine_similarity']))
                        
                        print(f"\nCosine similarity by context length:")
                        for ctx_len in sorted(context_cosines.keys()):
                            ctx_vals = context_cosines[ctx_len]
                            avg_ctx_cos = sum(ctx_vals) / len(ctx_vals)
                            low_count = sum(1 for v in ctx_vals if v < low_cosine_threshold)
                            print(f"  Context len {ctx_len}: mean={avg_ctx_cos:.4f}, low_cos={low_count}/{len(ctx_vals)} ({100*low_count/len(ctx_vals):.1f}%)")
                        
                        # Check for [SKIP] beliefs
                        skip_count = sum(1 for row in metrics_rows if row.get('chosen_belief', '') == '[SKIP]')
                        if skip_count > 0:
                            print(f"\n[WARNING] {skip_count}/{len(metrics_rows)} beliefs are [SKIP] placeholders")
                            skip_cosines = [float(row['cosine_similarity']) for row in metrics_rows if row.get('chosen_belief', '') == '[SKIP]']
                            if skip_cosines:
                                print(f"  [SKIP] beliefs mean cosine: {sum(skip_cosines)/len(skip_cosines):.4f}")
                        
                    print("="*80 + "\n")
                    sys.stdout.flush()
                
                # Run contrastive test
                # Use observations from evaluation episodes as test inputs
                # We need at least 2 episodes for contrastive test
                if len(filtered_eval_data) >= 2:
                    # Extract observations from evaluation episodes
                    # Use initial observations if available, otherwise fall back to ground truth texts
                    test_inputs = []
                    for ep_data, ground_truth_text in filtered_eval_data:
                        if 'initial_observation' in ep_data and ep_data['initial_observation']:
                            test_inputs.append(ep_data['initial_observation'])
                        else:
                            # Fallback: use ground truth text as proxy
                            test_inputs.append(ground_truth_text)
                    
                    # Ensure we have an even number (required for contrastive test pairing)
                    # If odd but >= 2, discard the last one to make it even (so we get at least 2)
                    if len(test_inputs) % 2 != 0:
                        if len(test_inputs) >= 2:
                            original_count = len(test_inputs)
                            test_inputs = test_inputs[:-1]  # Remove last one if odd
                            print(f"[Contrastive Test] Discarded 1 test input to make even number (had {original_count}, using {len(test_inputs)})")
                        else:
                            # If we only have 1, we can't do contrastive test
                            test_inputs = []
                    
                    if len(test_inputs) >= 2:
                        print(f"Running contrastive test on {len(test_inputs)} test inputs (from {len(filtered_eval_data)} evaluation episodes)...")
                        sys.stdout.flush()
                        try:
                            contrastive_results = run_contrastive_test(
                                test_inputs=test_inputs,
                                hl_agent=hl_agent,
                                value_function=value_function,
                                tokenizer=tokenizer,
                                config=config,
                                n_beliefs=5
                            )
                            
                            # Save contrastive test results
                            import csv
                            contrastive_csv_path = Path(config.output_dir) / f"contrastive_test_episode_{episodes_completed}.csv"
                            
                            # Flatten results for CSV
                            contrastive_rows = []
                            for result in contrastive_results:
                                row = {
                                    'test_idx': result['test_idx'],
                                    'is_contrastive': result['is_contrastive'],
                                    'observation': result['observation'][:500] if len(result['observation']) > 500 else result['observation'],  # Truncate long observations
                                    'belief_2': result.get('belief_2', ''),
                                    'belief_4': result.get('belief_4', ''),
                                    'error': result.get('error', '')
                                }
                                
                                # Add initial scores
                                if result.get('initial_scores'):
                                    for i, (cand, score) in enumerate(result['initial_scores'].items()):
                                        row[f'initial_score_{i}'] = score
                                        row[f'initial_belief_{i}'] = cand[:200] if len(cand) > 200 else cand
                                
                                # Add final scores and changes for contrastive tests
                                if result.get('is_contrastive') and result.get('final_scores'):
                                    row['swapped_belief_2'] = result.get('swapped_from_prev', {}).get('belief_2', '')
                                    row['swapped_belief_4'] = result.get('swapped_from_prev', {}).get('belief_4', '')
                                    
                                    for i, (cand, score) in enumerate(result['final_scores'].items()):
                                        row[f'final_score_{i}'] = score
                                        row[f'final_belief_{i}'] = cand[:200] if len(cand) > 200 else cand
                                    
                                    # Add score and rank changes
                                    if result.get('score_changes'):
                                        for swapped_belief, change in result['score_changes'].items():
                                            if change is not None:
                                                row[f'score_change_{swapped_belief[:50]}'] = change
                                    
                                    if result.get('rank_changes'):
                                        for swapped_belief, change in result['rank_changes'].items():
                                            if change is not None:
                                                row[f'rank_change_{swapped_belief[:50]}'] = change
                                
                                contrastive_rows.append(row)
                            
                            if contrastive_rows:
                                # Get all unique keys for CSV header
                                all_keys = set()
                                for row in contrastive_rows:
                                    all_keys.update(row.keys())
                                
                                with open(contrastive_csv_path, 'w', newline='') as f:
                                    writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
                                    writer.writeheader()
                                    writer.writerows(contrastive_rows)
                                
                                print(f"Saved contrastive test results to {contrastive_csv_path}")
                                
                                # Print summary statistics
                                contrastive_tests = [r for r in contrastive_results if r.get('is_contrastive') and not r.get('error')]
                                if contrastive_tests:
                                    score_decreases = []
                                    rank_increases = []
                                    
                                    for test in contrastive_tests:
                                        if test.get('score_changes'):
                                            for belief, change in test['score_changes'].items():
                                                if change is not None and change < 0:  # Score decreased
                                                    score_decreases.append(change)
                                        
                                        if test.get('rank_changes'):
                                            for belief, change in test['rank_changes'].items():
                                                if change is not None and change > 0:  # Rank increased (worse)
                                                    rank_increases.append(change)
                                    
                                    if score_decreases:
                                        avg_score_decrease = sum(score_decreases) / len(score_decreases)
                                        print(f"  Average score decrease for swapped beliefs: {avg_score_decrease:.4f} ({len(score_decreases)} beliefs)")
                                    
                                    if rank_increases:
                                        avg_rank_increase = sum(rank_increases) / len(rank_increases)
                                        print(f"  Average rank increase (worse) for swapped beliefs: {avg_rank_increase:.2f} ({len(rank_increases)} beliefs)")
                                
                                sys.stdout.flush()
                            
                        except Exception as e:
                            print(f"Error running contrastive test: {str(e)}")
                            import traceback
                            traceback.print_exc()
                            sys.stdout.flush()
                    else:
                        print(f"Skipping contrastive test: only {len(test_inputs)} valid test inputs (need at least 2)")
                        sys.stdout.flush()
                else:
                    print(f"Skipping contrastive test: only {len(evaluation_episodes)} evaluation episodes (need at least 2)")
                    sys.stdout.flush()
                
                # Debug: Print highest and lowest scoring HL candidates from stored example
                if debug_example and debug_example.get('candidates') and debug_example.get('q_values'):
                    candidates = debug_example['candidates']
                    q_vals = debug_example['q_values']
                    selected_action = debug_example['selected_action']
                    obs = debug_example['obs']
                    reward = debug_example.get('reward', 'N/A')
                    
                    # Sort candidates by Q-value
                    sorted_candidates = sorted(q_vals.items(), key=lambda x: x[1], reverse=True)
                    
                    # Print user action, agent action, and state
                    user_action = debug_example.get('user_action', 'N/A')
                    agent_action = debug_example.get('agent_action', 'N/A')
                    user_state = debug_example.get('user_state', {})
                    
                    print(f"\n  Observation used for Q computation:")
                    print(f"    {obs[:200]}{'...' if len(obs) > 200 else ''}")
                    print(f"\n  Agent Action (LL Response): {agent_action}")
                    print(f"  User Action (Response): {user_action}")
                    
                    if user_state:
                        print(f"\n  User State:")
                        if 'current_desire' in user_state:
                            cd = user_state['current_desire']
                            print(f"    Current Desire: {cd.get('intent', 'N/A')} ({cd.get('domain', 'N/A')})")
                            print(f"      Status: {cd.get('status', 'N/A')}, Attempts: {cd.get('attempt_count', 0)}")
                            req_slots = cd.get('required_slots', {})
                            if req_slots:
                                slots_str = ", ".join([f"{k}={v}" for k, v in list(req_slots.items())[:5]])
                                if len(req_slots) > 5:
                                    slots_str += f" ... ({len(req_slots)} total)"
                                print(f"      Required Slots: {slots_str}")
                        print(f"    Total Desires: {user_state.get('total_desires', 0)}")
                        print(f"    Satisfied: {user_state.get('satisfied_desires', 0)}, Failed: {user_state.get('failed_desires', 0)}")
                        goal_achieved_val = user_state.get('goal_achieved', 0.0)
                        print(f"    Goal Achieved: {goal_achieved_val:.2%} ({goal_achieved_val:.2f})")
                    else:
                        print(f"\n  User State: Not available (offline mode or no goal state)")
                    
                    print(f"\n  Highest Scoring Candidate (Q={sorted_candidates[0][1]:.4f}):")
                    print(f"    {sorted_candidates[0][0]}")
                    print(f"\n  Lowest Scoring Candidate (Q={sorted_candidates[-1][1]:.4f}):")
                    print(f"    {sorted_candidates[-1][0]}")
                    if len(sorted_candidates) > 2:
                        print(f"\n  All Candidates (sorted by Q-value):")
                        for i, (cand, q_val) in enumerate(sorted_candidates):
                            marker = " [SELECTED]" if cand == selected_action else ""
                            print(f"    {i+1}. Q={q_val:.4f}{marker}: {cand}")
                    print("=" * 80)
                    sys.stdout.flush()
                    
                    # Clear debug example after printing
                    debug_example = None
                print()  # Extra newline for readability
            
            # Print average stats whenever we cross an evaluation interval (training and evaluation mode)
            n_episodes = len(episode_stats['total_reward'])
            if n_episodes > 0:
                avg_reward = sum(episode_stats['total_reward']) / n_episodes
                avg_length = sum(episode_stats['episode_length']) / n_episodes
                goal_rate = sum(episode_stats['goal_achieved']) / n_episodes * 100
                avg_entropy = sum(episode_stats['final_entropy']) / n_episodes
                avg_vloss = sum(episode_stats['avg_value_loss']) / n_episodes
                avg_q_min_loss = sum(episode_stats['avg_q_min_loss']) / n_episodes
                avg_q_min_target_variance = sum(episode_stats['avg_q_min_target_variance']) / n_episodes
                avg_q_min_prediction_variance = sum(episode_stats['avg_q_min_prediction_variance']) / n_episodes
                max_gpu_memory = max(episode_stats['max_gpu_memory_gb']) if episode_stats['max_gpu_memory_gb'] else 0.0
                total_test_time_tokens = sum(episode_stats['test_time_agent_tokens_total'])
                avg_test_time_tokens_per_episode = total_test_time_tokens / n_episodes
                avg_test_time_tokens_per_turn = (
                    sum(episode_stats['test_time_agent_tokens_per_turn_avg']) / n_episodes
                )
                lengths = list(episode_stats['episode_length'])
                goals = list(episode_stats['goal_achieved'])
                successful_lengths = [lengths[i] for i in range(n_episodes) if goals[i] >= 1.0]
                avg_length_successful = sum(successful_lengths) / len(successful_lengths) if successful_lengths else None
                avg_cosine = 0.0
                avg_l2 = 0.0
                avg_log_prob_max_q = 0.0
                if episode_stats['avg_cosine_similarity']:
                    n_belief_episodes = len(episode_stats['avg_cosine_similarity'])
                    avg_cosine = sum(episode_stats['avg_cosine_similarity']) / n_belief_episodes if n_belief_episodes > 0 else 0.0
                    avg_l2 = sum(episode_stats['avg_l2_distance']) / n_belief_episodes if n_belief_episodes > 0 else 0.0
                if episode_stats['avg_log_prob_max_q_ll_action']:
                    n_log_prob_episodes = len(episode_stats['avg_log_prob_max_q_ll_action'])
                    avg_log_prob_max_q = sum(episode_stats['avg_log_prob_max_q_ll_action']) / n_log_prob_episodes if n_log_prob_episodes > 0 else 0.0
                window_size = min(stats_window_size, n_episodes)
                window_label = f"last {window_size} episode{'s' if window_size != 1 else ''}"
                print(f"\n{'='*80}")
                print(f"Episode {episodes_completed} / {total_episodes} - Average Stats ({window_label}):")
                print(f"  Average Reward: {avg_reward:.4f}")
                print(f"  Average Episode Length: {avg_length:.2f} turns")
                if avg_length_successful is not None:
                    print(f"  Average Episode Length (successful): {avg_length_successful:.2f} turns ({len(successful_lengths)} episodes)")
                print(f"  Goal Achievement Rate: {goal_rate:.2f}%")
                if config.environment_type in ("cares", "wildjailbreak", "redbench", "harmbench"):
                    harmful_levels = list(episode_stats.get('harmful_level', []))
                    if len(harmful_levels) == n_episodes:
                        non_adv_indices = [i for i in range(n_episodes) if harmful_levels[i] == 0]
                        adv_indices = [i for i in range(n_episodes) if harmful_levels[i] > 0]
                        if non_adv_indices:
                            non_adv_goals = sum(1 for i in non_adv_indices if goals[i] >= 0.5) / len(non_adv_indices) * 100
                            print(f"  Goal Completion (non-adversarial): {non_adv_goals:.2f}% ({len(non_adv_indices)} episodes)")
                        if adv_indices:
                            attack_success = sum(1 for i in adv_indices if goals[i] < 0.5) / len(adv_indices) * 100
                            print(f"  Attack Success Rate (adversarial): {attack_success:.2f}% ({len(adv_indices)} episodes)")
                            effective_goal_cap = 100.0 - attack_success
                            print(f"  Effective Goal Achievement (capped by 1 - attack_success): {min(goal_rate, effective_goal_cap):.2f}%")
                print(f"  Average Final Entropy: {avg_entropy:.4f}")
                print(f"  Average Value Loss: {avg_vloss:.4f}")
                if getattr(config, 'use_regret_critic', False):
                    print(f"  Average Q_min Loss: {avg_q_min_loss:.4f}")
                    print(f"  Average Q_min Target Variance (across steps): {avg_q_min_target_variance:.6f}")
                    print(f"  Average Q_min Prediction Variance (across steps): {avg_q_min_prediction_variance:.6f}")
                print(f"  Test-time Agent Tokens (window total): {total_test_time_tokens}")
                print(f"  Test-time Agent Tokens (avg/episode): {avg_test_time_tokens_per_episode:.2f}")
                print(f"  Test-time Agent Tokens (avg/turn): {avg_test_time_tokens_per_turn:.2f}")
                print(f"  Maximum GPU Memory: {max_gpu_memory:.2f} GB")
                if episode_stats['avg_cosine_similarity']:
                    print(f"  Average Belief Cosine Similarity: {avg_cosine:.4f}")
                    print(f"  Average Belief L2 Distance: {avg_l2:.4f}")
                if episode_stats['avg_log_prob_max_q_ll_action']:
                    print(f"  Average Log Prob of Correct LL Action (Max-Q Belief): {avg_log_prob_max_q:.4f}")
                print(f"{'='*80}")
                sys.stdout.flush()
                token_report_path = Path(config.output_dir) / f"test_time_token_eval_{episodes_completed}.json"
                with open(token_report_path, "w") as token_report_file:
                    json.dump(
                        {
                            "episodes_completed": episodes_completed,
                            "window_size": n_episodes,
                            "test_time_agent_tokens_total": total_test_time_tokens,
                            "test_time_agent_tokens_avg_per_episode": avg_test_time_tokens_per_episode,
                            "test_time_agent_tokens_avg_per_turn": avg_test_time_tokens_per_turn,
                        },
                        token_report_file,
                        indent=2,
                    )
            
            last_printed_episode_count = episodes_completed
    
    # Flush remaining transitions in buffer at end of training
    # Final batch update for remaining transitions (using Q_high with normalized transition probabilities)
    if (
        not _is_online_mode(config)
        and not getattr(config, "_evaluation_mode", False)
        and len(transition_buffer['observations']) > 0
    ):
        print(f"\nFlushing remaining {len(transition_buffer['observations'])} transitions from buffer...")
        # 1. Compute current Q-values: Q(o_t, a_t^HL)
        # Training mode: gradients needed for MLP head
        # Handle [SKIP] candidates by assigning 0.0
        hl_actions = transition_buffer['high_level_actions']
        valid_indices = [i for i, action in enumerate(hl_actions) if action != "[SKIP]"]
        
        if valid_indices:
            valid_obs = [transition_buffer['observations'][i] for i in valid_indices]
            valid_actions = [hl_actions[i] for i in valid_indices]
            
            # Chunked batching protocol
            chunk_size = _model_batch_chunk_size(config)
            num_chunks = (len(valid_obs) + chunk_size - 1) // chunk_size
            q_current_chunks = []
            
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, len(valid_obs))
                
                chunk_obs = valid_obs[start_idx:end_idx]
                chunk_actions = valid_actions[start_idx:end_idx]
                
                chunk_q_values = value_function.predict_q_value(
                    observations=chunk_obs,
                    high_level_actions=chunk_actions,
                    tokenizer=tokenizer,
                    requires_grad=True  # Training: gradients needed
                )
                
                q_current_chunks.append(chunk_q_values)
                
                # Cleanup after each chunk
                del chunk_obs, chunk_actions, chunk_q_values
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Concatenate all chunks
            if q_current_chunks:
                q_current_batch_valid = torch.cat(q_current_chunks, dim=0)
            else:
                q_current_batch_valid = torch.tensor([], device=value_function.device, dtype=standard_dtype)
            
            del q_current_chunks
        else:
            q_current_batch_valid = torch.tensor([], device=value_function.device, dtype=standard_dtype)
        
        # Create full batch with [SKIP] candidates assigned 0.0
        q_current_batch = torch.zeros(len(hl_actions), device=value_function.device, dtype=standard_dtype)
        for valid_idx, original_idx in enumerate(valid_indices):
            q_current_batch[original_idx] = q_current_batch_valid[valid_idx]
        # [SKIP] candidates already have 0.0 from initialization
        
        # 2. Collect all next observations and high-level actions for batch Q-value computation
        next_obs_list = []
        next_hl_actions_list = []
        transition_indices = []
        
        for i, (obs, hl_action, ll_action, next_obs, reward_val, terminal) in enumerate(zip(
            transition_buffer['observations'],
            transition_buffer['high_level_actions'],
            transition_buffer['low_level_actions'],
            transition_buffer['next_observations'],
            transition_buffer['rewards'],
            transition_buffer['terminals']
        )):
            if not terminal and next_obs:
                next_obs_list.append(next_obs)
                next_hl_actions_list.append(hl_action)
                transition_indices.append(i)
        
        # Batch compute Q-values for all next states at once - chunked batching protocol
        next_q_values = {}
        if next_obs_list:
            # Separate [SKIP] from valid actions
            next_valid_indices = [i for i, action in enumerate(next_hl_actions_list) if action != "[SKIP]"]
            
            if next_valid_indices:
                next_valid_obs = [next_obs_list[i] for i in next_valid_indices]
                next_valid_actions = [next_hl_actions_list[i] for i in next_valid_indices]
                
                chunk_size = _model_batch_chunk_size(config)
                num_chunks = (len(next_valid_obs) + chunk_size - 1) // chunk_size
                next_q_chunks = []
                
                for chunk_idx in range(num_chunks):
                    start_idx = chunk_idx * chunk_size
                    end_idx = min(start_idx + chunk_size, len(next_valid_obs))
                    
                    chunk_obs = next_valid_obs[start_idx:end_idx]
                    chunk_actions = next_valid_actions[start_idx:end_idx]
                    
                    chunk_q_values = value_function.predict_q_value(
                        observations=chunk_obs,
                        high_level_actions=chunk_actions,
                        tokenizer=tokenizer,
                        requires_grad=False  # Inference: no gradients needed
                    )
                    
                    next_q_chunks.append(chunk_q_values)
                    
                    # Cleanup after each chunk
                    del chunk_obs, chunk_actions, chunk_q_values
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Concatenate all chunks
                if next_q_chunks:
                    next_q_batch_valid = torch.cat(next_q_chunks, dim=0)
                else:
                    next_q_batch_valid = torch.tensor([], device=value_function.device)
                
                del next_q_chunks
                
                # Map valid Q-values back to transition indices
                for valid_idx, original_idx in enumerate(next_valid_indices):
                    transition_idx = transition_indices[original_idx]
                    next_q_values[transition_idx] = next_q_batch_valid[valid_idx].item()
            
            # [SKIP] candidates get 0.0 (already default in next_q_values dict)
        
        # 3. Prepare candidate low-level actions for batch computation
        # Compute P(a|b) directly from logits (unnormalized, proportional to true P(a|b))
        # P(a|b) = product of token probabilities computed from logits
        # We use unnormalized probability since normalizing over finite candidates is incorrect
        # (the true space of actions is infinite - all possible token sequences)
        belief_only = getattr(config, 'll_action_belief_only', True)
        
        transitions_list = list(zip(
            transition_buffer['observations'],
            transition_buffer['high_level_actions'],
            transition_buffer['low_level_actions'],
            transition_buffer['next_observations'],
            transition_buffer['rewards'],
            transition_buffer['terminals']
        ))
        
        # Prepare batch inputs for computing log P(a|b) from logits
        contexts_batch = []
        targets_batch = []
        transition_indices = []
        
        for i, (obs, hl_action, ll_action, next_obs, reward_val, terminal) in enumerate(transitions_list):
            if not ll_action or not ll_action.strip():
                continue
            
            if belief_only:
                context = f"High-Level Context: {hl_action}\nLow-Level Action:"
            else:
                context = f"Observation: {obs}\nHigh-Level Context: {hl_action}\nLow-Level Action:"
            
            contexts_batch.append(context)
            targets_batch.append(ll_action)
            transition_indices.append(i)
        
        # Batch compute log P(a|b) directly from logits (with chunking to prevent OOM)
        # We use mean log probability (normalized by action length) to avoid bias against longer actions
        # This prevents the policy from collapsing to empty/short actions
        # Mean log prob = (sum of token log probs) / length = geometric mean of token probabilities
        log_p_ab_list = []
        if contexts_batch:
            from src.utils.llm_utils import compute_log_prob_batch
            # Process in chunks to prevent OOM
            chunk_size = getattr(config, 'transition_prob_chunk_size', 8)
            if chunk_size <= 0:
                chunk_size = len(contexts_batch)
            
            for chunk_start in range(0, len(contexts_batch), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(contexts_batch))
                chunk_contexts = contexts_batch[chunk_start:chunk_end]
                chunk_targets = targets_batch[chunk_start:chunk_end]
                
                chunk_log_p_ab = compute_log_prob_batch(
                    ll_agent.model,
                    ll_agent.tokenizer,
                    chunk_contexts,
                    chunk_targets
                )
                log_p_ab_list.extend(chunk_log_p_ab)
                
                # Clean up GPU memory after each chunk
                del chunk_contexts, chunk_targets, chunk_log_p_ab
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Map log probabilities back to transitions
        log_p_ab_dict = {}
        for trans_idx, log_p_ab in zip(transition_indices, log_p_ab_list):
            log_p_ab_dict[trans_idx] = log_p_ab
        
        # Compute targets using length-normalized P(a|b)
        # Use exp(mean_log_P(a|b)) which is the geometric mean of token probabilities
        # This normalizes by action length, preventing bias against longer actions
        import math
        targets = []
        for i, (obs, hl_action, ll_action, next_obs, reward_val, terminal) in enumerate(transitions_list):
            # Get mean log P(a|b) for this transition's dataset action
            if i in log_p_ab_dict:
                log_p_ab = log_p_ab_dict[i]
                # Convert to probability (geometric mean, already length-normalized)
                # Clamp to prevent numerical issues
                log_p_ab = max(log_p_ab, -50.0)  # Prevent underflow
                p_ab = math.exp(log_p_ab)
            else:
                # Fallback if action is empty/invalid - use small probability
                p_ab = 1e-10
            
            # Get max Q(o_{t+1}, a') for next state (from batch computation)
            if terminal:
                max_q_next = 0.0
            else:
                max_q_next = next_q_values.get(i, 0.0)
            
            # Target: P(a|b) * [r_t + γ max Q]
            # Using length-normalized P(a|b) (geometric mean) to avoid bias against longer actions
            target = p_ab * (reward_val + config.discount_factor * max_q_next)
            targets.append(target)
        
        targets_tensor = torch.tensor(targets, device=config.device, dtype=standard_dtype)
        avg_entropy_batch = sum(transition_buffer['entropies']) / len(transition_buffer['entropies'])
        
        # Final batch update
        q_loss = value_function.compute_value_loss(
            value_predictions=q_current_batch,
            targets=targets_tensor,
            avg_entropy=avg_entropy_batch,
            entropy_coef=config.entropy_coef
        )
        value_function.update(q_loss)
        
        # Record Q loss in episodes that contributed to this final batch
        # Note: We need to track which episodes contributed to the final flush
        # For now, we'll append to all active episodes that have transitions
        q_loss_value = q_loss.item()
        for episode in active_episodes:
            if episode.is_active and len(episode.episode_value_losses) > 0:
                # Only append if episode has already contributed to batches
                episode.episode_value_losses.append(q_loss_value)
        
        print(f"Final batch update: {len(transition_buffer['observations'])} transitions, "
              f"avg_q_loss={q_loss_value:.4f}")
    
    # Print final training summary (from the last evaluation interval stored in deque)
    if len(episode_stats['total_reward']) > 0:
        n_episodes = len(episode_stats['total_reward'])
        final_avg_reward = sum(episode_stats['total_reward']) / n_episodes
        final_avg_length = sum(episode_stats['episode_length']) / n_episodes
        final_goal_rate = sum(episode_stats['goal_achieved']) / n_episodes * 100
        final_avg_entropy = sum(episode_stats['final_entropy']) / n_episodes
        final_avg_vloss = sum(episode_stats['avg_value_loss']) / n_episodes
        final_avg_q_min_loss = sum(episode_stats['avg_q_min_loss']) / n_episodes
        final_avg_q_min_target_variance = sum(episode_stats['avg_q_min_target_variance']) / n_episodes
        final_avg_q_min_prediction_variance = sum(episode_stats['avg_q_min_prediction_variance']) / n_episodes
        final_max_gpu_memory = max(episode_stats['max_gpu_memory_gb']) if episode_stats['max_gpu_memory_gb'] else 0.0
        final_total_test_time_tokens = sum(episode_stats['test_time_agent_tokens_total'])
        final_avg_test_time_tokens_per_episode = final_total_test_time_tokens / n_episodes
        final_avg_test_time_tokens_per_turn = (
            sum(episode_stats['test_time_agent_tokens_per_turn_avg']) / n_episodes
        )
        lengths = list(episode_stats['episode_length'])
        goals = list(episode_stats['goal_achieved'])
        successful_lengths = [lengths[i] for i in range(n_episodes) if goals[i] >= 1.0]
        final_avg_length_successful = sum(successful_lengths) / len(successful_lengths) if successful_lengths else None

        # Save final checkpoint - skip in evaluation mode
        if not getattr(config, "_evaluation_mode", False):
            checkpoint_dir = Path(config.output_dir) / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            final_checkpoint_path = checkpoint_dir / "value_function_final.pt"
            try:
                value_function.save_checkpoint(str(final_checkpoint_path))
                print(f"Saved final checkpoint to {final_checkpoint_path}")
            except Exception as e:
                print(f"ERROR: Failed to save final checkpoint: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETE - Final Statistics (last {n_episodes} episodes):")
        print(f"  Average Reward: {final_avg_reward:.4f}")
        print(f"  Average Episode Length: {final_avg_length:.2f} turns")
        if final_avg_length_successful is not None:
            print(f"  Average Episode Length (successful): {final_avg_length_successful:.2f} turns ({len(successful_lengths)} episodes)")
        print(f"  Goal Achievement Rate: {final_goal_rate:.2f}%")
        if config.environment_type in ("cares", "wildjailbreak", "redbench", "harmbench"):
            harmful_levels = list(episode_stats.get("harmful_level", []))
            if len(harmful_levels) == n_episodes:
                non_adv_indices = [i for i in range(n_episodes) if harmful_levels[i] == 0]
                adv_indices = [i for i in range(n_episodes) if harmful_levels[i] > 0]
                if non_adv_indices:
                    non_adv_goals = sum(1 for i in non_adv_indices if goals[i] >= 0.5) / len(non_adv_indices) * 100
                    print(f"  Goal Completion (non-adversarial): {non_adv_goals:.2f}% ({len(non_adv_indices)} episodes)")
                if adv_indices:
                    attack_success = sum(1 for i in adv_indices if goals[i] < 0.5) / len(adv_indices) * 100
                    print(f"  Attack Success Rate (adversarial): {attack_success:.2f}% ({len(adv_indices)} episodes)")
                    effective_goal_cap = 100.0 - attack_success
                    print(f"  Effective Goal Achievement (capped by 1 - attack_success): {min(final_goal_rate, effective_goal_cap):.2f}%")
        print(f"  Average Final Entropy: {final_avg_entropy:.4f}")
        print(f"  Average Value Loss: {final_avg_vloss:.4f}")
        if getattr(config, 'use_regret_critic', False):
            print(f"  Average Q_min Loss: {final_avg_q_min_loss:.4f}")
            print(f"  Average Q_min Target Variance (across steps): {final_avg_q_min_target_variance:.6f}")
            print(f"  Average Q_min Prediction Variance (across steps): {final_avg_q_min_prediction_variance:.6f}")
        print(f"  Test-time Agent Tokens (window total): {final_total_test_time_tokens}")
        print(f"  Test-time Agent Tokens (avg/episode): {final_avg_test_time_tokens_per_episode:.2f}")
        print(f"  Test-time Agent Tokens (avg/turn): {final_avg_test_time_tokens_per_turn:.2f}")
        print(f"  Maximum GPU Memory: {final_max_gpu_memory:.2f} GB")
        print(f"{'='*80}\n")
        final_token_report_path = Path(config.output_dir) / "test_time_token_eval_final.json"
        with open(final_token_report_path, "w") as final_token_report_file:
            json.dump(
                {
                    "window_size": n_episodes,
                    "test_time_agent_tokens_total": final_total_test_time_tokens,
                    "test_time_agent_tokens_avg_per_episode": final_avg_test_time_tokens_per_episode,
                    "test_time_agent_tokens_avg_per_turn": final_avg_test_time_tokens_per_turn,
                    "avg_q_min_loss": final_avg_q_min_loss,
                    "avg_q_min_target_variance": final_avg_q_min_target_variance,
                    "avg_q_min_prediction_variance": final_avg_q_min_prediction_variance,
                },
                final_token_report_file,
                indent=2,
            )


if __name__ == "__main__":
    import argparse
    import tempfile
    parser = argparse.ArgumentParser(description="LLM Context Belief Training")
    parser.add_argument("config_path", nargs="?", help="Path to config JSON file")
    parser.add_argument("output_dir", nargs="?", help="Output directory")
    parser.add_argument("--contrastive-coef", type=float, default=None, help="Override contrastive_coef from config")
    parser.add_argument("--contrastive-ablation-mode", type=str, default=None, help="Override contrastive_ablation_mode from config")
    parser.add_argument("--max-dialogues", type=int, default=None, help="Override max_dialogues from config (limit dataset size)")
    args = parser.parse_args()
    
    config_path = args.config_path
    output_dir = args.output_dir
    temp_config_file = None
    
    # If override parameters provided, modify config after loading
    if args.contrastive_coef is not None or args.contrastive_ablation_mode is not None or args.max_dialogues is not None:
        # Load config, apply overrides, then call main
        if config_path:
            config = BaseConfig.from_json(config_path)
        else:
            config = BaseConfig()
        
        if args.contrastive_coef is not None:
            config.contrastive_coef = args.contrastive_coef
        if args.contrastive_ablation_mode is not None:
            config.contrastive_ablation_mode = args.contrastive_ablation_mode
        if args.max_dialogues is not None:
            config.max_dialogues = args.max_dialogues
        
        # Save modified config to temp file and use that
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.to_json(f.name)
            temp_config_file = f.name
            config_path = f.name
    
    try:
        main(config_path, output_dir)
    finally:
        # Clean up temp config file if we created one
        if temp_config_file and os.path.exists(temp_config_file):
            os.unlink(temp_config_file)

