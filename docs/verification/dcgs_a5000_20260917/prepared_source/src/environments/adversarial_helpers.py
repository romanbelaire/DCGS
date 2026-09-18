"""Shared helpers for CARES and WildJailbreak (adversarial) environments.

Both environments use the same:
- CARESOnlineEnvironment / CARESOfflineEnvironment (from cares_env.py)
- compute_cares_reward for step rewards (LlamaGuard + fulfillment judge)
- compute_reward_for_marginal for marginal token rewards
"""

from typing import Dict

from ..training.episode_state import EpisodeState
from .cares_env import CARESOfflineEnvironment, CARESOnlineEnvironment

CONFIG_KEY_BY_ENV = {
    "cares": ("cares_online", "cares_use_lead_in"),
    "wildjailbreak": ("wildjailbreak_online", "wildjailbreak_use_lead_in"),
    "redbench": ("redbench_online", "redbench_use_lead_in"),
    "harmbench": ("harmbench_online", "harmbench_use_lead_in"),
}


def create_adversarial_episode_state(
    environment_type: str,
    example_idx: int,
    example: Dict,
    config,
    debug: bool = False,
    patient_agent=None,
    reward_model=None,
    reward_tokenizer=None,
    judge_model=None,
    judge_tokenizer=None,
) -> EpisodeState:
    """
    Create EpisodeState for CARES or WildJailbreak.

    Shared implementation; config keys vary by environment_type.
    """
    online_key, lead_in_key = CONFIG_KEY_BY_ENV[environment_type]
    use_online = getattr(config, online_key, True) and patient_agent is not None
    use_lead_in = getattr(config, lead_in_key, True)
    max_turns = getattr(config, "max_turns", 20)

    if use_online:
        env = CARESOnlineEnvironment(
            patient_agent=patient_agent,
            example=example,
            max_turns=max_turns,
            reward_model=reward_model,
            reward_tokenizer=reward_tokenizer,
            judge_model=judge_model,
            judge_tokenizer=judge_tokenizer,
            config=config,
            debug=debug,
        )
    else:
        env = CARESOfflineEnvironment(
            example=example,
            use_lead_in=use_lead_in,
            reward_model=reward_model,
            reward_tokenizer=reward_tokenizer,
            judge_model=judge_model,
            judge_tokenizer=judge_tokenizer,
            config=config,
            debug=debug,
        )

    initial_obs, initial_info = env.reset()
    ground_truth_goal = example.get("base_prompt", "")
    dialogue_id = example.get("id", f"{environment_type}_{example_idx}")

    episode = EpisodeState(
        dialogue_idx=example_idx,
        dialogue_id=dialogue_id,
        dialogue_data=example,
        ground_truth_goal=ground_truth_goal,
        env=env,
        initial_observation=initial_obs,
        initial_env_info=initial_info,
    )
    episode.initialize()
    return episode
