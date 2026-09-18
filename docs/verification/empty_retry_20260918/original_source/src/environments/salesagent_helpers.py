"""Helper functions for SalesAgent environment integration."""

import json
from pathlib import Path
from typing import Dict, List

from ..training.episode_state import EpisodeState
from .salesagent_env import SalesAgentEnvironment


def load_salesbot_dataset(data_path: str) -> List[Dict]:
    """Load SalesBot 2.0 dataset."""
    data_path = Path(data_path)
    with open(data_path, 'r') as f:
        if data_path.suffix == '.jsonl':
            return [json.loads(line) for line in f if line.strip()]
        else:
            return json.load(f)


def create_salesagent_episode_state(
    conversation_idx: int,
    conversation: Dict,
    config,
    debug: bool = False
) -> EpisodeState:
    """Create EpisodeState from SalesBot conversation."""
    env = SalesAgentEnvironment(conversation, debug=debug)
    
    initial_obs, initial_info = env.reset()
    
    # Get ground truth goal from intent
    ground_truth_goal = conversation.get("intent", "")
    if isinstance(ground_truth_goal, dict):
        ground_truth_goal = ground_truth_goal.get("description", str(ground_truth_goal))
    
    dialogue_id = conversation.get("id", f"salesbot_{conversation_idx}")
    
    episode = EpisodeState(
        dialogue_idx=conversation_idx,
        dialogue_id=dialogue_id,
        dialogue_data=conversation,
        ground_truth_goal=ground_truth_goal,
        env=env,
        initial_observation=initial_obs,
        initial_env_info=initial_info
    )
    episode.initialize()
    return episode

