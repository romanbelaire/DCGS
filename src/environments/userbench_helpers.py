"""Helper functions for UserBench environment integration."""

import sys
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List

# Add UserBench directory to Python path if not already present
_project_root = Path(__file__).parent.parent.parent
_userbench_path = _project_root / 'UserBench'
if _userbench_path.exists() and str(_userbench_path) not in sys.path:
    sys.path.insert(0, str(_userbench_path))

from ..training.episode_state import EpisodeState
from .userbench_env import UserBenchEnvironment


def load_userbench_data(env_name: str, data_root: str = None, one_choice: bool = False, split: str = "train") -> List[Dict]:
    """
    Load UserBench/TravelGym data.
    
    Args:
        env_name: Environment name (e.g., "travel22", "travel33", "travel44")
        data_root: Root directory for data (defaults to UserBench/data)
        one_choice: Whether to use one-choice variant
        split: Data split to load ("train", "test", "val") - defaults to "train" for training
    
    Returns:
        List of task data dictionaries
    """
    if data_root is None:
        data_root = _project_root / 'UserBench' / 'data'
    else:
        data_root = Path(data_root)
    
    # Determine file path - try train first, fallback to test if train doesn't exist
    if one_choice and "travel" in env_name:
        base_dir = data_root / f"{env_name}_multiturn_onechoice"
    else:
        base_dir = data_root / f"{env_name}_multiturn"
    
    # Try requested split first, then fallback to test if not found
    path = base_dir / f"{split}.parquet"
    if not path.exists():
        # Fallback to test if train/val doesn't exist
        if split != "test":
            test_path = base_dir / "test.parquet"
            if test_path.exists():
                path = test_path
                print(f"Warning: {split}.parquet not found, using test.parquet instead")
            else:
                raise FileNotFoundError(f"Data file not found: {path} (also tried {test_path})")
        else:
            raise FileNotFoundError(f"Data file not found: {path}")
    
    # Load parquet file
    df = pd.read_parquet(path)
    
    # Convert to list of dictionaries
    data = []
    for i in range(len(df)):
        data.append({
            "env_name": env_name,
            "gold": str(df.iloc[i]["reward_model"]["id"]) if (env_name == "intention" or env_name == "persuasion" or "travel" in env_name) else str(df.iloc[i]["reward_model"]["title"]),
            "messages": list(df.iloc[i]["prompt"]),
            "task_id": df.iloc[i].get("id", f"{env_name}_{i}"),
            "row_index": i
        })
    
    return data


def create_userbench_episode_state(
    task_data: Dict,
    config,
    debug: bool = False
) -> EpisodeState:
    """
    Create EpisodeState from UserBench task data.
    
    Args:
        task_data: Task data dictionary with env_name, gold, messages, etc.
        config: Configuration object
        debug: Whether to print debug statements
    """
    import travelgym
    
    env_name = task_data["env_name"]
    gold_id = task_data["gold"]
    
    # Create TravelGym config
    travel_config = travelgym.get_default_config()
    travel_config.max_steps = getattr(config, 'max_turns', 20)
    travel_config.data_mode = "single"
    travel_config.data_source = gold_id
    
    # Get UserBench-specific parameters from task_config
    task_config = getattr(config, 'task_config', {})
    travel_config.wrong_choice_number = task_config.get('wrong_choice_number', 10)
    travel_config.noise_choice_number = task_config.get('noise_choice_number', 5)
    travel_config.one_choice_per_aspect = task_config.get('one_choice_per_aspect', True)
    
    # Set user model (default to gpt-4o-mini for cost reduction)
    user_model_name = getattr(config, 'user_model_name', None)
    if user_model_name:
        travel_config.model_name = user_model_name
    else:
        travel_config.model_name = "gpt-4o-mini"  # Default user model
    
    # Set API key from environment or config
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        travel_config.api_key = api_key
    elif hasattr(config, 'openai_api_key') and config.openai_api_key:
        travel_config.api_key = config.openai_api_key
    
    # Set data path if specified in config
    if hasattr(config, 'userbench_data_path') and config.userbench_data_path:
        travel_config.data_path = config.userbench_data_path
    else:
        # Use environment variable or default
        data_path = os.environ.get("TRAVELGYM_DATA_PATH", None)
        if data_path:
            travel_config.data_path = data_path
    
    # Create TravelGym environment
    travel_env = travelgym.TravelEnv(config=travel_config)
    
    # Create adapter
    env = UserBenchEnvironment(
        travel_env=travel_env,
        task_data=task_data,
        max_steps=getattr(config, 'max_turns', 20),
        debug=debug
    )
    
    # Get ground truth goal from task data
    ground_truth_goal = f"Complete travel planning task: {gold_id}"
    if "messages" in task_data and len(task_data["messages"]) > 0:
        # Try to extract goal from first message
        first_msg = task_data["messages"][0]
        if isinstance(first_msg, dict):
            content = first_msg.get("content", "")
            if content:
                ground_truth_goal = content[:200]  # Truncate if too long
    
    # Create EpisodeState
    initial_obs, initial_info = env.reset()
    
    dialogue_id = task_data.get("task_id", f"userbench_{gold_id}")
    
    episode = EpisodeState(
        dialogue_idx=task_data.get("row_index", 0),
        dialogue_id=dialogue_id,
        dialogue_data=task_data,
        ground_truth_goal=ground_truth_goal,
        env=env,
        initial_observation=initial_obs,
        initial_env_info=initial_info
    )
    episode.initialize()
    return episode

