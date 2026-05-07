"""Logging and metrics tracking."""

import hashlib
import json
import sys
import torch
from pathlib import Path
from typing import Dict, List, Optional

from ..belief import BeliefState

DEFENSE_EPISODES_FILENAME = "defense_episodes.jsonl"


def harmful_level_for_defense_log(stats: Dict) -> int:
    """Same resolution as ``record_episode_statistics`` → ``episode_stats['harmful_level']``."""
    hl = stats.get("harmful_level")
    return hl if hl is not None else -1


def append_defense_episode_record(
    output_dir: str,
    dialogue_id: str,
    episode_idx: int,
    harmful_level: int,
    goal_achieved: float,
    turn_evaluation_data: List[Dict],
    defense_transcript: str,
    environment_type: str,
    base_prompt: Optional[str] = None,
) -> None:
    """
    Append one JSON line per CARES/WildJailbreak episode for offline diversity metrics.

    ``harmful_level`` matches ``episode_stats['harmful_level']`` / printed CARES splits
    (``0`` benign, ``>0`` adversarial, ``-1`` unknown — excluded from ASR / non-adv goal rates).

    Schema (stable keys for defense_embedding_metrics.py):
    - defense_last_ll_action: last low-level agent string in the episode
    - defense_transcript: full Agent/User transcript (same format as EpisodeState.format_dialogue_history)
    - defense_last_user_observation / defense_last_turn: last user message and paired last-turn block
    - attack_prompt_sha256: sha256 hex of base_prompt when present (group adversarial scenarios)
    """
    last_ll = ""
    last_obs = ""
    for turn in reversed(turn_evaluation_data):
        if not turn:
            continue
        ll_action = turn.get("ll_action", "")
        if ll_action and str(ll_action).strip():
            last_ll = str(ll_action).strip()
            last_obs = str(turn.get("observation", "") or "").strip()
            break

    attack_hash: Optional[str] = None
    if base_prompt:
        attack_hash = hashlib.sha256(base_prompt.encode("utf-8")).hexdigest()

    last_turn_block = ""
    if last_obs or last_ll:
        last_turn_block = f"User: {last_obs}\nAgent: {last_ll}".strip()

    record = {
        "dialogue_id": dialogue_id,
        "episode_idx": episode_idx,
        "environment_type": environment_type,
        "harmful_level": int(harmful_level),
        "goal_achieved": float(goal_achieved),
        "defense_last_ll_action": last_ll,
        "defense_last_user_observation": last_obs,
        "defense_last_turn": last_turn_block,
        "defense_transcript": defense_transcript,
        "attack_prompt_sha256": attack_hash,
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / DEFENSE_EPISODES_FILENAME
    with open(out_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_metrics(
    turn: int,
    belief_state: BeliefState,
    entropy: float,
    value_loss: float,
    reward: float,
    dialogue_id: Optional[str] = None,
    episode_idx: Optional[int] = None,
    belief_accuracy: Optional[float] = None,
    ground_truth_goal: Optional[str] = None,
    turn_evaluation: Optional[Dict] = None,
    output_dir: str = "outputs"
) -> None:
    """
    Log metrics for a dialogue turn (stored in memory, saved in batches).
    
    Args:
        turn: Current turn number
        belief_state: Current belief state
        entropy: Current entropy
        value_loss: Value function loss
        reward: Reward received
        dialogue_id: Dialogue identifier
        episode_idx: Episode index
        belief_accuracy: Belief accuracy vs. ground truth
        ground_truth_goal: Ground truth goal string
        output_dir: Output directory for logs
    """
    metrics = {
        "episode_idx": episode_idx,
        "dialogue_id": dialogue_id,
        "turn": turn,
        "entropy": entropy,
        "value_loss": float(value_loss),
        "reward": reward,
        "belief_accuracy": belief_accuracy,
        "ground_truth_goal": ground_truth_goal,
        "top_belief": belief_state.get_top_belief().summary if belief_state.candidates else None,
        "belief_probabilities": [c.probability for c in belief_state.candidates],
        "belief_summaries": [c.summary for c in belief_state.candidates],
        "turn_evaluation": turn_evaluation,
    }
    
    # Store in metrics buffer (will be saved in batches)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metrics_file = output_path / "all_metrics.jsonl"
    # Append to single file (JSONL format - one JSON object per line)
    with open(metrics_file, "a") as f:
        f.write(json.dumps(metrics) + "\n")


def save_rollout(
    rollout: List[Dict],
    dialogue_id: str,
    episode_idx: Optional[int] = None,
    output_dir: str = "outputs"
) -> None:
    """
    Save full rollout with beliefs at each turn (appended to single file).
    
    Args:
        rollout: List of turn data with beliefs
        dialogue_id: Dialogue identifier
        episode_idx: Episode index
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    rollout_entry = {
        "episode_idx": episode_idx,
        "dialogue_id": dialogue_id,
        "rollout": rollout
    }
    
    # Append to single file (JSONL format)
    rollout_file = output_path / "all_rollouts.jsonl"
    with open(rollout_file, "a") as f:
        f.write(json.dumps(rollout_entry) + "\n")


def save_dialogue_summary(
    dialogue_id: str,
    metrics: Dict,
    output_dir: str = "outputs"
) -> None:
    """Save summary metrics for a dialogue."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    summary_file = output_path / f"dialogue_{dialogue_id}_summary.json"
    with open(summary_file, "w") as f:
        json.dump(metrics, f, indent=2)


def print_gpu_memory(label: str = "GPU Memory", include_max: bool = False, restore_device: Optional[int] = None) -> None:
    """
    Print GPU memory usage for all available GPUs.
    
    Args:
        label: Label to print before memory information
        include_max: If True, also print max_memory_allocated (peak memory)
        restore_device: Device ID to restore as current device after printing (None = don't change)
    """
    if not torch.cuda.is_available():
        return
    
    print(f"\n[{label}]:")
    num_gpus = torch.cuda.device_count()
    original_device = torch.cuda.current_device() if restore_device is not None else None
    
    for device_id in range(num_gpus):
        torch.cuda.set_device(device_id)
        allocated_bytes = torch.cuda.memory_allocated(device_id)
        reserved_bytes = torch.cuda.memory_reserved(device_id)
        allocated_gb = allocated_bytes / (1024 ** 3)
        reserved_gb = reserved_bytes / (1024 ** 3)
        
        print(f"  GPU {device_id}:")
        print(f"    Allocated: {allocated_gb:.2f} GB")
        print(f"    Reserved: {reserved_gb:.2f} GB")
        
        if include_max:
            max_allocated_bytes = torch.cuda.max_memory_allocated(device_id)
            max_allocated_gb = max_allocated_bytes / (1024 ** 3)
            print(f"    Max allocated: {max_allocated_gb:.2f} GB")
    
    # Restore original device if specified
    if restore_device is not None and original_device is not None:
        torch.cuda.set_device(original_device)
    
    sys.stdout.flush()

