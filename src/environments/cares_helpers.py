"""Helper functions for CARES-18K environment integration."""

from typing import Dict, List, Optional

from ..data.dialogue_formatter import ensure_goal_json
from ..training.episode_state import EpisodeState
from .adversarial_helpers import create_adversarial_episode_state


def load_cares_dataset(split: str = "train") -> List[Dict]:
    """Load CARES-18K dataset from HuggingFace."""
    from datasets import load_dataset

    ds = load_dataset("HFXM/CARES-18K", split=split)
    examples = []
    for idx, row in enumerate(ds):
        base_prompt = row["base_prompt"]
        harmful_level = int(row["harmful_level"])
        example = {
            "id": f"cares_{split}_{idx}",
            "principle_index": int(row["principle_index"]),
            "generation_model": row.get("generation_model", ""),
            "harmful_level": harmful_level,
            "method": row.get("method", ""),
            "base_prompt": base_prompt,
            "prompt": row["prompt"],
        }
        example["goal_json"] = ensure_goal_json(base_prompt)
        examples.append(example)
    return examples


def create_cares_episode_state(
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
    """Create EpisodeState from CARES example."""
    return create_adversarial_episode_state(
        environment_type="cares",
        example_idx=example_idx,
        example=example,
        config=config,
        debug=debug,
        patient_agent=patient_agent,
        reward_model=reward_model,
        reward_tokenizer=reward_tokenizer,
        judge_model=judge_model,
        judge_tokenizer=judge_tokenizer,
    )
