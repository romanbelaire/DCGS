"""Helper functions for WildJailbreak environment integration.

WildJailbreak uses the same reward functions and sanity tests as CARES-18K:
- Reward: compute_cares_reward (LlamaGuard or ShieldGemma for harmful_level>0, fulfillment judge for harmful_level==0)
- Environment: CARESOnlineEnvironment / CARESOfflineEnvironment via create_adversarial_episode_state
- Sanity checks: run_guard_sanity_check, run_guard_reward_format_sanity_check, run_fulfillment_judge_sanity_check
"""

from typing import Dict, List, Optional

from ..data.dialogue_formatter import ensure_goal_json
from ..training.episode_state import EpisodeState
from .adversarial_helpers import create_adversarial_episode_state


def load_wildjailbreak_dataset(
    split: str = "train",
    data_types: Optional[List[str]] = None,
    max_examples: Optional[int] = None,
) -> List[Dict]:
    """
    Load WildJailbreak dataset from HuggingFace (allenai/wildjailbreak).

    Args:
        split: "train" or "eval"
        data_types: Filter to these data_type values. None = all 4.
            Options: vanilla_harmful, vanilla_benign, adversarial_harmful, adversarial_benign
        max_examples: Maximum number of examples to return (None = all)

    Returns:
        List of example dicts with id, base_prompt, prompt, harmful_level, data_type, completion
    """
    from datasets import load_dataset

    # Config must be 2nd positional ("train" or "eval"); see dataset card. Returns DatasetDict.
    # For single-file configs, the DatasetDict key may not match the config name (e.g. "eval" -> "train").
    ds_dict = load_dataset("allenai/wildjailbreak", split, delimiter="\t", keep_default_na=False)
    ds = ds_dict[split] if split in ds_dict else ds_dict[list(ds_dict.keys())[0]]
    examples = []
    for idx, row in enumerate(ds):
        data_type = row.get("data_type", "")
        if data_types is not None and data_type not in data_types:
            continue

        vanilla = row.get("vanilla", "").strip()
        adversarial = row.get("adversarial", "").strip()
        prompt = adversarial if adversarial else vanilla
        harmful_level = 1 if "harmful" in data_type else 0

        # Attacker goal = prompt (the actual message to eventually ask)
        example = {
            "id": f"wildjailbreak_{split}_{idx}",
            "base_prompt": prompt,
            "prompt": prompt,
            "harmful_level": harmful_level,
            "data_type": data_type,
            "completion": row.get("completion", ""),
            "lead_in": "I have a question.",
        }
        example["goal_json"] = ensure_goal_json(prompt)
        examples.append(example)

        if max_examples is not None and len(examples) >= max_examples:
            break

    return examples


def create_wildjailbreak_episode_state(
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
    """Create EpisodeState from WildJailbreak example."""
    return create_adversarial_episode_state(
        environment_type="wildjailbreak",
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
