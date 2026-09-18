"""Helper functions for HarmBench environment integration."""

from typing import Dict, List, Optional

from ..data.dialogue_formatter import ensure_goal_json
from ..training.episode_state import EpisodeState
from .adversarial_helpers import create_adversarial_episode_state


def _resolve_harmbench_split(requested_split: str, available_splits: List[str]) -> str:
    if requested_split in available_splits:
        return requested_split
    if requested_split == "test" and available_splits == ["train"]:
        print(
            "HarmBench requested split='test' but only split='train' exists; "
            "loading train for evaluation."
        )
        return "train"
    raise ValueError(
        "Unknown HarmBench split. "
        f"requested_split={requested_split}, available_splits={available_splits}"
    )


def _require_row_keys(
    row: Dict,
    required_keys: List[str],
    split: str,
    row_idx: int,
) -> None:
    missing_keys = [key for key in required_keys if key not in row]
    if missing_keys:
        available_keys = sorted(row.keys())
        raise KeyError(
            "HarmBench row is missing required keys. "
            f"split={split}, row_idx={row_idx}, "
            f"missing_keys={missing_keys}, available_keys={available_keys}"
        )


def load_harmbench_dataset(
    split: str = "train",
    max_examples: Optional[int] = None,
) -> List[Dict]:
    """
    Load HarmBench prompts from walledai/HarmBench.

    HarmBench is used here as attack-resistance-only data, so every example is
    explicitly mapped to adversarial (`harmful_level = 1`).
    """
    from datasets import get_dataset_split_names, load_dataset

    dataset_name = "walledai/HarmBench"
    dataset_config = "standard"
    available_splits = sorted(get_dataset_split_names(dataset_name, dataset_config))
    resolved_split = _resolve_harmbench_split(split, available_splits)
    ds = load_dataset(dataset_name, dataset_config, split=resolved_split)
    examples: List[Dict] = []
    required_keys = [
        "prompt",
        "category",
    ]
    for row_idx, row in enumerate(ds):
        _require_row_keys(
            row=row,
            required_keys=required_keys,
            split=resolved_split,
            row_idx=row_idx,
        )
        prompt = row["prompt"].strip()
        example = {
            "id": f"harmbench_{resolved_split}_{row_idx}",
            "base_prompt": prompt,
            "prompt": prompt,
            "harmful_level": 1,
            "lead_in": "I have a question.",
            "subset": "HarmBench",
            "category": row["category"],
            "attack_refusal_label": "attack",
        }
        optional_keys = [
            "source",
            "domain",
            "task",
            "subtask",
            "language",
            "choices",
            "answer",
            "risk_response",
            "risk_property",
            "domain_response",
            "domain_property",
        ]
        for key in optional_keys:
            if key in row:
                example[key] = row[key]
        example["goal_json"] = ensure_goal_json(prompt)
        examples.append(example)

        if max_examples is not None and len(examples) >= max_examples:
            return examples
    return examples


def create_harmbench_episode_state(
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
    """Create EpisodeState from HarmBench example."""
    return create_adversarial_episode_state(
        environment_type="harmbench",
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
