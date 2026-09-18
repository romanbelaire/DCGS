"""Helper functions for RedBench environment integration."""

from typing import Dict, List, Optional, Sequence, Set

from ..data.dialogue_formatter import ensure_goal_json
from ..training.episode_state import EpisodeState
from .adversarial_helpers import create_adversarial_episode_state


_DEFAULT_REFUSAL_SOURCES = [
    "CoCoNot",
    "ORBench",
    "SGXSTest",
    "XSTest",
]


def _normalize_source_name(source: str) -> str:
    return source.lower().replace("_", "").replace("-", "").replace(" ", "")


def _resolve_refusal_sources(refusal_sources: Optional[Sequence[str]]) -> Set[str]:
    selected_sources = refusal_sources if refusal_sources else _DEFAULT_REFUSAL_SOURCES
    return {_normalize_source_name(source) for source in selected_sources}


def _map_harmful_level(
    category: str,
    source: str,
    mapping_mode: str,
    refusal_sources_normalized: Set[str],
) -> int:
    source_is_refusal = _normalize_source_name(source) in refusal_sources_normalized

    if mapping_mode == "category":
        return 0 if category == "No Risk" else 1
    if mapping_mode == "source":
        return 0 if source_is_refusal else 1
    if mapping_mode == "category_with_source_override":
        if source_is_refusal:
            return 0
        return 0 if category == "No Risk" else 1
    raise ValueError(
        "redbench_mapping_mode must be one of: "
        "'category', 'source', 'category_with_source_override'"
    )


def _run_redbench_label_sanity(
    examples: List[Dict],
    mapping_mode: str,
    refusal_sources_normalized: Set[str],
) -> None:
    if not examples:
        raise ValueError("RedBench loader produced zero examples.")

    if mapping_mode == "category_with_source_override":
        for example in examples:
            source_is_refusal = (
                _normalize_source_name(example["source"]) in refusal_sources_normalized
            )
            if source_is_refusal and example["category"] != "No Risk":
                raise ValueError(
                    "RedBench label inconsistency detected: refusal source "
                    f"'{example['source']}' has category '{example['category']}'."
                )

    harmful_levels = {example["harmful_level"] for example in examples}
    if not harmful_levels.issubset({0, 1}):
        raise ValueError(
            f"RedBench harmful_level sanity failed. Found unexpected values: {harmful_levels}"
        )


def _require_row_keys(
    row: Dict,
    required_keys: Sequence[str],
    subset_name: str,
    split: str,
    row_idx: int,
) -> None:
    missing_keys = [key for key in required_keys if key not in row]
    if missing_keys:
        available_keys = sorted(row.keys())
        raise KeyError(
            "RedBench row is missing required keys. "
            f"subset={subset_name}, split={split}, row_idx={row_idx}, "
            f"missing_keys={missing_keys}, available_keys={available_keys}"
        )


def load_redbench_dataset(
    split: str = "train",
    subsets: Optional[List[str]] = None,
    max_examples: Optional[int] = None,
    mapping_mode: str = "category",
    refusal_sources: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Load RedBench dataset from HuggingFace (knoveleng/redbench).

    Args:
        split: dataset split (typically "train")
        subsets: subset names (dataset configs). None means all subsets.
        max_examples: maximum examples to load across all subsets (None = all)
        mapping_mode: one of "category", "source", "category_with_source_override"
        refusal_sources: source names considered refusal-focused when mapping uses source

    Returns:
        List of example dicts projected to the adversarial environment schema.
    """
    from datasets import get_dataset_config_names, load_dataset

    selected_subsets = subsets if subsets else get_dataset_config_names("knoveleng/redbench")
    refusal_sources_normalized = _resolve_refusal_sources(refusal_sources)
    examples: List[Dict] = []
    required_keys_core = [
        "prompt",
        "category",
        "source",
        "domain",
        "task",
        "language",
        "choices",
        "answer",
    ]
    required_keys_rich_metadata = [
        "risk_response",
        "risk_property",
        "domain_response",
        "domain_property",
    ]

    for subset_name in selected_subsets:
        ds = load_dataset("knoveleng/redbench", subset_name, split=split)
        if subset_name == "SGBench":
            required_keys = [*required_keys_core, *required_keys_rich_metadata]
        elif subset_name == "SGXSTest":
            required_keys = [*required_keys_core, "subtask"]
        else:
            required_keys = [*required_keys_core, "subtask", *required_keys_rich_metadata]
        for row_idx, row in enumerate(ds):
            _require_row_keys(
                row=row,
                required_keys=required_keys,
                subset_name=subset_name,
                split=split,
                row_idx=row_idx,
            )
            prompt = row["prompt"].strip()
            category = row["category"]
            source = row["source"]
            harmful_level = _map_harmful_level(
                category=category,
                source=source,
                mapping_mode=mapping_mode,
                refusal_sources_normalized=refusal_sources_normalized,
            )

            example = {
                "id": f"redbench_{subset_name}_{split}_{row_idx}",
                "base_prompt": prompt,
                "prompt": prompt,
                "harmful_level": harmful_level,
                "lead_in": "I have a question.",
                "subset": subset_name,
                "source": source,
                "category": category,
                "domain": row["domain"],
                "task": row["task"],
                "subtask": row["task"] if subset_name == "SGBench" else row["subtask"],
                "language": row["language"],
                "choices": row["choices"],
                "answer": row["answer"],
                "risk_response": "" if subset_name == "SGXSTest" else row["risk_response"],
                "risk_property": "" if subset_name == "SGXSTest" else row["risk_property"],
                "domain_response": "" if subset_name == "SGXSTest" else row["domain_response"],
                "domain_property": "" if subset_name == "SGXSTest" else row["domain_property"],
                "attack_refusal_label": "refusal" if harmful_level == 0 else "attack",
            }
            example["goal_json"] = ensure_goal_json(prompt)
            examples.append(example)

            if max_examples is not None and len(examples) >= max_examples:
                _run_redbench_label_sanity(
                    examples=examples,
                    mapping_mode=mapping_mode,
                    refusal_sources_normalized=refusal_sources_normalized,
                )
                return examples

    _run_redbench_label_sanity(
        examples=examples,
        mapping_mode=mapping_mode,
        refusal_sources_normalized=refusal_sources_normalized,
    )
    return examples


def create_redbench_episode_state(
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
    """Create EpisodeState from RedBench example."""
    return create_adversarial_episode_state(
        environment_type="redbench",
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
