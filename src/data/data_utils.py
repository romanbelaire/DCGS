"""Data utilities for train/val splits, sampling, and statistics."""

import random
from typing import Dict, List, Optional, Tuple

from .multiwoz_loader import MultiWOZDialogue


def split_dataset(
    dialogues: List[MultiWOZDialogue],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[List[MultiWOZDialogue], List[MultiWOZDialogue], List[MultiWOZDialogue]]:
    """
    Split dialogues into train/val/test.
    
    Args:
        dialogues: List of dialogues
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train, val, test) lists
    """
    if seed is not None:
        random.seed(seed)
    
    shuffled = dialogues.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    
    return train, val, test


def sample_dialogues(
    dialogues: List[MultiWOZDialogue],
    n: int,
    seed: Optional[int] = None
) -> List[MultiWOZDialogue]:
    """Sample n dialogues for training/testing."""
    if seed is not None:
        random.seed(seed)
    return random.sample(dialogues, min(n, len(dialogues)))


def get_dialogue_statistics(dialogues: List[MultiWOZDialogue]) -> Dict:
    """
    Compute statistics: avg length, domain distribution, etc.
    
    Returns:
        Dictionary with statistics
    """
    if not dialogues:
        return {}
    
    total_turns = sum(len(d.turns) for d in dialogues)
    avg_length = total_turns / len(dialogues)
    
    domain_counts = {}
    for dialogue in dialogues:
        for domain in dialogue.goal.keys():
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    return {
        "total_dialogues": len(dialogues),
        "avg_turns": avg_length,
        "domain_distribution": domain_counts,
    }


def compute_belief_accuracy(
    predicted_belief: str,
    ground_truth_goal: str,
    method: str = "semantic"
) -> float:
    """
    Compare predicted belief vs. ground truth.
    
    Args:
        predicted_belief: Predicted belief summary (natural language text)
        ground_truth_goal: Ground truth goal summary (natural language text)
        method: "exact", "semantic", or "partial"
    
    Returns:
        Accuracy score (0.0 to 1.0)
    """
    if method == "exact":
        return 1.0 if predicted_belief.strip().lower() == ground_truth_goal.strip().lower() else 0.0
    
    elif method == "partial":
        pred_words = set(predicted_belief.lower().split())
        gt_words = set(ground_truth_goal.lower().split())
        if not gt_words:
            return 0.0
        overlap = len(pred_words & gt_words)
        return overlap / len(gt_words)
    
    elif method == "semantic":
        # Simple semantic similarity using word overlap with weights
        pred_words = set(predicted_belief.lower().split())
        gt_words = set(ground_truth_goal.lower().split())
        if not gt_words:
            return 0.0
        overlap = len(pred_words & gt_words)
        union = len(pred_words | gt_words)
        if union == 0:
            return 0.0
        return overlap / union  # Jaccard similarity
    
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_belief_accuracy_with_conversion(
    predicted_belief_structure,
    ground_truth_belief_structure,
    model,
    tokenizer,
    prompt_manager,
    method: str = "semantic",
    temperature: float = 0.0
) -> float:
    """
    Compare structured beliefs by converting both to text, then comparing.
    
    Args:
        predicted_belief_structure: Structured belief (dict or BeliefState)
        ground_truth_belief_structure: Structured ground truth belief (dict)
        model: Language model for conversion
        tokenizer: Tokenizer
        prompt_manager: PromptManager instance
        method: "exact", "semantic", or "partial"
        temperature: Temperature for LLM conversion (0.0 for deterministic)
    
    Returns:
        Accuracy score (0.0 to 1.0)
    """
    from ..utils.belief_converter import (
        convert_structured_belief_to_text,
        convert_belief_state_to_text,
    )
    from ..belief import BeliefState
    
    # Convert predicted belief to text
    if isinstance(predicted_belief_structure, BeliefState):
        predicted_text = convert_belief_state_to_text(
            predicted_belief_structure, model, tokenizer, prompt_manager, temperature
        )
    else:
        predicted_text = convert_structured_belief_to_text(
            predicted_belief_structure, model, tokenizer, prompt_manager, temperature
        )
    
    # Convert ground truth to text
    ground_truth_text = convert_structured_belief_to_text(
        ground_truth_belief_structure, model, tokenizer, prompt_manager, temperature
    )
    
    # Compare using text-based method
    return compute_belief_accuracy(predicted_text, ground_truth_text, method=method)

