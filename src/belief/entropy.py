"""Entropy regularization for belief distributions."""

import math
from typing import List


def compute_entropy(probabilities: List[float]) -> float:
    """
    Compute entropy of belief distribution.
    
    H(B) = -Σ p(b_i) log p(b_i)
    
    Args:
        probabilities: List of probabilities (should sum to 1.0)
    
    Returns:
        Entropy value
    """
    entropy = 0.0
    for prob in probabilities:
        if prob > 0:
            entropy -= prob * math.log(prob)
    return entropy


def compute_average_entropy(context_entropies: List[float]) -> float:
    """
    Compute average entropy during the context/dialogue.
    
    Args:
        context_entropies: List of entropy values over dialogue turns
    
    Returns:
        Average entropy
    """
    if not context_entropies:
        return 0.0
    return sum(context_entropies) / len(context_entropies)


def get_entropy_regularization_term(
    avg_entropy: float,
    model_entropy: float,
    entropy_coef: float = 0.1
) -> float:
    """
    Compute entropy regularization term for value function loss.
    
    Term: λ * (H_avg - H_model)
    
    This term is subtracted from the value function loss to encourage
    the value function to account for uncertainty in belief distribution.
    
    Args:
        avg_entropy: Average entropy during context
        model_entropy: Model entropy (reference)
        entropy_coef: Entropy coefficient λ
    
    Returns:
        Regularization term
    """
    return entropy_coef * (avg_entropy - model_entropy)

