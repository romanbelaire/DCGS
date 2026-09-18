"""Belief update logic using DPO-based ranking."""

import math
from typing import List

from .belief_state import BeliefState


def log_sum_exp(log_probs: List[float]) -> float:
    """
    Numerically stable computation of log(sum(exp(log_probs))).
    
    Uses log-sum-exp trick: subtract max before exponentiating to prevent overflow.
    """
    if not log_probs:
        return float("-inf")
    
    max_log_prob = max(log_probs)
    if max_log_prob == float("-inf"):
        return float("-inf")
    
    # Subtract max before exponentiating
    shifted_log_probs = [lp - max_log_prob for lp in log_probs]
    exp_sum = sum(math.exp(lp) for lp in shifted_log_probs)
    
    return max_log_prob + math.log(exp_sum)


def softmax_log_probs(log_probs: List[float], temperature: float = 1.0) -> List[float]:
    """
    Compute softmax probabilities from log-probabilities.
    
    Uses log-sum-exp trick for numerical stability.
    
    Args:
        log_probs: List of log-probabilities
        temperature: Temperature scaling parameter
    
    Returns:
        List of normalized probabilities that sum to 1.0
    """
    if not log_probs:
        return []
    
    # Scale by temperature
    log_probs_scaled = [lp / temperature for lp in log_probs]
    
    # Compute log-sum-exp for stability
    log_sum_exp_value = log_sum_exp(log_probs_scaled)
    
    if log_sum_exp_value == float("-inf"):
        # All probabilities are zero, return uniform
        uniform_prob = 1.0 / len(log_probs)
        return [uniform_prob] * len(log_probs)
    
    # Compute probabilities: p_i = exp(log_prob_i - log_sum_exp)
    probabilities = [math.exp(lp - log_sum_exp_value) for lp in log_probs_scaled]
    
    # Ensure they sum to 1.0 (within floating-point tolerance)
    total = sum(probabilities)
    if total > 0:
        probabilities = [p / total for p in probabilities]
    else:
        # Fallback to uniform if all zero
        uniform_prob = 1.0 / len(log_probs)
        probabilities = [uniform_prob] * len(log_probs)
    
    return probabilities


def update_belief_distribution_dpo(
    belief_state: BeliefState,
    log_probs: List[float],
    temperature: float = 1.0
) -> BeliefState:
    """
    Update belief probabilities using DPO-based multi-candidate ranking.
    
    Applies softmax over log-probabilities (listwise Bradley-Terry model).
    
    Args:
        belief_state: Current belief state
        log_probs: List of log-probabilities for each candidate
        temperature: DPO temperature parameter (controls sharpness)
    
    Returns:
        Updated belief state with new probabilities
    """
    if len(log_probs) != len(belief_state.candidates):
        raise ValueError(
            f"Mismatch: {len(log_probs)} log_probs but {len(belief_state.candidates)} candidates"
        )
    
    # Compute softmax probabilities
    probabilities = softmax_log_probs(log_probs, temperature=temperature)
    
    # Update candidate probabilities
    for i, candidate in enumerate(belief_state.candidates):
        candidate.probability = probabilities[i]
    
    # Ensure probabilities sum to 1.0
    belief_state.normalize_probabilities()
    
    return belief_state

