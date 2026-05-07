"""Predictive consistency computation (DPO-based belief surrogate)."""

from typing import List

from .belief_state import BeliefCandidate
from ..utils.llm_utils import compute_log_prob_batch


def compute_log_probs_batch(
    beliefs: List[BeliefCandidate],
    action: str,
    observation: str,
    model,
    tokenizer
) -> List[float]:
    """Compute log-probabilities for each candidate belief."""
    if not beliefs:
        return []

    contexts = [f"{belief.context}\nAgent: {action}\nUser:" for belief in beliefs]
    targets = [observation] * len(beliefs)

    log_probs = compute_log_prob_batch(model, tokenizer, contexts, targets)
    for belief, log_prob in zip(beliefs, log_probs):
        belief.log_prob = log_prob

    return log_probs

