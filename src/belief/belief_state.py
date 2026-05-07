"""Belief state data structures."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class BeliefCandidate:
    """A candidate belief summary with associated metadata."""
    
    summary: str  # Natural language belief summary
    context: str  # Full context for LLM
    probability: float  # Current belief probability (updated via DPO ranking)
    log_prob: float = 0.0  # Last computed log-probability P(o_{t+1} | b_i, a_t)
    
    def __post_init__(self):
        """Validate initial probability."""
        if not (0.0 <= self.probability <= 1.0):
            raise ValueError(f"Probability must be in [0, 1], got {self.probability}")


@dataclass
class BeliefState:
    """Container for belief state with candidates and dialogue history."""
    
    candidates: List[BeliefCandidate] = field(default_factory=list)
    turn: int = 0
    history: List[Tuple[str, str]] = field(default_factory=list)  # (action, observation) pairs
    
    def get_top_belief(self) -> BeliefCandidate:
        """Return the candidate with highest probability."""
        if not self.candidates:
            raise ValueError("No candidates in belief state")
        return max(self.candidates, key=lambda c: c.probability)
    
    def get_weighted_context(self) -> str:
        """Get weighted context string from all candidates."""
        if not self.candidates:
            return ""
        sorted_candidates = sorted(self.candidates, key=lambda c: c.probability, reverse=True)
        context_parts = []
        for i, candidate in enumerate(sorted_candidates):
            context_parts.append(f"[Prob: {candidate.probability:.3f}] {candidate.summary}")
        return "\n".join(context_parts)
    
    def get_observation_context(self) -> str:
        """Get context string from entire dialogue history (agent and user messages)."""
        if not self.history:
            return ""
        parts = []
        for action, observation in self.history:
            parts.append(f"Agent: {action}\nUser: {observation}")
        return "\n".join(parts)
    
    def normalize_probabilities(self) -> None:
        """Ensure probabilities sum to 1.0."""
        total = sum(c.probability for c in self.candidates)
        if total > 0:
            for candidate in self.candidates:
                candidate.probability /= total
        else:
            # Uniform distribution if all zero
            uniform_prob = 1.0 / len(self.candidates) if self.candidates else 0.0
            for candidate in self.candidates:
                candidate.probability = uniform_prob

