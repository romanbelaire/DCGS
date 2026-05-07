"""Dialogue state data structure."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .belief_state import BeliefState


@dataclass
class DialogueState:
    """Full dialogue state including belief state, values, and rewards."""
    
    belief_state: BeliefState
    value_estimates: Dict[str, float] = field(default_factory=dict)  # Value for each belief candidate
    rewards: List[float] = field(default_factory=list)  # Reward history
    goal_achieved: float = 0.0  # Fraction of goals achieved (0.0 to 1.0)
    ground_truth_goal: Optional[Dict] = None  # Ground truth goal from MultiWOZ or persona

