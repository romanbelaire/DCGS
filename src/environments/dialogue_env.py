"""Dialogue environment interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class StepResult:
    """
    Unified return type for env.step().

    All dialogue envs return StepResult for consistent unpacking in main.py.
    agent_action: When set (MultiWOZ offline), the dataset-provided system action.
    reward_task / reward_harm: CARES dual-channel decomposition when set; else None.
    """

    observation: str
    reward: float
    done: bool
    info: Dict[str, Any]
    agent_action: Optional[str] = None
    reward_task: Optional[float] = None
    reward_harm: Optional[float] = None


class DialogueEnvironment(ABC):
    """Abstract interface for dialogue environments."""

    @abstractmethod
    def reset(self):
        """Reset environment to initial state."""
        pass

    @abstractmethod
    def step(self, action: str = None) -> StepResult:
        """
        Execute action and return StepResult.

        For offline MultiWOZ, action is None and agent_action in result is the dataset action.
        """
        pass

    @abstractmethod
    def check_goal(self, observation: str) -> bool:
        """Check if user goal is satisfied."""
        pass

    def compute_reward_for_marginal(
        self, action: str, actual_reward: Optional[float] = None
    ) -> float:
        """
        Compute reward for marginal token contribution.
        Override in envs that need custom logic (MultiWOZ, VitaBench, SalesAgent, CARES).
        """
        return actual_reward if actual_reward is not None else 0.0

