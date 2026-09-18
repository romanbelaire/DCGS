"""Reward computation for dialogue outcomes."""

from ..belief import DialogueState


def compute_reward(dialogue_state: DialogueState, goal_achieved: bool) -> float:
    """
    Compute reward based on dialogue outcomes.
    
    Args:
        dialogue_state: Current dialogue state
        goal_achieved: Whether user goal is satisfied
    
    Returns:
        Reward value
    """
    if goal_achieved:
        return 1.0  # Success reward
    else:
        # Small intermediate reward per informative turn
        return 0.0  # No reward if goal not achieved


def compute_reward_shaped(
    dialogue_state: DialogueState,
    goal_achieved: bool,
    turn_reward: float = 0.1
) -> float:
    """
    Compute shaped reward with intermediate rewards.
    
    Args:
        dialogue_state: Current dialogue state
        goal_achieved: Whether user goal is satisfied
        turn_reward: Reward per informative turn
    
    Returns:
        Shaped reward value
    """
    if goal_achieved:
        return 1.0 + len(dialogue_state.rewards) * turn_reward
    else:
        # Small reward for continuing dialogue
        return turn_reward

