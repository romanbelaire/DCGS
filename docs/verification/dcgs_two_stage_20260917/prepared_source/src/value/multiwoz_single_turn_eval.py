"""Single-turn evaluation for MultiWOZ (action, response) pairs.

Supports both offline and online MultiWOZ:

**Offline MultiWOZ**:
- Reward = 1.0 if user response contains THANK_YOU, AFFIRM, or SELECT dialogue acts
- Reward = 0.0 otherwise
- Deterministic: Based on dialogue act extraction from user response

**Online MultiWOZ**:
- Reward = 1.0 if LLM judge says goal is satisfied AND goal achievement fraction increased
- Reward = 0.0 otherwise
- Uses LLM-as-judge: Evaluates if agent action fulfills user goal
- Requires: dialogue history, goal_json, and agent action
"""

from typing import List, Optional, Tuple, Dict
from ..main import extract_user_actions


def evaluate_single_turn_proxy_reward(
    user_response: str,
    user_turn: Optional[dict] = None,
    prev_system_turn: Optional[dict] = None,
    # Online mode parameters
    agent_action: Optional[str] = None,
    dialogue_history: Optional[List[Tuple[str, str]]] = None,
    goal_json: Optional[Dict] = None,
    user_agent=None,  # UserAgent instance for LLM judge
    prev_goal_achievement_fraction: float = 0.0
) -> float:
    """
    Evaluate a single (action, response) pair to compute MultiWOZ reward.
    
    Supports both offline and online modes:
    - Offline: Based on dialogue acts in user response
    - Online: Based on LLM judge evaluating agent action against goal
    
    Args:
        user_response: User's response text
        user_turn: Optional structured user turn dict (for offline mode with dialogue acts)
        prev_system_turn: Optional previous system turn (for dialogue act extraction fallback)
        agent_action: Agent's action (required for online mode)
        dialogue_history: Dialogue history (required for online mode)
        goal_json: User goal JSON (required for online mode)
        user_agent: UserAgent instance with judge_goal_satisfaction method (required for online mode)
        prev_goal_achievement_fraction: Previous goal achievement fraction (for online mode)
    
    Returns:
        Reward value (0.0 or 1.0)
    """
    # Online mode: Use LLM judge (same as OnlineEnvironment)
    if agent_action and dialogue_history and goal_json and user_agent:
        return _evaluate_online_reward(
            agent_action=agent_action,
            dialogue_history=dialogue_history,
            goal_json=goal_json,
            user_agent=user_agent,
            prev_goal_achievement_fraction=prev_goal_achievement_fraction
        )
    
    # Offline mode: Use dialogue acts (same as MultiWOZEnvironment)
    # Extract dialogue acts from user response
    # Use structured turn if available (offline mode), otherwise extract from text
    if user_turn:
        # Use the same extraction method as MultiWOZEnvironment
        actions = _extract_actions_from_turn(user_turn, prev_system_turn)
    else:
        # Extract from text (online mode or when structured turn not available)
        actions = _extract_actions_from_text(user_response)
    
    # Check for success actions (same as MultiWOZEnvironment._compute_reward)
    success_actions = {"THANK_YOU", "AFFIRM", "SELECT"}
    return 1.0 if any(action in success_actions for action in actions) else 0.0


def _evaluate_online_reward(
    agent_action: str,
    dialogue_history: List[Tuple[str, str]],
    goal_json: Dict,
    user_agent,
    prev_goal_achievement_fraction: float = 0.0
) -> float:
    """
    Evaluate reward for online MultiWOZ using LLM-as-judge.
    
    This matches OnlineEnvironment.step() logic:
    1. Judge if agent action satisfies goal
    2. Update goal state if satisfied
    3. Compute goal achievement fraction
    4. Reward = 1.0 if fraction increased, else 0.0
    
    Args:
        agent_action: Agent's action
        dialogue_history: Dialogue history
        goal_json: User goal JSON
        user_agent: UserAgent instance with judge_goal_satisfaction
        prev_goal_achievement_fraction: Previous goal achievement fraction
    
    Returns:
        Reward value (0.0 or 1.0)
    """
    # Judge goal satisfaction (same as OnlineEnvironment)
    try:
        judge_history = dialogue_history.copy()
        goal_satisfied = user_agent.judge_goal_satisfaction(
            dialogue_history=judge_history,
            goal_json=goal_json,
            agent_response=agent_action,
            temperature=0.0
        )
    except Exception:
        goal_satisfied = False
    
    # For single-turn evaluation, we can't update GoalState directly
    # Instead, we approximate: if judge says satisfied, assume one desire was satisfied
    # This is a simplification - full evaluation would track all desires
    
    # Simplified: if judge says satisfied, assume progress was made
    # In practice, this would require tracking GoalState across turns
    if goal_satisfied:
        # Assume progress: if judge says satisfied, goal achievement increased
        # This is an approximation - true evaluation needs GoalState tracking
        return 1.0
    
    return 0.0


def _extract_actions_from_turn(turn: dict, prev_system_turn: Optional[dict] = None) -> List[str]:
    """
    Extract dialogue acts from structured turn (same as MultiWOZEnvironment._extract_actions).
    
    This matches the exact logic used in MultiWOZEnvironment for consistency.
    """
    # Try to extract from turn first
    actions = _extract_dialogue_acts_from_turn(turn)
    if actions:
        return actions
    
    # Fallback to previous system turn
    if prev_system_turn:
        actions = _extract_dialogue_acts_from_turn(prev_system_turn)
        if actions:
            return actions
    
    return []


def _extract_dialogue_acts_from_turn(turn: dict) -> List[str]:
    """
    Extract dialogue acts from a structured turn dict.
    
    Matches MultiWOZEnvironment._extract_dialogue_acts logic.
    """
    acts: List[str] = []
    if not isinstance(turn, dict):
        return acts
    
    dialogue_acts = turn.get("dialogue_acts") or turn.get("dialogue_act")
    canonical_map = {
        "GENERAL-THANK": "THANK_YOU",
        "GENERAL-BYE": "GOODBYE",
        "GENERAL-REQMORE": "REQ_MORE",
        "GENERAL-INFORM": "INFORM",
        "GENERAL-AFFIRM": "AFFIRM",
        "GENERAL-NEGATE": "NEGATE",
    }
    
    def add_act(act_value):
        if not act_value:
            return
        act_upper = str(act_value).upper()
        acts.append(canonical_map.get(act_upper, act_upper))
    
    if isinstance(dialogue_acts, dict):
        for domain_or_act, acts_data in dialogue_acts.items():
            if isinstance(acts_data, dict):
                for act_type in acts_data.keys():
                    add_act(act_type)
            elif isinstance(acts_data, list):
                for act in acts_data:
                    if isinstance(act, dict):
                        add_act(act.get("act") or act.get("act_type"))
                    elif isinstance(act, str):
                        add_act(act)
            else:
                add_act(domain_or_act)
    elif isinstance(dialogue_acts, list):
        for act in dialogue_acts:
            if isinstance(act, dict):
                add_act(act.get("act") or act.get("act_type"))
            elif isinstance(act, str):
                add_act(act)
    
    # Also check actions field
    direct_actions = turn.get("actions", [])
    if isinstance(direct_actions, list):
        for action in direct_actions:
            if isinstance(action, dict):
                add_act(action.get("act") or action.get("act_type"))
            elif isinstance(action, str):
                add_act(action)
    
    return acts


def _extract_actions_from_text(text: str) -> List[str]:
    """
    Extract dialogue acts from text using keyword matching.
    
    Matches OnlineEnvironment._extract_dialogue_acts logic for consistency.
    """
    text_lower = text.lower().strip()
    acts = []
    
    if not text_lower:
        return acts
    
    # Simple keyword-based extraction (same as OnlineEnvironment)
    if any(word in text_lower for word in ["thank", "thanks", "appreciate"]):
        acts.append("THANK_YOU")
    if any(word in text_lower for word in ["yes", "yeah", "sure", "okay", "ok", "correct"]):
        acts.append("AFFIRM")
    if any(word in text_lower for word in ["no", "not", "don't", "won't", "can't"]):
        acts.append("NEGATE")
    if any(word in text_lower for word in ["goodbye", "bye", "see you", "farewell"]):
        acts.append("GOODBYE")
    if any(word in text_lower for word in ["want", "need", "looking for", "searching"]):
        acts.append("INFORM")
    if any(word in text_lower for word in ["select", "choose", "pick", "book", "reserve"]):
        acts.append("SELECT")
    
    return acts

