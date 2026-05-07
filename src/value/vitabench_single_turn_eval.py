"""Single-turn evaluation for VitaBench (action, response) pairs.

This module provides proxy rewards for single turns without requiring the full trajectory.
This is useful for marginal token rewards where masking an action would change the user response.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add vitabench directory to Python path if not already present
_project_root = Path(__file__).parent.parent.parent
_vitabench_path = _project_root / 'vitabench' / 'src'
if _vitabench_path.exists() and str(_vitabench_path) not in sys.path:
    sys.path.insert(0, str(_vitabench_path))

from vita.data_model.message import Message, AssistantMessage, UserMessage
from vita.data_model.tasks import Task, ExpectedState
from vita.environment.environment import Environment
from vita.utils.llm_utils import generate
from vita.config import DEFAULT_LLM_EVALUATOR, models
from vita.data_model.message import SystemMessage


def evaluate_single_turn_proxy_reward(
    task: Task,
    agent_action: str,
    user_response: str,
    environment: Environment,
    previous_trajectory: Optional[List[Message]] = None,
    llm_evaluator: Optional[str] = None,
    llm_args_evaluator: Optional[dict] = None,
    language: Optional[str] = None
) -> float:
    """
    Evaluate a single (action, response) pair to compute a proxy reward.
    
    This evaluates whether the turn makes progress toward satisfying task criteria
    without requiring the full trajectory. This is useful for marginal rewards where
    masking an action would change the user response.
    
    Args:
        task: VitaBench task with evaluation criteria
        agent_action: Agent's action text or tool call
        user_response: User's response text
        environment: Environment instance (to check database state)
        previous_trajectory: Optional previous messages for context
        llm_evaluator: LLM model for evaluation
        llm_args_evaluator: LLM arguments
        language: Language for evaluation
    
    Returns:
        Proxy reward (0.0 to 1.0) indicating progress toward task completion
    """
    if task.evaluation_criteria is None:
        return 0.0
    
    evaluation_criteria = task.evaluation_criteria
    
    # The full evaluation checks:
    # 1. state_rubrics from expected_states (via _initialize_rubric_states)
    # 2. overall_rubrics (via _initialize_rubric_states)
    # Both are evaluated via LLM in sliding windows
    
    # For single-turn evaluation, we check the same rubrics but with minimal context
    all_rubrics = []
    
    # 1. Extract state_rubrics from expected_states (same as full evaluation)
    if evaluation_criteria.expected_states:
        for expected_state in evaluation_criteria.expected_states:
            if hasattr(expected_state, 'state_rubrics') and expected_state.state_rubrics:
                all_rubrics.extend(expected_state.state_rubrics)
    
    # 2. Add overall_rubrics (same as full evaluation)
    if evaluation_criteria.overall_rubrics:
        all_rubrics.extend(evaluation_criteria.overall_rubrics)
    
    # 3. Evaluate all rubrics with minimal context (single turn)
    if all_rubrics:
        rubric_score = _evaluate_rubrics_single_turn(
            task=task,
            rubrics=all_rubrics,
            agent_action=agent_action,
            user_response=user_response,
            previous_trajectory=previous_trajectory,
            llm_evaluator=llm_evaluator,
            llm_args_evaluator=llm_args_evaluator,
            language=language
        )
        return rubric_score
    
    # If no rubrics, check expected_states directly (fallback)
    # This is a simplified check - full evaluation doesn't do this directly,
    # but it's useful as a proxy when rubrics aren't available
    if evaluation_criteria.expected_states:
        for expected_state in evaluation_criteria.expected_states:
            if expected_state.required_orders:
                state_score = _check_expected_state_single_turn(
                    expected_state=expected_state,
                    agent_action=agent_action,
                    environment=environment
                )
                if state_score > 0:
                    return state_score
    
    return 0.0
    
    # Return maximum score (if any criterion is satisfied, give credit)
    # Or average if we want to be more conservative
    if scores:
        return max(scores)  # Optimistic: any progress counts
        # Alternative: return sum(scores) / len(scores) if scores else 0.0  # Average
    return 0.0


def _check_expected_state_single_turn(
    expected_state: ExpectedState,
    agent_action: str,
    environment: Environment
) -> float:
    """
    Check if a single turn satisfies any part of an expected state.
    
    This is a simplified check - we look for indicators that the action
    might have created an order or satisfied a requirement.
    """
    if not expected_state.required_orders:
        return 0.0
    
    # Check if environment database has orders that match expected state
    # This is a simplified check - in practice, we'd need to match order details
    try:
        db = environment.tools.db if hasattr(environment, 'tools') and hasattr(environment.tools, 'db') else None
        if db and hasattr(db, 'orders') and db.orders:
            # If there are orders in the database, this turn might have created one
            # We can't fully verify without matching order details, so we give partial credit
            # if the action looks like it could create an order
            if any(keyword in agent_action.lower() for keyword in ['order', 'book', 'reserve', 'purchase', 'buy']):
                return 0.5  # Partial credit - might satisfy expected state
    except Exception:
        pass
    
    return 0.0


def _check_assertions_single_turn(
    assertions: List,
    environment: Environment
) -> float:
    """
    Check if any assertion passes after this turn.
    
    Assertions check environment state, which we can verify after a single turn.
    """
    if not assertions:
        return 0.0
    
    passed_count = 0
    for assertion in assertions:
        try:
            if environment.run_env_assertion(assertion, raise_assertion_error=False):
                passed_count += 1
        except Exception:
            pass
    
    if passed_count > 0:
        return passed_count / len(assertions)  # Fraction of assertions passed
    
    return 0.0


def _evaluate_rubrics_single_turn(
    task: Task,
    rubrics: List[str],
    agent_action: str,
    user_response: str,
    previous_trajectory: Optional[List[Message]] = None,
    llm_evaluator: Optional[str] = None,
    llm_args_evaluator: Optional[dict] = None,
    language: Optional[str] = None
) -> float:
    """
    Evaluate if a single turn makes progress on any rubric.
    
    This uses LLM evaluation with minimal context (task instructions + this turn).
    """
    if not task.evaluation_criteria or not task.evaluation_criteria.overall_rubrics:
        return 0.0
    
    if llm_evaluator is None:
        llm_evaluator = DEFAULT_LLM_EVALUATOR
    if llm_args_evaluator is None:
        llm_args_evaluator = models[DEFAULT_LLM_EVALUATOR]
    
    # Build minimal context
    context = f"Task Instructions: {task.instructions}\n\n"
    
    if previous_trajectory:
        # Include last few messages for context (but keep it minimal)
        recent_messages = previous_trajectory[-4:] if len(previous_trajectory) > 4 else previous_trajectory
        context += "Previous dialogue:\n"
        for msg in recent_messages:
            role = getattr(msg, 'role', 'unknown')
            content = getattr(msg, 'content', '')
            if content:
                context += f"{role}: {content}\n"
        context += "\n"
    
    context += f"Current turn:\nAssistant: {agent_action}\nUser: {user_response}\n"
    
    # Format rubrics (same format as full evaluation)
    rubrics_text = "\n".join([f"- {i+1}. {rubric}" for i, rubric in enumerate(rubrics)])
    
    # Create evaluation prompt
    system_prompt = f"""You are evaluating whether a single dialogue turn makes progress toward completing task rubrics.

Task Instructions:
{task.instructions}

Rubrics to evaluate:
{rubrics_text}

Evaluate if the current turn makes progress toward satisfying ANY of these rubrics.
Respond with a JSON object:
{{
    "progress_made": true/false,
    "rubrics_advanced": [list of rubric indices (1-based) that show progress],
    "justification": "brief explanation"
}}"""

    user_prompt = f"Dialogue turn:\n{context}"

    messages = [
        SystemMessage(role="system", content=system_prompt),
        UserMessage(role="user", content=user_prompt)
    ]
    
    try:
        assistant_message = generate(
            model=llm_evaluator,
            messages=messages,
            **llm_args_evaluator
        )
        
        # Parse response
        import json
        import re
        content = assistant_message.content
        
        # Try to extract JSON
        json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            if result.get("progress_made", False):
                # Return fraction of rubrics advanced
                advanced = result.get("rubrics_advanced", [])
                return len(advanced) / len(rubrics) if rubrics else 0.0
    except Exception as e:
        # If evaluation fails, return 0.0
        pass
    
    return 0.0

