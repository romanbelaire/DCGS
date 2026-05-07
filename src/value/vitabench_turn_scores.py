"""Turn-level scoring for VitaBench trajectories using incremental evaluation."""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Add vitabench directory to Python path if not already present
_project_root = Path(__file__).parent.parent.parent
_vitabench_path = _project_root / 'vitabench' / 'src'
if _vitabench_path.exists() and str(_vitabench_path) not in sys.path:
    sys.path.insert(0, str(_vitabench_path))

from vita.data_model.message import Message
from vita.data_model.tasks import Task
from vita.evaluator.evaluator_traj import TrajectoryEvaluator
from vita.data_model.simulation import RewardInfo, RewardType


def compute_turn_level_scores(
    task: Task,
    full_trajectory: List[Message],
    final_state: dict,
    llm_evaluator: Optional[str] = None,
    llm_args_evaluator: Optional[dict] = None,
    language: Optional[str] = None,
    window_size: int = 10,
    overlap: int = 2
) -> Dict[int, float]:
    """
    Compute score at each turn level by evaluating trajectory incrementally.
    
    For each turn, evaluates the trajectory up to that turn and returns the
    rubric score (fraction of met rubrics).
    
    Args:
        task: VitaBench task with evaluation criteria
        full_trajectory: Complete trajectory (all messages)
        final_state: Final environment state
        llm_evaluator: LLM model for evaluation
        llm_args_evaluator: LLM arguments
        language: Language for evaluation
        window_size: Window size for sliding window evaluation
        overlap: Overlap between windows
    
    Returns:
        Dictionary mapping turn index to score (0.0 to 1.0)
        - Key: Turn index (0-based, corresponds to message index)
        - Value: Rubric score (fraction of met rubrics at that turn)
    """
    if task.evaluation_criteria is None:
        # No evaluation criteria - return 0.0 for all turns
        return {i: 0.0 for i in range(len(full_trajectory))}
    
    evaluation_criteria = task.evaluation_criteria
    if not evaluation_criteria.expected_states and not evaluation_criteria.overall_rubrics:
        return {i: 0.0 for i in range(len(full_trajectory))}
    
    turn_scores = {}
    
    # Evaluate trajectory incrementally
    # For each turn, evaluate trajectory up to that turn
    for turn_idx in range(len(full_trajectory)):
        partial_trajectory = full_trajectory[:turn_idx + 1]
        
        # Evaluate partial trajectory
        reward_info = TrajectoryEvaluator.calculate_reward(
            task=task,
            full_trajectory=partial_trajectory,
            final_state=final_state,  # Use final state (may not be accurate for partial, but needed for API)
            window_size=window_size,
            overlap=overlap,
            llm_evaluator=llm_evaluator,
            llm_args_evaluator=llm_args_evaluator,
            language=language
        )
        
        # Extract rubric score (fraction of met rubrics)
        rubric_score = reward_info.reward_breakdown.get(RewardType.NL_ASSERTION, 0.0)
        turn_scores[turn_idx] = rubric_score
    
    return turn_scores


def compute_marginal_reward_with_turn_scores(
    task: Task,
    full_trajectory: List[Message],
    masked_trajectory: List[Message],
    final_state: dict,
    turn_idx: int,
    llm_evaluator: Optional[str] = None,
    llm_args_evaluator: Optional[dict] = None,
    language: Optional[str] = None,
    window_size: int = 10,
    overlap: int = 2
) -> float:
    """
    Compute marginal reward by comparing scores of full vs masked trajectory at a specific turn.
    
    Args:
        task: VitaBench task
        full_trajectory: Original trajectory
        masked_trajectory: Trajectory with masked action
        final_state: Final environment state
        turn_idx: Turn index where action was masked
        llm_evaluator: LLM model for evaluation
        llm_args_evaluator: LLM arguments
        language: Language for evaluation
        window_size: Window size for sliding window evaluation
        overlap: Overlap between windows
    
    Returns:
        Marginal reward: score(full) - score(masked) at turn_idx
    """
    # Compute scores for both trajectories up to turn_idx
    full_scores = compute_turn_level_scores(
        task=task,
        full_trajectory=full_trajectory[:turn_idx + 1],
        final_state=final_state,
        llm_evaluator=llm_evaluator,
        llm_args_evaluator=llm_args_evaluator,
        language=language,
        window_size=window_size,
        overlap=overlap
    )
    
    masked_scores = compute_turn_level_scores(
        task=task,
        full_trajectory=masked_trajectory[:turn_idx + 1],
        final_state=final_state,
        llm_evaluator=llm_evaluator,
        llm_args_evaluator=llm_args_evaluator,
        language=language,
        window_size=window_size,
        overlap=overlap
    )
    
    # Get scores at turn_idx
    full_score = full_scores.get(turn_idx, 0.0)
    masked_score = masked_scores.get(turn_idx, 0.0)
    
    # Marginal reward = difference in scores
    marginal_reward = full_score - masked_score
    
    return marginal_reward

