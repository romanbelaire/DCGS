"""Episode state container for parallel episode processing."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union

from ..belief import BeliefState, DialogueState
from ..data.multiwoz_loader import MultiWOZDialogue
from ..environments.multiwoz_env import MultiWOZEnvironment
from ..environments.dialogue_env import DialogueEnvironment


@dataclass
class EpisodeState:
    """
    Container for all state associated with a single episode.
    
    This allows multiple episodes to be processed in parallel.
    """
    # Episode metadata
    dialogue_idx: int
    dialogue_id: str
    dialogue_data: Union[MultiWOZDialogue, Any]  # Can be MultiWOZDialogue, Task, or conversation dict
    ground_truth_goal: str
    
    # Environment
    env: Union[MultiWOZEnvironment, DialogueEnvironment]  # Support different environment types
    initial_observation: str
    initial_env_info: Dict
    
    # Belief and dialogue state
    belief_state: BeliefState = field(default_factory=BeliefState)
    dialogue_state: DialogueState = field(default_factory=lambda: DialogueState(belief_state=BeliefState()))
    
    # Per-episode tracking
    context_entropies: List[float] = field(default_factory=list)
    rollout: List[Dict] = field(default_factory=list)
    chosen_beliefs_per_turn: List[str] = field(default_factory=list)
    
    # Per-turn evaluation data entries include:
    # 'observation', 'll_action', 'candidate_beliefs', 'q_values',
    # 'selected_belief', 'top_probability_belief', 'random_baseline_belief',
    # 'baseline_metrics', 'ground_truth_goal'
    turn_evaluation_data: List[Dict] = field(default_factory=list)
    
    # Episode statistics
    episode_total_reward: float = 0.0
    episode_value_losses: List[float] = field(default_factory=list)
    episode_q_min_losses: List[float] = field(default_factory=list)
    episode_q_min_target_variances: List[float] = field(default_factory=list)
    episode_q_min_prediction_variances: List[float] = field(default_factory=list)
    actual_turns_processed: int = 0
    
    # Current turn state
    turn: int = 0
    observation: str = ""
    agent_response: str = ""
    reward: float = 0.0
    goal_achieved: float = 0.0  # Fraction of goals achieved (0.0 to 1.0)
    done_from_env: bool = False
    env_info: Dict = field(default_factory=dict)
    reward_from_env: float = 0.0
    reward_task: Optional[float] = None
    reward_harm: Optional[float] = None
    user_actions: List[str] = field(default_factory=list)
    current_obs_for_action: str = ""
    previous_obs_for_action: str = ""
    
    # Forward-fill tracking for user state (goal_state)
    # This ensures Q scoring has accurate user state even when env_info doesn't have goal_state
    last_known_goal_state: Optional[Any] = None  # Last known goal_state from env_info
    
    # Debug: Store belief generation debug info when all candidates are [SKIP]
    belief_gen_debug_prompt: str = ""
    belief_gen_debug_raw_output: str = ""
    belief_gen_debug_history: List = field(default_factory=list)
    
    # Episode status
    is_active: bool = True  # True if episode is still running
    is_initialized: bool = False  # True if episode has been initialized
    is_frozen: bool = False  # True if episode failed belief generation and should be retried
    frozen_retry_count: int = 0  # Number of times this episode has been frozen (max 3 before termination)
    # CARES/WildJailbreak: fulfillment judge parse failure — retry env.step next iteration (no belief/Q regen)
    judge_step_pending: bool = False
    pending_env_action: Optional[str] = None
    judge_step_retry_count: int = 0
    
    def initialize(self) -> None:
        """Initialize the episode state."""
        if self.is_initialized:
            return
        
        # Initialize dialogue state
        self.dialogue_state = DialogueState(
            belief_state=self.belief_state,
            ground_truth_goal={"goal": self.ground_truth_goal}
        )
        
        # Set initial observation
        self.current_obs_for_action = self.initial_observation
        self.observation = self.initial_observation
        
        # Initialize last known goal_state from initial_env_info if available
        if self.initial_env_info and self.initial_env_info.get('goal_state'):
            self.last_known_goal_state = self.initial_env_info.get('goal_state')
        
        self.is_initialized = True
        self.update_current_observation_context()
    
    def finalize(self) -> Dict:
        """
        Finalize episode and return statistics.
        
        Returns:
            Dictionary with episode statistics
        """
        self.is_active = False
        
        avg_value_loss = (
            sum(self.episode_value_losses) / len(self.episode_value_losses)
            if self.episode_value_losses else 0.0
        )
        avg_q_min_loss = (
            sum(self.episode_q_min_losses) / len(self.episode_q_min_losses)
            if self.episode_q_min_losses else 0.0
        )
        avg_q_min_target_variance = (
            sum(self.episode_q_min_target_variances) / len(self.episode_q_min_target_variances)
            if self.episode_q_min_target_variances else 0.0
        )
        avg_q_min_prediction_variance = (
            sum(self.episode_q_min_prediction_variances) / len(self.episode_q_min_prediction_variances)
            if self.episode_q_min_prediction_variances else 0.0
        )
        
        from ..belief import compute_entropy
        final_entropy = (
            compute_entropy([c.probability for c in self.belief_state.candidates])
            if self.belief_state.candidates else 0.0
        )
        
        harmful_level = (
            self.dialogue_data.get("harmful_level", 0)
            if isinstance(self.dialogue_data, dict)
            else None
        )
        return {
            'dialogue_idx': self.dialogue_idx,
            'dialogue_id': self.dialogue_id,
            'episode_total_reward': self.episode_total_reward,
            'episode_length': self.actual_turns_processed,
            'goal_achieved': self.dialogue_state.goal_achieved,
            'final_entropy': final_entropy,
            'avg_value_loss': avg_value_loss,
            'avg_q_min_loss': avg_q_min_loss,
            'avg_q_min_target_variance': avg_q_min_target_variance,
            'avg_q_min_prediction_variance': avg_q_min_prediction_variance,
            'chosen_beliefs': self.chosen_beliefs_per_turn.copy(),
            # goal is a required field in MultiWOZDialogue, so it should always exist
            # However, it might be an empty dict {} in rare cases:
            # - Annotation errors/inconsistencies in earlier MultiWOZ versions
            # - Missing goal data in the source JSON
            # - Dialogue with no domain-specific goals
            # Empty goals are converted to None so they're excluded from evaluation
            # For non-MultiWOZ environments, goal_json may not exist
            'goal_json': getattr(self.dialogue_data, 'goal', None) if hasattr(self.dialogue_data, 'goal') else None,
            'initial_observation': self.initial_observation,
            'turn_evaluation_data': [turn_data.copy() for turn_data in self.turn_evaluation_data],
            'harmful_level': harmful_level,
        }
    
    def reset_for_new_episode(
        self,
        new_dialogue_idx: int,
        new_dialogue_data: Union[MultiWOZDialogue, Any],
        new_ground_truth_goal: str,
        new_env: Union[MultiWOZEnvironment, DialogueEnvironment],
        new_initial_observation: str,
        new_initial_env_info: Dict
    ) -> None:
        """Reset this episode state for a new episode (reuse container)."""
        self.dialogue_idx = new_dialogue_idx
        # dialogue_id may be an attribute or a dict key
        if hasattr(new_dialogue_data, 'dialogue_id'):
            self.dialogue_id = new_dialogue_data.dialogue_id
        elif isinstance(new_dialogue_data, dict):
            self.dialogue_id = new_dialogue_data.get('id', new_dialogue_data.get('dialogue_id', str(new_dialogue_idx)))
        elif hasattr(new_dialogue_data, 'id'):
            self.dialogue_id = new_dialogue_data.id
        else:
            self.dialogue_id = str(new_dialogue_idx)
        self.dialogue_data = new_dialogue_data
        self.ground_truth_goal = new_ground_truth_goal
        self.env = new_env
        self.initial_observation = new_initial_observation
        self.initial_env_info = new_initial_env_info
        
        # Reset all state
        self.belief_state = BeliefState()
        self.dialogue_state = DialogueState(belief_state=self.belief_state)
        self.context_entropies = []
        self.rollout = []
        self.chosen_beliefs_per_turn = []
        self.turn_evaluation_data = []
        self.episode_total_reward = 0.0
        self.episode_value_losses = []
        self.episode_q_min_losses = []
        self.episode_q_min_target_variances = []
        self.episode_q_min_prediction_variances = []
        self.actual_turns_processed = 0
        self.turn = 0
        self.observation = ""
        self.agent_response = ""
        self.reward = 0.0
        self.goal_achieved = 0.0
        self.done_from_env = False
        self.env_info = {}
        self.reward_from_env = 0.0
        self.reward_task = None
        self.reward_harm = None
        self.user_actions = []
        self.current_obs_for_action = ""
        self.previous_obs_for_action = ""
        self.belief_gen_debug_prompt = ""
        self.belief_gen_debug_raw_output = ""
        self.belief_gen_debug_history = []
        self.last_known_goal_state = None
        self.is_active = True
        self.is_initialized = False
        self.is_frozen = False
        self.frozen_retry_count = 0
        self.judge_step_pending = False
        self.pending_env_action = None
        self.judge_step_retry_count = 0
        
        # Re-initialize
        self.initialize()
    
    def format_dialogue_history(self) -> str:
        """
        Format the dialogue history as a multi-turn transcript.
        
        Returns:
            A string containing all agent/user exchanges in chronological order.
        """
        parts = []
        if self.belief_state.history:
            for idx, (agent_action, user_observation) in enumerate(self.belief_state.history, start=1):
                agent_text = agent_action.strip() if agent_action and agent_action.strip() else "[NO_AGENT_ACTION]"
                user_text = user_observation.strip() if user_observation and user_observation.strip() else "[NO_USER_RESPONSE]"
                parts.append(f"Turn {idx}:")
                parts.append(f"Agent: {agent_text}")
                parts.append(f"User: {user_text}")
        else:
            initial_obs = self.initial_observation.strip() if self.initial_observation and self.initial_observation.strip() else "[NO_USER_RESPONSE]"
            parts.append("Turn 1:")
            parts.append("Agent: [NO_AGENT_ACTION]")
            parts.append(f"User: {initial_obs}")
        
        return "\n".join(parts).strip()
    
    def update_current_observation_context(self) -> None:
        """
        Refresh the observation context used by the high-level agent.
        """
        prev_context = self.current_obs_for_action
        new_context = self.format_dialogue_history()
        if self.previous_obs_for_action:
            self.previous_obs_for_action = prev_context if prev_context else new_context
        else:
            self.previous_obs_for_action = new_context
        self.current_obs_for_action = new_context



