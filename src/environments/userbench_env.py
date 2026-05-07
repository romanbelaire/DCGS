"""UserBench environment adapter for dialogue Q-learning."""

import sys
from pathlib import Path
from typing import Dict, Tuple

# Add UserBench directory to Python path if not already present
_project_root = Path(__file__).parent.parent.parent
_userbench_path = _project_root / 'UserBench'
if _userbench_path.exists() and str(_userbench_path) not in sys.path:
    sys.path.insert(0, str(_userbench_path))

from .dialogue_env import DialogueEnvironment


class UserBenchEnvironment(DialogueEnvironment):
    """
    Adapter for UserBench/TravelGym environments.
    
    Wraps TravelGym TravelEnv to provide DialogueEnvironment interface.
    """
    
    def __init__(
        self,
        travel_env,
        task_data: Dict,
        max_steps: int = 20,
        debug: bool = False
    ):
        """
        Initialize UserBench environment adapter.
        
        Args:
            travel_env: TravelGym TravelEnv instance
            task_data: Task data dictionary
            max_steps: Maximum number of steps
            debug: Whether to print debug statements
        """
        self.travel_env = travel_env
        self.task_data = task_data
        self.max_steps = max_steps
        self.debug = debug
        self._turn_count = 0
        self._last_observation = None
        self._last_info = None
        self._last_agent_action = ""  # Track last agent action for offline mode
        self._agent_action_queue = []  # Queue of agent actions from messages (if available)
        self._message_idx = 0  # Index into messages for extracting agent actions
    
    def reset(self) -> Tuple[str, Dict]:
        """Reset environment and return initial observation."""
        obs_dict, info = self.travel_env.reset()
        self._turn_count = 0
        self._last_observation = obs_dict
        self._last_info = info
        self._last_agent_action = ""
        self._message_idx = 0
        
        # Extract agent actions from messages if available (for offline mode)
        self._agent_action_queue = []
        if "messages" in self.task_data:
            messages = self.task_data["messages"]
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "").lower()
                    content = msg.get("content", "")
                    if role in ["assistant", "agent", "system"] and content:
                        self._agent_action_queue.append(content)
        
        # Extract feedback text as observation
        observation = obs_dict.get("feedback", "")
        return observation, info
    
    def step(self, action: str = None) -> Tuple:
        """
        Execute agent action and return observation, reward, done, info (online mode)
        or agent_action, observation, reward, done, info (offline mode).
        
        For offline mode compatibility, this method can be called without an action.
        In that case, it returns the last agent action that was executed.
        
        Args:
            action: Agent action (text). If None, returns last action (offline mode).
        
        Returns:
            Online mode (action provided): (observation, reward, done, info)
            Offline mode (action is None): (agent_action, observation, reward, done, info)
        """
        # For offline mode: if no action provided, extract from queue or return last action
        if action is None:
            # Try to get agent action from queue (if available from messages)
            if self._message_idx < len(self._agent_action_queue):
                agent_action = self._agent_action_queue[self._message_idx]
                self._message_idx += 1
            else:
                # No more actions in queue, return last action (or empty if first turn)
                agent_action = self._last_agent_action
            
            # Return agent action and current observation (offline mode)
            observation = self._last_observation.get("feedback", "") if self._last_observation else ""
            reward = self._last_observation.get("last_reward", 0.0) if self._last_observation else 0.0
            done = (
                (self._last_observation and self._last_observation.get("episode_complete", False)) or
                self._turn_count >= self.max_steps
            ) if self._last_observation else False
            info = self._last_info if self._last_info else {}
            from .dialogue_env import StepResult
            return StepResult(observation, reward, done, info, agent_action=agent_action)
        
        # Store agent action for offline mode compatibility
        self._last_agent_action = action
        
        # TravelEnv.step returns (observation, reward, terminated, truncated, info)
        obs_dict, reward, terminated, truncated, info = self.travel_env.step(action)
        
        self._turn_count += 1
        self._last_observation = obs_dict
        self._last_info = info
        
        # Extract feedback text as observation
        observation = obs_dict.get("feedback", "")
        
        # Check if done
        done = terminated or truncated or self._turn_count >= self.max_steps
        
        # Get goal achievement fraction and add to info dict
        goal_achieved_fraction = self.get_goal_achievement_fraction()
        info["goal_achieved"] = goal_achieved_fraction
        
        from .dialogue_env import StepResult
        return StepResult(observation, reward, done, info)
    
    def get_goal_achievement_fraction(self) -> float:
        """Get goal achievement as fraction (0.0 to 1.0)."""
        if self._last_observation:
            episode_complete = self._last_observation.get("episode_complete", False)
            if episode_complete:
                return 1.0
            
            # Return elicitation ratio directly (already 0.0 to 1.0)
            elicitation_ratio = self._last_observation.get("elicitation_ratio", 0.0)
            return elicitation_ratio
        
        return 0.0
    
    def check_goal(self, observation: str) -> bool:
        """Check if task goal is satisfied (binary, for backward compatibility)."""
        return self.get_goal_achievement_fraction() >= 1.0

