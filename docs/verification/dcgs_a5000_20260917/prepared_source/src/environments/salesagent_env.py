"""SalesAgent environment adapter for dialogue Q-learning."""

from typing import Dict, List, Optional, Tuple


from .dialogue_env import DialogueEnvironment, StepResult


class SalesAgentEnvironment(DialogueEnvironment):
    """
    Adapter for SalesAgent (SalesBot 2.0) conversations.
    
    Replays conversation dataset similar to MultiWOZEnvironment.
    """
    
    def __init__(
        self,
        conversation: Dict,
        debug: bool = False
    ):
        """
        Initialize with SalesBot conversation.
        
        Args:
            conversation: SalesBot conversation dict with turns
            debug: Whether to print debug statements
        """
        self.conversation = conversation
        self.debug = debug
        self._turn_idx = 0
        self._turns = self._parse_conversation(conversation)
        self._goal_achieved = False
    
    def reset(self) -> Tuple[str, Dict]:
        """Reset to first user turn."""
        self._turn_idx = 0
        self._goal_achieved = False
        
        # Find first user turn
        first_obs = ""
        for turn in self._turns:
            if turn["role"] == "user":
                first_obs = turn["content"]
                break
        
        info = {
            "conversation_id": self.conversation.get("id", ""),
            "intent": self.conversation.get("intent", ""),
            "goal_achieved": False
        }
        return first_obs, info
    
    def step(self, action: str) -> StepResult:
        """
        Execute agent action and return next user observation.
        
        Args:
            action: Agent response text
        
        Returns:
            observation: Next user message
            reward: Reward based on conversation success
            done: Whether conversation ended
            info: Additional information
        """
        # Find next user turn
        user_turn = None
        for i in range(self._turn_idx, len(self._turns)):
            if self._turns[i]["role"] == "user":
                user_turn = self._turns[i]
                self._turn_idx = i + 1
                break
        
        if user_turn is None:
            return StepResult("", 0.0, True, {"goal_achieved": self._goal_achieved})
        
        observation = user_turn["content"]
        
        # Check for conversation end signals
        done = self._check_conversation_end(observation)
        
        # Compute reward (could be based on intent fulfillment, conversation quality, etc.)
        reward = self._compute_reward(observation, done)
        
        if done:
            self._goal_achieved = True
        
        info = {
            "conversation_id": self.conversation.get("id", ""),
            "intent": self.conversation.get("intent", ""),
            "turn": self._turn_idx,
            "goal_achieved": self._goal_achieved
        }

        return StepResult(observation, reward, done, info)
    
    def check_goal(self, observation: str) -> bool:
        """Check if conversation goal (intent) is satisfied."""
        return self._goal_achieved

    def compute_reward_for_marginal(
        self, action: str, actual_reward: Optional[float] = None
    ) -> float:
        base_reward = 1.0 if self._goal_achieved else 0.0
        if actual_reward is not None:
            return actual_reward
        return base_reward * 0.5

    def _parse_conversation(self, conversation: Dict) -> List[Dict]:
        """Parse SalesBot conversation format into turns."""
        turns = []
        
        # Handle different SalesBot formats
        if "conversation" in conversation:
            conv_data = conversation["conversation"]
        elif "messages" in conversation:
            conv_data = conversation["messages"]
        elif "dialog" in conversation:
            # Handle dialog format where each item is a string like "User: ..." or "Agent: ..."
            conv_data = []
            for item in conversation["dialog"]:
                if isinstance(item, str):
                    if item.startswith("User: "):
                        conv_data.append({"from": "human", "value": item[6:]})
                    elif item.startswith("Agent: "):
                        conv_data.append({"from": "gpt", "value": item[7:]})
                else:
                    conv_data.append(item)
        else:
            conv_data = conversation
        
        for item in conv_data:
            if isinstance(item, dict):
                role = item.get("role", item.get("from", ""))
                content = item.get("content", item.get("value", ""))
                if role and content:
                    # Normalize role names
                    if role in ["human", "user", "usr"]:
                        role = "user"
                    elif role in ["gpt", "assistant", "agent", "system"]:
                        role = "agent"
                    turns.append({"role": role, "content": content})
        
        return turns
    
    def _compute_reward(self, observation: str, done: bool) -> float:
        """Compute reward based on conversation quality."""
        # Simple reward: 1.0 if conversation completed successfully
        if done and self._goal_achieved:
            return 1.0
        return 0.0
    
    def _check_conversation_end(self, observation: str) -> bool:
        """Check if conversation has ended."""
        # Look for end signals in user message
        end_signals = ["goodbye", "bye", "thanks", "thank you", "see you"]
        obs_lower = observation.lower()
        return any(signal in obs_lower for signal in end_signals)

