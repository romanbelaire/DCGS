"""User simulation protocol for dialogue environments."""

from typing import Dict, List, Protocol, Tuple


class UserSimulator(Protocol):
    """
    Protocol for user simulation agents.

    Both UserAgent (MultiWOZ) and PatientAgent (CARES/WildJailbreak) implement
    this interface. Environments depend on UserSimulator instead of concrete types.
    """

    def generate_response(
        self,
        goal_json: Dict,
        dialogue_history: List[Tuple[str, str]],
        agent_action: str,
        **kwargs,
    ) -> str:
        """
        Generate next user/patient message given goal and dialogue context.

        Args:
            goal_json: Goal dict (MultiWOZ domain structure or CARES base_prompt).
            dialogue_history: Previous (agent_action, user_response) pairs.
            agent_action: Agent's last message.
            **kwargs: Env-specific (persona, goal_progress_summary, harmful_level, etc.).

        Returns:
            User's observation/response.
        """
        ...
