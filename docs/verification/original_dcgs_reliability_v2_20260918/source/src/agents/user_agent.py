"""User agent with persona-based simulation."""

import random
from typing import Dict, List, Optional, Tuple

from .base_agent import BaseAgent
from ..utils.llm_utils import batch_generate


class UserAgent(BaseAgent):
    """Simulates user responses using personas (ground truth beliefs)."""

    def generate_response(
        self,
        goal_json: Dict,
        dialogue_history: List[Tuple[str, str]],
        agent_action: str,
        **kwargs,
    ) -> str:
        """UserSimulator protocol: generate user response from goal and context."""
        persona = kwargs.get("persona")
        goal_progress_summary = kwargs.get("goal_progress_summary")
        if persona is None:
            raise ValueError("persona required for UserAgent.generate_response")
        return self.generate_user_response(
            persona=persona,
            dialogue_history=dialogue_history,
            agent_action=agent_action,
            goal_json=goal_json,
            goal_progress_summary=goal_progress_summary,
            temperature=kwargs.get("temperature", 0.7),
            chunk_size=kwargs.get("chunk_size"),
        )

    def load_persona(self, persona_id: str) -> Dict:
        """Load persona definition."""
        return self.prompt_manager.load_persona(persona_id)
    
    def generate_user_response(
        self,
        persona: Dict,
        dialogue_history: List[Tuple[str, str]],
        agent_action: str,
        temperature: float = 0.7,
        goal_json: Optional[Dict] = None,
        goal_progress_summary: Optional[str] = None,
        chunk_size: int = None
    ) -> str:
        """
        Generate user response based on persona and dialogue context.
        
        Args:
            persona: Persona dictionary with ground truth beliefs
            dialogue_history: Previous dialogue turns
            agent_action: Agent's last action
            temperature: Sampling temperature
            goal_json: MultiWOZ goal JSON (optional)
        
        Returns:
            User's observation/response
        """
        prompt = self.prompt_manager.get_user_prompt(
            template_name="response_generation",
            persona=persona,
            dialogue_history=dialogue_history,
            agent_action=agent_action,
            goal_json=goal_json,
            goal_progress_summary=goal_progress_summary,
            model=self.model,
            tokenizer=self.tokenizer
        )
        
        responses = batch_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=[prompt],
            max_new_tokens=256,
            temperature=temperature,
            do_sample=True,
            prefill_suffix="[RESPONSE]",
            chunk_size=chunk_size
        )
        
        return responses[0] if responses else ""
    
    def act(
        self,
        persona: Dict,
        dialogue_history: List[Tuple[str, str]],
        agent_action: str,
        goal_json: Optional[Dict] = None,
        goal_progress_summary: Optional[str] = None
    ) -> str:
        """Interface method for acting."""
        return self.generate_user_response(
            persona,
            dialogue_history,
            agent_action,
            goal_json=goal_json,
            goal_progress_summary=goal_progress_summary
        )

    def generate_user_response_batch(
        self,
        personas: List[Dict],
        dialogue_histories: List[List[Tuple[str, str]]],
        agent_actions: List[str],
        temperature: float = 0.7,
        goal_jsons: Optional[List[Dict]] = None,
        goal_progress_summaries: Optional[List[str]] = None,
        chunk_size: int = None
    ) -> List[str]:
        """
        Batch user response generation.
        """
        prompts: List[str] = []
        for persona, history, agent_action, goal_json, goal_prog in zip(
            personas,
            dialogue_histories,
            agent_actions,
            goal_jsons or [None] * len(personas),
            goal_progress_summaries or [None] * len(personas)
        ):
            prompts.append(
                self.prompt_manager.get_user_prompt(
                    template_name="response_generation",
                    persona=persona,
                    dialogue_history=history,
                    agent_action=agent_action,
                    goal_json=goal_json,
                    goal_progress_summary=goal_prog,
                    model=self.model,
                    tokenizer=self.tokenizer
                )
            )
        if not prompts:
            return []
        responses = batch_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=prompts,
            max_new_tokens=256,
            temperature=temperature,
            do_sample=True,
            prefill_suffix="[RESPONSE]",
            chunk_size=chunk_size
        )
        return responses

    def judge_goal_satisfaction_batch(
        self,
        dialogue_histories: List[List[Tuple[str, str]]],
        goal_jsons: List[Dict],
        agent_responses: List[str],
        temperature: float = 0.0,
        chunk_size: int = None
    ) -> List[bool]:
        """
        Batch judge goal satisfaction across multiple dialogues.
        """
        from ..data.dialogue_formatter import format_multiwoz_goal
        prompts: List[str] = []
        for history, goal_json, agent_response in zip(dialogue_histories, goal_jsons, agent_responses):
            formatted_history = []
            for agent_action, user_response in history:
                if agent_action:
                    formatted_history.append(f"Agent: {agent_action}")
                if user_response:
                    formatted_history.append(f"User: {user_response}")
            dialogue_text = "\n".join(formatted_history) if formatted_history else "No previous dialogue."
            goal_text = format_multiwoz_goal(goal_json)
            judge_prompt = f"""Does this agent response fulfill the user's goal?

Chat History:
{dialogue_text}

Agent's Last Response:
{agent_response}

User's Goal:
{goal_text}

Provide a YES or a NO:"""
            prompts.append(judge_prompt)
        if not prompts:
            return []
        responses = batch_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=prompts,
            max_new_tokens=10,
            temperature=temperature,
            do_sample=False,
            prefill_suffix="",
            chunk_size=chunk_size
        )
        results: List[bool] = []
        for judgment in responses:
            j = judgment.strip().upper() if judgment else ""
            if "YES" in j:
                results.append(True)
            elif "NO" in j:
                results.append(False)
            else:
                results.append(False)
        return results
    
    def judge_goal_satisfaction(
        self,
        dialogue_history: List[Tuple[str, str]],
        goal_json: Dict,
        agent_response: str,
        temperature: float = 0.0,
        chunk_size: int = None
    ) -> bool:
        """
        Judge whether the agent's response fulfills the user's goal.
        
        This is a separate step from response generation to ensure goal state
        is updated before the user generates their next response.
        
        Args:
            dialogue_history: Previous dialogue turns
            goal_json: MultiWOZ goal JSON
            agent_response: Agent's last response
            temperature: Sampling temperature (default 0.0 for deterministic)
        
        Returns:
            True if goal is satisfied, False otherwise
        """
        from ..data.dialogue_formatter import format_multiwoz_goal
        
        # Format dialogue history
        formatted_history = []
        for agent_action, user_response in dialogue_history:
            if agent_action:
                formatted_history.append(f"Agent: {agent_action}")
            if user_response:
                formatted_history.append(f"User: {user_response}")
        
        dialogue_text = "\n".join(formatted_history) if formatted_history else "No previous dialogue."
        
        # Format goal
        goal_text = format_multiwoz_goal(goal_json)
        
        # Create judge prompt
        judge_prompt = f"""Does this agent response fulfill the user's goal?

Chat History:
{dialogue_text}

Agent's Last Response:
{agent_response}

User's Goal:
{goal_text}

Provide a YES or a NO:"""
        
        # Generate judgment
        responses = batch_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=[judge_prompt],
            max_new_tokens=10,
            temperature=temperature,
            do_sample=False,  # Deterministic for judgment
            prefill_suffix="",
            chunk_size=chunk_size
        )
        
        if not responses:
            return False
        
        judgment = responses[0].strip().upper()
        
        # Check for YES/NO
        if "YES" in judgment:
            return True
        elif "NO" in judgment:
            return False
        else:
            # Default to False if unclear
            return False

