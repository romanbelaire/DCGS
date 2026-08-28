"""GPT-based agent implementations using OpenAI API."""

from typing import Dict, List, Optional, Tuple

from .base_agent import BaseAgent
from .high_level_agent import HighLevelAgent
from .low_level_agent import LowLevelAgent
from .user_agent import UserAgent
from ..belief import BeliefCandidate, BeliefState
from ..utils.llm_utils import batch_generate_gpt


class GPTHighLevelAgent(HighLevelAgent):
    """High-level agent using GPT API instead of local model."""
    
    def __init__(self, tokenizer, prompt_manager=None, template_name: str = "belief_generation", 
                 model_name: str = "gpt-4o-mini", api_key: Optional[str] = None,
                 reasoning_effort: Optional[str] = None):
        """Initialize GPT-based high-level agent.
        
        Args:
            tokenizer: Tokenizer (still needed for prompt formatting, but model not used)
            prompt_manager: Prompt manager
            template_name: Name of the belief generation template to use
            model_name: GPT model name (e.g., "gpt-4o-mini")
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
        """
        # Initialize with None model (we'll override batch_generate calls)
        super().__init__(model=None, tokenizer=tokenizer, prompt_manager=prompt_manager, template_name=template_name)
        self.gpt_model_name = model_name
        self.gpt_api_key = api_key
        self.gpt_reasoning_effort = reasoning_effort
    
    def generate_candidate_beliefs_batch(
        self,
        histories: List[List[Tuple[str, str]]],
        n_candidates: int,
        temperature: float = 0.7,
        max_new_tokens: int = 200,
        return_debug_info: bool = False,
        chunk_size: Optional[int] = None,
        base_prompts: Optional[List[Optional[str]]] = None,
    ):
        """Batch generate candidate belief summaries using GPT API."""
        if not histories:
            return ([], []) if return_debug_info else []

        if base_prompts is not None and len(base_prompts) != len(histories):
            raise ValueError(
                f"base_prompts length ({len(base_prompts) if base_prompts else 0}) must match histories length ({len(histories)})"
            )
        
        # Generate prompts for all histories (adversarial template when base_prompt provided)
        prompts = []
        for idx, history in enumerate(histories):
            bp = base_prompts[idx] if base_prompts is not None else None
            prompt = self.prompt_manager.get_high_level_prompt(
                template_name=self.template_name,
                dialogue_history=history,
                n_candidates=n_candidates,
                base_prompt=bp,
            )
            prompts.append(prompt)
        
        # Create prefill suffix to force numbered list format
        prefill_suffix = f"Here are {n_candidates} candidate beliefs about the user's desires:\n\n1. "
        
        # Use GPT API for generation
        responses = batch_generate_gpt(
            prompts=prompts,
            model_name=self.gpt_model_name,
            api_key=self.gpt_api_key,
            reasoning_effort=self.gpt_reasoning_effort,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            prefill_suffix=prefill_suffix,
            chunk_size=chunk_size
        )
        
        # Parse each response into candidates (same as parent class)
        all_belief_candidates = []
        for idx, (generated_text, prompt, history) in enumerate(zip(responses, prompts, histories)):
            candidates = self._parse_belief_candidates(generated_text, n_candidates)
            
            # Initialize with uniform probabilities
            uniform_prob = 1.0 / len(candidates) if candidates else 0.0
            belief_candidates = []
            for summary in candidates:
                context = f"Belief: {summary}"
                candidate = BeliefCandidate(
                    summary=summary,
                    context=context,
                    probability=uniform_prob
                )
                belief_candidates.append(candidate)
            
            all_belief_candidates.append(belief_candidates)
        
        if return_debug_info:
            return all_belief_candidates, responses
        else:
            return all_belief_candidates


class GPTLowLevelAgent(LowLevelAgent):
    """Low-level agent using GPT API instead of local model."""
    
    def __init__(self, tokenizer, prompt_manager=None, 
                 model_name: str = "gpt-4o-mini", api_key: Optional[str] = None,
                 reasoning_effort: Optional[str] = None):
        """Initialize GPT-based low-level agent.
        
        Args:
            tokenizer: Tokenizer (still needed for prompt formatting, but model not used)
            prompt_manager: Prompt manager
            model_name: GPT model name (e.g., "gpt-4o-mini")
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
        """
        # Initialize with None model (we'll override batch_generate calls)
        super().__init__(model=None, tokenizer=tokenizer, prompt_manager=prompt_manager)
        self.gpt_model_name = model_name
        self.gpt_api_key = api_key
        self.gpt_reasoning_effort = reasoning_effort
    
    def generate_actions_from_prompts(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        do_sample: bool = False,
        chunk_size: int = None,
        template_name: str = "action_generation"
    ) -> List[str]:
        """Generate actions for a batch of prompts using GPT API."""
        if not prompts:
            return []
        
        # Do not force a GPT prefill suffix for chat completions.
        # Appending "[RESPONSE]" to the user message can cause degenerate outputs like only "[/RESPONSE]".
        # The prompt template already specifies the output format.
        prefill_suffix = None
        
        # Use GPT API for generation
        responses = batch_generate_gpt(
            prompts=prompts,
            model_name=self.gpt_model_name,
            api_key=self.gpt_api_key,
            reasoning_effort=self.gpt_reasoning_effort,
            max_new_tokens=128,  # Reduced to prevent long responses
            temperature=temperature,
            do_sample=do_sample,
            prefill_suffix=prefill_suffix,
            chunk_size=chunk_size
        )
        
        # Parse responses using tag-based extraction (same as parent class)
        cleaned_responses = []
        is_userbench = (template_name in ["action_generation_userbench", "action_generation_userbench_baseline"])
        for idx, (response, prompt) in enumerate(zip(responses, prompts)):
            cleaned = self._parse_response_from_tags(response, is_userbench=is_userbench)
            # Validate that parsing didn't result in empty string
            if not cleaned or not cleaned.strip():
                import sys
                print(f"\n[DEBUG] Empty action after parsing (index {idx}):")
                print(f"  Raw response: {repr(response[:500])}")
                print(f"  Prompt length: {len(prompt)}")
                print(f"  Parsed result: {repr(cleaned)}")
                sys.stdout.flush()
            cleaned_responses.append(cleaned)
        return cleaned_responses


class GPTUserAgent(UserAgent):
    """User agent using GPT API instead of local model."""
    
    def __init__(self, tokenizer, prompt_manager=None, 
                 model_name: str = "gpt-4o-mini", api_key: Optional[str] = None):
        """Initialize GPT-based user agent.
        
        Args:
            tokenizer: Tokenizer (still needed for prompt formatting, but model not used)
            prompt_manager: Prompt manager
            model_name: GPT model name (e.g., "gpt-4o-mini")
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
        """
        # Initialize with None model (we'll override batch_generate calls)
        super().__init__(model=None, tokenizer=tokenizer, prompt_manager=prompt_manager)
        self.gpt_model_name = model_name
        self.gpt_api_key = api_key
    
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
        """Generate user response using GPT API."""
        prompt = self.prompt_manager.get_user_prompt(
            template_name="response_generation",
            persona=persona,
            dialogue_history=dialogue_history,
            agent_action=agent_action,
            goal_json=goal_json,
            goal_progress_summary=goal_progress_summary,
            model=None,  # Not used for GPT
            tokenizer=self.tokenizer
        )
        
        responses = batch_generate_gpt(
            prompts=[prompt],
            model_name=self.gpt_model_name,
            api_key=self.gpt_api_key,
            max_new_tokens=256,
            temperature=temperature,
            do_sample=True,
            prefill_suffix="[RESPONSE]",
            chunk_size=chunk_size
        )
        
        return responses[0] if responses else ""
    
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
        """Batch user response generation using GPT API."""
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
                    model=None,  # Not used for GPT
                    tokenizer=self.tokenizer
                )
            )
        if not prompts:
            return []
        responses = batch_generate_gpt(
            prompts=prompts,
            model_name=self.gpt_model_name,
            api_key=self.gpt_api_key,
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
        """Batch judge goal satisfaction using GPT API."""
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
        responses = batch_generate_gpt(
            prompts=prompts,
            model_name=self.gpt_model_name,
            api_key=self.gpt_api_key,
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
        """Judge goal satisfaction using GPT API."""
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
        
        # Generate judgment using GPT
        responses = batch_generate_gpt(
            prompts=[judge_prompt],
            model_name=self.gpt_model_name,
            api_key=self.gpt_api_key,
            max_new_tokens=10,
            temperature=temperature,
            do_sample=False,
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


class GPTPatientAgent:
    """CARES/WJB patient simulator via CMU AI Gateway (no local LLM)."""

    def __init__(
        self,
        tokenizer,
        prompts_path: str = "src/prompts/cares_patient_prompts.json",
        model_name: str = "gpt-5.4-nano",
        api_key: Optional[str] = None,
    ):
        from .patient_agent import _load_patient_prompts, _format_dialogue_history
        from ..data.dialogue_formatter import format_multiwoz_goal

        self.tokenizer = tokenizer
        self.gpt_model_name = model_name
        self.gpt_api_key = api_key
        self._format_dialogue_history = _format_dialogue_history
        self._format_multiwoz_goal = format_multiwoz_goal
        self.prompt_templates = _load_patient_prompts(prompts_path)
        default_id = self.prompt_templates.get("meta", {}).get(
            "default_patient_prompt", "patient_with_goal"
        )
        self._default_template = None
        self._templates_by_id: dict = {}
        for p in self.prompt_templates.get("patient_prompts", []):
            tid = p.get("id")
            if tid:
                self._templates_by_id[tid] = p.get("template", "")
            if tid == default_id:
                self._default_template = p.get("template", "")

    def generate_response(
        self,
        goal_json: Dict,
        dialogue_history: List[Tuple[str, str]],
        agent_action: str,
        **kwargs,
    ) -> str:
        return self.generate_patient_response(
            goal_json=goal_json,
            dialogue_history=dialogue_history,
            agent_action=agent_action,
            temperature=kwargs.get("temperature", 0.7),
            chunk_size=kwargs.get("chunk_size"),
            harmful_level=kwargs.get("harmful_level"),
        )

    def generate_patient_response(
        self,
        goal_json: Dict,
        dialogue_history: List[Tuple[str, str]],
        agent_action: str,
        temperature: float = 0.7,
        chunk_size: int = None,
        harmful_level: Optional[int] = None,
    ) -> str:
        goal_text = self._format_multiwoz_goal(goal_json) if goal_json else "User goal not specified"
        state_text = self._format_dialogue_history(dialogue_history)
        template = self._default_template
        if harmful_level is not None:
            if harmful_level > 0:
                for template_id in ("attacker_harmful_healthcare", "attacker_harmful"):
                    template = self._templates_by_id.get(template_id)
                    if template:
                        break
                if not template:
                    raise ValueError(
                        "Missing adversarial attacker template. Expected one of "
                        "['attacker_harmful_healthcare', 'attacker_harmful'] in prompts file."
                    )
            else:
                for template_id in (
                    "attacker_benign_healthcare",
                    "attacker_benign",
                    "patient_with_goal",
                ):
                    template = self._templates_by_id.get(template_id)
                    if template:
                        break
                if not template:
                    raise ValueError(
                        "Missing benign user template. Expected one of "
                        "['attacker_benign_healthcare', 'attacker_benign', 'patient_with_goal'] "
                        "in prompts file."
                    )
        if template is None:
            template = (
                "You are a patient. Your goal is to eventually ask: {goal_text}\n\n"
                "Conversation:\n{state_text}\n\nAssistant: {agent_action}\n\nYour message:"
            )
        prompt = template.format(
            base_prompt=goal_text,
            goal_text=goal_text,
            state_text=state_text,
            agent_action=agent_action or "(Conversation start)",
        )
        responses = batch_generate_gpt(
            prompts=[prompt],
            model_name=self.gpt_model_name,
            api_key=self.gpt_api_key,
            max_new_tokens=256,
            temperature=temperature,
            do_sample=True,
            chunk_size=chunk_size,
        )
        return responses[0].strip() if responses else ""

