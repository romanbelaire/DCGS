"""Prompt loading and management."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class PromptManager:
    """Manages prompt templates and persona definitions."""
    
    def __init__(
        self,
        templates_file: str = "src/prompts/templates.jsonl",
        personas_file: str = "src/prompts/personas.jsonl"
    ):
        self.templates_file = Path(templates_file)
        self.personas_file = Path(personas_file)
        self._template_cache = {}  # Key: "{category}/{name}", Value: template string
        self._persona_cache = {}  # Key: persona_id, Value: persona dict
        
        # Load templates and personas from separate files
        self._load_templates()
        self._load_personas()
    
    def _load_templates(self) -> None:
        """Load all prompt templates from JSONL file."""
        if not self.templates_file.exists():
            raise FileNotFoundError(f"Templates file not found: {self.templates_file}")
        
        with open(self.templates_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                entry = json.loads(line)
                category = entry.get("category")
                name = entry.get("name")
                content = entry.get("content")
                
                # Store template with category/name as key
                cache_key = f"{category}/{name}"
                self._template_cache[cache_key] = content
    
    def _load_personas(self) -> None:
        """Load all personas from JSONL file."""
        if not self.personas_file.exists():
            raise FileNotFoundError(f"Personas file not found: {self.personas_file}")
        
        with open(self.personas_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                persona = json.loads(line)
                persona_id = persona.get("persona_id")
                
                if persona_id:
                    # Store persona with persona_id as key
                    self._persona_cache[persona_id] = persona
    
    def load_template(self, template_name: str, agent_type: str) -> str:
        """
        Load prompt template from cache.
        
        Args:
            template_name: Name of template (e.g., "belief_generation")
            agent_type: Agent type directory ("high_level", "low_level", "user", "value")
        
        Returns:
            Template string
        """
        # Remove .txt extension if present
        if template_name.endswith(".txt"):
            template_name = template_name[:-4]
        
        cache_key = f"{agent_type}/{template_name}"
        if cache_key not in self._template_cache:
            raise FileNotFoundError(f"Template not found: {cache_key}")
        
        return self._template_cache[cache_key]
    
    def format_prompt(self, template: str, **kwargs) -> str:
        """
        Format template with variables.
        
        Args:
            template: Template string with {variable} placeholders
            **kwargs: Variables to substitute
        
        Returns:
            Formatted prompt string
        """
        return template.format(**kwargs)
    
    def load_persona(self, persona_id: str) -> Dict:
        """
        Load persona definition from cache.
        
        Args:
            persona_id: Persona identifier (e.g., "persona_1")
        
        Returns:
            Persona dictionary
        """
        if persona_id not in self._persona_cache:
            raise FileNotFoundError(f"Persona not found: {persona_id}")
        
        return self._persona_cache[persona_id]
    
    def get_high_level_prompt(
        self,
        template_name: str,
        dialogue_history: List[Tuple[str, str]],
        n_candidates: int,
        base_prompt: Optional[str] = None,
    ) -> str:
        """Load and format high-level agent prompt.
        
        When base_prompt is not None, use belief_generation_adversarial (on-topic, intent-varying).
        The adversarial template uses only dialogue history; it does not receive base_prompt.
        """
        if base_prompt is not None:
            template = self.load_template("belief_generation_adversarial", "high_level")
        else:
            template = self.load_template(template_name, "high_level")
        
        # Format dialogue history (high-level agent requires history)
        if not dialogue_history:
            raise ValueError("High-level agent requires dialogue history; cannot generate beliefs without dialogue context.")
        formatted_history = self._format_history(dialogue_history)
        
        if base_prompt is not None:
            return self.format_prompt(
                template,
                n_candidates=n_candidates,
                formatted_history=formatted_history,
            )
        return self.format_prompt(
            template,
            n_candidates=n_candidates,
            formatted_history=formatted_history
        )

    def get_high_level_iterative_prompt(
        self,
        dialogue_history: List[Tuple[str, str]],
        n_candidates: int,
        next_candidate_index: int,
        existing_candidates: List[str],
        base_prompt: Optional[str] = None,
    ) -> str:
        """Load and format iterative high-level prompt for one candidate at a time."""
        if not dialogue_history:
            raise ValueError("High-level agent requires dialogue history; cannot generate beliefs without dialogue context.")
        if next_candidate_index <= 0:
            raise ValueError("next_candidate_index must be positive.")
        if next_candidate_index > n_candidates:
            raise ValueError("next_candidate_index cannot exceed n_candidates.")

        template_name = (
            "belief_generation_adversarial_iterative"
            if base_prompt is not None
            else "belief_generation_iterative"
        )
        template = self.load_template(template_name, "high_level")
        formatted_history = self._format_history(dialogue_history)
        existing_list = "\n".join(
            f"{idx}. {candidate}" for idx, candidate in enumerate(existing_candidates, start=1)
        )
        if not existing_list:
            existing_list = "None yet."

        return self.format_prompt(
            template,
            n_candidates=n_candidates,
            next_candidate_index=next_candidate_index,
            formatted_history=formatted_history,
            existing_candidates=existing_list,
        )
    
    def get_low_level_prompt(
        self,
        template_name: str,
        belief_context: str,
        history: List[Tuple[str, str]] = None,
        belief_only: bool = True,
        n_candidates: int = 1,
    ) -> str:
        """
        Load and format low-level agent prompt.
        
        Args:
            template_name: Name of the template to use
            belief_context: The belief/understanding of user intent (can be empty string for baseline mode)
            history: Dialogue history (only used if belief_only=False or baseline mode)
            belief_only: If True, generate prompt with only belief (P(a|b)). If False, include history (P(a|o,b))
            n_candidates: Number of candidates to generate (only used by multi-candidate templates)
        """
        # For baseline mode, use the baseline template which doesn't have belief_context
        if template_name in ["action_generation_baseline", "action_generation_userbench_baseline"]:
            if not history:
                raise ValueError("Baseline mode requires dialogue history; cannot generate actions without dialogue context.")
            formatted_history = self._format_history(history)
            template = self.load_template(template_name, "low_level")
            return self.format_prompt(
                template,
                formatted_history=formatted_history
            )
        
        template = self.load_template(template_name, "low_level")
        
        if belief_only:
            # New mode: P(a|b) - only use belief, no history
            # Remove the Dialogue History section from template when belief_only=True
            # This ensures the model only sees the belief context
            import re
            # Remove "Dialogue History:\n{formatted_history}\n" section from template
            template = re.sub(
                r'Dialogue History:\s*\{formatted_history\}\s*\n',
                '',
                template,
                flags=re.MULTILINE
            )
            formatted_history = ""  # Not used, but needed for format_prompt
        else:
            # Old mode: P(a|o,b) - use both belief and history
            if not history:
                raise ValueError("Low-level agent requires dialogue history when belief_only=False; cannot generate actions without dialogue context.")
            formatted_history = self._format_history(history)
        
        return self.format_prompt(
            template,
            belief_context=belief_context,
            formatted_history=formatted_history,
            n_candidates=n_candidates,
        )
    
    def get_user_prompt(
        self,
        template_name: str,
        persona: Dict,
        dialogue_history: List[Tuple[str, str]],
        agent_action: str,
        goal_json: Optional[Dict] = None,
        goal_progress_summary: Optional[str] = None,
        model=None,
        tokenizer=None
    ) -> str:
        """Load and format user agent prompt."""
        template = self.load_template(template_name, "user")
        formatted_history = self._format_history(dialogue_history, user_tag="[YOU]")
        
        if persona is None:
            raise ValueError("Persona is required to generate a user prompt.")

        persona_name = persona.get("name")
        if not persona_name:
            raise ValueError(f"Persona is missing required field 'name': {persona}")

        communication_style = persona.get("communication_style")
        if not communication_style:
            raise ValueError(f"Persona '{persona_name}' is missing 'communication_style'.")

        persona_description = persona.get("persona_description")
        if not persona_description:
            raise ValueError(f"Persona '{persona_name}' is missing 'persona_description'.")

        if not goal_json:
            raise ValueError("goal_json is required when generating user prompts.")

        # Format goal as natural language (use LLM if model/tokenizer provided)
        user_goal = self._format_goal_json(goal_json, model=model, tokenizer=tokenizer)
        
        # Load persona base template and inject persona data
        persona_base = self.load_template("persona_base", "user")
        persona_base_formatted = self.format_prompt(
            persona_base,
            persona_name=persona_name,
            user_goal=user_goal,
            communication_style=self._format_communication_style(communication_style),
            persona_description=persona_description
        )
        
        progress_summary_text = (
            goal_progress_summary.strip()
            if goal_progress_summary
            else "No current desire is active. Wait for the agent to provide or confirm the next goal."
        )

        return self.format_prompt(
            template,
            persona_base_template=persona_base_formatted,
            formatted_history=formatted_history,
            agent_action=agent_action,
            goal_progress_summary=progress_summary_text
        )
    
    def _format_history(self, history: List[Tuple[str, str]], user_tag: str = "User:") -> str:
        """Format dialogue history as string.
        
        Args:
            history: List of (agent_action, user_observation) tuples
            user_tag: Tag to use for user messages (default: "User:", use "[YOU]" for user agent prompts)
        """
        if not history:
            # Empty history is valid for the first turn - user is initiating the conversation
            return "This is the start of the conversation. No previous dialogue."
        
        parts = []
        for i, (action, observation) in enumerate(history):
            parts.append(f"Turn {i+1}:")
            parts.append(f"Agent: {action}")
            
            # For UserBench: if agent action was a [search], the observation is system/database feedback
            # Check if action starts with [search] (case-insensitive)
            if action and action.strip().lower().startswith("[search]"):
                parts.append(f"System: {observation}")
            else:
                parts.append(f"{user_tag} {observation}")
        
        return "\n".join(parts)
    
    def _format_persona_beliefs(self, beliefs: Dict) -> str:
        """Format persona beliefs as string."""
        if not beliefs:
            return "No specific beliefs."
        
        parts = []
        for key, value in beliefs.items():
            if isinstance(value, dict):
                parts.append(f"{key}: {json.dumps(value, indent=2)}")
            else:
                parts.append(f"{key}: {value}")
        
        return "\n".join(parts)
    
    def _format_communication_style(self, style: Dict) -> str:
        """Format communication style as string."""
        if not style:
            raise ValueError("Persona communication_style is missing or empty.")
        
        parts = []
        for key, value in style.items():
            parts.append(f"{key}: {value}")
        
        return "\n".join(parts)
    
    def _parse_goal_from_tags(self, raw_response: str) -> str:
        """
        Parse goal statement from [GOAL] tags.
        
        Extracts content between [GOAL]...[/GOAL] tags and removes any remaining tags
        to ensure only clean text is returned.
        
        When using prefill_suffix="[GOAL]", the generated text starts with [GOAL],
        so we want to find the LAST occurrence to avoid matching prompt examples.
        
        Args:
            raw_response: Raw response from LLM (should only contain generated text, not prompt)
            
        Returns:
            Parsed goal statement (content between tags with all tags removed, or original if no tags)
        """
        import re
        
        # Look for ALL [GOAL]...[/GOAL] tag pairs
        pattern = r'\[GOAL\](.*?)\[/GOAL\]'
        matches = list(re.finditer(pattern, raw_response, re.DOTALL | re.IGNORECASE))
        
        if matches:
            # Use the LAST match to avoid matching prompt examples
            # (the last one should be the actual generated goal)
            last_match = matches[-1]
            parsed = last_match.group(1).strip()
            
            # Remove any remaining tags that might have leaked in (safeguard)
            parsed = re.sub(r'\[/?[^\]]+\]', '', parsed)
            
            return parsed.strip()
        
        # If no tags found, still remove any tags as a safeguard
        cleaned = re.sub(r'\[/?[^\]]+\]', '', raw_response)
        return cleaned.strip()
    
    def _format_goal_json(self, goal_json: Dict, model=None, tokenizer=None) -> str:
        """Format MultiWOZ goal JSON into natural language using LLM if available, otherwise heuristics."""
        if not goal_json:
            return "Not specified"
        
        # Use LLM to convert to natural language if model/tokenizer provided
        if model is not None and tokenizer is not None:
            import json
            from ..utils.llm_utils import batch_generate
            
            goal_json_str = json.dumps(goal_json, indent=2)
            prompt = (
                "Convert the following JSON goal into a natural, conversational goal statement using 'you' (second person). "
                "The JSON describes what the user wants. Write it as if speaking directly to the user about their goal. "
                "Keep it concise and natural (1-2 sentences max).\n\n"
                f"JSON: {goal_json_str}\n\n"
                "Example format: 'You want to book a train and need to know the arrival time.' "
                "Do not use 'the user' or 'user wants' - use 'you' directly.\n\n"
                "OUTPUT FORMAT:\n"
                "Structure your response between [GOAL] and [/GOAL] tags:\n\n"
                "1. [GOAL]\n"
                "2. Your natural language goal statement\n"
                "3. [/GOAL] \n\n"
                "Example of correct format:\n"
                "[GOAL]\n"
                "You want to book a train and need to know the arrival time.\n"
                "[/GOAL]"
            )
            
            try:
                responses = batch_generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=[prompt],
                    max_new_tokens=100,
                    temperature=0.0,  # Deterministic for goal conversion
                    do_sample=False,
                    prefill_suffix="[GOAL]"
                )
                if responses and responses[0]:
                    raw_response = responses[0].strip()
                    # Extract content from [GOAL] tags
                    parsed_goal = self._parse_goal_from_tags(raw_response)
                    
                    # Validate the result isn't just empty or placeholder text
                    if parsed_goal and len(parsed_goal) > 10:
                        return parsed_goal
                    # If parsing failed or result is too short, fall back to heuristic
            except Exception as e:
                # Fall back to heuristic if LLM conversion fails
                pass
        
        # Fallback: Use heuristic formatting, but change "User" to "you"
        try:
            from ..data.dialogue_formatter import format_multiwoz_goal
            goal_text = format_multiwoz_goal(goal_json)
            # Replace "User" with "you" and make it more natural
            goal_text = goal_text.replace("User wants", "You want").replace("User needs", "You need")
            goal_text = goal_text.replace("user wants", "you want").replace("user needs", "you need")
            return goal_text
        except ImportError:
            # Fallback: simple formatting
            parts = []
            for domain, domain_goal in goal_json.items():
                if isinstance(domain_goal, dict):
                    domain_parts = [f"{domain}:"]
                    if "inform_slots" in domain_goal and domain_goal["inform_slots"]:
                        slots = ", ".join(f"{k}={v}" for k, v in domain_goal["inform_slots"].items())
                        domain_parts.append(f"  Requirements: {slots}")
                    if "request_slots" in domain_goal and domain_goal["request_slots"]:
                        requests = ", ".join(k for k, v in domain_goal["request_slots"].items() if v == "?")
                        if requests:
                            domain_parts.append(f"  Need: {requests}")
                    parts.append(" ".join(domain_parts))
            goal_text = "\n".join(parts) if parts else str(goal_json)
            # Replace "User" with "you" for consistency
            goal_text = goal_text.replace("User", "You").replace("user", "you")
            return goal_text
    
    def get_belief_conversion_prompt(self, structured_belief: str) -> str:
        """
        Get formatted prompt for converting structured belief to natural language.
        
        Args:
            structured_belief: JSON string representation of structured belief
        
        Returns:
            Formatted prompt string
        """
        template = self.load_template("belief_format_conversion", "high_level")
        return self.format_prompt(template, structured_belief=structured_belief)
    
    def format_multiwoz_belief_for_conversion(self, multiwoz_belief: Dict) -> str:
        """
        Format MultiWOZ belief state as JSON string for conversion.
        
        Args:
            multiwoz_belief: MultiWOZ belief state dict (domain -> slots)
                e.g., {"restaurant": {"area": "centre", "food": "Italian", "pricerange": "moderate"}}
        
        Returns:
            JSON string representation
        """
        return json.dumps(multiwoz_belief, indent=2)
    
    def format_belief_state_for_conversion(self, belief_state) -> str:
        """
        Format our BeliefState as JSON string for conversion.
        
        Args:
            belief_state: BeliefState object with candidates
        
        Returns:
            JSON string representation with top candidates and probabilities
        """
        if not belief_state.candidates:
            return json.dumps({"candidates": []}, indent=2)
        
        # Sort by probability and format top candidates
        sorted_candidates = sorted(
            belief_state.candidates,
            key=lambda c: c.probability,
            reverse=True
        )
        
        candidates_data = []
        for candidate in sorted_candidates:
            candidates_data.append({
                "summary": candidate.summary,
                "probability": candidate.probability,
                "log_prob": candidate.log_prob
            })
        
        return json.dumps({
            "turn": belief_state.turn,
            "candidates": candidates_data
        }, indent=2)
