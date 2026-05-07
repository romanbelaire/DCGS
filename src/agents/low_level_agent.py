"""Low-level agent that generates actions based on belief context and dialogue history."""

import re
from typing import Dict, List, Optional, Tuple

from .base_agent import BaseAgent
from ..utils.llm_utils import batch_generate, SuppressWordsLogitsProcessor


class LowLevelAgent(BaseAgent):
    """Produces actions/responses given belief context and dialogue context."""

    def __init__(
        self,
        model,
        tokenizer,
        prompt_manager=None,
        enable_thinking: Optional[bool] = None,
    ):
        super().__init__(model, tokenizer, prompt_manager)
        self.enable_thinking = enable_thinking

    def build_prompt(
        self,
        belief_context: str,
        history: List[Tuple[str, str]] = None,
        belief_only: bool = True,
        template_name: str = "action_generation"
    ) -> str:
        """
        Construct low-level generation prompt.
        
        Args:
            belief_context: The belief/understanding of user intent
            history: Dialogue history (only used if belief_only=False)
            belief_only: If True, generate prompt with only belief (P(a|b)). If False, include history (P(a|o,b))
            template_name: Name of the template to use (default: "action_generation", use "action_generation_userbench" for UserBench)
        """
        return self.prompt_manager.get_low_level_prompt(
            template_name=template_name,
            belief_context=belief_context,
            history=history,
            belief_only=belief_only
        )

    def generate_actions_from_prompts(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        do_sample: bool = False,
        chunk_size: int = None,
        template_name: str = "action_generation"
    ) -> List[str]:
        """
        Generate actions for a batch of prompts.
        
        Args:
            prompts: List of prompt strings
            temperature: Generation temperature
            do_sample: Whether to use sampling
            chunk_size: Batch size for generation
            template_name: Name of the template being used (to determine prefill_suffix)
        """
        if not prompts:
            return []
        
        # For UserBench template, don't use prefill_suffix (model should generate [action]/[search]/[answer] directly)
        # For other templates, use [RESPONSE] prefill
        prefill_suffix = None if template_name == "action_generation_userbench" else "[RESPONSE]"
        
        # Create logits processor to suppress "Example", "example", "Examples", "examples"
        logits_processor = SuppressWordsLogitsProcessor(
            tokenizer=self.tokenizer,
            words_to_suppress=["Example", "example", "Examples", "examples"]
        )
        
        responses = batch_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=prompts,
            max_new_tokens=128,  # Reduced to prevent long responses with code/examples
            temperature=temperature,
            do_sample=do_sample,
            prefill_suffix=prefill_suffix,
            chunk_size=chunk_size,
            logits_processor=logits_processor,
            enable_thinking=self.enable_thinking,
        )
        # Parse responses using tag-based extraction (same as user agent)
        cleaned_responses = []
        is_userbench = (template_name == "action_generation_userbench")
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
    
    def _parse_response_from_tags(self, raw_response: str, is_userbench: bool = False) -> str:
        """
        Parse agent response from tags.
        
        For UserBench/TravelEnv: Extracts actions in format [action], [search], or [answer] followed by content.
        For other environments: Extracts content between [RESPONSE]...[/RESPONSE] tags.
        
        Args:
            raw_response: Raw response from LLM (should only contain generated text, not prompt)
            is_userbench: Whether this is for UserBench environment (ensures proper format)
            
        Returns:
            Parsed response (TravelEnv format preserved, or content between tags with tags removed)
        """
        import re
        
        # First check if this is TravelEnv format ([action], [search], or [answer] prefix)
        # Check if response starts with one of the TravelEnv prefixes (case-insensitive)
        stripped = raw_response.strip()
        if re.match(r'^\[(action|search|answer)\]\s*', stripped, re.IGNORECASE):
            # Return the full TravelEnv format (prefix + content), preserving original formatting
            return stripped
        
        # For UserBench, if we don't have the proper format, wrap with [action] as fallback
        if is_userbench:
            # Remove any leading/trailing whitespace and wrap with [action]
            content = stripped
            # Remove any existing malformed tags
            content = re.sub(r'^\[/?[^\]]+\]\s*', '', content, flags=re.IGNORECASE)
            if content.strip():
                return f"[action] {content.strip()}"
            # If content is empty, raise an error
            raise ValueError(f"Empty action after parsing: {raw_response}")
        
        # Otherwise, use the standard [RESPONSE] tag parsing
        # Look for ALL [RESPONSE]...[/RESPONSE] tag pairs
        pattern = r'\[RESPONSE\](.*?)\[/RESPONSE\]'
        matches = list(re.finditer(pattern, raw_response, re.DOTALL | re.IGNORECASE))
        
        if matches:
            # Filter out matches that only contain whitespace or are very short
            # (these are likely malformed or from metacommentary)
            valid_matches = []
            for match in matches:
                content = match.group(1).strip()
                # Remove tags from content to check actual text length
                text_only = re.sub(r'\[/?[^\]]+\]', '', content).strip()
                if len(text_only) >= 10:  # Require at least 10 characters of actual text
                    valid_matches.append((match, len(text_only)))
            
            if valid_matches:
                # Prefer the first valid match (usually the actual response)
                # If multiple valid matches, use the longest one
                best_match = max(valid_matches, key=lambda x: x[1])[0]
                parsed = best_match.group(1).strip()
            else:
                # If no valid matches, use the first match anyway (might be short but valid)
                parsed = matches[0].group(1).strip()
            
            # Remove any remaining tags that might have leaked in (safeguard)
            parsed = re.sub(r'\[/?[^\]]+\]', '', parsed)
            
            result = parsed.strip()
            # If result is still empty, try fallback strategies
            if not result:
                # Try to extract content from an opening tag even without closing tag
                opening_match = re.search(r'\[RESPONSE\](.*)', raw_response, re.DOTALL | re.IGNORECASE)
                if opening_match:
                    content = opening_match.group(1).strip()
                    # Remove all tags and get text
                    result = re.sub(r'\[/?[^\]]+\]', '', content).strip()
                    # Take first reasonable chunk (before any metacommentary)
                    if result:
                        # Stop at common metacommentary patterns
                        for stop_pattern in [r'\[/RESPONSE\]', r'is not needed', r'Here is the revised']:
                            stop_match = re.search(stop_pattern, result, re.IGNORECASE)
                            if stop_match:
                                result = result[:stop_match.start()].strip()
                                break
            
            return result if result else raw_response.strip()
        
        # If no tags found, try to extract from opening tag without closing tag
        opening_match = re.search(r'\[RESPONSE\](.*)', raw_response, re.DOTALL | re.IGNORECASE)
        if opening_match:
            content = opening_match.group(1).strip()
            # Remove all tags
            cleaned = re.sub(r'\[/?[^\]]+\]', '', content).strip()
            if cleaned:
                # For UserBench, wrap with [action] if not already formatted
                if is_userbench and not re.match(r'^\[(action|search|answer)\]\s*', cleaned, re.IGNORECASE):
                    return f"[action] {cleaned}"
                return cleaned
        
        # Final fallback: remove all tags from raw response
        cleaned = re.sub(r'\[/?[^\]]+\]', '', raw_response).strip()
        # For UserBench, ensure proper format
        if is_userbench:
            if cleaned and not re.match(r'^\[(action|search|answer)\]\s*', cleaned, re.IGNORECASE):
                return f"[action] {cleaned}"
            elif not cleaned:
                return "[action] I understand. How can I help you?"
        return cleaned

    def generate_action(
        self,
        belief_context: str,
        history: List[Tuple[str, str]] = None,
        temperature: float = 0.7,
        belief_only: bool = True,
        template_name: str = "action_generation"
    ) -> str:
        """
        Generate action/response given belief context and optionally dialogue history.
        
        Args:
            belief_context: The belief/understanding of user intent
            history: Dialogue history (only used if belief_only=False)
            temperature: Generation temperature
            belief_only: If True, generate from belief only (P(a|b)). If False, use history too (P(a|o,b))
            template_name: Name of the template to use (default: "action_generation", use "action_generation_userbench" for UserBench)
        """
        prompt = self.build_prompt(
            belief_context=belief_context,
            history=history,
            belief_only=belief_only,
            template_name=template_name
        )
        responses = self.generate_actions_from_prompts(
            prompts=[prompt],
            temperature=temperature,
            do_sample=False,
            template_name=template_name
        )
        return responses[0] if responses else ""

    def generate_action_batch(
        self,
        belief_contexts: List[str],
        histories: List[List[Tuple[str, str]]],
        temperature: float = 0.7,
        belief_only: bool = True,
        chunk_size: int = None,
        template_name: str = "action_generation"
    ) -> List[str]:
        """
        Batch generate actions for multiple (belief_context, history) pairs.
        
        Args:
            belief_contexts: List of belief contexts
            histories: List of dialogue histories
            temperature: Generation temperature
            belief_only: If True, generate from belief only (P(a|b)). If False, use history too (P(a|o,b))
            chunk_size: Batch size for generation
            template_name: Name of the template to use (default: "action_generation", use "action_generation_userbench" for UserBench)
        """
        prompts: List[str] = []
        for belief_context, history in zip(belief_contexts, histories):
            prompts.append(
                self.build_prompt(
                    belief_context=belief_context,
                    history=history,
                    belief_only=belief_only,
                    template_name=template_name
                )
            )
        return self.generate_actions_from_prompts(
            prompts=prompts,
            temperature=temperature,
            do_sample=False,
            chunk_size=chunk_size,
            template_name=template_name
        )

    def generate_ll_candidates(
        self,
        belief_context: str,
        history: List[Tuple[str, str]],
        n_candidates: int,
        template_name: str,
        temperature: float = 0.7,
        belief_only: bool = True,
        chunk_size: int = None,
    ) -> List[str]:
        """Generate n_candidates diverse LL responses in a single LLM call.

        Mirrors the HL belief generation pattern: one forward pass produces all
        candidates as a numbered list, ensuring semantic diversity.  The multi-
        candidate template name is derived by appending ``_multi`` to
        ``template_name`` (e.g. ``action_generation_cares_multi``).

        Args:
            belief_context: Selected high-level belief string.
            history: Dialogue history for the episode.
            n_candidates: Number of diverse responses to generate.
            template_name: Base single-candidate template name; ``_multi`` is
                appended to resolve the multi-candidate variant.
            temperature: Sampling temperature.
            belief_only: If True, omit dialogue history from prompt (P(a|b)).
            chunk_size: Passed to batch_generate for OOM control.

        Returns:
            List of exactly n_candidates response strings.  Falls back to the
            single-candidate path if parsing yields fewer than n_candidates.
        """
        multi_template_name = template_name + "_multi"
        prompt = self.prompt_manager.get_low_level_prompt(
            template_name=multi_template_name,
            belief_context=belief_context,
            history=history,
            belief_only=belief_only,
            n_candidates=n_candidates,
        )

        # Prefill forces the model to start the numbered list immediately.
        prefill_suffix = "1. [RESPONSE]\n"

        logits_processor = SuppressWordsLogitsProcessor(
            tokenizer=self.tokenizer,
            words_to_suppress=["Example", "example", "Examples", "examples"],
        )

        raw_output = batch_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=[prompt],
            max_new_tokens=128 * n_candidates,
            temperature=temperature,
            do_sample=True,
            prefill_suffix=prefill_suffix,
            chunk_size=chunk_size,
            logits_processor=logits_processor,
            enable_thinking=self.enable_thinking,
        )[0]

        candidates = self._parse_multi_candidate_responses(raw_output, n_candidates, template_name)

        # Pad with the single-candidate fallback if parsing fell short.
        if len(candidates) < n_candidates:
            fallback = self.generate_action(
                belief_context=belief_context,
                history=history,
                temperature=temperature,
                belief_only=belief_only,
                template_name=template_name,
            )
            while len(candidates) < n_candidates:
                candidates.append(fallback)

        return candidates[:n_candidates]

    def _parse_multi_candidate_responses(
        self, raw_output: str, n_candidates: int, template_name: str
    ) -> List[str]:
        """Parse n_candidates [RESPONSE]...[/RESPONSE] blocks from a numbered list output."""
        # Match numbered items: "N. [RESPONSE] ... [/RESPONSE]"
        pattern = r'\d+\.\s*\[RESPONSE\](.*?)\[/RESPONSE\]'
        matches = re.findall(pattern, raw_output, re.DOTALL | re.IGNORECASE)

        is_userbench = "userbench" in template_name
        candidates = []
        for raw_text in matches[:n_candidates]:
            cleaned = raw_text.strip()
            # Re-use the existing tag-stripping logic for consistency.
            cleaned = self._parse_response_from_tags(
                "[RESPONSE]" + cleaned + "[/RESPONSE]",
                is_userbench=is_userbench,
            )
            if cleaned and cleaned.strip():
                candidates.append(cleaned.strip())

        return candidates

    def act(self, belief_context: str, history: List[Tuple[str, str]] = None, belief_only: bool = True) -> str:
        """
        Interface method for acting.
        
        Args:
            belief_context: The belief/understanding of user intent
            history: Dialogue history (only used if belief_only=False)
            belief_only: If True, generate from belief only (P(a|b)). If False, use history too (P(a|o,b))
        """
        return self.generate_action(belief_context, history, belief_only=belief_only)

