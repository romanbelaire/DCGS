"""Utilities for converting structured beliefs to natural language using LLM."""

import json
from typing import Dict

from ..prompts.prompt_manager import PromptManager
from ..utils.llm_utils import batch_generate
from ..utils.belief_evaluation import _parse_translation_from_tags


def convert_multiwoz_belief_to_text(
    multiwoz_belief: Dict,
    model,
    tokenizer,
    prompt_manager: PromptManager,
    temperature: float = 0.0
) -> str:
    """
    Convert MultiWOZ structured belief to natural language text using LLM.
    
    Args:
        multiwoz_belief: MultiWOZ belief state dict
            e.g., {"restaurant": {"area": "centre", "food": "Italian", "pricerange": "moderate"}}
        model: Language model (high-level LLM)
        tokenizer: Tokenizer
        prompt_manager: PromptManager instance
        temperature: Sampling temperature (0.0 for deterministic)
    
    Returns:
        Natural language belief summary
    """
    # Format belief as JSON string
    structured_belief = prompt_manager.format_multiwoz_belief_for_conversion(multiwoz_belief)
    
    # Get conversion prompt
    prompt = prompt_manager.get_belief_conversion_prompt(structured_belief)
    
    # Generate text using LLM
    responses = batch_generate(
        model=model,
        tokenizer=tokenizer,
        prompts=[prompt],
        max_new_tokens=256,
        temperature=temperature,
        do_sample=temperature > 0.0,
        prefill_suffix="[TRANSLATION]"
    )
    
    raw_result = responses[0] if responses else ""
    
    # Extract content from [TRANSLATION] tags
    result = _parse_translation_from_tags(raw_result)
    
    return result


def convert_belief_state_to_text(
    belief_state,
    model,
    tokenizer,
    prompt_manager: PromptManager,
    temperature: float = 0.0,
    use_top_k: int = 1
) -> str:
    """
    Convert our BeliefState to natural language text using LLM.
    
    Args:
        belief_state: BeliefState object
        model: Language model (high-level LLM)
        tokenizer: Tokenizer
        prompt_manager: PromptManager instance
        temperature: Sampling temperature (0.0 for deterministic)
        use_top_k: Number of top candidates to include in conversion (default: 1)
    
    Returns:
        Natural language belief summary
    """
    # Format belief state as JSON string (with top candidates)
    structured_belief = prompt_manager.format_belief_state_for_conversion(belief_state)
    
    # Get conversion prompt
    prompt = prompt_manager.get_belief_conversion_prompt(structured_belief)
    
    # Generate text using LLM
    responses = batch_generate(
        model=model,
        tokenizer=tokenizer,
        prompts=[prompt],
        max_new_tokens=256,
        temperature=temperature,
        do_sample=temperature > 0.0,
        prefill_suffix="[TRANSLATION]"
    )
    
    raw_result = responses[0] if responses else ""
    
    # Extract content from [TRANSLATION] tags
    result = _parse_translation_from_tags(raw_result)
    
    return result


def convert_structured_belief_to_text(
    structured_belief: Dict,
    model,
    tokenizer,
    prompt_manager: PromptManager,
    temperature: float = 0.0
) -> str:
    """
    Generic function to convert any structured belief format to natural language.
    
    Automatically detects MultiWOZ format vs. our BeliefState format.
    
    Args:
        structured_belief: Structured belief dict (MultiWOZ format or custom format)
        model: Language model (high-level LLM)
        tokenizer: Tokenizer
        prompt_manager: PromptManager instance
        temperature: Sampling temperature (0.0 for deterministic)
    
    Returns:
        Natural language belief summary
    """
    # Check if it's MultiWOZ format (domain-based structure)
    if isinstance(structured_belief, dict):
        # Check for MultiWOZ format: top-level keys are domains (restaurant, hotel, etc.)
        multiwoz_domains = ["restaurant", "hotel", "train", "taxi", "attraction", "police", "hospital"]
        if any(domain in structured_belief for domain in multiwoz_domains):
            return convert_multiwoz_belief_to_text(
                structured_belief, model, tokenizer, prompt_manager, temperature
            )
    
    # Otherwise, treat as generic structured belief
    structured_belief_str = json.dumps(structured_belief, indent=2)
    prompt = prompt_manager.get_belief_conversion_prompt(structured_belief_str)
    
    responses = batch_generate(
        model=model,
        tokenizer=tokenizer,
        prompts=[prompt],
        max_new_tokens=256,
        temperature=temperature,
        do_sample=temperature > 0.0,
        prefill_suffix="[TRANSLATION]"
    )
    
    raw_result = responses[0] if responses else ""
    
    # Extract content from [TRANSLATION] tags
    result = _parse_translation_from_tags(raw_result)
    
    return result

