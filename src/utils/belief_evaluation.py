"""Utilities for evaluating beliefs against ground truth goals."""

import json
import re
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple

from ..utils.llm_utils import batch_generate, batch_generate_gpt


def _parse_translation_from_tags(raw_response: str) -> str:
    """
    Parse translation from [TRANSLATION] tags.
    
    Extracts content between [TRANSLATION]...[/TRANSLATION] tags and removes any remaining tags
    to ensure only clean text is returned.
    
    When using prefill_suffix="[TRANSLATION]", the generated text starts with [TRANSLATION],
    so we want to find the LAST occurrence to avoid matching prompt examples.
    
    Args:
        raw_response: Raw response from LLM (should only contain generated text, not prompt)
        
    Returns:
        Parsed translation (content between tags with all tags removed, or original if no tags)
    """
    # Look for ALL [TRANSLATION]...[/TRANSLATION] tag pairs
    pattern = r'\[TRANSLATION\](.*?)\[/TRANSLATION\]'
    matches = list(re.finditer(pattern, raw_response, re.DOTALL | re.IGNORECASE))
    
    if matches:
        # Use the LAST match to avoid matching prompt examples
        # (the last one should be the actual generated translation)
        last_match = matches[-1]
        parsed = last_match.group(1).strip()
        
        # Remove any remaining tags that might have leaked in (safeguard)
        parsed = re.sub(r'\[/?[^\]]+\]', '', parsed)
        
        return parsed.strip()
    
    # If no tags found, still remove any tags as a safeguard
    cleaned = re.sub(r'\[/?[^\]]+\]', '', raw_response)
    return cleaned.strip()


def convert_goal_json_to_natural_language(
    goal_json: Dict,
    hl_agent,
    max_new_tokens: int = 200,
    model=None
) -> str:
    """
    Convert MultiWOZ goal JSON to natural language using HL agent.
    Uses tag-and-extract method to get clean output.
    
    Args:
        goal_json: MultiWOZ goal dictionary
        hl_agent: High-level agent for generation
        max_new_tokens: Maximum tokens to generate
        model: Optional model to use (if hl_agent.model is None, e.g., for GPT agents)
    
    Returns:
        Natural language description of the goal, or a placeholder if conversion fails.
    """
    # Check if goal_json is empty or invalid
    if not goal_json or (isinstance(goal_json, dict) and not goal_json):
        return "[NO_GOAL_SPECIFIED]"
    
    goal_json_str = json.dumps(goal_json, indent=2)
    
    # Validate that the JSON string isn't just empty braces
    if goal_json_str.strip() in ["{}", "null", ""]:
        return "[NO_GOAL_SPECIFIED]"
    
    prompt = (
        "Your job is to convert the following JSON into natural language text. "
        "The JSON describes the desires of a user. "
        "Your output should be a natural-language description of the user's desire, i.e. 'The user wants...'. "
        f"Here is the JSON: {goal_json_str}"
    )
    
    # Check if hl_agent is a GPT agent (has gpt_model_name attribute)
    if hasattr(hl_agent, 'gpt_model_name') and hl_agent.model is None:
        # Use GPT API
        responses = batch_generate_gpt(
            prompts=[prompt],
            model_name=hl_agent.gpt_model_name,
            api_key=getattr(hl_agent, 'gpt_api_key', None),
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            prefill_suffix="[TRANSLATION]"
        )
    elif model is not None:
        # Use provided model (e.g., main model for value function)
        responses = batch_generate(
            model=model,
            tokenizer=hl_agent.tokenizer,
            prompts=[prompt],
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            prefill_suffix="[TRANSLATION]"
        )
    else:
        # Use hl_agent's model (regular agent)
        responses = batch_generate(
            model=hl_agent.model,
            tokenizer=hl_agent.tokenizer,
            prompts=[prompt],
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            prefill_suffix="[TRANSLATION]"
        )
    
    raw_result = responses[0] if responses else ""
    
    # Extract content from [TRANSLATION] tags
    result = _parse_translation_from_tags(raw_result)
    
    # Validate the result isn't just empty or placeholder text
    if not result or result.strip() == "" or result.strip().startswith("{}"):
        return "[CONVERSION_FAILED]"
    
    return result


def get_text_embeddings(
    texts: List[str],
    model,
    tokenizer,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Get embeddings for texts using the frozen base LLM.
    
    Only encodes the text itself (no prompt contamination).
    
    Args:
        texts: List of text strings to embed
        model: Frozen base LLM model
        tokenizer: Tokenizer
        device: Device to run on
    
    Returns:
        Tensor of shape [batch_size, hidden_size] with mean-pooled embeddings
    """
    if not texts:
        return torch.empty(0, device=device)
    
    # Tokenize texts (just the texts, no prompts)
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)
    
    # Get hidden states from base model (no gradients)
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)
        # Use last hidden state, mean pooling
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]
            pooled = hidden_states.mean(dim=1)  # [batch, hidden_size]
        else:
            # Fallback: use last_hidden_state or embeddings
            if hasattr(outputs, 'last_hidden_state'):
                pooled = outputs.last_hidden_state.mean(dim=1)
            else:
                # Use embeddings
                pooled = model.get_input_embeddings()(encoded['input_ids']).mean(dim=1)
    
    return pooled.detach()


def compute_cosine_similarity(
    embeddings1: torch.Tensor,
    embeddings2: torch.Tensor
) -> torch.Tensor:
    """
    Compute cosine similarity between two sets of embeddings.
    
    Args:
        embeddings1: Tensor of shape [batch_size, hidden_size]
        embeddings2: Tensor of shape [batch_size, hidden_size] or [1, hidden_size]
    
    Returns:
        Tensor of shape [batch_size] with cosine similarities
    """
    # Normalize embeddings
    emb1_norm = F.normalize(embeddings1, p=2, dim=1)
    emb2_norm = F.normalize(embeddings2, p=2, dim=1)
    
    # Compute cosine similarity
    if embeddings2.shape[0] == 1:
        # Broadcast single embedding to all
        similarity = (emb1_norm * emb2_norm).sum(dim=1)
    else:
        # Pairwise similarity
        similarity = (emb1_norm * emb2_norm).sum(dim=1)
    
    return similarity


def compute_l2_distance(
    embeddings1: torch.Tensor,
    embeddings2: torch.Tensor
) -> torch.Tensor:
    """
    Compute L2 (Euclidean) distance between two sets of embeddings.
    
    Args:
        embeddings1: Tensor of shape [batch_size, hidden_size]
        embeddings2: Tensor of shape [batch_size, hidden_size] or [1, hidden_size]
    
    Returns:
        Tensor of shape [batch_size] with L2 distances
    """
    if embeddings2.shape[0] == 1:
        # Broadcast single embedding to all
        distances = torch.norm(embeddings1 - embeddings2, p=2, dim=1)
    else:
        # Pairwise distance
        distances = torch.norm(embeddings1 - embeddings2, p=2, dim=1)
    
    return distances


def evaluate_beliefs_against_ground_truth(
    chosen_beliefs: List[str],
    ground_truth_text: str,
    model,
    tokenizer,
    device: str = "cuda"
) -> List[Dict[str, float]]:
    """
    Evaluate chosen beliefs against ground truth goal.
    
    Args:
        chosen_beliefs: List of chosen belief texts (one per turn)
        ground_truth_text: Natural language description of ground truth goal
        model: Frozen base LLM model
        tokenizer: Tokenizer
        device: Device to run on
    
    Returns:
        List of dictionaries with metrics for each turn:
        [{"cosine_similarity": float, "l2_distance": float}, ...]
    """
    if not chosen_beliefs:
        return []
    
    # Get embeddings for chosen beliefs (batched)
    belief_embeddings = get_text_embeddings(chosen_beliefs, model, tokenizer, device)
    
    # Get embedding for ground truth (single)
    gt_embeddings = get_text_embeddings([ground_truth_text], model, tokenizer, device)
    
    # Compute metrics
    cosine_sims = compute_cosine_similarity(belief_embeddings, gt_embeddings)
    l2_dists = compute_l2_distance(belief_embeddings, gt_embeddings)
    
    # Convert to list of dicts
    metrics = []
    for i in range(len(chosen_beliefs)):
        metrics.append({
            "cosine_similarity": cosine_sims[i].item(),
            "l2_distance": l2_dists[i].item()
        })
    
    return metrics

