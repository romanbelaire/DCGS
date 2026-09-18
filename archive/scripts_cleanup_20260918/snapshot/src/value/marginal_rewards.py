"""Marginal token contribution reward computation."""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy
from ..utils.llm_utils import batch_generate


def select_top_k_tokens(
    sequence_critic,
    tokenizer,
    action: str,
    top_k: int = 5,
    device: str = "cuda"
) -> List[int]:
    """
    Select top-k tokens using attention-guided selection.
    
    Args:
        sequence_critic: Model to extract hidden states from (sequence-level critic)
        tokenizer: Tokenizer for encoding action
        action: Agent action text
        top_k: Number of top tokens to select
        device: Device for computation
    
    Returns:
        List of token positions (indices) selected
    """
    # Tokenize action
    tokens = tokenizer.encode(action, return_tensors="pt").to(device)
    
    # Forward pass through sequence critic to get hidden states
    with torch.no_grad():
        outputs = sequence_critic(tokens)
        # Get last hidden states (shape: [batch_size, seq_len, hidden_size])
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            hidden_states = outputs.hidden_states[-1]
        elif hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        else:
            # Fallback: use logits to compute importance
            logits = outputs.logits
            hidden_states = logits
    
    # Compute token importance as L2 norm of hidden states
    # Shape: [seq_len]
    token_importance = torch.norm(hidden_states[0], dim=1)
    
    # Select top-k most important tokens
    _, top_indices = torch.topk(token_importance, min(top_k, len(token_importance)), dim=0)
    
    # Convert to list of integers (excluding special tokens at start/end)
    selected_positions = top_indices.cpu().tolist()
    
    # Filter out special tokens (typically at position 0 and -1)
    # Keep only positions that are valid content tokens
    valid_positions = [pos for pos in selected_positions if 0 < pos < len(token_importance) - 1]
    
    return valid_positions[:top_k]


def compute_reward_with_reward_model(
    reward_model,
    reward_tokenizer,
    text: str,
    reward_model_type: str = "llamaguard",
    device: str = "cuda"
) -> float:
    """
    Compute reward using a reward model (LlamaGuard or Skywork).
    
    Args:
        reward_model: Reward model instance
        reward_tokenizer: Tokenizer for reward model
        text: Text to evaluate
        reward_model_type: Type of reward model ("llamaguard", "shieldgemma", or "skywork")
        device: Device for computation
    
    Returns:
        Reward score (float)
    """
    with torch.no_grad():
        if reward_model_type == "llamaguard":
            # LlamaGuard is a causal LM; expects [user, assistant] chat format
            chat = [
                {"role": "user", "content": "User request."},
                {"role": "assistant", "content": text},
            ]
            batch_inputs = reward_tokenizer.apply_chat_template(
                chat,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
            ).to(device)
            outputs = reward_model(**batch_inputs)
            next_token_logits = outputs.logits[:, -1, :]  # (batch, vocab_size)
            unsafe_ids = reward_tokenizer.encode("unsafe", add_special_tokens=False)
            safe_ids = reward_tokenizer.encode("safe", add_special_tokens=False)
            if not unsafe_ids or not safe_ids:
                probs = F.softmax(next_token_logits[0], dim=-1)
                return probs[-1].item()
            unsafe_id, safe_id = unsafe_ids[0], safe_ids[0]
            unsafe_logit = next_token_logits[0, unsafe_id]
            safe_logit = next_token_logits[0, safe_id]
            probs = F.softmax(torch.stack([unsafe_logit, safe_logit], dim=0), dim=0)
            return probs[1].item()
        elif reward_model_type == "shieldgemma":
            from .cares_reward import SHIELDGEMMA_GUIDELINE

            text_trunc = text[-2000:] if len(text) > 2000 else text
            chats = [
                [
                    {"role": "user", "content": "[Text to classify]"},
                    {"role": "assistant", "content": text_trunc},
                ]
            ]
            batch_inputs = reward_tokenizer.apply_chat_template(
                chats,
                guideline=SHIELDGEMMA_GUIDELINE,
                return_tensors="pt",
                return_dict=True,
                padding=True,
                truncation=True,
                max_length=1024,
            )
            batch_inputs = batch_inputs.to(device)
            outputs = reward_model(**batch_inputs)
            last_logits = outputs.logits[:, -1, :]
            vocab = reward_tokenizer.get_vocab()
            yes_id = vocab["Yes"]
            no_id = vocab["No"]
            selected = last_logits[:, [yes_id, no_id]]
            probs = F.softmax(selected, dim=-1)
            p_no = probs[0, 1].item()
            return p_no
        elif reward_model_type == "skywork":
            inputs = reward_tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
            outputs = reward_model(**inputs)
            if hasattr(outputs, 'reward'):
                return outputs.reward.item()
            elif hasattr(outputs, 'logits'):
                return outputs.logits.mean().item()
            else:
                return outputs.last_hidden_state.mean().item()
        else:
            raise ValueError(f"Unknown reward model type: {reward_model_type}")


def compute_instruction_fulfillment_score(
    action: str,
    instruction: str,
    model,
    tokenizer,
    device: str = "cuda"
) -> float:
    """
    Judge whether low-level response fulfills high-level instruction.
    
    Args:
        action: Agent action text
        instruction: Instruction/context text
        model: Model to extract embeddings from
        tokenizer: Tokenizer
        device: Device for computation
    
    Returns:
        1.0 when instruction is fulfilled, 0.0 otherwise
    """
    prompt = (
        "You are a strict instruction-following judge.\n"
        "Determine if the assistant response fulfills the instruction.\n\n"
        f"Instruction:\n{instruction[:1200]}\n\n"
        f"Assistant response:\n{action[:1200]}\n\n"
        "OUTPUT FORMAT:\n"
        "[JUDGE]\n"
        "YES or NO\n"
        "[/JUDGE]"
    )
    max_attempts = 3
    last_raw = ""
    for attempt_idx in range(max_attempts):
        attempt_prompt = prompt
        if attempt_idx > 0:
            attempt_prompt = (
                prompt
                + "\n\nRETRY REQUIREMENT:\n"
                + "Output exactly one label after [JUDGE]: YES or NO.\n"
                + "Do not include explanations."
            )
        responses = batch_generate(
            model=model,
            tokenizer=tokenizer,
            prompts=[attempt_prompt],
            max_new_tokens=20,
            temperature=0.0,
            do_sample=False,
            prefill_suffix="[JUDGE]",
            chunk_size=1,
        )
        raw = responses[0]
        last_raw = raw
        lower = raw.lower()
        if "[/judge]" in lower:
            parsed = lower.split("[/judge]", 1)[0]
        else:
            parsed = lower
        parsed = parsed.replace("[judge]", " ").strip()
        match = re.search(r"\b(yes|no)\b", parsed)
        if match:
            return 1.0 if match.group(1) == "yes" else 0.0
    raise ValueError(
        "Instruction fulfillment judge produced invalid output after retries. "
        f"Expected YES/NO, got: {last_raw!r}"
    )


def create_masked_attention_mask(
    token_ids: torch.Tensor,
    mask_positions: List[int],
    device: str = "cuda"
) -> torch.Tensor:
    """
    Create attention mask that masks out specified token positions.
    
    Args:
        token_ids: Token IDs tensor [batch_size, seq_len]
        mask_positions: List of token positions to mask
        device: Device for computation
    
    Returns:
        Attention mask tensor [batch_size, seq_len]
    """
    batch_size, seq_len = token_ids.shape
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    
    # Set masked positions to 0
    for pos in mask_positions:
        if 0 <= pos < seq_len:
            attention_mask[:, pos] = 0
    
    return attention_mask


def compute_reward_with_environment(
    env,
    action: str,
    current_observation: str,
    dialogue_history: List[Tuple[str, str]],
    env_state_snapshot: Optional[Dict] = None,
    actual_reward: Optional[float] = None
) -> float:
    """
    Compute reward using environment's compute_reward_for_marginal (polymorphic).

    Args:
        env: Environment instance
        action: Agent action to evaluate
        current_observation: Current observation/state
        dialogue_history: Current dialogue history
        env_state_snapshot: Optional snapshot of environment state for restoration
        actual_reward: Actual reward from the transition (for unmasked action)

    Returns:
        Reward value using environment's actual computation methods
    """
    return env.compute_reward_for_marginal(action, actual_reward)


def compute_sequence_level_reward(
    state: str,
    action: str,
    tokenizer,
    sequence_critic,
    reward_model=None,
    reward_tokenizer=None,
    reward_model_type: str = "llamaguard",
    environment=None,
    current_observation: Optional[str] = None,
    dialogue_history: Optional[List[Tuple[str, str]]] = None,
    actual_reward: Optional[float] = None,
    device: str = "cuda",
) -> float:
    """
    Sequence-level reward R for an action (same priority as unmasked marginal base reward).
    Priority: environment > reward_model > sequence_critic logits mean.
    """
    action_tokens = tokenizer.encode(action, return_tensors="pt").to(device)
    if environment is not None:
        return compute_reward_with_environment(
            env=environment,
            action=action,
            current_observation=current_observation or state,
            dialogue_history=dialogue_history or [],
            actual_reward=actual_reward,
        )
    if reward_model is not None and reward_tokenizer is not None:
        return compute_reward_with_reward_model(
            reward_model=reward_model,
            reward_tokenizer=reward_tokenizer,
            text=action,
            reward_model_type=reward_model_type,
            device=device,
        )
    with torch.no_grad():
        outputs = sequence_critic(action_tokens)
        if hasattr(outputs, "logits"):
            return outputs.logits.mean().item()
        return 0.0


def compute_alternative_token_rewards(
    schema: str,
    action: str,
    tokenizer,
    sequence_reward: float,
    decay_gamma: float,
    use_task_reward: bool,
    instruction: Optional[str],
    sequence_critic,
    device: str,
) -> Dict[int, float]:
    """
    Uniform or decayed token targets from sequence reward R over interior token indices
    (same convention as marginal: positions i with 0 < i < L - 1).
    """
    if schema not in ("uniform", "decayed"):
        raise ValueError(
            f"compute_alternative_token_rewards expects schema 'uniform' or 'decayed', got {schema!r}"
        )
    action_tokens = tokenizer.encode(action, return_tensors="pt")
    action_length = action_tokens.shape[1]
    if action_length < 3:
        return {}
    interior = list(range(1, action_length - 1))
    if not interior:
        return {}
    rewards: Dict[int, float] = {}
    if schema == "uniform":
        for pos in interior:
            rewards[pos] = sequence_reward
    else:
        last = action_length - 2
        for pos in interior:
            rewards[pos] = sequence_reward * (decay_gamma ** (last - pos))
    if use_task_reward and instruction is not None and sequence_critic is not None:
        fulfillment_score = compute_instruction_fulfillment_score(
            action=action,
            instruction=instruction,
            model=sequence_critic,
            tokenizer=tokenizer,
            device=device,
        )
        n = len(rewards)
        if n > 0:
            per_token = fulfillment_score / n
            for pos in rewards:
                rewards[pos] += per_token
    return rewards


def compute_marginal_rewards(
    state: str,
    action: str,
    base_state: str,
    tokenizer,
    sequence_critic,
    reward_model=None,
    reward_tokenizer=None,
    reward_model_type: str = "llamaguard",
    environment=None,
    current_observation: Optional[str] = None,
    dialogue_history: Optional[List[Tuple[str, str]]] = None,
    actual_reward: Optional[float] = None,
    top_k_tokens: int = 5,
    pairwise_masking: bool = False,
    use_task_reward: bool = True,
    instruction: Optional[str] = None,
    device: str = "cuda",
    chunk_size: int = 4
) -> Dict[int, float]:
    """
    Compute marginal token contribution rewards.
    
    Args:
        state: Current state/observation
        action: Agent action text
        base_state: Base state for comparison
        tokenizer: Tokenizer for action encoding
        sequence_critic: Sequence-level critic model for token selection
        reward_model: Optional reward model (LlamaGuard or Skywork) - only used if environment is None
        reward_tokenizer: Tokenizer for reward model
        reward_model_type: Type of reward model ("llamaguard" or "skywork") - only used if environment is None
        environment: Optional environment instance to use for reward computation (takes precedence over reward_model)
        current_observation: Current observation for environment reward computation
        dialogue_history: Current dialogue history for environment reward computation
        actual_reward: Actual reward from transition (for unmasked action) - used when environment is provided
        top_k_tokens: Number of top tokens to process
        pairwise_masking: Whether to use pairwise masking
        use_task_reward: Whether to include instruction-following reward
        instruction: Optional instruction text for task reward
        device: Device for computation
        chunk_size: Batch size for processing masked versions
    
    Returns:
        Dictionary mapping token positions to marginal rewards
    """
    # Initialize marginal rewards dictionary
    marginal_rewards: Dict[int, float] = {}
    
    # Select top-k tokens using attention-guided selection
    selected_tokens = select_top_k_tokens(
        sequence_critic=sequence_critic,
        tokenizer=tokenizer,
        action=action,
        top_k=top_k_tokens,
        device=device
    )
    
    if not selected_tokens:
        return marginal_rewards
    
    # Tokenize action
    action_tokens = tokenizer.encode(action, return_tensors="pt").to(device)
    action_length = action_tokens.shape[1]

    base_reward = compute_sequence_level_reward(
        state=state,
        action=action,
        tokenizer=tokenizer,
        sequence_critic=sequence_critic,
        reward_model=reward_model,
        reward_tokenizer=reward_tokenizer,
        reward_model_type=reward_model_type,
        environment=environment,
        current_observation=current_observation,
        dialogue_history=dialogue_history,
        actual_reward=actual_reward,
        device=device,
    )

    # Process tokens in chunks to avoid OOM
    if pairwise_masking:
        # Generate all pairs of selected tokens
        token_pairs = []
        for i in range(len(selected_tokens)):
            for j in range(i + 1, len(selected_tokens)):
                token_pairs.append((selected_tokens[i], selected_tokens[j]))
        
        # Process pairs in chunks
        for chunk_start in range(0, len(token_pairs), chunk_size):
            chunk_pairs = token_pairs[chunk_start:chunk_start + chunk_size]
            
            masked_rewards = []
            for token_i, token_j in chunk_pairs:
                # Create masked version with both tokens masked
                mask_positions = [token_i, token_j]
                attention_mask = create_masked_attention_mask(
                    action_tokens, mask_positions, device=device
                )
                
                # Compute reward for masked version
                # Reconstruct text without masked tokens
                token_list = action_tokens[0].cpu().tolist()
                masked_token_list = [t for idx, t in enumerate(token_list) if idx not in mask_positions]
                masked_text = tokenizer.decode(masked_token_list, skip_special_tokens=True)
                
                if environment is not None:
                    masked_reward = compute_reward_with_environment(
                        env=environment,
                        action=masked_text,
                        current_observation=current_observation or state,
                        dialogue_history=dialogue_history or []
                    )
                elif reward_model is not None and reward_tokenizer is not None:
                    masked_reward = compute_reward_with_reward_model(
                        reward_model=reward_model,
                        reward_tokenizer=reward_tokenizer,
                        text=masked_text,
                        reward_model_type=reward_model_type,
                        device=device
                    )
                else:
                    with torch.no_grad():
                        outputs = sequence_critic(action_tokens, attention_mask=attention_mask)
                        if hasattr(outputs, 'logits'):
                            masked_reward = outputs.logits.mean().item()
                        else:
                            masked_reward = base_reward
                
                masked_rewards.append(masked_reward)
                
                # Compute marginal contribution (split equally between two tokens)
                marginal_contribution = (base_reward - masked_reward) / 2.0
                
                # Update marginal rewards for both tokens
                if token_i not in marginal_rewards:
                    marginal_rewards[token_i] = 0.0
                if token_j not in marginal_rewards:
                    marginal_rewards[token_j] = 0.0
                
                marginal_rewards[token_i] += marginal_contribution
                marginal_rewards[token_j] += marginal_contribution
            
            # Cleanup
            del masked_rewards
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        # Single-token masking
        for chunk_start in range(0, len(selected_tokens), chunk_size):
            chunk_tokens = selected_tokens[chunk_start:chunk_start + chunk_size]
            
            for token_pos in chunk_tokens:
                # Create masked version
                mask_positions = [token_pos]
                attention_mask = create_masked_attention_mask(
                    action_tokens, mask_positions, device=device
                )
                
                # Compute reward for masked version
                # Reconstruct text without masked token
                token_list = action_tokens[0].cpu().tolist()
                masked_token_list = [t for idx, t in enumerate(token_list) if idx != token_pos]
                masked_text = tokenizer.decode(masked_token_list, skip_special_tokens=True)
                
                if environment is not None:
                    masked_reward = compute_reward_with_environment(
                        env=environment,
                        action=masked_text,
                        current_observation=current_observation or state,
                        dialogue_history=dialogue_history or []
                    )
                elif reward_model is not None and reward_tokenizer is not None:
                    masked_reward = compute_reward_with_reward_model(
                        reward_model=reward_model,
                        reward_tokenizer=reward_tokenizer,
                        text=masked_text,
                        reward_model_type=reward_model_type,
                        device=device
                    )
                else:
                    with torch.no_grad():
                        outputs = sequence_critic(action_tokens, attention_mask=attention_mask)
                        if hasattr(outputs, 'logits'):
                            masked_reward = outputs.logits.mean().item()
                        else:
                            masked_reward = base_reward
                
                # Compute marginal contribution
                marginal_contribution = base_reward - masked_reward
                marginal_rewards[token_pos] = marginal_contribution
            
            # Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Add instruction-following reward if enabled.
    # In HRL mode this is the low-level helpfulness signal:
    # does the response fulfill the selected high-level instruction?
    if use_task_reward and instruction is not None and sequence_critic is not None:
        fulfillment_score = compute_instruction_fulfillment_score(
            action=action,
            instruction=instruction,
            model=sequence_critic,
            tokenizer=tokenizer,
            device=device
        )
        
        # Distribute instruction-fulfillment reward equally across all valid tokens.
        num_valid_tokens = len([pos for pos in marginal_rewards.keys() if 0 < pos < action_length - 1])
        if num_valid_tokens > 0:
            per_token_reward = fulfillment_score / num_valid_tokens
            for token_pos in marginal_rewards.keys():
                marginal_rewards[token_pos] += per_token_reward
    
    return marginal_rewards


def train_token_critic_with_marginal_rewards(
    token_critic,
    token_critic_values: torch.Tensor,
    marginal_rewards: Dict[int, float],
    token_positions: List[int],
    device: str = "cuda"
) -> torch.Tensor:
    """
    Compute loss for token-level critic using marginal rewards as targets.
    
    Args:
        token_critic: Token-level critic model (optional, for future use)
        token_critic_values: Predicted per-token values [batch_size, seq_len] or [seq_len]
        marginal_rewards: Dictionary mapping token positions to marginal rewards
        token_positions: List of token positions that have marginal rewards
        device: Device for computation
    
    Returns:
        MSE loss tensor
    """
    # Convert marginal rewards to tensor
    targets = []
    values = []
    
    for pos in token_positions:
        if pos in marginal_rewards:
            targets.append(marginal_rewards[pos])
            # Extract corresponding predicted value
            if len(token_critic_values.shape) == 1:
                # [seq_len]
                if pos < token_critic_values.shape[0]:
                    values.append(token_critic_values[pos].item())
                else:
                    values.append(0.0)
            else:
                # [batch_size, seq_len] - assume batch_size=1
                if pos < token_critic_values.shape[1]:
                    values.append(token_critic_values[0, pos].item())
                else:
                    values.append(0.0)
    
    if not targets:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Convert to tensors
    targets_tensor = torch.tensor(targets, device=device, dtype=torch.float32)
    values_tensor = torch.tensor(values, device=device, dtype=torch.float32)
    
    # Compute MSE loss
    loss = F.mse_loss(values_tensor, targets_tensor)
    
    return loss

