"""Integration utilities for marginal token rewards in training."""

import torch
from typing import Dict, List, Optional, Tuple

from .cares_reward import normalize_reward_model_type
from .marginal_rewards import (
    compute_alternative_token_rewards,
    compute_marginal_rewards,
    compute_sequence_level_reward,
    train_token_critic_with_marginal_rewards,
)


def compute_and_apply_marginal_rewards(
    value_function,
    actions: List[str],
    states: List[str],
    base_states: List[str],
    tokenizer,
    config,
    reward_model=None,
    reward_tokenizer=None,
    environments: Optional[List] = None,
    dialogue_histories: Optional[List[List[Tuple[str, str]]]] = None,
    actual_rewards: Optional[List[float]] = None,
    instructions: Optional[List[str]] = None,
    device: str = "cuda"
) -> Tuple[Optional[torch.Tensor], Dict]:
    """
    Compute marginal token rewards and train token-level critic.
    
    Args:
        value_function: ValueFunction instance with token critic
        actions: List of agent actions
        states: List of current states/observations
        base_states: List of base states for comparison
        tokenizer: Tokenizer for action encoding
        config: Configuration object with marginal reward parameters
        reward_model: Optional reward model (LlamaGuard or Skywork) - only used if environments is None
        reward_tokenizer: Optional tokenizer for reward model
        environments: Optional list of environment instances (one per action) - takes precedence over reward_model
        dialogue_histories: Optional list of dialogue histories (one per action) for environment reward computation
        actual_rewards: Optional list of actual rewards from transitions (for unmasked actions)
        instructions: Optional list of instruction texts
        device: Device for computation
    
    Returns:
        Tuple of (token_critic_loss, stats_dict)
        - token_critic_loss: Loss tensor if marginal rewards enabled, None otherwise
        - stats_dict: Dictionary with statistics
    """
    if not getattr(config, 'use_marginal_token_rewards', False):
        return None, {}
    
    # Use sequence critic (base model) for token selection if no reward model
    sequence_critic = value_function.model
    
    # Collect marginal rewards for all actions
    all_marginal_rewards: List[Dict[int, float]] = []
    all_token_positions: List[List[int]] = []
    
    reward_model_type_norm = normalize_reward_model_type(
        getattr(config, "reward_model_type", None),
        getattr(config, "reward_model_name", None),
    )
    schema = getattr(config, "token_level_reward_schema", "marginal")
    decay_gamma_cfg = getattr(config, "token_reward_decay_gamma", None)
    discount_factor = getattr(config, "discount_factor", 0.9)
    decay_gamma = decay_gamma_cfg if decay_gamma_cfg is not None else discount_factor

    for i, (action, state, base_state) in enumerate(zip(actions, states, base_states)):
        instruction = instructions[i] if instructions and i < len(instructions) else None
        env = environments[i] if environments and i < len(environments) else None
        dialogue_history = dialogue_histories[i] if dialogue_histories and i < len(dialogue_histories) else None
        actual_reward = actual_rewards[i] if actual_rewards and i < len(actual_rewards) else None

        if schema == "marginal":
            marginal_rewards = compute_marginal_rewards(
                state=state,
                action=action,
                base_state=base_state,
                tokenizer=tokenizer,
                sequence_critic=sequence_critic,
                reward_model=reward_model if not environments else None,
                reward_tokenizer=reward_tokenizer if not environments else None,
                reward_model_type=reward_model_type_norm,
                environment=env,
                current_observation=state,
                dialogue_history=dialogue_history,
                actual_reward=actual_reward,
                top_k_tokens=getattr(config, 'num_masked_marginal_tokens', 5),
                pairwise_masking=getattr(config, 'pairwise_marginal_masking', True),
                use_task_reward=getattr(config, 'use_task_reward', True),
                instruction=instruction,
                device=device,
                chunk_size=getattr(config, 'marginal_reward_chunk_size', 4)
            )
        else:
            R = compute_sequence_level_reward(
                state=state,
                action=action,
                tokenizer=tokenizer,
                sequence_critic=sequence_critic,
                reward_model=reward_model if not environments else None,
                reward_tokenizer=reward_tokenizer if not environments else None,
                reward_model_type=reward_model_type_norm,
                environment=env,
                current_observation=state,
                dialogue_history=dialogue_history,
                actual_reward=actual_reward,
                device=device,
            )
            marginal_rewards = compute_alternative_token_rewards(
                schema=schema,
                action=action,
                tokenizer=tokenizer,
                sequence_reward=R,
                decay_gamma=decay_gamma,
                use_task_reward=getattr(config, 'use_task_reward', True),
                instruction=instruction,
                sequence_critic=sequence_critic,
                device=device,
            )

        all_marginal_rewards.append(marginal_rewards)
        all_token_positions.append(sorted(marginal_rewards.keys()))
    
    # Predict token-level values using token critic
    token_critic_values = value_function.predict_token_values(
        actions=actions,
        tokenizer=tokenizer,
        requires_grad=True
    )
    
    # Compute loss for each action
    losses = []
    total_tokens_processed = 0
    
    for i, (marginal_rewards, token_positions) in enumerate(zip(all_marginal_rewards, all_token_positions)):
        if not token_positions:
            continue
        
        # Get predicted values for this action
        action_token_values = token_critic_values[i]  # [seq_len]
        
        # Compute loss
        loss = train_token_critic_with_marginal_rewards(
            token_critic=None,  # Not needed, we use value_function directly
            token_critic_values=action_token_values,
            marginal_rewards=marginal_rewards,
            token_positions=token_positions,
            device=device
        )
        
        losses.append(loss)
        total_tokens_processed += len(token_positions)
    
    # Average loss across all actions
    if losses:
        token_critic_loss = torch.stack(losses).mean()
    else:
        token_critic_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    # Statistics
    stats = {
        'num_actions_processed': len(actions),
        'total_tokens_processed': total_tokens_processed,
        'avg_tokens_per_action': total_tokens_processed / len(actions) if actions else 0.0,
        'token_critic_loss': token_critic_loss.item() if isinstance(token_critic_loss, torch.Tensor) else 0.0,
        'token_level_reward_schema': schema,
    }
    
    return token_critic_loss, stats

