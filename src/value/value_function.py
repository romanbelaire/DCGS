"""Value function (Q-learning) with MLP head."""

import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional

from ..utils.llm_utils import compute_log_prob_batch


def _pad_prepared_samples_to_tensors(
    prepared_samples: List[Dict[str, List[int]]],
    pad_token_id: int,
    device: str,
) -> Dict[str, torch.Tensor]:
    if not prepared_samples:
        raise ValueError("prepared_samples cannot be empty")
    max_len = max(len(sample["input_ids"]) for sample in prepared_samples)
    batch_tensors: Dict[str, torch.Tensor] = {}
    for key in prepared_samples[0].keys():
        pad_value = pad_token_id if key == "input_ids" else 0
        rows = []
        for sample in prepared_samples:
            values = sample[key]
            rows.append(values + [pad_value] * (max_len - len(values)))
        batch_tensors[key] = torch.tensor(rows, dtype=torch.long, device=device)
    return batch_tensors


class ValueFunction:
    """Value function with MLP head on top of base LLM."""

    def _make_mlp_head(self, device: str, dtype: torch.dtype):
        """Create a standard MLP head for value prediction."""
        return nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1)
        ).to(device=device, dtype=dtype)

    def __init__(
        self,
        model,
        hidden_size: int = 4096,
        learning_rate: float = 1e-4,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        use_regret_critic: bool = False,
    ):
        self.model = model
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        self.use_regret_critic = use_regret_critic

        # Freeze base model parameters (no gradients needed)
        base_param_count = 0
        for param in self.model.parameters():
            param.requires_grad = False
            base_param_count += 1
        self.model.eval()  # Ensure model is in eval mode

        # Initialize separate MLP heads for Q and V in specified dtype
        self.q_mlp_head = self._make_mlp_head(device, dtype)
        self.v_mlp_head = self._make_mlp_head(device, dtype)

        # Token-level critic head (for marginal token rewards)
        self.token_critic_head = self._make_mlp_head(device, dtype)

        # Min-value critic heads (Q_min, V_min) - for adversarial robustness
        self.q_min_mlp_head = self._make_mlp_head(device, dtype) if use_regret_critic else None
        self.v_min_mlp_head = self._make_mlp_head(device, dtype) if use_regret_critic else None

        # Regret critic head - learns cumulative value gap (value - min_value)
        self.regret_mlp_head = self._make_mlp_head(device, dtype) if use_regret_critic else None

        # Ensure MLP head parameters require gradients
        for param in self.q_mlp_head.parameters():
            param.requires_grad = True
        for param in self.v_mlp_head.parameters():
            param.requires_grad = True
        for param in self.token_critic_head.parameters():
            param.requires_grad = True

        all_trainable_params = (
            list(self.q_mlp_head.parameters()) +
            list(self.v_mlp_head.parameters()) +
            list(self.token_critic_head.parameters())
        )

        if use_regret_critic:
            for param in self.q_min_mlp_head.parameters():
                param.requires_grad = True
            for param in self.v_min_mlp_head.parameters():
                param.requires_grad = True
            for param in self.regret_mlp_head.parameters():
                param.requires_grad = True
            all_trainable_params += (
                list(self.q_min_mlp_head.parameters()) +
                list(self.v_min_mlp_head.parameters()) +
                list(self.regret_mlp_head.parameters())
            )

        self.optimizer = torch.optim.Adam(all_trainable_params, lr=learning_rate)

        # Verify setup
        print(f"[ValueFunction] Initialized:")
        print(f"  Base model: {base_param_count} parameters, all frozen (requires_grad=False)")
        print(f"  Q MLP head: trainable")
        print(f"  V MLP head: trainable")
        print(f"  Token critic head: trainable")
        if use_regret_critic:
            print(f"  Q_min MLP head: trainable (adversarial robustness)")
            print(f"  V_min MLP head: trainable (adversarial robustness)")
            print(f"  Regret MLP head: trainable (adversarial robustness)")
        print(f"  Optimizer: Adam with lr={learning_rate}")
    
    def predict_q_value(
        self,
        observations: List[str],
        high_level_actions: List[str],
        tokenizer,
        requires_grad: bool = False
    ) -> torch.Tensor:
        """
        Predict Q-value for observation-high_level_action pairs (Q_high).

        Q_high takes: (observation, high_level_action) where high_level_action is the context/instruction.

        Args:
            observations: List of observation strings
            high_level_actions: List of high-level actions/contexts (belief summaries)
            tokenizer: Tokenizer for encoding
            requires_grad: If True, enable gradients for MLP head (for training).
                          Base model is always frozen.

        Returns:
            Tensor of Q-value predictions
        """
        return self._predict_q_like(observations, high_level_actions, tokenizer, self.q_mlp_head, requires_grad)

    def predict_q_min_value(
        self,
        observations: List[str],
        high_level_actions: List[str],
        tokenizer,
        requires_grad: bool = False
    ) -> torch.Tensor:
        """
        Predict Q_min-value (adversarial / min-value critic) for (obs, action) pairs.
        Same interface as predict_q_value but uses q_min_mlp_head.
        """
        if not self.use_regret_critic or self.q_min_mlp_head is None:
            raise RuntimeError("predict_q_min_value requires use_regret_critic=True")
        return self._predict_q_like(observations, high_level_actions, tokenizer, self.q_min_mlp_head, requires_grad)

    def predict_regret_value(
        self,
        observations: List[str],
        high_level_actions: List[str],
        tokenizer,
        requires_grad: bool = False
    ) -> torch.Tensor:
        """
        Predict regret value Regret(s,a) for (obs, action) pairs.
        Learns cumulative value gap: value - min_value.
        """
        if not self.use_regret_critic or self.regret_mlp_head is None:
            raise RuntimeError("predict_regret_value requires use_regret_critic=True")
        return self._predict_q_like(observations, high_level_actions, tokenizer, self.regret_mlp_head, requires_grad)

    def _predict_q_like(
        self,
        observations: List[str],
        high_level_actions: List[str],
        tokenizer,
        head: nn.Module,
        requires_grad: bool
    ) -> torch.Tensor:
        """Shared encoding + head forward for Q-style (obs, action) prediction."""
        max_seq_length = 1500
        input_texts = []
        for obs, action in zip(observations, high_level_actions):
            if not obs or not obs.strip():
                raise ValueError("Empty observation provided to Q function.")
            obs_context = f"Observation: {obs}\nHigh-Level Context:"
            belief_text = action or ""
            input_texts.append(f"{obs_context} {belief_text}")
        if not input_texts:
            raise ValueError("No valid observation/action pairs provided.")
        encoded = tokenizer(
            input_texts,
            add_special_tokens=True,
            max_length=max_seq_length,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        if requires_grad:
            outputs = self.model(**encoded, output_hidden_states=True)
        else:
            with torch.inference_mode():
                outputs = self.model(**encoded, output_hidden_states=True)
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            hidden_states = outputs.hidden_states[-1]
            pooled = hidden_states.mean(dim=1)
            del hidden_states
        else:
            pooled = (
                outputs.last_hidden_state.mean(dim=1)
                if hasattr(outputs, 'last_hidden_state')
                else self.model.get_input_embeddings()(encoded['input_ids']).mean(dim=1)
            )
        del outputs, encoded
        pooled = pooled.detach().to(dtype=self.dtype)
        if requires_grad:
            values = head(pooled)
        else:
            with torch.no_grad():
                values = head(pooled)
        return values.squeeze(-1)

    def predict_v_min_value(
        self,
        observations: List[str],
        tokenizer,
        requires_grad: bool = False
    ) -> torch.Tensor:
        """
        Predict V_min-value (adversarial state value) for observations.
        Same interface as predict_v_value but uses v_min_mlp_head.
        """
        if not self.use_regret_critic or self.v_min_mlp_head is None:
            raise RuntimeError("predict_v_min_value requires use_regret_critic=True")
        return self._predict_v_like(observations, tokenizer, self.v_min_mlp_head, requires_grad)

    def _predict_v_like(
        self,
        observations: List[str],
        tokenizer,
        head: nn.Module,
        requires_grad: bool
    ) -> torch.Tensor:
        """Shared encoding + head forward for V-style (obs only) prediction."""
        max_seq_length = 1500
        input_texts = []
        for obs in observations:
            if not obs or not obs.strip():
                raise ValueError("Empty observation provided to V function.")
            obs_context = f"Observation: {obs}"
            input_texts.append(obs_context)
        if not input_texts:
            raise ValueError("No valid observations provided.")
        encoded = tokenizer(
            input_texts,
            add_special_tokens=True,
            max_length=max_seq_length,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        if requires_grad:
            outputs = self.model(**encoded, output_hidden_states=True)
        else:
            with torch.inference_mode():
                outputs = self.model(**encoded, output_hidden_states=True)
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            hidden_states = outputs.hidden_states[-1]
            pooled = hidden_states.mean(dim=1)
            del hidden_states
        else:
            pooled = (
                outputs.last_hidden_state.mean(dim=1)
                if hasattr(outputs, 'last_hidden_state')
                else self.model.get_input_embeddings()(encoded['input_ids']).mean(dim=1)
            )
        del outputs, encoded
        pooled = pooled.detach().to(dtype=self.dtype)
        if requires_grad:
            values = head(pooled)
        else:
            with torch.no_grad():
                values = head(pooled)
        return values.squeeze(-1)

    def predict_v_value(
        self,
        observations: List[str],
        tokenizer,
        requires_grad: bool = False
    ) -> torch.Tensor:
        """
        Predict V-value for observations (state value function).

        V(s) estimates the expected return from state s under the optimal policy.
        This is faster than Q(s,a) because it doesn't require generating action candidates.

        Args:
            observations: List of observation strings
            tokenizer: Tokenizer for encoding
            requires_grad: If True, enable gradients for MLP head (for training).
                          Base model is always frozen.

        Returns:
            Tensor of V-value predictions
        """
        return self._predict_v_like(observations, tokenizer, self.v_mlp_head, requires_grad)
    
    def predict_token_values(
        self,
        actions: List[str],
        tokenizer,
        requires_grad: bool = False
    ) -> torch.Tensor:
        """
        Predict per-token values using token-level critic.
        
        Args:
            actions: List of action strings
            tokenizer: Tokenizer for encoding
            requires_grad: If True, enable gradients for MLP head (for training).
                          Base model is always frozen.
        
        Returns:
            Tensor of per-token value predictions [batch_size, seq_len]
        """
        max_seq_length = 1500
        for action in actions:
            if not action or not action.strip():
                raise ValueError("Empty action provided to token critic.")
        if not actions:
            raise ValueError("No valid actions provided for token value prediction.")
        encoded = tokenizer(
            actions,
            add_special_tokens=True,
            max_length=max_seq_length,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Get hidden states from base model
        if requires_grad:
            outputs = self.model(**encoded, output_hidden_states=True)
        else:
            with torch.inference_mode():
                outputs = self.model(**encoded, output_hidden_states=True)
        
        # Use last hidden state (per-token, not pooled)
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]
        elif hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = self.model.get_input_embeddings()(encoded['input_ids'])
        
        del outputs
        
        # Detach from computation graph since base model is frozen
        hidden_states = hidden_states.detach().to(dtype=self.dtype)
        
        # Pass through token critic head (per-token predictions)
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_flat = hidden_states.reshape(batch_size * seq_len, hidden_size)
        
        if requires_grad:
            token_values_flat = self.token_critic_head(hidden_flat)  # [batch*seq, 1]
        else:
            with torch.no_grad():
                token_values_flat = self.token_critic_head(hidden_flat)
        
        token_values = token_values_flat.reshape(batch_size, seq_len)  # [batch, seq_len]
        
        # Mask out padding tokens
        attention_mask = encoded.get('attention_mask', torch.ones_like(encoded['input_ids']))
        token_values = token_values * attention_mask.float()
        
        del encoded, hidden_states
        
        return token_values  # [batch, seq_len]

    def predict_ll_candidate_scores(
        self,
        actions: List[str],
        tokenizer,
    ) -> Dict[str, float]:
        """Score LL candidates via length-normalized mean token critic value.

        Encodes each action, runs predict_token_values (which zeros padding), then
        divides the per-sequence sum by the number of non-padding tokens to get the
        length-normalized mean — the LL analog of a sequence-level Q value.

        Returns:
            Dict mapping each action string to its scalar score.
        """
        encoded = tokenizer(
            actions,
            padding=True,
            truncation=True,
            max_length=1500,
            return_attention_mask=True,
            return_tensors="pt",
        )
        mask = encoded["attention_mask"].to(self.device).float()  # [B, L]
        token_values = self.predict_token_values(actions, tokenizer, requires_grad=False)  # [B, L], padding zeroed
        scores = token_values.sum(dim=1) / mask.sum(dim=1)  # [B], length-normalized mean
        return {action: scores[i].item() for i, action in enumerate(actions)}

    def get_q_value(
        self, 
        observation: str, 
        high_level_action: str, 
        tokenizer,
        requires_grad: bool = False
    ) -> float:
        """
        Get Q-value for observation-high_level_action pair (Q_high).
        
        Args:
            observation: Current observation text
            high_level_action: High-level action/context (belief summary)
            tokenizer: Tokenizer for encoding
            requires_grad: If True, enable gradients (default False for inference)
        
        Returns:
            Q-value estimate (float)
        """
        q_tensor = self.predict_q_value(
            observations=[observation], 
            high_level_actions=[high_level_action], 
            tokenizer=tokenizer,
            requires_grad=requires_grad
        )
        return q_tensor[0].item()
    
    def compute_likelihood(
        self,
        observation: str,
        high_level_action: str,
        low_level_action: str,
        ll_model,
        ll_tokenizer,
        belief_only: bool = True
    ) -> float:
        """
        Compute log-likelihood: log P_π_LL(a_t^LL | observation, a_t^HL) or log P_π_LL(a_t^LL | a_t^HL).
        
        Args:
            observation: Current observation (only used if belief_only=False)
            high_level_action: High-level action (belief candidate)
            low_level_action: Low-level action to compute probability for
            ll_model: Low-level model
            ll_tokenizer: Low-level tokenizer
            belief_only: If True, compute P(a|b). If False, compute P(a|o,b) (old mode)
        """
        if not low_level_action:
            return float('-inf')

        if belief_only:
            # New mode: P(a|b) - only use belief, no observation
            context = f"High-Level Context: {high_level_action}\nLow-Level Action:"
        else:
            # Old mode: P(a|o,b) - use both observation and belief
            context = f"Observation: {observation}\nHigh-Level Context: {high_level_action}\nLow-Level Action:"
        
        log_probs = compute_log_prob_batch(
            ll_model,
            ll_tokenizer,
            [context],
            [low_level_action]
        )
        return log_probs[0] if log_probs else float('-inf')

    def compute_likelihood_batch(
        self,
        observations: List[str],
        high_level_actions: List[str],
        low_level_actions: List[str],
        ll_model,
        ll_tokenizer,
        belief_only: bool = True
    ) -> List[float]:
        """
        Batch version of compute_likelihood for multiple (belief, action) pairs.
        
        Args:
            observations: List of observations (only used if belief_only=False)
            high_level_actions: List of high-level actions (belief candidates)
            low_level_actions: List of low-level actions to compute probability for
            ll_model: Low-level model
            ll_tokenizer: Low-level tokenizer
            belief_only: If True, compute P(a|b). If False, compute P(a|o,b)
        
        Returns:
            List of log-probabilities
        """
        if len(high_level_actions) != len(low_level_actions):
            raise ValueError(f"Mismatch: {len(high_level_actions)} beliefs vs {len(low_level_actions)} actions")
        if not belief_only and len(observations) != len(high_level_actions):
            raise ValueError(f"Mismatch: {len(observations)} observations vs {len(high_level_actions)} beliefs")
        
        contexts = []
        targets = []
        valid_indices = []
        
        for i, (hl_action, ll_action) in enumerate(zip(high_level_actions, low_level_actions)):
            if not ll_action:
                continue
            
            if belief_only:
                context = f"High-Level Context: {hl_action}\nLow-Level Action:"
            else:
                obs = observations[i] if i < len(observations) else ""
                context = f"Observation: {obs}\nHigh-Level Context: {hl_action}\nLow-Level Action:"
            
            contexts.append(context)
            targets.append(ll_action)
            valid_indices.append(i)
        
        if not contexts:
            return [float('-inf')] * len(high_level_actions)
        
        # Batch compute log probabilities
        log_probs = compute_log_prob_batch(
            ll_model,
            ll_tokenizer,
            contexts,
            targets
        )
        
        # Map back to original indices
        results = [float('-inf')] * len(high_level_actions)
        for valid_idx, orig_idx in enumerate(valid_indices):
            if valid_idx < len(log_probs):
                results[orig_idx] = log_probs[valid_idx]
        
        return results

    def compute_normalized_transition_probs(
        self,
        observation: str,
        high_level_action: str,
        candidate_low_level_actions: List[str],
        ll_model,
        ll_tokenizer,
        belief_only: bool = True
    ) -> List[float]:
        """
        Compute normalized transition probabilities P̂(o_{t+1} | o_t, a_t^HL).
        
        Args:
            observation: Current observation (only used if belief_only=False)
            high_level_action: High-level action (belief candidate)
            candidate_low_level_actions: List of candidate low-level actions
            ll_model: Low-level model
            ll_tokenizer: Low-level tokenizer
            belief_only: If True, compute P(a|b). If False, compute P(a|o,b) (old mode)
        """
        if not candidate_low_level_actions:
            return []

        if belief_only:
            # New mode: P(a|b) - only use belief, no observation
            context_template = f"High-Level Context: {high_level_action}\nLow-Level Action:"
        else:
            # Old mode: P(a|o,b) - use both observation and belief
            context_template = f"Observation: {observation}\nHigh-Level Context: {high_level_action}\nLow-Level Action:"
        
        log_likelihoods = [float('-inf')] * len(candidate_low_level_actions)

        contexts_batch: List[str] = []
        targets_batch: List[str] = []
        batch_indices: List[int] = []

        for idx, ll_action in enumerate(candidate_low_level_actions):
            if ll_action:
                contexts_batch.append(context_template)
                targets_batch.append(ll_action)
                batch_indices.append(idx)

        if contexts_batch:
            batch_log_probs = compute_log_prob_batch(
                ll_model,
                ll_tokenizer,
                contexts_batch,
                targets_batch
            )
            for idx, log_prob in zip(batch_indices, batch_log_probs):
                log_likelihoods[idx] = log_prob

        # Normalize using log-sum-exp trick for numerical stability
        max_log_prob = max(log_likelihoods)
        if max_log_prob == float('-inf'):
            return [1.0 / len(candidate_low_level_actions)] * len(candidate_low_level_actions)

        exp_terms = [math.exp(log_prob - max_log_prob) if log_prob != float('-inf') else 0.0
                     for log_prob in log_likelihoods]

        sum_exp = sum(exp_terms)
        if sum_exp == 0:
            return [1.0 / len(candidate_low_level_actions)] * len(candidate_low_level_actions)

        probabilities = [exp_term / sum_exp for exp_term in exp_terms]

        return probabilities
    
    def compute_normalized_transition_probs_batch(
        self,
        observations: List[str],
        high_level_actions: List[str],
        candidate_low_level_actions_list: List[List[str]],
        ll_model,
        ll_tokenizer,
        chunk_size: int = 8,
        belief_only: bool = True
    ) -> List[List[float]]:
        """
        Batch compute normalized transition probabilities for multiple transitions.
        
        Args:
            observations: List of observations (one per transition, only used if belief_only=False)
            high_level_actions: List of high-level actions (one per transition)
            candidate_low_level_actions_list: List of candidate low-level action lists (one per transition)
            ll_model: Low-level model for likelihood computation
            ll_tokenizer: Tokenizer for low-level model
            chunk_size: Chunk size for batch processing
            belief_only: If True, compute P(a|b). If False, compute P(a|o,b) (old mode)
        
        Returns:
            List of probability lists (one per transition)
        """
        if len(observations) != len(high_level_actions) or len(observations) != len(candidate_low_level_actions_list):
            raise ValueError("All input lists must have the same length")
        
        if not observations:
            return []
        
        # Collect all (context, target) pairs across all transitions
        contexts_batch: List[str] = []
        targets_batch: List[str] = []
        transition_indices: List[int] = []  # Which transition this pair belongs to
        candidate_indices: List[int] = []  # Which candidate within that transition
        
        for trans_idx, (obs, hl_action, candidate_ll_actions) in enumerate(
            zip(observations, high_level_actions, candidate_low_level_actions_list)
        ):
            if not candidate_ll_actions:
                continue
            
            if belief_only:
                # New mode: P(a|b) - only use belief, no observation
                context_template = f"High-Level Context: {hl_action}\nLow-Level Action:"
            else:
                # Old mode: P(a|o,b) - use both observation and belief
                context_template = f"Observation: {obs}\nHigh-Level Context: {hl_action}\nLow-Level Action:"
            
            for cand_idx, ll_action in enumerate(candidate_ll_actions):
                if ll_action:
                    contexts_batch.append(context_template)
                    targets_batch.append(ll_action)
                    transition_indices.append(trans_idx)
                    candidate_indices.append(cand_idx)
        
        # Compute all log probabilities in chunks to prevent memory issues
        # Process in smaller chunks to avoid OOM with long sequences
        all_log_probs: List[float] = []
        if contexts_batch:
            num_chunks = (len(contexts_batch) + chunk_size - 1) // chunk_size
            
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, len(contexts_batch))
                
                chunk_contexts = contexts_batch[start_idx:end_idx]
                chunk_targets = targets_batch[start_idx:end_idx]
                
                chunk_log_probs = compute_log_prob_batch(
                    ll_model,
                    ll_tokenizer,
                    chunk_contexts,
                    chunk_targets
                )
                
                all_log_probs.extend(chunk_log_probs)
                
                # Explicit cleanup after each chunk
                del chunk_contexts, chunk_targets, chunk_log_probs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Group results by transition and normalize each separately
        results: List[List[float]] = []
        
        for trans_idx in range(len(observations)):
            candidate_ll_actions = candidate_low_level_actions_list[trans_idx]
            
            if not candidate_ll_actions:
                results.append([])
                continue
            
            # Collect log probabilities for this transition's candidates
            log_likelihoods = [float('-inf')] * len(candidate_ll_actions)
            
            for batch_idx, (trans_idx_batch, cand_idx_batch) in enumerate(zip(transition_indices, candidate_indices)):
                if trans_idx_batch == trans_idx:
                    log_likelihoods[cand_idx_batch] = all_log_probs[batch_idx]
            
            # Normalize using log-sum-exp trick for numerical stability
            max_log_prob = max(log_likelihoods)
            if max_log_prob == float('-inf'):
                results.append([1.0 / len(candidate_ll_actions)] * len(candidate_ll_actions))
                continue
            
            exp_terms = [math.exp(log_prob - max_log_prob) if log_prob != float('-inf') else 0.0
                        for log_prob in log_likelihoods]
            
            sum_exp = sum(exp_terms)
            if sum_exp == 0:
                results.append([1.0 / len(candidate_ll_actions)] * len(candidate_ll_actions))
                continue
            
            probabilities = [exp_term / sum_exp for exp_term in exp_terms]
            results.append(probabilities)
        
        return results
    
    def compute_contrastive_loss(
        self,
        observations: List[str],
        positive_beliefs: List[str],
        negative_beliefs: List[str],
        dataset_actions: List[str],
        ll_model,
        ll_tokenizer,
        contrastive_coef: float = 1.0,
        belief_only: bool = True
    ) -> torch.Tensor:
        """
        Compute contrastive loss: maximize log P(action | obs, good_belief) - log P(action | obs, bad_belief).
        
        This encourages the Q-function to learn that beliefs that make the dataset action likely
        should have higher Q-values than beliefs that make it unlikely.
        
        Args:
            observations: List of observation strings
            positive_beliefs: List of "good" beliefs (chosen beliefs from Q-function)
            negative_beliefs: List of "bad" beliefs (from other dialogues' ground truths)
            dataset_actions: List of dataset low-level actions (ground truth actions)
            ll_model: Low-level model for computing log probabilities
            ll_tokenizer: Low-level tokenizer
            contrastive_coef: Coefficient for contrastive loss term
        
        Returns:
            Contrastive loss tensor (scalar)
        """
        if not observations or not positive_beliefs or not negative_beliefs or not dataset_actions:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)
        
        # Filter out invalid entries (empty strings, etc.)
        valid_pairs = []
        for obs, pos_belief, neg_belief, action in zip(observations, positive_beliefs, negative_beliefs, dataset_actions):
            if obs and pos_belief and neg_belief and action:
                valid_pairs.append((obs, pos_belief, neg_belief, action))
        
        if not valid_pairs:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)
        
        # Build batched contexts and targets for positive and negative beliefs
        # Combine into single batch for efficiency (one model forward pass instead of two)
        all_contexts = []
        all_targets = []
        pos_indices = []  # Track which indices correspond to positive beliefs
        neg_indices = []  # Track which indices correspond to negative beliefs
        
        # Add positive belief contexts
        for obs, pos_belief, _, action in valid_pairs:
            if belief_only:
                # New mode: P(a|b) - only use belief, no observation
                context = f"High-Level Context: {pos_belief}\nLow-Level Action:"
            else:
                # Old mode: P(a|o,b) - use both observation and belief
                context = f"Observation: {obs}\nHigh-Level Context: {pos_belief}\nLow-Level Action:"
            all_contexts.append(context)
            all_targets.append(action)
            pos_indices.append(len(all_contexts) - 1)
        
        # Add negative belief contexts
        for obs, _, neg_belief, action in valid_pairs:
            if belief_only:
                # New mode: P(a|b) - only use belief, no observation
                context = f"High-Level Context: {neg_belief}\nLow-Level Action:"
            else:
                # Old mode: P(a|o,b) - use both observation and belief
                context = f"Observation: {obs}\nHigh-Level Context: {neg_belief}\nLow-Level Action:"
            all_contexts.append(context)
            all_targets.append(action)
            neg_indices.append(len(all_contexts) - 1)
        
        # Compute all log probabilities in a single batch
        all_log_probs = compute_log_prob_batch(
            ll_model,
            ll_tokenizer,
            all_contexts,
            all_targets
        )
        
        # Split results back into positive and negative
        pos_log_probs = [all_log_probs[i] for i in pos_indices]
        neg_log_probs = [all_log_probs[i] for i in neg_indices]
        
        # Filter out -inf results
        pos_log_probs_valid = [lp for lp in pos_log_probs if lp != float('-inf')]
        neg_log_probs_valid = [lp for lp in neg_log_probs if lp != float('-inf')]
        
        if not pos_log_probs_valid or not neg_log_probs_valid:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)
        
        # Contrastive loss: we want log P(action | good_belief) > log P(action | bad_belief)
        # So we minimize: -[log P(action | good_belief) - log P(action | bad_belief)]
        # Which is equivalent to: log P(action | bad_belief) - log P(action | good_belief)
        pos_log_prob_tensor = torch.tensor(pos_log_probs_valid, device=self.device, dtype=self.dtype)
        neg_log_prob_tensor = torch.tensor(neg_log_probs_valid, device=self.device, dtype=self.dtype)
        
        # Average over batch
        avg_pos_log_prob = pos_log_prob_tensor.mean()
        avg_neg_log_prob = neg_log_prob_tensor.mean()
        
        # Contrastive loss: minimize (neg_log_prob - pos_log_prob)
        # This encourages pos_log_prob to be higher than neg_log_prob
        contrastive_loss = avg_neg_log_prob - avg_pos_log_prob
        
        return contrastive_coef * contrastive_loss
    
    def compute_value_loss(
        self,
        value_predictions: torch.Tensor,
        targets: torch.Tensor,
        avg_entropy: float,
        entropy_coef: float = 0.1,
        model_entropy: float = 0.0,
        contrastive_loss: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute value function loss with entropy regularization and optional contrastive loss.
        
        Loss: L = (V(s) - target)² - λ * (H_avg - H_model) + α * L_contrastive
        
        Args:
            value_predictions: Predicted values V(s)
            targets: Bellman targets (reward + γ * V(s'))
            avg_entropy: Average entropy during context
            entropy_coef: Entropy coefficient λ
            model_entropy: Model entropy (reference)
            contrastive_loss: Optional contrastive loss tensor
        
        Returns:
            Loss tensor
        """
        # MSE loss
        mse_loss = nn.functional.mse_loss(value_predictions, targets)
        
        # Entropy regularization term
        entropy_term = entropy_coef * (avg_entropy - model_entropy)
        
        # Total loss (subtract entropy term to encourage accounting for uncertainty)
        total_loss = mse_loss - entropy_term
        
        # Add contrastive loss if provided
        if contrastive_loss is not None:
            total_loss = total_loss + contrastive_loss
        
        return total_loss
    
    def update(self, loss: torch.Tensor) -> None:
        """
        Update value function via backpropagation.
        
        Verifies that gradients are computed for MLP head parameters.
        """
        self.optimizer.zero_grad()
        loss.backward()
        
        # Verify gradients are computed for MLP heads
        has_gradients = False
        grad_norms = []
        for name, param in self.q_mlp_head.named_parameters():
            if param.grad is not None:
                has_gradients = True
                grad_norms.append((f"Q.{name}", param.grad.norm().item()))
        for name, param in self.v_mlp_head.named_parameters():
            if param.grad is not None:
                has_gradients = True
                grad_norms.append((f"V.{name}", param.grad.norm().item()))
        if self.use_regret_critic:
            for name, param in self.q_min_mlp_head.named_parameters():
                if param.grad is not None:
                    has_gradients = True
                    grad_norms.append((f"Q_min.{name}", param.grad.norm().item()))
            for name, param in self.v_min_mlp_head.named_parameters():
                if param.grad is not None:
                    has_gradients = True
                    grad_norms.append((f"V_min.{name}", param.grad.norm().item()))
            for name, param in self.regret_mlp_head.named_parameters():
                if param.grad is not None:
                    has_gradients = True
                    grad_norms.append((f"Regret.{name}", param.grad.norm().item()))
        
        if not has_gradients:
            raise RuntimeError(
                "No gradients computed for MLP heads! "
                "Ensure predict_q_value or predict_v_value is called with requires_grad=True during training."
            )
        
        # Log gradient norms for debugging (first few updates)
        if not hasattr(self, '_update_count'):
            self._update_count = 0
        self._update_count += 1
        if self._update_count <= 3:
            print(f"  [ValueFunction] Update {self._update_count}: MLP head gradients computed")
            for name, grad_norm in grad_norms:
                print(f"    {name}: grad_norm={grad_norm:.6f}")
        
        # Verify base model has no gradients (should be frozen)
        base_model_has_grads = False
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                base_model_has_grads = True
                if self._update_count <= 3:
                    print(f"    WARNING: Base model parameter {name} has requires_grad=True!")
        
        if base_model_has_grads:
            raise RuntimeError(
                "Base model parameters have requires_grad=True! "
                "Base model should be frozen."
            )
        
        self.optimizer.step()
    
    def save_checkpoint(self, checkpoint_path: str) -> None:
        """
        Save value function checkpoint (MLP head and optimizer state).
        
        Args:
            checkpoint_path: Path to save checkpoint
        """
        from pathlib import Path
        checkpoint_file = Path(checkpoint_path)
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'q_mlp_head_state_dict': self.q_mlp_head.state_dict(),
            'v_mlp_head_state_dict': self.v_mlp_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hidden_size': self.hidden_size,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'device': self.device,
            'dtype': str(self.dtype),
            'use_regret_critic': self.use_regret_critic,
        }
        if self.use_regret_critic:
            checkpoint['q_min_mlp_head_state_dict'] = self.q_min_mlp_head.state_dict()
            checkpoint['v_min_mlp_head_state_dict'] = self.v_min_mlp_head.state_dict()
            checkpoint['regret_mlp_head_state_dict'] = self.regret_mlp_head.state_dict()
        
        try:
            torch.save(checkpoint, checkpoint_path)
            # Verify file was created
            if checkpoint_file.exists():
                file_size = checkpoint_file.stat().st_size
                print(f"[ValueFunction] Saved checkpoint to {checkpoint_path} ({file_size} bytes)")
            else:
                raise RuntimeError(f"Checkpoint file was not created: {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save checkpoint to {checkpoint_path}: {e}")
    
    def load_checkpoint(self, checkpoint_path: str, strict: bool = True, load_optimizer: bool = True) -> None:
        """
        Load value function checkpoint (MLP head and optimizer state).
        
        Args:
            checkpoint_path: Path to checkpoint file
            strict: If True, raise error if checkpoint structure doesn't match
            load_optimizer: If False, skip loading optimizer state (useful for evaluation)
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load MLP head state dicts (handle both old and new format for backward compatibility)
        try:
            if 'q_mlp_head_state_dict' in checkpoint and 'v_mlp_head_state_dict' in checkpoint:
                self.q_mlp_head.load_state_dict(checkpoint['q_mlp_head_state_dict'], strict=strict)
                self.v_mlp_head.load_state_dict(checkpoint['v_mlp_head_state_dict'], strict=strict)
                print(f"[ValueFunction] Loaded Q and V MLP heads from checkpoint")
            elif 'mlp_head_state_dict' in checkpoint:
                self.q_mlp_head.load_state_dict(checkpoint['mlp_head_state_dict'], strict=strict)
                self.v_mlp_head.load_state_dict(checkpoint['mlp_head_state_dict'], strict=strict)
                print(f"[ValueFunction] Loaded shared MLP head from checkpoint (applied to both Q and V)")
            else:
                raise RuntimeError("Checkpoint missing MLP head state dict(s)")
            # Load regret critic heads if present in checkpoint and we have them
            if self.use_regret_critic and 'q_min_mlp_head_state_dict' in checkpoint:
                self.q_min_mlp_head.load_state_dict(checkpoint['q_min_mlp_head_state_dict'], strict=strict)
                self.v_min_mlp_head.load_state_dict(checkpoint['v_min_mlp_head_state_dict'], strict=strict)
                self.regret_mlp_head.load_state_dict(checkpoint['regret_mlp_head_state_dict'], strict=strict)
                print(f"[ValueFunction] Loaded Q_min, V_min, Regret MLP heads from checkpoint")
        except RuntimeError as e:
            if strict:
                raise RuntimeError(f"Failed to load MLP head state dict: {e}")
            else:
                print(f"[ValueFunction] Warning: Could not fully load MLP head state dict: {e}")
        
        # Load optimizer state dict (if available and requested)
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"[ValueFunction] Loaded optimizer state from checkpoint")
            except (RuntimeError, ValueError) as e:
                print(f"[ValueFunction] Warning: Could not load optimizer state dict: {e}")
        elif not load_optimizer:
            print(f"[ValueFunction] Skipping optimizer state loading (evaluation mode)")
        
        print(f"[ValueFunction] Loaded checkpoint from {checkpoint_path}")

