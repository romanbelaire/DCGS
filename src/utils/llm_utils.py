"""LLM inference utilities with singleton pattern."""

import os
import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor
from typing import Optional, List, Tuple, Dict, Union
from openai import OpenAI
from peft import PeftConfig, PeftModel

# Singleton instances
_model_instance: Optional[AutoModelForCausalLM] = None
_user_model_instance: Optional[AutoModelForCausalLM] = None
_tokenizer_instance: Optional[AutoTokenizer] = None
_judge_model_instance: Optional[AutoModelForCausalLM] = None
_judge_tokenizer_instance: Optional[AutoTokenizer] = None
_current_main_model_name: Optional[str] = None
_current_user_model_name: Optional[str] = None
_current_tokenizer_name: Optional[str] = None
_current_judge_model_name: Optional[str] = None

# VitaBench model instances (singleton pattern, same as main codebase)
_vitabench_models: Dict[str, AutoModelForCausalLM] = {}
_vitabench_tokenizers: Dict[str, AutoTokenizer] = {}

# Fast tokenizer is not thread-safe (Rust "Already borrowed"). Serialize tokenizer/model use.
_tokenizer_model_lock = threading.Lock()


def _is_peft_adapter_repo(model_name: str) -> bool:
    try:
        PeftConfig.from_pretrained(model_name)
        return True
    except Exception:
        return False


def _resolve_tokenizer_name(model_name: str) -> str:
    if _is_peft_adapter_repo(model_name):
        peft_config = PeftConfig.from_pretrained(model_name)
        return peft_config.base_model_name_or_path
    return model_name


def _load_model_or_adapter(
    model_name: str,
    device: str,
    use_bf16: bool
):
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    if _is_peft_adapter_repo(model_name):
        peft_config = PeftConfig.from_pretrained(model_name)
        base_model_name = peft_config.base_model_name_or_path
        print(f"Detected PEFT adapter: {model_name}")
        print(f"Loading base model for adapter: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True
        )
        merged_model = PeftModel.from_pretrained(base_model, model_name)
        merged_model = merged_model.merge_and_unload()
        merged_model.eval()
        print(f"Adapter merged successfully: {model_name}")
        return merged_model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()
    return model


class SuppressWordsLogitsProcessor(LogitsProcessor):
    """Logits processor that sets logits to -inf for specified words."""

    def __init__(self, tokenizer, words_to_suppress: List[str]):
        """
        Initialize the logits processor.

        Args:
            tokenizer: The tokenizer to use for encoding words
            words_to_suppress: List of words (strings) to suppress
        """
        self.tokenizer = tokenizer
        self.suppress_token_ids = set()
        if not words_to_suppress:
            return
        # Batch-tokenize all words in one call under lock to avoid tokenizer deadlock (Rust "Already borrowed")
        with _tokenizer_model_lock:
            encoded = tokenizer(
                words_to_suppress,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_tensors=None,
            )
            for ids in encoded["input_ids"]:
                self.suppress_token_ids.update(ids)
    
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        Set logits to -inf for suppressed tokens.
        
        Args:
            input_ids: Current input token IDs [batch_size, seq_len]
            scores: Logits for next token [batch_size, vocab_size]
        
        Returns:
            Modified scores with suppressed tokens set to -inf
        """
        # Set logits to -inf for all suppressed token IDs
        for token_id in self.suppress_token_ids:
            if token_id < scores.shape[-1]:  # Ensure token_id is within vocab size
                scores[:, token_id] = float('-inf')
        
        return scores


def get_model_instance(
    model_name: str,
    device: str = "cuda",
    use_bf16: bool = True
) -> AutoModelForCausalLM:
    """
    Get main model instance with singleton pattern (for HL/LL agents and value function).
    
    Loads on GPU 0 (or specified device).
    Ensures only one model instance is loaded and cached.
    """
    global _model_instance, _current_main_model_name
    
    if _model_instance is None or _current_main_model_name != model_name:
        print(f"Loading main model on {device}: {model_name}")
        _model_instance = _load_model_or_adapter(
            model_name=model_name,
            device=device,
            use_bf16=use_bf16
        )
        _current_main_model_name = model_name
        print(f"Main model loaded successfully on {device}")
    
    return _model_instance


def get_user_model_instance(
    user_model_name: str,
    main_model_name: str,
    device: str = "cuda:1",
    use_bf16: bool = True,
    fallback_to_main: bool = True
) -> AutoModelForCausalLM:
    """
    Get user model instance with singleton pattern (for user agent).
    
    If user_model_name == main_model_name, returns the main model singleton (single LLM).
    Otherwise, loads a separate model instance on GPU 1.
    
    Args:
        user_model_name: User model identifier
        main_model_name: Main model identifier (to check if same)
        device: Target device (default: "cuda:1")
        use_bf16: Use bfloat16 precision
        fallback_to_main: If True, return main model if GPU 1 unavailable
    
    Returns:
        Model instance (same as main if same model name, separate if different)
    """
    global _user_model_instance, _current_user_model_name
    
    # If same model name, use main model singleton (single LLM)
    if user_model_name == main_model_name:
        print(f"User model same as main model ({user_model_name}), using main model singleton")
        return get_model_instance(main_model_name, device="cuda:0", use_bf16=use_bf16)
    
    # Different model requested - load separately
    # Check if GPU 1 is available
    if device.startswith("cuda:1") and torch.cuda.device_count() < 2:
        print(f"Warning: GPU 1 not available ({torch.cuda.device_count()} GPUs), falling back to main model")
        if fallback_to_main:
            return get_model_instance(main_model_name, device="cuda:0", use_bf16=use_bf16)
        else:
            device = "cuda:0"
    
    if _user_model_instance is None or _current_user_model_name != user_model_name:
        print(f"Loading separate user model on {device}: {user_model_name}")
        _user_model_instance = _load_model_or_adapter(
            model_name=user_model_name,
            device=device,
            use_bf16=use_bf16
        )
        _current_user_model_name = user_model_name
        print(f"User model loaded successfully on {device}")
    
    return _user_model_instance


def get_tokenizer_instance(model_name: str) -> AutoTokenizer:
    """
    Get tokenizer instance with singleton pattern.
    """
    global _tokenizer_instance, _current_tokenizer_name
    tokenizer_name = _resolve_tokenizer_name(model_name)
    
    if _tokenizer_instance is None or _current_tokenizer_name != tokenizer_name:
        print(f"Loading tokenizer: {tokenizer_name}")
        _tokenizer_instance = AutoTokenizer.from_pretrained(tokenizer_name)
        if _tokenizer_instance.pad_token is None:
            _tokenizer_instance.pad_token = _tokenizer_instance.eos_token
        # Set padding side to left for decoder-only models
        _tokenizer_instance.padding_side = "left"
        _current_tokenizer_name = tokenizer_name
    
    return _tokenizer_instance


def get_judge_model_and_tokenizer(
    model_name: str,
    device: str = "cuda",
    use_bf16: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Get judge model/tokenizer.

    If model_name is a PEFT adapter, load the base checkpoint (adapter-unloaded)
    for judging so policy fine-tuning behavior does not affect judge outputs.
    """
    global _judge_model_instance, _judge_tokenizer_instance, _current_judge_model_name

    if _is_peft_adapter_repo(model_name):
        peft_config = PeftConfig.from_pretrained(model_name)
        judge_model_name = peft_config.base_model_name_or_path
    else:
        judge_model_name = model_name

    if judge_model_name == model_name:
        return get_model_instance(model_name, device=device, use_bf16=use_bf16), get_tokenizer_instance(model_name)

    if _judge_model_instance is None or _current_judge_model_name != judge_model_name:
        print(f"Loading judge base model on {device}: {judge_model_name}")
        _judge_model_instance = _load_model_or_adapter(
            model_name=judge_model_name,
            device=device,
            use_bf16=use_bf16,
        )
        _judge_tokenizer_instance = AutoTokenizer.from_pretrained(judge_model_name)
        if _judge_tokenizer_instance.pad_token is None:
            _judge_tokenizer_instance.pad_token = _judge_tokenizer_instance.eos_token
        _judge_tokenizer_instance.padding_side = "left"
        _current_judge_model_name = judge_model_name
        print(f"Judge base model loaded successfully on {device}")

    return _judge_model_instance, _judge_tokenizer_instance


def extract_log_probs(
    model,
    tokenizer,
    context_ids: torch.Tensor,
    observation_ids: torch.Tensor
) -> float:
    """
    Extract log-probabilities for observation tokens given context.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        context_ids: Token IDs for context (belief + action)
        observation_ids: Token IDs for observation to compute probability of
    
    Returns:
        Mean log-probability (normalized by sequence length)
    """
    # Concatenate context and observation
    input_ids = torch.cat([context_ids, observation_ids], dim=-1)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
    
    # Extract log-probabilities for observation tokens
    # Shift logits by 1 for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    
    # Get log-probabilities using log_softmax
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    
    # Extract log-probs for observation tokens only
    context_len = context_ids.shape[-1]
    observation_start_pos = context_len - 1  # Position in shift_logits (after shifting)
    
    # Get log-prob for each token in observation
    token_log_probs = []
    for i, token_id in enumerate(observation_ids[0]):
        pos = observation_start_pos + i
        if pos < log_probs.shape[1]:
            token_log_prob = log_probs[0, pos, token_id].item()
            token_log_probs.append(token_log_prob)
    
    # Return mean log-probability (normalized by sequence length)
    if not token_log_probs:
        return float("-inf")
    
    return sum(token_log_probs) / len(token_log_probs)


def compute_log_prob_batch(
    model,
    tokenizer,
    contexts: List[str],
    targets: List[str],
    max_length: int = 2048
) -> List[float]:
    """
    Compute log-probabilities for a batch of (context, target) pairs.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        contexts: List of context strings
        targets: List of target strings
        max_length: Maximum total sequence length (context + target). 
                   If exceeded, context will be truncated to fit target.
    
    Returns:
        List of log-probability values
    """
    if len(contexts) != len(targets):
        raise ValueError("contexts and targets must have the same length")
    if not contexts:
        return []
    
    if model is None:
        raise ValueError(
            "model is None. When using GPT agents (ll_agent.model is None), "
            "you must pass the main model instance for log probability computation."
        )

    with _tokenizer_model_lock:
        device = getattr(model, "device", None)
        if device is None:
            device = next(model.parameters()).device

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

        # Tokenize with truncation to prevent memory issues from growing dialogue history
        context_token_lists = []
        target_token_lists = []
        for ctx, tgt in zip(contexts, targets):
            ctx_tokens = tokenizer.encode(ctx, add_special_tokens=False, max_length=max_length, truncation=True)
            tgt_tokens = tokenizer.encode(tgt, add_special_tokens=False)
            
            # If context + target exceeds max_length, truncate to fit
            total_len = len(ctx_tokens) + len(tgt_tokens)
            if total_len > max_length:
                # Try to truncate target first (preserve context as much as possible)
                if len(tgt_tokens) > (max_length - len(ctx_tokens)):
                    # Target is too long, truncate it
                    max_tgt_len = max(0, max_length - len(ctx_tokens))
                    tgt_tokens = tgt_tokens[:max_tgt_len]
                # If still too long, truncate context
                if len(ctx_tokens) + len(tgt_tokens) > max_length:
                    max_ctx_len = max(1, max_length - len(tgt_tokens))  # Keep at least 1 token for context
                    ctx_tokens = ctx_tokens[:max_ctx_len]
            
            context_token_lists.append(ctx_tokens)
            target_token_lists.append(tgt_tokens)

        max_total_len = 0
        for ctx_tokens, tgt_tokens in zip(context_token_lists, target_token_lists):
            total_len = len(ctx_tokens) + len(tgt_tokens)
            if total_len > max_total_len:
                max_total_len = total_len
        
        # Cap max_total_len to max_length to prevent OOM
        max_total_len = min(max_total_len, max_length)

        batch_size = len(contexts)
        input_ids = torch.full((batch_size, max_total_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_total_len), dtype=torch.long)
        labels = torch.full((batch_size, max_total_len), -100, dtype=torch.long)

        for i, (ctx_tokens, tgt_tokens) in enumerate(zip(context_token_lists, target_token_lists)):
            ctx_tensor = torch.tensor(ctx_tokens, dtype=torch.long)
            tgt_tensor = torch.tensor(tgt_tokens, dtype=torch.long)
            total_len = ctx_tensor.numel() + tgt_tensor.numel()
            if total_len == 0:
                continue
            # Truncate if total_len exceeds max_total_len (can happen if target is very long after encoding)
            if total_len > max_total_len:
                # Truncate target to fit within max_total_len
                available_len = max_total_len - ctx_tensor.numel()
                if available_len > 0:
                    tgt_tensor = tgt_tensor[:available_len]
                    total_len = ctx_tensor.numel() + tgt_tensor.numel()
                else:
                    # If context itself exceeds max, truncate context
                    ctx_tensor = ctx_tensor[:max_total_len]
                    tgt_tensor = torch.tensor([], dtype=torch.long)
                    total_len = ctx_tensor.numel()
            input_ids[i, :total_len] = torch.cat([ctx_tensor, tgt_tensor], dim=0)
            attention_mask[i, :total_len] = 1
            if tgt_tensor.numel() > 0:
                labels[i, ctx_tensor.numel():total_len] = tgt_tensor

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = labels[:, 1:] != -100

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        token_log_probs = token_log_probs.masked_fill(~shift_mask, 0.0)

        sum_log_probs = token_log_probs.sum(dim=1)
        counts = shift_mask.sum(dim=1)

        results: List[float] = []
        for sum_lp, count in zip(sum_log_probs, counts):
            if count.item() == 0:
                results.append(float("-inf"))
            else:
                results.append((sum_lp / count).item())

        # Explicit cleanup of large tensors to prevent memory accumulation
        del log_probs, token_log_probs, shift_logits, shift_labels, shift_mask
        del outputs, logits, input_ids, attention_mask, labels
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results


def compute_joint_log_prob_batch(
    model,
    tokenizer,
    contexts: List[str],
    targets: List[str],
    max_length: int = 2048
) -> List[float]:
    """
    Compute joint log-probabilities (sum, not mean) for a batch of (context, target) pairs.
    
    This returns the actual joint probability P(a|b) = P(t1|b) * P(t2|b,t1) * ... * P(tL|b,t1...tL-1)
    as log P(a|b) = sum of log probabilities (not normalized by length).
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        contexts: List of context strings
        targets: List of target strings
        max_length: Maximum total sequence length (context + target). 
                   If exceeded, context will be truncated to fit target.
    
    Returns:
        List of joint log-probability values (sum of token log probs, not mean)
    """
    if len(contexts) != len(targets):
        raise ValueError("contexts and targets must have the same length")
    if not contexts:
        return []

    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    # Tokenize with truncation to prevent memory issues from growing dialogue history
    context_token_lists = []
    target_token_lists = []
    for ctx, tgt in zip(contexts, targets):
        ctx_tokens = tokenizer.encode(ctx, add_special_tokens=False, max_length=max_length, truncation=True)
        tgt_tokens = tokenizer.encode(tgt, add_special_tokens=False)
        
        # If context + target exceeds max_length, truncate context further to fit target
        if len(ctx_tokens) + len(tgt_tokens) > max_length:
            max_ctx_len = max(1, max_length - len(tgt_tokens))  # Keep at least 1 token for context
            ctx_tokens = tokenizer.encode(ctx, add_special_tokens=False, max_length=max_ctx_len, truncation=True)
        
        context_token_lists.append(ctx_tokens)
        target_token_lists.append(tgt_tokens)

    max_total_len = 0
    for ctx_tokens, tgt_tokens in zip(context_token_lists, target_token_lists):
        total_len = len(ctx_tokens) + len(tgt_tokens)
        if total_len > max_total_len:
            max_total_len = total_len
    
    # Cap max_total_len to max_length to prevent OOM
    max_total_len = min(max_total_len, max_length)

    batch_size = len(contexts)
    input_ids = torch.full((batch_size, max_total_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_total_len), dtype=torch.long)
    labels = torch.full((batch_size, max_total_len), -100, dtype=torch.long)

    for i, (ctx_tokens, tgt_tokens) in enumerate(zip(context_token_lists, target_token_lists)):
        ctx_tensor = torch.tensor(ctx_tokens, dtype=torch.long)
        tgt_tensor = torch.tensor(tgt_tokens, dtype=torch.long)
        total_len = ctx_tensor.numel() + tgt_tensor.numel()
        if total_len == 0:
            continue
        # Truncate if total_len exceeds max_total_len (can happen if target is very long after encoding)
        if total_len > max_total_len:
            # Truncate target to fit within max_total_len
            available_len = max_total_len - ctx_tensor.numel()
            if available_len > 0:
                tgt_tensor = tgt_tensor[:available_len]
                total_len = ctx_tensor.numel() + tgt_tensor.numel()
            else:
                # If context itself exceeds max, truncate context
                ctx_tensor = ctx_tensor[:max_total_len]
                tgt_tensor = torch.tensor([], dtype=torch.long)
                total_len = ctx_tensor.numel()
        input_ids[i, :total_len] = torch.cat([ctx_tensor, tgt_tensor], dim=0)
        attention_mask[i, :total_len] = 1
        if tgt_tensor.numel() > 0:
            labels[i, ctx_tensor.numel():total_len] = tgt_tensor

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = labels[:, 1:] != -100

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs.masked_fill(~shift_mask, 0.0)

    # Return sum (joint probability), not mean
    sum_log_probs = token_log_probs.sum(dim=1)

    results: List[float] = []
    for sum_lp in sum_log_probs:
        if sum_lp.item() == 0.0:
            results.append(float("-inf"))
        else:
            results.append(sum_lp.item())

    # Explicit cleanup of large tensors to prevent memory accumulation
    del log_probs, token_log_probs, shift_logits, shift_labels, shift_mask
    del outputs, logits, input_ids, attention_mask, labels
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def compute_log_prob(
    model,
    tokenizer,
    belief_context: str,
    action: str,
    observation: str
) -> float:
    """
    Compute log-probability of observation given belief context and action.
    """
    context = f"{belief_context}\nAgent: {action}\nUser:"
    log_probs = compute_log_prob_batch(model, tokenizer, [context], [observation])
    return log_probs[0] if log_probs else float("-inf")


def batch_generate(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    do_sample: bool = True,
    prefill_suffix: Optional[str] = None,
    chunk_size: Optional[int] = None,
    logits_processor: Optional[LogitsProcessor] = None,
    enable_thinking: Optional[bool] = None
) -> List[str]:
    """
    Batch processing for text generation with optional chunking to prevent OOM.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: List of prompt strings
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling
        prefill_suffix: Optional text to append to each prompt's tokenized input
        chunk_size: Optional chunk size for processing. If None, processes all prompts at once.
        enable_thinking: Optional stage-specific thinking control for models that support it.
    
    Returns:
        List of generated texts
    """
    if not prompts:
        return []
    
    # If chunk_size is provided, process in chunks (minibatching)
    if chunk_size is not None and len(prompts) > chunk_size:
        all_generated_texts = []
        num_chunks = (len(prompts) + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(prompts))
            chunk_prompts = prompts[start_idx:end_idx]
            
            chunk_results = batch_generate(
                model=model,
                tokenizer=tokenizer,
                prompts=chunk_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                prefill_suffix=prefill_suffix,
                chunk_size=None,  # Don't recurse further
                logits_processor=logits_processor,  # Pass logits processor to recursive call
                enable_thinking=enable_thinking,
            )
            all_generated_texts.extend(chunk_results)
            
            # Cleanup after each chunk
            del chunk_prompts, chunk_results
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return all_generated_texts

    model_name_or_path = model.name_or_path
    is_qwen_35 = model_name_or_path.startswith("Qwen/Qwen3.5")
    enable_thinking_set = enable_thinking in (True, False)
    if enable_thinking is True and not is_qwen_35:
        raise ValueError(
            f"enable_thinking was set to {enable_thinking} for model '{model_name_or_path}', "
            "but only Qwen3.5 local generation currently supports stage-specific thinking control."
        )

    prompt_texts = prompts
    if is_qwen_35 and enable_thinking_set:
        prompt_texts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            try:
                rendered_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except TypeError as exc:
                raise RuntimeError(
                    "Qwen3.5 tokenizer does not accept enable_thinking in apply_chat_template. "
                    "Cannot honor stage-specific thinking configuration."
                ) from exc
            prompt_texts.append(rendered_prompt)

    # One block: tokenize all prompts (and optional suffix) under lock to avoid tokenizer deadlock
    with _tokenizer_model_lock:
        inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        if prefill_suffix:
            suffix_tokens = tokenizer(prefill_suffix, add_special_tokens=False, return_tensors="pt")

    prompt_lengths: List[int]
    original_prompt_lengths: List[int]  # Prompt lengths without suffix (for decoding)

    if prefill_suffix:
        suffix_ids = suffix_tokens["input_ids"].squeeze(0)
        suffix_attention = torch.ones_like(suffix_ids, dtype=inputs["attention_mask"].dtype)
        suffix_ids = suffix_ids.to(model.device)
        suffix_attention = suffix_attention.to(model.device)

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

        new_input_ids = []
        new_attention_masks = []
        prompt_lengths = []
        original_prompt_lengths = []

        for input_ids, attention_mask in zip(inputs["input_ids"], inputs["attention_mask"]):
            # Store original prompt length (without suffix) for decoding
            original_prompt_lengths.append(int(attention_mask.sum().item()))
            # Combine with suffix for generation
            combined_ids = torch.cat([input_ids, suffix_ids], dim=-1)
            combined_attention = torch.cat([attention_mask, suffix_attention], dim=-1)
            new_input_ids.append(combined_ids)
            new_attention_masks.append(combined_attention)
            prompt_lengths.append(int(combined_attention.sum().item()))

        inputs["input_ids"] = torch.nn.utils.rnn.pad_sequence(
            new_input_ids,
            batch_first=True,
            padding_value=pad_token_id
        )
        inputs["attention_mask"] = torch.nn.utils.rnn.pad_sequence(
            new_attention_masks,
            batch_first=True,
            padding_value=0
        )
    else:
        prompt_lengths = inputs["attention_mask"].sum(dim=-1).tolist()
        original_prompt_lengths = prompt_lengths
    
    # Generate
    with torch.no_grad():
        generate_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id
        }
        
        # Add logits processor if provided
        if logits_processor is not None:
            from transformers import LogitsProcessorList
            generate_kwargs["logits_processor"] = LogitsProcessorList([logits_processor])
        
        outputs = model.generate(**generate_kwargs)
    
    # Decode (one block under lock: all decode calls use tokenizer)
    generated_texts = []
    with _tokenizer_model_lock:
        for i, output_ids in enumerate(outputs):
            # Account for left padding: find where actual input starts
            attention_mask = inputs["attention_mask"][i]

            # Find the first non-padding token (where actual input starts)
            # With left padding, padding tokens are at the beginning
            input_start_idx = 0
            if tokenizer.padding_side == "left":
                # Find first position where attention_mask is 1 (non-padding)
                non_zero_indices = (attention_mask == 1).nonzero(as_tuple=True)[0]
                if len(non_zero_indices) > 0:
                    input_start_idx = int(non_zero_indices[0].item())

            # Calculate where generated tokens start
            # input_start_idx + input_len gives us the end of input, start of generation
            input_len = original_prompt_lengths[i]
            generation_start_idx = input_start_idx + input_len

            # Extract generated tokens (everything after the input)
            generated_ids = output_ids[generation_start_idx:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            generated_texts.append(generated_text)
    
    # Cleanup
    del inputs, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return generated_texts


def _gpt_completion_limit_kwargs(model_name: str, max_new_tokens: int) -> dict:
    """
    Return the correct completion-limit parameter for the OpenAI chat completions API.
    GPT-5 and newer models use max_completion_tokens; older models use max_tokens.
    GPT-5 reasoning models use part of the budget for internal reasoning (not in content);
    we scale up the limit so the visible reply has at least max_new_tokens headroom.
    """
    if model_name.startswith("gpt-5"):
        return {"max_completion_tokens": 2048}
    return {"max_tokens": max_new_tokens}

def _gpt_temperature_kwargs(model_name: str, temperature: float) -> dict:
    """
    Return the correct temperature parameter for the OpenAI chat completions API.
    GPT-5 only uses default (1.0)
    """
    if model_name.startswith("gpt-5"):
        return {"temperature": 1.0}
    return {"temperature": temperature}

def batch_generate_gpt(
    prompts: List[str],
    model_name: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    do_sample: bool = True,
    prefill_suffix: Optional[str] = None,
    chunk_size: Optional[int] = None
) -> List[str]:
    """
    Batch processing for text generation using OpenAI GPT API.
    
    Args:
        prompts: List of prompt strings
        model_name: GPT model name (e.g., "gpt-4o-mini", "gpt-4o")
        api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
        reasoning_effort: Optional reasoning effort for GPT-5 family (e.g., "none", "medium")
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling (ignored for GPT, always uses temperature)
        prefill_suffix: Optional text to append to each prompt
        chunk_size: Optional chunk size for processing. If None, processes all prompts at once.
    
    Returns:
        List of generated texts
    """
    if not prompts:
        return []
    
    # Get API key
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("OpenAI API key not provided and OPENAI_API_KEY environment variable not set")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # If chunk_size is provided, process in chunks
    if chunk_size is not None and len(prompts) > chunk_size:
        all_generated_texts = []
        num_chunks = (len(prompts) + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(prompts))
            chunk_prompts = prompts[start_idx:end_idx]
            
            chunk_results = batch_generate_gpt(
                prompts=chunk_prompts,
                model_name=model_name,
                api_key=api_key,
                reasoning_effort=reasoning_effort,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                prefill_suffix=prefill_suffix,
                chunk_size=None  # Don't recurse further
            )
            all_generated_texts.extend(chunk_results)
        
        return all_generated_texts
    
    # Prepare prompts with optional suffix
    full_prompts = []
    for prompt in prompts:
        if prefill_suffix:
            full_prompt = prompt + prefill_suffix
        else:
            full_prompt = prompt
        full_prompts.append(full_prompt)
    
    # Make API calls (can be parallelized in future, but for now sequential for simplicity)
    generated_texts = []
    for full_prompt in full_prompts:
        try:
            completion_kwargs = _gpt_completion_limit_kwargs(model_name, max_new_tokens)
            temperature_kwargs = _gpt_temperature_kwargs(model_name, temperature)
            extra_kwargs = {}
            if reasoning_effort is not None and model_name.startswith("gpt-5"):
                extra_kwargs["reasoning_effort"] = reasoning_effort
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": full_prompt}
                ],
                **temperature_kwargs,
                **completion_kwargs,
                **extra_kwargs,
            )
            raw_content = response.choices[0].message.content
            if raw_content is None:
                raw_content = ""
            generated_text = raw_content.strip()
            
            # If prefill_suffix was used, remove it from the generated text
            if prefill_suffix and generated_text.startswith(prefill_suffix):
                generated_text = generated_text[len(prefill_suffix):].strip()
            
            # Strip any role labels that GPT might include (e.g., "Agent:", "User:", "Assistant:")
            # These can confuse downstream processing when responses are used as agent actions
            import re
            # Remove leading role labels (case-insensitive)
            generated_text = re.sub(r'^(Agent|User|Assistant|System):\s*', '', generated_text, flags=re.IGNORECASE).strip()
            
            generated_texts.append(generated_text)
        except Exception as e:
            print(f"[ERROR] GPT API call failed: {e}")
            # Return empty string as fallback
            generated_texts.append("")
    
    return generated_texts


def is_local_model(model_name: str) -> bool:
    """
    Check if a model name refers to a local HuggingFace model (not an API model).
    
    Args:
        model_name: Model identifier
    
    Returns:
        True if local model, False if API model
    """
    openai_models = {"gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo", "gpt-4.1"}
    claude_models = {"claude-3-opus", "claude-3-sonnet", "claude-3-haiku"}
    api_models = openai_models | claude_models
    
    if model_name in api_models:
        return False
    
    if "/" in model_name:
        return True
    
    if model_name.startswith(("gpt-", "claude-")):
        return False
    
    return True


def get_vitabench_model_instance(
    model_name: str,
    device: str = "cuda",
    use_bf16: bool = True
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Get VitaBench model instance with singleton pattern (same as main codebase).
    
    Model is loaded once per model_name and shared across all uses (user, agent, judge).
    This matches the pattern in multiwoz where policy, user, and judge all use the same base model.
    
    Args:
        model_name: Model identifier (used as key for singleton)
        device: Target device
        use_bf16: Use bfloat16 precision
    
    Returns:
        Tuple of (model, tokenizer)
    """
    global _vitabench_models, _vitabench_tokenizers
    
    # Load model if not already loaded
    if model_name not in _vitabench_models:
        print(f"Loading VitaBench model on {device}: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if use_bf16 else torch.float32,
            device_map=device,
            trust_remote_code=True
        )
        model.eval()
        _vitabench_models[model_name] = model
        print(f"VitaBench model loaded successfully on {device}")
    
    model = _vitabench_models[model_name]
    
    # Load tokenizer if not already loaded
    if model_name not in _vitabench_tokenizers:
        print(f"Loading VitaBench tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        _vitabench_tokenizers[model_name] = tokenizer
    
    tokenizer = _vitabench_tokenizers[model_name]
    
    return model, tokenizer


def messages_to_prompt(messages: List) -> str:
    """
    Convert vitabench Message objects to a formatted prompt string.
    
    Args:
        messages: List of Message objects (SystemMessage, UserMessage, AssistantMessage, ToolMessage)
    
    Returns:
        Formatted prompt string
    """
    prompt_parts = []
    
    for msg in messages:
        role = msg.role
        content = msg.content or ""
        
        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            # Handle tool calls if present
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_info = []
                for tc in msg.tool_calls:
                    tool_info.append(f"{tc.name}({tc.arguments})")
                content = content or ""
                if content:
                    prompt_parts.append(f"Assistant: {content} [Tool calls: {', '.join(tool_info)}]")
                else:
                    prompt_parts.append(f"Assistant: [Tool calls: {', '.join(tool_info)}]")
            else:
                prompt_parts.append(f"Assistant: {content}")
        elif role == "tool":
            tool_name = getattr(msg, 'name', 'tool')
            prompt_parts.append(f"Tool ({tool_name}): {content}")
    
    return "\n".join(prompt_parts)

