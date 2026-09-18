"""Local frozen HF execution for native TPO; no API server or paid calls."""

import math
import time

import run_safedial_baseline as base


def tokenizers_and_preflight(lock, rows, config, max_input_tokens=None):
    from transformers import AutoConfig, AutoTokenizer
    tokenizers = {}
    limits = {}
    for kind in ("actor", "reward"):
        path = lock[kind]["path"]
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        cfg = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        if not tok.chat_template:
            raise ValueError(f"{kind} has no chat template")
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        limits[kind] = min(v for v in (cfg.max_position_embeddings, tok.model_max_length)
                           if isinstance(v, int) and 0 < v < 1_000_000)
        tokenizers[kind] = tok
    maximum = {"actor": 0, "reward": 0}
    for row in rows:
        for index in range(len(row["history"])):
            messages = base.gold_messages(row["history"], index)
            for kind, tok in tokenizers.items():
                rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                size = len(tok.encode(rendered, add_special_tokens=False))
                maximum[kind] = max(maximum[kind], size)
                available = limits[kind] - (config["response_tokens"] if kind == "actor" else 1)
                if max_input_tokens is not None and kind == "actor":
                    available = min(available, max_input_tokens)
                if size > available:
                    raise ValueError(f"{kind} gold-history overflow: dialogue {row['id']} turn {index}: {size}>{available}")
    return tokenizers, limits, {"gold_history_max_tokens": maximum, "context_windows": limits,
                                "dynamic_feedback_and_reward_inputs": "checked_before_every_call; no_truncation"}


class LocalBackend:
    def __init__(self, args, lock, tokenizers, limits):
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
        self.torch, self.args, self.tokenizers, self.limits = torch, args, tokenizers, limits
        self.models = {}
        self.loading = {}
        for kind, cls in (("actor", AutoModelForCausalLM), ("reward", AutoModelForSequenceClassification)):
            print(f"Loading frozen {kind}: {lock[kind]['repo']} on {args.device}", flush=True)
            model, info = cls.from_pretrained(
                lock[kind]["path"], local_files_only=True, trust_remote_code=False,
                torch_dtype=base.dtype_from_name(args.dtype, torch), device_map=args.device,
                attn_implementation="sdpa", output_loading_info=True,
            )
            unexpected = {key: info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if info.get(key)}
            if unexpected:
                raise RuntimeError(f"Incomplete {kind} checkpoint loading: {unexpected}")
            model.eval().requires_grad_(False)
            if kind == "reward" and (model.config.num_labels != 1 or tuple(model.score.weight.shape) != (1, 4096)):
                raise ValueError("Expected one trained reward logit")
            self.models[kind] = model
            self.loading[kind] = {"passed": True, "model_class": type(model).__name__,
                                  "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}

    def __call__(self, request):
        from transformers import set_seed
        torch = self.torch
        kind = "actor" if request["kind"] == "generate" else "reward"
        model, tok = self.models[kind], self.tokenizers[kind]
        rendered = tok.apply_chat_template(request["messages"], tokenize=False, add_generation_prompt=kind == "actor")
        encoded = tok(rendered, return_tensors="pt", add_special_tokens=False)
        length = int(encoded["input_ids"].shape[1])
        limit = self.limits[kind] - (request["max_new_tokens"] if kind == "actor" else 0)
        if self.args.max_input_tokens is not None and kind == "actor":
            limit = min(limit, self.args.max_input_tokens)
        if length > limit:
            raise ValueError(f"{kind} {request['stage']} context overflow: {length}>{limit}; input preserved")
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        base.synchronize_if_cuda(torch, self.args.device)
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                if kind == "actor":
                    set_seed(request["seed"])
                    output = model.generate(**encoded, do_sample=True, temperature=request["temperature"],
                                            top_p=request["top_p"], top_k=0,
                                            max_new_tokens=request["max_new_tokens"],
                                            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                                            use_cache=True)
                    tokens = output[0, length:]
                    result = {"message": tok.decode(tokens, skip_special_tokens=True).strip(),
                              "completion_tokens": int(tokens.shape[0]),
                              "hit_token_cap": int(tokens.shape[0]) == request["max_new_tokens"]}
                else:
                    logits = model(**encoded, use_cache=False).logits.float()
                    if tuple(logits.shape) != (1, 1):
                        raise ValueError(f"Expected one raw reward logit, got {tuple(logits.shape)}")
                    score = float(logits[0, 0].item())
                    if not math.isfinite(score):
                        raise ValueError("Nonfinite reward-model output")
                    result = {"score": score}
            base.synchronize_if_cuda(torch, self.args.device)
        except RuntimeError:
            if self.args.device.startswith("cuda"):
                torch.cuda.empty_cache()
            raise
        return {**result, "prompt_tokens": length, "original_prompt_tokens": length,
                "input_truncated": False, "latency_seconds": time.perf_counter() - started}
