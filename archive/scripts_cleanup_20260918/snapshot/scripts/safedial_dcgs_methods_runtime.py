"""Pinned frozen actor/value and contextual token-critic execution for RDCGS."""
import json
import math
import time
from pathlib import Path

import run_safedial_baseline as base
from safedial_generation_retry import generation_diagnostics
from safedial_dcgs_methods import Prompts, defense_config, history_from_messages, format_dcgs_observation
from src.value.ll_token_critic import LLTokenCritic, make_head, PREFIX_TEMPLATE
from safedial_dcgs_native_context import tokenize_action_span

DEFAULT_LOCK = Path("configs/safedial/dcgs_two_stage_aug11.lock.json")
HL_HEADS = ("q_mlp_head_state_dict", "v_mlp_head_state_dict", "q_min_mlp_head_state_dict",
            "v_min_mlp_head_state_dict", "regret_mlp_head_state_dict")


def load_heads(lock, device="cpu", method="rdcgs"):
    import torch
    high = torch.load(lock["high_level"]["path"], map_location="cpu", weights_only=True)
    low = torch.load(lock["token_critic"]["path"], map_location="cpu", weights_only=True)
    if high.get("hidden_size") != 4096 or high.get("use_regret_critic") is not True or high.get("dtype") != "torch.bfloat16":
        raise ValueError("Unexpected WildJailbreak high-level checkpoint")
    if (low.get("hidden_size"), low.get("mlp_width_mult"), low.get("objective")) != (4096, 2.0, "shapley"):
        raise ValueError("Unexpected WildJailbreak token checkpoint")
    if not isinstance(low.get("n_examples"), int) or low["n_examples"] <= 0:
        raise ValueError("Token checkpoint has no recorded training examples")
    heads = {}
    required = ("q_mlp_head_state_dict",) + (("regret_mlp_head_state_dict",) if defense_config(method)["use_regret"] else ())
    for name in required + ("harm_head", "follow_head"):
        is_high = name in HL_HEADS
        state = (high if is_high else low)[name]
        dtype = torch.bfloat16 if is_high else torch.float32
        head = make_head(4096, 1.0 if is_high else 2.0).to(dtype=dtype)
        head.load_state_dict(state, strict=True)
        for key, value in head.state_dict().items():
            if not torch.equal(value, state[key].to(dtype=dtype)) or not torch.isfinite(value).all():
                raise ValueError(f"Invalid or incorrectly loaded head: {name}/{key}")
        heads[name] = head.eval().requires_grad_(False).to(device)
    return heads, low


def validate_lock(path, method="rdcgs"):
    lock = json.loads(Path(path).read_text())
    if lock.get("schema_version") != 1 or lock.get("method") != "RDCGS_two_stage":
        raise ValueError("Unknown DCGS artifact lock")
    actor = lock["actor"]
    if (actor["repo"], actor["revision"]) != ("HuggingFaceH4/zephyr-7b-beta", "892b3d7a7b1cf10c7a701c60881cd93df615734c"):
        raise ValueError("Unexpected DCGS backbone")
    folder = Path(actor["path"])
    index = json.loads((folder / "model.safetensors.index.json").read_text())
    required = set(index["weight_map"].values()) | {"model.safetensors.index.json", "config.json", "tokenizer.json", "tokenizer_config.json"}
    if not required.issubset(actor["sha256"]):
        raise ValueError("Incomplete actor hash manifest")
    for name, digest in actor["sha256"].items():
        if Path(name).name != name or base.file_sha256(folder / name) != digest:
            raise ValueError(f"Actor artifact hash mismatch: {name}")
    for key in ("high_level", "token_critic"):
        if base.file_sha256(Path(lock[key]["path"])) != lock[key]["sha256"]:
            raise ValueError(f"{key} checkpoint hash mismatch")
    if lock["token_critic"]["sha256"] != "a0a4cc34e4a096122adeb21d5cff2acfe6cf760c22f70b9a76733555b9e1313f":
        raise ValueError("Token checkpoint differs from the user-selected artifact")
    heads, _ = load_heads(lock, method=method)
    del heads
    return lock


def generation_inputs(tok, prompt, prefill):
    # Match the repository's raw-template encoding and separately encoded prefill.
    ids = tok.encode(prompt, add_special_tokens=True)
    ids += tok.encode(prefill, add_special_tokens=False)
    return ids


def tokenizer_and_preflight(lock, rows, method="rdcgs"):
    from transformers import AutoConfig, AutoTokenizer
    cfg = AutoConfig.from_pretrained(lock["actor"]["path"], local_files_only=True, trust_remote_code=False)
    tok = AutoTokenizer.from_pretrained(lock["actor"]["path"], local_files_only=True, trust_remote_code=False)
    if not tok.is_fast or cfg.hidden_size != 4096 or cfg.max_position_embeddings < 32768:
        raise ValueError("Fast offset tokenizer and 4096-wide backbone required")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    policy, prompts = defense_config(method), Prompts()
    maxima = {"intent_generation": 0, "response_generation_placeholder": 0, "hl_placeholder": 0, "ll_placeholder": 0}
    for row in rows:
        for i in range(len(row["history"])):
            history = history_from_messages(base.gold_messages(row["history"], i))
            observation = format_dcgs_observation(history)
            for side in ([False, True] if policy["use_regret"] else [False]):
                ids = generation_inputs(tok, prompts.intent(history, side), "Here are 5 candidate beliefs about the user's desires:\n\n1. ")
                maxima["intent_generation"] = max(maxima["intent_generation"], len(ids))
                if len(ids) + policy["intent_tokens_per_side"] > cfg.max_position_embeddings:
                    raise ValueError(f"Intent prompt overflow: {row['id']}/{i}")
            ids = generation_inputs(tok, prompts.response(history, "candidate intent"), "[RESPONSE]")
            maxima["response_generation_placeholder"] = max(maxima["response_generation_placeholder"], len(ids))
            if len(ids) + policy["response_tokens"] > cfg.max_position_embeddings:
                raise ValueError(f"Response prompt overflow: {row['id']}/{i}")
            high_len = len(tok.encode(f"Observation: {observation}\nHigh-Level Context: candidate intent", add_special_tokens=True))
            maxima["hl_placeholder"] = max(maxima["hl_placeholder"], high_len)
            if high_len > policy["hl_context_limit"]:
                raise ValueError(f"HL gold prefix exceeds 32768-token limit: {row['id']}/{i} ({high_len})")
            prefix = PREFIX_TEMPLATE.format(observation=observation, selected_belief="candidate intent")
            encoded, _, _ = tokenize_action_span(tok, prefix, prefix + "candidate response")
            maxima["ll_placeholder"] = max(maxima["ll_placeholder"], encoded["input_ids"].shape[1])
    # Size a synthetic startup capacity check for the longest observed prefix plus
    # both generation budgets and token-boundary slack. Actual inputs still have
    # exact runtime guards; this estimate is not a proof of every dynamic length.
    probe_tokens = ((max(maxima.values()) + policy["intent_tokens_per_side"]
                     + policy["response_tokens"] + 128 + 511) // 512) * 512
    if probe_tokens > min(policy["hl_context_limit"], policy["ll_context_limit"], cfg.max_position_embeddings):
        raise ValueError("Insufficient native context headroom for full-benchmark preflight")
    return tok, int(cfg.max_position_embeddings), {"passed": True, "dialogues": len(rows),
        "turns": sum(len(r["history"]) for r in rows), "max_input_tokens": maxima,
        "actor_context_tokens": int(cfg.max_position_embeddings), "hl_context_tokens": 32768, "ll_context_tokens": 32768,
        "startup_probe_tokens": probe_tokens,
        "dynamic_inputs": "every actual prompt, belief and action checked at runtime without truncation"}


class LocalBackend:
    def __init__(self, args, lock, tokenizer, actor_limit):
        import torch
        from transformers import AutoModelForCausalLM
        from src.utils.llm_utils import SuppressWordsLogitsProcessor
        self.torch, self.args, self.tokenizer, self.actor_limit = torch, args, tokenizer, actor_limit
        self.ll_device = getattr(args, 'll_device', args.device)
        self.loading = {}
        models = {}
        # Keep each critic's expected residual precision; do not convert the shared actor.
        for name, dtype in (("actor_and_hl", torch.bfloat16), ("ll_encoder", torch.float16)):
            device = args.device if name == 'actor_and_hl' else self.ll_device
            model, info = AutoModelForCausalLM.from_pretrained(lock["actor"]["path"],
                local_files_only=True, trust_remote_code=False, torch_dtype=dtype,
                device_map=device, attn_implementation="sdpa", output_loading_info=True)
            if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
                raise ValueError(f"Incomplete {name} loading: {info}")
            model.eval().requires_grad_(False)
            if model.dtype != dtype:
                raise ValueError(f"Wrong {name} backbone precision")
            models[name] = model
            self.loading[name] = {"passed": True, "dtype": str(dtype), "device": device,
                                  "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
        self.actor = models["actor_and_hl"]
        heads, low = load_heads(lock, 'cpu', args.method)
        for name, head in heads.items():
            head.to(args.device if name in HL_HEADS else self.ll_device)
        self.q, self.regret = heads["q_mlp_head_state_dict"], heads.get("regret_mlp_head_state_dict")
        self.ll = LLTokenCritic(heads["harm_head"], heads["follow_head"], models["ll_encoder"].model,
                               tokenizer, self.ll_device, low["objective"], int(low["k"]))
        self.loading["heads"] = {"passed": True, "required_heads_strictly_loaded": sorted(heads),
                                 "method": args.method,
                                 "devices": {name: args.device if name in HL_HEADS else self.ll_device for name in heads},
                                 "trainable_parameters": sum(p.numel() for head in heads.values() for p in head.parameters() if p.requires_grad),
                                 "token_checkpoint_sha256": lock["token_critic"]["sha256"],
                                 "high_level_checkpoint_sha256": lock["high_level"]["sha256"]}
        self.suppress = SuppressWordsLogitsProcessor(tokenizer, ["Example", "example", "Examples", "examples"])

    def probe_context(self, tokens):
        """Synthetic forwards with both backbones resident; no benchmark calls."""
        torch = self.torch
        if not 1 <= tokens <= min(self.actor_limit, 32768):
            raise ValueError("Invalid capacity probe length")
        started = time.perf_counter()
        with torch.inference_mode():
            ids = torch.full((1, tokens), self.tokenizer.eos_token_id,
                             dtype=torch.long, device=self.args.device)
            mask = torch.ones_like(ids)
            # Include actor logits and KV allocation, then release before LL.
            output = self.actor(input_ids=ids, attention_mask=mask, use_cache=True)
            if not torch.isfinite(output.logits[:, -1]).all():
                raise ValueError("Nonfinite actor capacity probe")
            del output
            hidden = self.ll.encoder(input_ids=ids.to(self.ll_device), attention_mask=mask.to(self.ll_device), use_cache=False).last_hidden_state
            score = self.ll._score_action_hiddens(hidden[0, -1024:].float())
            if not torch.isfinite(score).all():
                raise ValueError("Nonfinite token critic capacity probe")
            del hidden, score
        base.synchronize_if_cuda(torch, self.args.device)
        base.synchronize_if_cuda(torch, self.ll_device)
        return {"passed": True, "sequence_tokens": tokens,
                "elapsed_seconds": time.perf_counter() - started,
                "scope": "synthetic_actor_and_ll_forwards_with_both_backbones_resident; excluded_from_benchmark_calls"}

    def __call__(self, request):
        from transformers import set_seed, LogitsProcessorList
        torch, tok = self.torch, self.tokenizer
        device = self.ll.device if request['kind'] == 'll_score' else self.args.device
        base.synchronize_if_cuda(torch, device)
        started = time.perf_counter()
        with torch.inference_mode():
            if request["kind"] == "generate":
                ids = generation_inputs(tok, request["prompt"], request["prefill"])
                length = len(ids)
                if length + request["max_new_tokens"] > self.actor_limit:
                    raise ValueError("Generation input exceeds actor context; refusing truncation")
                encoded = torch.tensor([ids], dtype=torch.long, device=self.args.device)
                set_seed(request["seed"])
                output = self.actor.generate(input_ids=encoded, attention_mask=torch.ones_like(encoded),
                    do_sample=True, temperature=request["temperature"], top_p=request["top_p"], top_k=request["top_k"],
                    max_new_tokens=request["max_new_tokens"], pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                    logits_processor=LogitsProcessorList([self.suppress]), use_cache=True)
                tokens = output[0, length:]
                result = {"message": tok.decode(tokens, skip_special_tokens=True), "completion_tokens": len(tokens),
                          "hit_token_cap": len(tokens) == request["max_new_tokens"],
                          **generation_diagnostics(tok, tokens, request["max_new_tokens"])}
            elif request["kind"] == "hl_score":
                text = f"Observation: {request['observation']}\nHigh-Level Context: {request['belief']}"
                encoded = tok(text, add_special_tokens=True, truncation=False, return_tensors="pt")
                length = encoded["input_ids"].shape[1]
                if length > defense_config(self.args.method)["hl_context_limit"]:
                    raise ValueError(f"HL context {length}>32768; refusing truncation")
                encoded = {k: v.to(self.args.device) for k, v in encoded.items()}
                hidden = self.actor.model(**encoded, use_cache=False).last_hidden_state
                pooled = hidden.mean(dim=1).to(dtype=torch.bfloat16)
                if request["use_regret"] != (self.regret is not None):
                    raise ValueError("Request critic heads differ from configured method")
                result = {"q": float(self.q(pooled).item())}
                if request["use_regret"]:
                    result["regret"] = float(self.regret(pooled).item())
            elif request["kind"] == "ll_score":
                prefix = PREFIX_TEMPLATE.format(observation=request["observation"], selected_belief=request["belief"])
                encoded, start, end = tokenize_action_span(tok, prefix, prefix + request["action"])
                length = encoded["input_ids"].shape[1]
                encoded = {k: v.to(device) for k, v in encoded.items()}
                hidden = self.ll.encoder(**encoded, use_cache=False).last_hidden_state
                if hidden.shape[-1] != 4096:
                    raise ValueError("Incorrect LL residual width")
                score = float(self.ll._score_action_hiddens(hidden[0, start:end].float()).item())
                if not math.isfinite(score):
                    raise ValueError("Nonfinite token critic score")
                result = {"score": score, "action_start": start, "action_end": end}
            else:
                raise ValueError("Unknown DCGS call kind")
        base.synchronize_if_cuda(torch, device)
        return {**result, "prompt_tokens": int(length), "original_prompt_tokens": int(length),
                "input_truncated": False, "latency_seconds": time.perf_counter() - started}
