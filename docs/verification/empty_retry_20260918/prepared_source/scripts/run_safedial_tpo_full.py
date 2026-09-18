#!/usr/bin/env python3
"""Full TPO entrypoint: preserve gold history within the reward model's native 8K window.

The validated smoke entrypoint and runtime remain unchanged. This entrypoint
pins its context policy and source hash separately and tests one 8K reward
forward after both models load, before the first missing benchmark call.
"""
import json
import math
import sys
import time
from pathlib import Path

import run_safedial_baseline as base
import run_safedial_tpo as smoke
from run_safedial_tpo import generate_selected, single_writer
from safedial_tpo_artifacts import validate_lock
from safedial_tpo_runtime import LocalBackend

REWARD_CONTEXT_TOKENS = 8192
CONTEXT_POLICY = {
    "reward_context_tokens": REWARD_CONTEXT_TOKENS,
    "limit_source": "pinned_model_max_position_embeddings; override_tokenizer_4096_metadata",
    "input_truncation": False,
    "startup_probe": "one_synthetic_reward_forward_at_limit_with_both_models_resident; separate_costs",
}


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not any(a == "--output-dir" or a.startswith("--output-dir=") for a in argv):
        argv += ["--output-dir", "outputs/safedial_baseline/tpo_zephyr_full"]
    return smoke.parse_args(argv)


def implementation_hashes():
    return {**smoke.implementation_hashes(), Path(__file__).name: base.file_sha256(Path(__file__))}


def manifest_for(args, selected, lock):
    manifest = smoke.manifest_for(args, selected, lock)
    manifest.update(implementation_sha256=implementation_hashes(), full_run_context=CONTEXT_POLICY.copy())
    return manifest


def reward_context_limit(model_limit):
    if type(model_limit) is not int or model_limit < REWARD_CONTEXT_TOKENS:
        raise ValueError(f"Reward checkpoint does not support {REWARD_CONTEXT_TOKENS} positions: {model_limit}")
    return REWARD_CONTEXT_TOKENS


def tokenizers_and_preflight(lock, rows, config, max_input_tokens=None):
    from transformers import AutoConfig, AutoTokenizer
    tokenizers, limits, declared = {}, {}, {}
    for kind in ("actor", "reward"):
        path = lock[kind]["path"]
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        cfg = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        if not tok.chat_template:
            raise ValueError(f"{kind} has no chat template")
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        declared[kind] = tok.model_max_length
        if kind == "reward":
            limits[kind] = reward_context_limit(cfg.max_position_embeddings)
            tok.model_max_length = limits[kind]  # Metadata only; tokenization/truncation settings unchanged.
        else:
            limits[kind] = min(v for v in (cfg.max_position_embeddings, tok.model_max_length)
                               if isinstance(v, int) and 0 < v < 1_000_000)
        tokenizers[kind] = tok
    maximum = {"actor": 0, "reward": 0}
    longest = {}
    for row in rows:
        for index in range(len(row["history"])):
            messages = base.gold_messages(row["history"], index)
            for kind, tok in tokenizers.items():
                rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                size = len(tok.encode(rendered, add_special_tokens=False))
                if size > maximum[kind]:
                    maximum[kind] = size
                    longest[kind] = {"dialogue_id": row["id"], "turn_index": index}
                available = limits[kind] - (config["response_tokens"] if kind == "actor" else 1)
                if max_input_tokens is not None and kind == "actor":
                    available = min(available, max_input_tokens)
                if size > available:
                    raise ValueError(f"{kind} gold-history overflow: dialogue {row['id']} turn {index}: {size}>{available}")
    return tokenizers, limits, {"gold_history_max_tokens": maximum, "context_windows": limits,
                              "longest_gold_history": longest, "declared_tokenizer_max_tokens": declared,
                              "dynamic_feedback_and_reward_inputs": "checked_before_every_call; no_truncation"}


def probe_reward_context(backend, tokens):
    """Capacity/finite-output check; synthetic score is never an optimization reward."""
    torch = backend.torch
    model = backend.models["reward"]
    if tokens > model.config.max_position_embeddings:
        raise ValueError("Context probe exceeds reward model configuration")
    token = backend.tokenizers["reward"].encode("test", add_special_tokens=False)[0]
    inputs = torch.full((1, tokens), token, dtype=torch.long, device=model.device)
    attention = torch.ones_like(inputs)
    base.synchronize_if_cuda(torch, backend.args.device)
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(input_ids=inputs, attention_mask=attention, use_cache=False).logits.float()
    base.synchronize_if_cuda(torch, backend.args.device)
    if tuple(logits.shape) != (1, 1) or not math.isfinite(float(logits[0, 0].item())):
        raise ValueError("Reward context probe did not produce one finite scalar")
    return {"passed": True, "sequence_tokens": tokens, "synthetic": True,
            "not_a_benchmark_reward": True, "latency_seconds": time.perf_counter() - started,
            "raw_score": float(logits[0, 0].item())}


def main(argv=None):
    args = parse_args(argv)
    rows = base.load_jsonl(args.dataset)
    base.validate_dataset(rows, args.dataset)
    selected = base.select_dialogues(rows, args)
    if not selected or len({r["id"] for r in selected}) != len(selected):
        raise ValueError("Selection must be nonempty with unique IDs")
    print("Checking pinned actor/reward artifacts", flush=True)
    lock = validate_lock(args.artifact_lock)
    manifest = manifest_for(args, selected, lock)
    tokenizers, limits, preflight = tokenizers_and_preflight(lock, selected, manifest["defense"], args.max_input_tokens)
    count = sum(len(r["history"]) for r in selected)
    print(f"Validated {len(selected)} dialogues / {count} turns / {count * args.sample_size * (args.max_iters + 1)} candidates", flush=True)
    print(json.dumps(preflight), flush=True)
    if args.validate_only:
        return 0
    with single_writer(args.output_dir):
        if not (args.output_dir / "run_config.json").exists() and any((args.output_dir / name).exists() for name in ("answers.jsonl", "turns.jsonl", "events.jsonl")):
            raise ValueError("Existing output has no TPO manifest")
        base.ensure_manifest(args.output_dir / "run_config.json", manifest)
        (args.output_dir / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
        import torch
        started = time.perf_counter()
        backend = None
        gpu = args.device.startswith("cuda")
        gpu_ready = False
        gpu_name = None
        failure = None
        stage = "device_setup"
        actual_calls = {"generate": 0, "reward": 0}
        startup_reward_probe_calls = 0

        def execute(request):
            nonlocal backend, startup_reward_probe_calls
            if backend is None:
                backend = LocalBackend(args, lock, tokenizers, limits)
                (args.output_dir / "model_loading.json").write_text(json.dumps(backend.loading, indent=2) + "\n")
                startup_reward_probe_calls += 1
                probe = {"passed": False, "sequence_tokens": REWARD_CONTEXT_TOKENS}
                try:
                    probe = probe_reward_context(backend, REWARD_CONTEXT_TOKENS)
                except (RuntimeError, ValueError) as exc:
                    probe["error"] = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    (args.output_dir / "context_probe.json").write_text(json.dumps(probe, indent=2) + "\n")
                print(f"Reward context probe PASS: {REWARD_CONTEXT_TOKENS} tokens", flush=True)
            actual_calls[request["kind"]] += 1
            return backend(request)

        status = 2
        try:
            if gpu:
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA unavailable; submit the prepared GPU batch manually")
                # Explicit cuda:0 bypasses the lazy initialization performed by
                # current_device(). The allocator must exist before resetting it.
                torch.cuda.init()
                device = torch.device(args.device)
                torch.cuda.set_device(device.index if device.index is not None else torch.cuda.current_device())
                gpu_name = torch.cuda.get_device_name(args.device)
                torch.cuda.reset_peak_memory_stats(args.device)
                gpu_ready = True
                print(f"CUDA ready: {args.device} / {gpu_name} / torch {torch.__version__}", flush=True)
            stage = "generation"
            status = generate_selected(args, manifest, selected, execute)
            return status
        except Exception as exc:
            failure = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
            print(f"TPO {stage} failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            raise
        finally:
            stats = {"elapsed_seconds": time.perf_counter() - started, "device": args.device,
                     "gpu_name": gpu_name, "cuda_ready": gpu_ready, "failure": failure,
                     "torch_version": torch.__version__, "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(args.device) if gpu_ready else None,
                     "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(args.device) if gpu_ready else None,
                     "scope": "current_invocation_includes_loading_and_context_probe; excludes_hashing_and_tokenizer_preflight", "actual_calls": actual_calls,
                     "startup_reward_probe_calls": startup_reward_probe_calls,
                     "model_loading_performed": backend is not None, "exit_code": status}
            # Preserve measurements from the real generation invocation on a no-op resume.
            if backend is not None or not (args.output_dir / "runtime_stats.json").exists():
                (args.output_dir / "runtime_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
            base.append_jsonl(args.output_dir / "invocations.jsonl", [stats])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)
