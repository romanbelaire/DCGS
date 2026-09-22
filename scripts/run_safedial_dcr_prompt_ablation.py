#!/usr/bin/env python3
"""Isolated DCR diagnostic: training chat template versus raw history text.

Derived from the frozen original DCR runner so job 271289 remains auditable.
Raw mode joins the same message contents with two newlines, with no role labels,
assistant cue, replacement chat template, or automatically added special tokens.
"""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace

import run_safedial_baseline as baseline
from prepare_safedial_dcr import BASE, BASE_REVISION, DEFAULT_LOCK, load_tokenizer, validate_lock, validate_lock_data
from safedial_adapters import attach_adapter, verify_adapter_effect
from validate_safedial_generation import validate_run

SEED_MODEL_ID = "qwen2.5-1.5b-dcr-sft-171151"
SOURCES = ("run_safedial_dcr_prompt_ablation.py", "prepare_safedial_dcr.py", "run_safedial_baseline.py",
           "safedial_adapters.py", "validate_safedial_generation.py", "judge_safedial.py")


def identities(prompt_format):
    if prompt_format not in ("training", "raw"):
        raise ValueError("DCR prompt format must be training or raw")
    return (f"safedial_dcr_prompt_ablation_{prompt_format}_v1",
            f"{SEED_MODEL_ID}-prompt-{prompt_format}-v1")


def prompt_policy(prompt_format):
    identities(prompt_format)
    return {"tokenizer": "uploaded_adapter", "prompt_format": prompt_format,
            "template": "chat_template.jinja" if prompt_format == "training" else None,
            "serialization": "training_template" if prompt_format == "training" else "content_join_double_newline",
            "system_message": None, "truncation": "reject_overflow", "add_special_tokens": False}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--prompt-format", choices=("training", "raw"), default="training")
    parser.add_argument("--dataset", type=Path, default=baseline.DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--ids", help="Comma-separated dialogue IDs; smoke launcher selects 1.")
    selection.add_argument("--per-task", type=int)
    selection.add_argument("--limit", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true", help="Check artifacts and every selected prompt on CPU.")
    mode.add_argument("--audit-only", action="store_true", help="Validate completed saved outputs without inference.")
    parser.add_argument("--require-gpu", action="store_true", help="Require saved GPU memory evidence when auditing.")
    parser.add_argument("--retry-errors", action="store_true", help="Regenerate entire failed dialogues, retaining an attempt journal.")
    args = parser.parse_args(argv)
    if args.require_gpu and not args.audit_only:
        parser.error("--require-gpu is only valid with --audit-only")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    # These method/protocol settings are deliberately fixed.
    args.model = BASE
    args.protocol, args.model_id = identities(args.prompt_format)
    args.revision = BASE_REVISION
    args.adapter_lock = None
    args.temperature = 0.0
    args.top_p = 1.0
    args.num_choices = 1
    args.max_input_tokens = None
    return args


def source_hashes():
    directory = Path(__file__).resolve().parent
    return {name: baseline.file_sha256(directory / name) for name in SOURCES}


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def render_prompt(tokenizer, history, index, prompt_format):
    messages = baseline.gold_messages(history, index)
    identities(prompt_format)
    if prompt_format == "training":
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        rendered = "\n\n".join(message["content"] for message in messages)
    return messages, rendered


def preflight(selected, tokenizer, config, max_new_tokens, prompt_format):
    limit = baseline.effective_context_limit(SimpleNamespace(config=SimpleNamespace(**config)),
                                             tokenizer, None, max_new_tokens)
    maximum, turns, overflows = 0, 0, []
    for row in selected:
        for index in range(len(row["history"])):
            _, rendered = render_prompt(tokenizer, row["history"], index, prompt_format)
            ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
            if not ids or min(ids) < 0 or max(ids) >= config["vocab_size"]:
                raise ValueError("DCR prompt has invalid token IDs for the standard base")
            maximum = max(maximum, len(ids))
            turns += 1
            if len(ids) > limit:
                overflows.append((row["id"], index, len(ids)))
    if overflows:
        raise ValueError(f"DCR context overflow; history will not be truncated: {overflows[:10]}")
    return {"passed": True, "prompt_format": prompt_format, "dialogues": len(selected), "turns": turns,
            "max_prompt_tokens": maximum, "input_token_limit": limit,
            "max_new_tokens": max_new_tokens, "truncated_turns": 0,
            "system_message": None, "history": "dataset_reference_assistant"}


def manifest_for(args, selected, lock):
    manifest = baseline.manifest_for(args, args.dataset, selected)
    manifest.update({
        "protocol": args.protocol, "adapter": lock,
        "prompt_policy": prompt_policy(args.prompt_format), "prompt_format": args.prompt_format,
        "seed_model_id": SEED_MODEL_ID,
        "source_sha256": source_hashes(),
        "package_versions": {name: importlib.metadata.version(name)
                             for name in ("torch", "transformers", "peft", "tokenizers")},
    })
    return manifest


def audit_run(output_dir, require_gpu=False):
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "run_config.json").read_text())
    prompt_format = manifest.get("prompt_format")
    protocol, model_id = identities(prompt_format)
    if manifest.get("prompt_policy") != prompt_policy(prompt_format) or manifest.get("seed_model_id") != SEED_MODEL_ID:
        raise ValueError("DCR prompt policy mismatch")
    if (manifest.get("protocol"), manifest.get("model"), manifest.get("model_id"),
            manifest.get("revision"), manifest.get("temperature"), manifest.get("num_choices")) != (
        protocol, BASE, model_id, BASE_REVISION, 0.0, 1
    ):
        raise ValueError("DCR run identity/decoding mismatch")
    if manifest.get("source_sha256") != source_hashes():
        raise ValueError("DCR runtime source hash mismatch; audit with the original source version")
    lock = validate_lock_data(manifest["adapter"])
    tokenizer = load_tokenizer(lock)
    report = validate_run(output_dir, require_adapter=require_gpu)
    effect = json.loads((output_dir / "adapter_validation.json").read_text())
    if not effect.get("passed") or effect.get("max_abs_logit_difference", 0) <= 1e-6:
        raise ValueError("DCR adapter effect verification missing or failed")
    rows = {row["id"]: row for row in baseline.load_jsonl(Path(manifest["dataset"]))}
    metrics = []
    for record in baseline.load_jsonl(output_dir / "turns.jsonl"):
        _, rendered = render_prompt(tokenizer, rows[record["dialogue_id"]]["history"], record["turn_index"], prompt_format)
        count = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
        if (record.get("rendered_prompt_sha256") != text_hash(rendered)
                or record.get("prompt_tokens") != count or record.get("original_prompt_tokens") != count
                or record.get("rendered_prompt") != rendered
                or record.get("protocol") != protocol or record.get("prompt_format") != prompt_format
                or record.get("input_truncated")):
            raise ValueError("DCR rendered prompt audit failed")
        ids = record.get("completion_token_ids", [])
        if (not ids or len(ids) != record["completion_tokens"] or len(ids) > manifest["max_new_tokens"]
                or tokenizer.decode(ids, skip_special_tokens=True).strip() != record["generated_response"]):
            raise ValueError("DCR generated token audit failed")
        expected_seed = baseline.turn_seed(manifest["seed"], SEED_MODEL_ID, record["dialogue_id"], 0, record["turn_index"])
        if record.get("seed") != expected_seed:
            raise ValueError("DCR paired turn seed mismatch")
        metrics.append({"dialogue_id": record["dialogue_id"], "turn_index": record["turn_index"],
                        "completion_tokens": len(ids), "hit_token_cap": len(ids) == manifest["max_new_tokens"],
                        "last_token_is_eos": ids[-1] == tokenizer.eos_token_id,
                        "cjk_characters": len(re.findall("[\u4e00-\u9fff]", record["generated_response"])),
                        "generated_role_lines": len(re.findall(r"(?m)^(?:USER|ASSISTANT|SYSTEM):", record["generated_response"]))})
    return {**report, "protocol": protocol, "prompt_format": prompt_format,
            "prompt_format_verified": True, "turn_metrics": metrics}


def generate_rendered(*, model, tokenizer, rendered, args, seed, torch_module, input_limit):
    """Tokenize exactly the audited text; raw mode never invokes a chat template."""
    from transformers import set_seed

    set_seed(seed)
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    count = int(encoded["input_ids"].shape[1])
    if count > input_limit:
        raise ValueError("DCR prompt exceeds preflight context limit")
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    baseline.synchronize_if_cuda(torch_module, args.device)
    started = time.perf_counter()
    with torch_module.inference_mode():
        output_ids = model.generate(
            **encoded, max_new_tokens=args.max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    baseline.synchronize_if_cuda(torch_module, args.device)
    elapsed = time.perf_counter() - started
    ids = output_ids[0, count:].tolist()
    return {"message": tokenizer.decode(ids, skip_special_tokens=True).strip(),
            "prompt_tokens": count, "original_prompt_tokens": count,
            "completion_tokens": len(ids), "completion_token_ids": ids,
            "latency_seconds": elapsed, "input_truncated": False}


@contextmanager
def output_lock(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another DCR process owns this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_generation(args, selected, lock, tokenizer, checked):
    import torch
    from transformers import AutoModelForCausalLM

    manifest = manifest_for(args, selected, lock)
    baseline.ensure_manifest(args.output_dir / "run_config.json", manifest)
    answers_path, turns_path = args.output_dir / "answers.jsonl", args.output_dir / "turns.jsonl"
    completed = baseline.load_latest_answers(answers_path, args.model_id)
    pending = [row for row in selected if row["id"] not in completed
               or (args.retry_errors and baseline.answer_has_error(completed[row["id"]]))]
    if not pending:
        # Preserve original GPU/runtime evidence on a no-op resume.
        print(json.dumps(audit_run(args.output_dir, args.device.startswith("cuda")), indent=2))
        return 0
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; submit the prepared smoke launcher on a GPU node")
        torch.cuda.set_device(torch.device(args.device))
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(args.device)
    started = time.perf_counter()
    (args.output_dir / "preflight.json").write_text(json.dumps(checked, indent=2) + "\n")
    print(f"Loading standard {BASE} at {BASE_REVISION}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        lock["base_path"], local_files_only=True, trust_remote_code=False,
        torch_dtype=baseline.dtype_from_name(args.dtype, torch), device_map=args.device,
    )
    model = attach_adapter(model, lock)
    effect = verify_adapter_effect(model, tokenizer, torch)
    (args.output_dir / "adapter_validation.json").write_text(json.dumps(effect, indent=2) + "\n")
    print(f"DCR adapter active; max logit difference {effect['max_abs_logit_difference']:.6g}", flush=True)
    # Override inherited generation defaults with the benchmark's greedy policy.
    model.generation_config.do_sample = False
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 50
    run_id = baseline.stable_id(manifest["dataset_sha256"], args.model_id, args.seed, manifest["selected_ids"])
    for number, row in enumerate(pending, 1):
        records, generated_turns = [], []
        answer_id = baseline.stable_id(run_id, row["id"])
        for index, source_turn in enumerate(row["history"]):
            messages, rendered = render_prompt(tokenizer, row["history"], index, args.prompt_format)
            seed = baseline.turn_seed(args.seed, SEED_MODEL_ID, row["id"], 0, index)
            error = None
            try:
                result = generate_rendered(model=model, tokenizer=tokenizer, rendered=rendered,
                    args=args, seed=seed, torch_module=torch, input_limit=checked["input_token_limit"])
                if result["input_truncated"]:
                    raise RuntimeError("DCR unexpectedly truncated a preflight-checked prompt")
                if not result["message"].strip():
                    raise ValueError("DCR generated an empty response")
            except (RuntimeError, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                result = {"message": "ERROR", "prompt_tokens": None, "original_prompt_tokens": None,
                          "completion_tokens": 0, "latency_seconds": None, "input_truncated": None}
                print(f"ERROR dialogue={row['id']} turn={index}: {error}", flush=True)
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            generated = {"role": "assistant", "message": result["message"]}
            if error:
                generated["error"] = error
            generated_turns.append(generated)
            record = {
                "run_id": run_id, "answer_id": answer_id, "benchmark": "SafeDialBench",
                "protocol": args.protocol, "model": BASE, "model_id": args.model_id,
                "dialogue_id": row["id"], "task": row["task"], "method": row["method"],
                "scene": row["scene"], "dataset_model_type": row.get("model_type"),
                "choice_index": 0, "turn_index": index, "seed": seed,
                "prompt_history": messages, "rendered_prompt_sha256": text_hash(rendered),
                "rendered_prompt": rendered, "prompt_format": args.prompt_format,
                "user_message": source_turn["user"], "reference_response": source_turn["bot"],
                "generated_response": result["message"], "error": error,
                **{key: value for key, value in result.items() if key != "message"},
            }
            # Append-only evidence survives interrupted dialogues and explicit retries.
            baseline.append_jsonl(args.output_dir / "attempts.jsonl", [{**record, "tstamp": time.time()}])
            records.append(record)
        answer = {"id": row["id"], "task": row["task"], "answer_id": answer_id,
                  "model_id": args.model_id, "method": row["method"],
                  "choices": [{"index": 0, "turns": generated_turns}], "tstamp": time.time()}
        baseline.append_jsonl(turns_path, records)
        baseline.append_jsonl(answers_path, [answer])
        print(f"[{number}/{len(pending)}] saved dialogue {row['id']}", flush=True)
    baseline.compact_answers(answers_path, args.model_id)
    baseline.compact_turns(turns_path)
    runtime = {
        "elapsed_seconds": time.perf_counter() - started, "device": args.device,
        "gpu_name": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(args.device) if args.device.startswith("cuda") else None,
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(args.device) if args.device.startswith("cuda") else None,
        "includes_model_loading": True, "scope": "current_invocation",
    }
    (args.output_dir / "runtime_stats.json").write_text(json.dumps(runtime, indent=2) + "\n")
    report = audit_run(args.output_dir, require_gpu=args.device.startswith("cuda"))
    (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.audit_only:
        print(json.dumps(audit_run(args.output_dir, args.require_gpu), indent=2))
        return 0
    lock = validate_lock(args.artifact_lock)
    rows = baseline.load_jsonl(args.dataset)
    baseline.validate_dataset(rows, args.dataset)
    selected = baseline.select_dialogues(rows, args)
    if len({row["id"] for row in selected}) != len(selected):
        raise ValueError("Duplicate selected dialogue IDs")
    tokenizer = load_tokenizer(lock)
    config = json.loads((Path(lock["base_path"]) / "config.json").read_text())
    checked = preflight(selected, tokenizer, config, args.max_new_tokens, args.prompt_format)
    print(json.dumps(checked, indent=2), flush=True)
    if args.validate_only:
        return 0
    with output_lock(args.output_dir):
        return run_generation(args, selected, lock, tokenizer, checked)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)
