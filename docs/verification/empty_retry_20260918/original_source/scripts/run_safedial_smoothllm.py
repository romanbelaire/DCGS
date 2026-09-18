#!/usr/bin/env python3
"""Generate a separate SmoothLLM SafeDialBench arm; never invoke a judge.

Reuses the native baseline's read-only helpers, without modifying its runner or
the pending CAT configuration. Model-free --validate-only and injected decoder
tests do not need CUDA. The batch script owns the single-writer file lock.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import run_safedial_baseline as base
from safedial_smoothllm import CandidateFailure, defense_config, generate_smoothed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=base.DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/safedial_baseline/smoothllm_zephyr_smoke"))
    parser.add_argument("--model", default=base.DEFAULT_MODEL)
    parser.add_argument("--model-id", default="zephyr-7b-beta-smoothllm")
    parser.add_argument("--revision", default="892b3d7a7b1cf10c7a701c60881cd93df615734c")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-input-tokens", type=int)
    parser.add_argument("--copies", type=int, default=8)
    parser.add_argument("--perturbation-percent", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--ids")
    selection.add_argument("--per-task", type=int)
    selection.add_argument("--limit", type=int)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    # Explicit greedy candidate decoding, one benchmark answer per turn.
    parser.set_defaults(temperature=0.0, top_p=1.0, num_choices=1)
    args = parser.parse_args(argv)
    base.validate_args(args)
    defense_config(args.copies, args.perturbation_percent)
    return args


def manifest_for(args, selected):
    manifest = base.manifest_for(args, args.dataset, selected)
    manifest["defense"] = defense_config(args.copies, args.perturbation_percent)
    manifest["token_accounting"] = "turn_totals_all_copies; individual_costs_in_smoothllm.candidates"
    manifest["implementation_sha256"] = {
        name: base.file_sha256(Path(__file__).parent / name)
        for name in ("run_safedial_smoothllm.py", "safedial_smoothllm.py", "run_safedial_baseline.py")
    }
    return manifest


def pending_rows(selected, answers_path, args):
    completed = base.load_latest_answers(answers_path, args.model_id)
    return [row for row in selected if row["id"] not in completed
            or (args.retry_errors and base.answer_has_error(completed[row["id"]]))]


def generate_dialogues(args, manifest, pending, generate):
    """Dialogue-level append/compaction supports deterministic interrupted resume."""
    run_id = base.stable_id(json.dumps(manifest, sort_keys=True))
    errors = 0
    for number, row in enumerate(pending, 1):
        answer_id = base.stable_id(run_id, row["id"])
        records, official_turns = [], []
        for index, source in enumerate(row["history"]):
            seed = base.turn_seed(args.seed, args.model_id, row["id"], 0, index)
            messages = base.gold_messages(row["history"], index)
            error = None
            try:
                result = generate_smoothed(messages, manifest["defense"], seed, generate)
            except CandidateFailure as exc:
                errors += 1
                error = str(exc)
                result = {"message": "ERROR", "smoothllm": exc.audit,
                          "prompt_tokens": None, "original_prompt_tokens": None,
                          "completion_tokens": None, "latency_seconds": None,
                          "input_truncated": None}
                print(f"ERROR dialogue={row['id']} turn={index}: {error}", file=sys.stderr)
            official = {"role": "assistant", "message": result["message"]}
            if error:
                official["error"] = error
            official_turns.append(official)
            records.append({
                "run_id": run_id, "answer_id": answer_id, "benchmark": "SafeDialBench",
                "protocol": base.PROTOCOL, "model": args.model, "model_id": args.model_id,
                "dialogue_id": row["id"], "task": row["task"], "method": row["method"],
                "scene": row["scene"], "dataset_model_type": row.get("model_type"),
                "choice_index": 0, "turn_index": index, "seed": seed,
                "prompt_history": messages, "user_message": source["user"],
                "reference_response": source["bot"], "generated_response": result["message"],
                "error": error, **{key: value for key, value in result.items() if key != "message"},
            })
        answer = {"id": row["id"], "task": row["task"], "method": row["method"],
                  "answer_id": answer_id, "model_id": args.model_id,
                  "choices": [{"index": 0, "turns": official_turns}], "tstamp": time.time()}
        base.append_jsonl(args.output_dir / "turns.jsonl", records)
        base.append_jsonl(args.output_dir / "answers.jsonl", [answer])
        print(f"[{number}/{len(pending)}] completed dialogue {row['id']} ({len(records)} turns)")
    base.compact_answers(args.output_dir / "answers.jsonl", args.model_id)
    base.compact_turns(args.output_dir / "turns.jsonl")
    return errors


def main(argv=None):
    args = parse_args(argv)
    rows = base.load_jsonl(args.dataset)
    base.validate_dataset(rows, args.dataset)
    selected = base.select_dialogues(rows, args)
    if not selected or len({row["id"] for row in selected}) != len(selected):
        raise ValueError("Selection must be nonempty with unique IDs")
    manifest = manifest_for(args, selected)
    turns = sum(len(row["history"]) for row in selected)
    print(f"Validated {len(selected)} dialogues / {turns} turns / {turns * args.copies} candidate generations")
    print(f"SmoothLLM: {args.copies} copies, {args.perturbation_percent}% random swap, greedy, current user only")
    if args.validate_only:
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base.ensure_manifest(args.output_dir / "run_config.json", manifest)
    pending = pending_rows(selected, args.output_dir / "answers.jsonl", args)
    print(f"Already complete: {len(selected) - len(pending)}; pending: {len(pending)}")
    if not pending:
        base.compact_answers(args.output_dir / "answers.jsonl", args.model_id)
        base.compact_turns(args.output_dir / "turns.jsonl")
        return 0

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; submit the prepared GPU batch manually")
    started = time.perf_counter()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = tokenizer.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=base.dtype_from_name(args.dtype, torch),
        device_map=args.device, trust_remote_code=True,
    ).eval()
    input_limit = base.effective_context_limit(model, tokenizer, args.max_input_tokens, args.max_new_tokens)

    def generate(messages, seed):
        try:
            return base.generate_one(model=model, tokenizer=tokenizer, messages=messages,
                                     args=args, seed=seed, torch_module=torch, input_limit=input_limit)
        except RuntimeError:
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
            raise

    errors = generate_dialogues(args, manifest, pending, generate)
    stats = {
        "elapsed_seconds": time.perf_counter() - started, "device": args.device,
        "gpu_name": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
        "torch_version": torch.__version__,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(args.device) if args.device.startswith("cuda") else None,
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(args.device) if args.device.startswith("cuda") else None,
        "includes_model_loading": True, "scope": "current_invocation", "turn_errors": errors,
    }
    (args.output_dir / "runtime_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"Done: {args.output_dir}; turn errors={errors}; no judging performed")
    return 2 if errors else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)
