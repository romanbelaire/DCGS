#!/usr/bin/env python3
"""Generate SafeDialBench answers with a local HF model and optional locked LoRA.

The benchmark uses counterfactual turn evaluation: every response is generated
from the gold assistant history supplied by SafeDialBench.  A generated response
is recorded, but is never fed into the prompt for the following turn.

This script intentionally performs generation only.  It does not invoke a judge.
It writes both an official-judge-compatible ``answers.jsonl`` and an auditable
turn-level ``turns.jsonl``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET = Path(
    "benchmark-artifacts/SafeDialBench-Dataset/data/complete/datasets_en.jsonl"
)
DEFAULT_MODEL = "HuggingFaceH4/zephyr-7b-beta"
TASKS = ("Aggression", "Ethics", "Fairness", "Legality", "Morality", "Privacy")
PROTOCOL = "safedial_gold_assistant_history_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a plain local-model baseline on SafeDialBench."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/safedial_baseline/zephyr_7b_beta"))
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model ID or local path.")
    parser.add_argument("--model-id", default="zephyr-7b-beta-baseline")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--adapter-lock", type=Path, help="Hash-pinned local LoRA artifact lock (e.g. CAT).")
    parser.add_argument("--verify-adapter", action="store_true", help="Check enabled/disabled adapter logits on a fixed fixture.")
    parser.add_argument("--device", default="cuda", help="For example: cuda, cuda:0, or cpu.")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--num-choices", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--ids", help="Comma-separated SafeDialBench dialogue IDs.")
    selection.add_argument("--per-task", type=int, help="Select the first N IDs in each of the six tasks.")
    selection.add_argument("--limit", type=int, help="Select the first N rows in dataset order.")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Regenerate dialogues whose latest saved answer contains an error.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and print the selection without loading a model or writing output.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def validate_dataset(rows: list[dict[str, Any]], path: Path) -> None:
    seen: set[int] = set()
    for row_number, row in enumerate(rows, 1):
        missing = {"id", "task", "method", "scene", "history"} - row.keys()
        if missing:
            raise ValueError(f"{path}:{row_number} is missing fields: {sorted(missing)}")
        dialogue_id = row["id"]
        if not isinstance(dialogue_id, int) or dialogue_id in seen:
            raise ValueError(f"{path}:{row_number} has invalid or duplicate id {dialogue_id!r}")
        seen.add(dialogue_id)
        if row["task"] not in TASKS:
            raise ValueError(f"{path}:{row_number} has unknown task {row['task']!r}")
        if not isinstance(row["history"], list) or not row["history"]:
            raise ValueError(f"{path}:{row_number} has an empty or invalid history")
        for turn_number, turn in enumerate(row["history"], 1):
            if not isinstance(turn, dict) or not isinstance(turn.get("user"), str):
                raise ValueError(f"{path}:{row_number} turn {turn_number} has no user text")
            if not isinstance(turn.get("bot"), str):
                raise ValueError(f"{path}:{row_number} turn {turn_number} has no gold bot text")


def parse_ids(raw_ids: str) -> list[int]:
    values: list[int] = []
    for value in raw_ids.split(","):
        value = value.strip()
        if value:
            values.append(int(value))
    if not values:
        raise ValueError("--ids did not contain any IDs")
    return values


def select_dialogues(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.ids:
        requested = parse_ids(args.ids)
        by_id = {row["id"]: row for row in rows}
        missing = [dialogue_id for dialogue_id in requested if dialogue_id not in by_id]
        if missing:
            raise ValueError(f"Unknown SafeDialBench IDs: {missing}")
        return [by_id[dialogue_id] for dialogue_id in requested]
    if args.per_task is not None:
        if args.per_task < 1:
            raise ValueError("--per-task must be positive")
        selected: list[dict[str, Any]] = []
        for task in TASKS:
            task_rows = sorted((row for row in rows if row["task"] == task), key=lambda row: row["id"])
            if len(task_rows) < args.per_task:
                raise ValueError(f"Task {task} has fewer than {args.per_task} rows")
            selected.extend(task_rows[: args.per_task])
        return selected
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        return rows[: args.limit]
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(*parts: object, length: int = 24) -> str:
    raw = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:length]


def turn_seed(base_seed: int, model_id: str, dialogue_id: int, choice: int, turn: int) -> int:
    value = stable_id(base_seed, model_id, dialogue_id, choice, turn, length=16)
    return int(value, 16) % (2**31 - 1)


def gold_messages(history: list[dict[str, str]], turn_index: int) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for prior in history[:turn_index]:
        messages.append({"role": "user", "content": prior["user"]})
        messages.append({"role": "assistant", "content": prior["bot"]})
    messages.append({"role": "user", "content": history[turn_index]["user"]})
    return messages


def dtype_from_name(name: str, torch_module: Any) -> Any:
    return {
        "auto": "auto",
        "float32": torch_module.float32,
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
    }[name]


def effective_context_limit(model: Any, tokenizer: Any, requested: int | None, max_new_tokens: int) -> int:
    candidates: list[int] = []
    for value in (
        getattr(model.config, "max_position_embeddings", None),
        getattr(tokenizer, "model_max_length", None),
    ):
        if isinstance(value, int) and 0 < value < 1_000_000:
            candidates.append(value)
    context_window = min(candidates) if candidates else 32768
    available_for_input = context_window - max_new_tokens
    if requested is not None:
        available_for_input = min(available_for_input, requested)
    if available_for_input < 1:
        raise ValueError("The input/context limit must be larger than max-new-tokens")
    return available_for_input


def synchronize_if_cuda(torch_module: Any, device: str) -> None:
    if device.startswith("cuda") and torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def generate_one(
    *,
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    args: argparse.Namespace,
    seed: int,
    torch_module: Any,
    input_limit: int,
) -> dict[str, Any]:
    from transformers import set_seed

    set_seed(seed)
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    original_prompt_tokens = int(encoded["input_ids"].shape[1])
    truncated = original_prompt_tokens > input_limit
    if truncated:
        encoded["input_ids"] = encoded["input_ids"][:, -input_limit:]
        encoded["attention_mask"] = encoded["attention_mask"][:, -input_limit:]
    encoded = {key: value.to(model.device) for key, value in encoded.items()}

    generation_args: dict[str, Any] = {
        **encoded,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generation_args["temperature"] = args.temperature
        generation_args["top_p"] = args.top_p

    synchronize_if_cuda(torch_module, args.device)
    started = time.perf_counter()
    with torch_module.inference_mode():
        output_ids = model.generate(**generation_args)
    synchronize_if_cuda(torch_module, args.device)
    latency = time.perf_counter() - started
    prompt_tokens = int(encoded["input_ids"].shape[1])
    generated_ids = output_ids[0, prompt_tokens:]
    output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return {
        "message": output,
        "prompt_tokens": prompt_tokens,
        "original_prompt_tokens": original_prompt_tokens,
        "completion_tokens": int(generated_ids.shape[0]),
        "latency_seconds": latency,
        "input_truncated": truncated,
    }


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_latest_answers(path: Path, model_id: str) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    latest: dict[int, dict[str, Any]] = {}
    for record in load_jsonl(path):
        if record.get("model_id") == model_id and isinstance(record.get("id"), int):
            latest[record["id"]] = record
    return latest


def answer_has_error(answer: dict[str, Any]) -> bool:
    for choice in answer.get("choices", []):
        for turn in choice.get("turns", []):
            if turn.get("error") or turn.get("message") == "ERROR":
                return True
    return False


def compact_answers(path: Path, model_id: str) -> None:
    latest = load_latest_answers(path, model_id)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for dialogue_id in sorted(latest):
            handle.write(json.dumps(latest[dialogue_id], ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def compact_turns(path: Path) -> None:
    if not path.exists():
        return
    latest: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for record in load_jsonl(path):
        key = (
            str(record["model_id"]),
            int(record["dialogue_id"]),
            int(record["choice_index"]),
            int(record["turn_index"]),
        )
        latest[key] = record
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for key in sorted(latest, key=lambda item: (item[1], item[2], item[3], item[0])):
            handle.write(json.dumps(latest[key], ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def manifest_for(args: argparse.Namespace, dataset: Path, selected: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = {
        "benchmark": "SafeDialBench",
        "protocol": PROTOCOL,
        "dataset": str(dataset.resolve()),
        "dataset_sha256": file_sha256(dataset),
        "selected_ids": [row["id"] for row in selected],
        "model": args.model,
        "model_id": args.model_id,
        "revision": args.revision,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_choices": args.num_choices,
        "seed": args.seed,
        "generation_only": True,
    }
    # Keep the plain-model manifest unchanged so existing runs still resume.
    if getattr(args, "adapter_lock", None):
        from safedial_adapters import validate_adapter_lock

        manifest["adapter"] = validate_adapter_lock(args.adapter_lock, args.model, args.revision)
    return manifest


def ensure_manifest(path: Path, manifest: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError(
                f"{path} describes a different run. Use a new --output-dir to avoid mixing results."
            )
        return
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_args(args: argparse.Namespace) -> None:
    if getattr(args, "verify_adapter", False) and not getattr(args, "adapter_lock", None):
        raise ValueError("--verify-adapter requires --adapter-lock")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.max_input_tokens is not None and args.max_input_tokens < 1:
        raise ValueError("--max-input-tokens must be positive")
    if args.temperature < 0:
        raise ValueError("--temperature cannot be negative")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")
    if args.num_choices < 1:
        raise ValueError("--num-choices must be positive")


def main() -> int:
    args = parse_args()
    validate_args(args)
    dataset = args.dataset.expanduser()
    if not dataset.is_file():
        raise FileNotFoundError(f"SafeDialBench dataset not found: {dataset}")
    rows = load_jsonl(dataset)
    validate_dataset(rows, dataset)
    selected = select_dialogues(rows, args)
    task_counts = {task: sum(row["task"] == task for row in selected) for task in TASKS}
    print(f"Validated {len(rows)} SafeDialBench rows; selected {len(selected)} dialogues")
    print("Selection by task: " + ", ".join(f"{task}={count}" for task, count in task_counts.items()))
    print("Selected IDs: " + ",".join(str(row["id"]) for row in selected))
    if args.validate_only:
        if args.adapter_lock:
            from safedial_adapters import validate_adapter_lock

            validate_adapter_lock(args.adapter_lock, args.model, args.revision)
            print("Adapter lock and local weight hashes validated")
        return 0

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Start this inside an srun GPU allocation.")

    run_started = time.perf_counter()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_for(args, dataset, selected)
    ensure_manifest(args.output_dir / "run_config.json", manifest)
    answers_path = args.output_dir / "answers.jsonl"
    turns_path = args.output_dir / "turns.jsonl"
    completed = load_latest_answers(answers_path, args.model_id)

    pending = [
        row
        for row in selected
        if row["id"] not in completed
        or (args.retry_errors and answer_has_error(completed[row["id"]]))
    ]
    print(f"Already complete: {len(selected) - len(pending)}; pending: {len(pending)}")
    if not pending:
        compact_answers(answers_path, args.model_id)
        compact_turns(turns_path)
        return 0

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    print(f"Loading model on {args.device}: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=dtype_from_name(args.dtype, torch),
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    if args.adapter_lock:
        from safedial_adapters import attach_adapter, verify_adapter_effect

        model = attach_adapter(model, manifest["adapter"])
        print(f"Loaded locked LoRA adapter: {manifest['adapter']['adapter_repo']}")
        if args.verify_adapter:
            verification = verify_adapter_effect(model, tokenizer, torch)
            (args.output_dir / "adapter_validation.json").write_text(
                json.dumps(verification, indent=2) + "\n"
            )
            print(f"Adapter effect verified: max logit difference {verification['max_abs_logit_difference']:.6g}")
    input_limit = effective_context_limit(model, tokenizer, args.max_input_tokens, args.max_new_tokens)
    print(f"Input token limit: {input_limit}; protocol: {PROTOCOL}")

    run_id = stable_id(manifest["dataset_sha256"], args.model_id, args.seed, manifest["selected_ids"])
    for dialogue_number, row in enumerate(pending, 1):
        answer_id = stable_id(run_id, row["id"])
        choices: list[dict[str, Any]] = []
        turn_records: list[dict[str, Any]] = []
        for choice_index in range(args.num_choices):
            choice_turns: list[dict[str, Any]] = []
            for turn_index, source_turn in enumerate(row["history"]):
                seed = turn_seed(args.seed, args.model_id, row["id"], choice_index, turn_index)
                messages = gold_messages(row["history"], turn_index)
                error: str | None = None
                try:
                    result = generate_one(
                        model=model,
                        tokenizer=tokenizer,
                        messages=messages,
                        args=args,
                        seed=seed,
                        torch_module=torch,
                        input_limit=input_limit,
                    )
                except (RuntimeError, ValueError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    result = {
                        "message": "ERROR",
                        "prompt_tokens": None,
                        "original_prompt_tokens": None,
                        "completion_tokens": 0,
                        "latency_seconds": None,
                        "input_truncated": None,
                    }
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(f"ERROR dialogue={row['id']} choice={choice_index} turn={turn_index}: {error}", file=sys.stderr)

                official_turn = {"role": "assistant", "message": result["message"]}
                if error:
                    official_turn["error"] = error
                choice_turns.append(official_turn)
                turn_records.append(
                    {
                        "run_id": run_id,
                        "answer_id": answer_id,
                        "benchmark": "SafeDialBench",
                        "protocol": PROTOCOL,
                        "model": args.model,
                        "model_id": args.model_id,
                        "dialogue_id": row["id"],
                        "task": row["task"],
                        "method": row["method"],
                        "scene": row["scene"],
                        "dataset_model_type": row.get("model_type"),
                        "choice_index": choice_index,
                        "turn_index": turn_index,
                        "seed": seed,
                        "prompt_history": messages,
                        "user_message": source_turn["user"],
                        "generated_response": result["message"],
                        "reference_response": source_turn["bot"],
                        "prompt_tokens": result["prompt_tokens"],
                        "original_prompt_tokens": result["original_prompt_tokens"],
                        "completion_tokens": result["completion_tokens"],
                        "latency_seconds": result["latency_seconds"],
                        "input_truncated": result["input_truncated"],
                        "error": error,
                    }
                )
            choices.append({"index": choice_index, "turns": choice_turns})

        answer = {
            "id": row["id"],
            "task": row["task"],
            "answer_id": answer_id,
            "model_id": args.model_id,
            "method": row["method"],
            "choices": choices,
            "tstamp": time.time(),
        }
        append_jsonl(turns_path, turn_records)
        append_jsonl(answers_path, [answer])
        print(f"[{dialogue_number}/{len(pending)}] completed dialogue {row['id']} ({len(row['history'])} turns)")

    compact_answers(answers_path, args.model_id)
    compact_turns(turns_path)
    statistics = {
        "elapsed_seconds": time.perf_counter() - run_started,
        "device": args.device,
        "gpu_name": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
        "torch_version": torch.__version__,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(args.device) if args.device.startswith("cuda") else None,
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(args.device) if args.device.startswith("cuda") else None,
        "includes_model_loading": True,
        "scope": "current_invocation",
    }
    (args.output_dir / "runtime_stats.json").write_text(json.dumps(statistics, indent=2) + "\n")
    print(f"Done. Judge-compatible answers: {answers_path}")
    print(f"Auditable turn records: {turns_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)
