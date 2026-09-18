#!/usr/bin/env python3
"""SafeDial dataset/output boundary around the original WildJailbreak DCGS policy."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import uuid

from run_safedial_baseline import (DEFAULT_DATASET, append_jsonl, file_sha256,
    gold_messages, load_jsonl, select_dialogues, stable_id, turn_seed, validate_dataset)
from safedial_dcgs_wildjailbreak import (ROOT, REFERENCE, LocalBackend, PolicyFailure,
    configuration, generate_turn, validate_turn)

PROTOCOL = "safedial_original_wildjailbreak_v1"
LOCK = ROOT / "configs/safedial/dcgs_two_stage_aug11.lock.json"


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_records(path):
    # Reject damaged journals rather than silently losing an attempted model call.
    return load_jsonl(path) if path.exists() else []


def source_hashes():
    files = list((ROOT / "src").rglob("*.py")) + list((ROOT / "src/prompts").glob("*.jsonl"))
    files += [REFERENCE] + [ROOT / "scripts" / name for name in (
        "run_safedial_dcgs.py", "run_safedial_baseline.py", "judge_safedial.py",
        "validate_safedial_generation.py", "safedial_dcgs_wildjailbreak.py",
        "run_safedial_dcgs_wildjailbreak.py", "validate_safedial_dcgs_wildjailbreak.py")]
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in sorted(files)}


def verify_artifacts(lock):
    for name, digest in lock["actor"]["sha256"].items():
        if file_sha256(Path(lock["actor"]["path"]) / name) != digest:
            raise ValueError(f"Actor artifact mismatch: {name}")
    if file_sha256(Path(lock["high_level"]["path"])) != lock["high_level"]["sha256"]:
        raise ValueError("High-level checkpoint mismatch")


def manifest_for(args, selected, lock):
    config = configuration(args.method, args.device)
    return {"benchmark": "SafeDialBench", "protocol": PROTOCOL, "method": args.method,
            "dataset": str(args.dataset.resolve()), "dataset_sha256": file_sha256(args.dataset),
            "selected_ids": [r["id"] for r in selected], "num_choices": 1, "seed": args.seed,
            "model_id": "zephyr-7b-beta-" + args.method + "-original-wildjailbreak",
            "reference_config": str(REFERENCE), "effective_config": vars(config),
            "artifacts": {k: lock[k] for k in ("actor", "high_level")},
            "source_sha256": source_hashes(), "generation_only": True,
            "resume_unit": "complete turn; interrupted turn restarts from its original seed",
            "custom_empty_retries": 0, "critic_truncation": "original 1500-token cutoff"}


def tokenizer_for(lock):
    from src.utils.llm_utils import get_tokenizer_instance
    return get_tokenizer_instance(lock["actor"]["path"])


def preflight(selected, config, tokenizer):
    # Obtain prompts/parameters through the original policy, using clearly marked
    # synthetic beliefs/scores. This does not load the backbone or generate answers.
    maximum = 0
    calls = 0
    def probe(request):
        nonlocal maximum, calls
        if request["kind"] == "score":
            return {"scores": [0.0] * len(request["observations"])}
        calls += 1
        for prompt in request["prompts"]:
            suffix = len(tokenizer(request["prefill_suffix"], add_special_tokens=False)["input_ids"]) if request.get("prefill_suffix") else 0
            maximum = max(maximum, len(tokenizer(prompt)["input_ids"]) + suffix + request["max_new_tokens"])
        text = "\n".join(f"{i}. Synthetic preflight instruction {i}." for i in range(1, 6))
        return {"texts": [text if request["do_sample"] else "Synthetic preflight response."] * len(request["prompts"])}
    for row in selected:
        for turn in range(len(row["history"])):
            generate_turn(row, turn, 0, config, tokenizer, probe)
    if maximum > 32768:
        raise ValueError("Original prompts exceed pinned actor context")
    return {"passed": True, "dialogues": len(selected), "turns": sum(len(r["history"]) for r in selected),
            "synthetic_prompt_calls": calls, "max_prompt_plus_generation_tokens": maximum,
            "model_calls": 0, "critic_max_length": 1500,
            "note": "Synthetic belief lengths; actual actor context is guarded at generation time."}


def export_answers(folder, selected, manifest, records):
    by_key = {(r["dialogue_id"], r["turn_index"]): r for r in records}
    answers = []
    for row in selected:
        turns = [by_key.get((row["id"], i)) for i in range(len(row["history"]))]
        if all(turns):
            answers.append({"id": row["id"], "task": row["task"], "method": row["method"],
                "model_id": manifest["model_id"], "answer_id": stable_id(manifest, row["id"]),
                "choices": [{"index": 0, "turns": [{"role": "assistant", "message": t["generated_response"]} for t in turns]}]})
    path = folder / "answers.jsonl"
    content = "".join(json.dumps(answer) + "\n" for answer in answers)
    if not path.exists() or path.read_text() != content:
        temporary = path.with_suffix(".tmp")
        with temporary.open("w") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)


def run_selected(folder, selected, manifest, config, tokenizer, backend_factory):
    """Resume completed turns only; never reset a failed policy's retry budget."""
    records = read_records(folder / "turns.jsonl")
    rows = {row["id"]: row for row in selected}
    done = set()
    for record in records:
        key = (record["dialogue_id"], record["turn_index"])
        if key in done or key[0] not in rows:
            raise ValueError("Duplicate or unselected saved turn")
        if record["seed"] != turn_seed(manifest["seed"], manifest["model_id"], key[0], 0, key[1]):
            raise ValueError("Saved seed mismatch")
        validate_turn(record, rows[key[0]], config, tokenizer)
        done.add(key)
    export_answers(folder, selected, manifest, records)
    if (folder / "failure.json").exists():
        raise ValueError("A policy failure is saved; inspect it before preparing a new experiment")
    pending = [(row, i) for row in selected for i in range(len(row["history"])) if (row["id"], i) not in done]
    if not pending:
        return 0
    # Validate journal syntax before making any new calls. Interrupted invocations
    # remain in this append-only journal with completed-call costs. A killed
    # in-flight call may have unknown cost; attempts.jsonl records its start.
    read_records(folder / "events.jsonl")
    invocation = uuid.uuid4().hex
    started = time.perf_counter()
    backend = None
    success = False
    try:
        backend = backend_factory()
        if hasattr(backend, "loading"):
            write_json(folder / "model_loading.json", backend.loading)
        for row, turn in pending:
            seed = turn_seed(manifest["seed"], manifest["model_id"], row["id"], 0, turn)
            def save_event(event):
                append_jsonl(folder / "events.jsonl", [{"invocation": invocation,
                    "dialogue_id": row["id"], "turn_index": turn, "seed": seed, **event}])
            call_index = 0
            def execute(request):
                nonlocal call_index
                append_jsonl(folder / "attempts.jsonl", [{"invocation": invocation,
                    "dialogue_id": row["id"], "turn_index": turn, "seed": seed,
                    "index": call_index, "request": request}])
                call_index += 1
                return backend(request)
            result = generate_turn(row, turn, seed, config, tokenizer, execute, save_event)
            record = {"dialogue_id": row["id"], "turn_index": turn, "choice_index": 0,
                      "seed": seed, "model_id": manifest["model_id"], "invocation": invocation,
                      "prompt_history": gold_messages(row["history"], turn),
                      "generated_response": result["message"], "dcgs_original": result["dcgs_original"]}
            append_jsonl(folder / "turns.jsonl", [record])
            records.append(record)
            export_answers(folder, selected, manifest, records)
            print(f"Saved dialogue {row['id']} turn {turn}", flush=True)
        success = True
        return 0
    except Exception as exc:
        write_json(folder / "failure.json", {"invocation": invocation, "error": f"{type(exc).__name__}: {exc}",
                   "audit": getattr(exc, "audit", None)})
        raise
    finally:
        runtime = {"invocation": invocation, "success": success, "elapsed_seconds": time.perf_counter() - started}
        if backend is not None and hasattr(backend, "torch") and config.device.startswith("cuda"):
            runtime.update(peak_gpu_allocated_bytes=backend.torch.cuda.max_memory_allocated(config.device),
                           peak_gpu_reserved_bytes=backend.torch.cuda.max_memory_reserved(config.device))
        append_jsonl(folder / "runtime.jsonl", [runtime])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("vdcgs", "rdcgs"), required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--ids")
    selection.add_argument("--limit", type=int)
    selection.add_argument("--per-task", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    rows = load_jsonl(args.dataset)
    validate_dataset(rows, args.dataset)
    selected = select_dialogues(rows, args)
    lock = json.loads(LOCK.read_text())
    config = configuration(args.method, args.device)
    verify_artifacts(lock)
    tokenizer = tokenizer_for(lock)
    manifest = json.loads(json.dumps(manifest_for(args, selected, lock)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.output_dir / "run_config.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("Manifest/source mismatch; use a fresh output directory")
        if not path.exists():
            write_json(path, manifest)
        if args.validate_only:
            report = preflight(selected, config, tokenizer)
            write_json(args.output_dir / "preflight.json", report)
            print(json.dumps(report))
            return 0
        return run_selected(args.output_dir, selected, manifest, config, tokenizer, lambda: LocalBackend(config, lock))


if __name__ == "__main__":
    raise SystemExit(main())
