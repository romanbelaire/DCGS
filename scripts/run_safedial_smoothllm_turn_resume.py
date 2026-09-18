#!/usr/bin/env python3
"""SmoothLLM continuation with durable per-turn resume and a locked legacy import.

The original runner and shared inference/defense sources remain unchanged.
Import is performed only into a fresh target, while holding the legacy writer
lock. Raw legacy files and sources are archived before identifiers are migrated.
"""
import argparse
import contextlib
import copy
import fcntl
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import run_safedial_baseline as base
import run_safedial_smoothllm as legacy
from safedial_smoothllm import CandidateFailure, generate_smoothed, validate_audit

RESUME_POLICY = "per_turn_fsync; reuse_valid_successes; explicit_failed_turn_retry; gold_history"
DEFAULT_OUTPUT = "outputs/safedial_baseline/smoothllm_zephyr_full_turn_resume"


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--resume-from", type=Path)
    options, rest = extra.parse_known_args(argv)
    if not any(v == "--output-dir" or v.startswith("--output-dir=") for v in rest):
        rest += ["--output-dir", DEFAULT_OUTPUT]
    args = legacy.parse_args(rest)
    args.resume_from = options.resume_from
    if args.resume_from and args.resume_from.resolve() == args.output_dir.resolve():
        raise ValueError("Legacy import requires a separate output directory")
    return args


def manifest_for(args, selected):
    manifest = legacy.manifest_for(args, selected)
    manifest["implementation_sha256"][Path(__file__).name] = base.file_sha256(Path(__file__))
    manifest["resume_policy"] = RESUME_POLICY
    return manifest


@contextlib.contextmanager
def single_writer(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".generation.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another writer is active in {folder}; wait for that job to finish") from exc
        yield


def atomic_jsonl(path, records):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_journal(path):
    """Repair only an interrupted last append, preserving its exact bytes."""
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        end = data.rfind(b"\n") + 1
        tail = data[end:]
        try:
            json.loads(tail)
        except (ValueError, UnicodeDecodeError):
            recovery = path.parent / "recovery"
            recovery.mkdir(exist_ok=True)
            (recovery / f"{path.name}.{time.time_ns()}.partial").write_bytes(tail)
            with path.open("r+b") as handle:
                handle.truncate(end)
                handle.flush()
                os.fsync(handle.fileno())
        else:
            with path.open("ab") as handle:
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
    return base.load_jsonl(path)


def checked_turns(records, args, manifest, selected):
    run_id = base.stable_id(json.dumps(manifest, sort_keys=True))
    by_id = {row["id"]: row for row in selected}
    latest = {}
    for record in records:
        key = (record["dialogue_id"], record["turn_index"])
        if (key[0] not in by_id or type(key[1]) is not int or not 0 <= key[1] < len(by_id[key[0]]["history"])
                or record["model_id"] != args.model_id or record["choice_index"] != 0
                or record["run_id"] != run_id or record["answer_id"] != base.stable_id(run_id, key[0])):
            raise ValueError("Foreign SmoothLLM turn or run identifier")
        if record["seed"] != base.turn_seed(args.seed, args.model_id, key[0], 0, key[1]):
            raise ValueError("Saved SmoothLLM seed changed")
        if record["prompt_history"] != base.gold_messages(by_id[key[0]]["history"], key[1]):
            raise ValueError("Saved SmoothLLM gold history changed")
        if not record.get("error"):
            validate_audit(record, manifest["defense"])
        latest[key] = record
    return latest


def load_turns(args, manifest, selected):
    return checked_turns(read_journal(args.output_dir / "turns.jsonl"), args, manifest, selected)


def pending_rows(selected, turns, args):
    return [row for row in selected if any(
        (row["id"], i) not in turns or (args.retry_errors and turns[row["id"], i].get("error"))
        for i in range(len(row["history"])))]


def export_answers(args, manifest, selected, turns):
    """Rebuild complete dialogue exports from the authoritative turn journal."""
    run_id = base.stable_id(json.dumps(manifest, sort_keys=True))
    answers = []
    for row in selected:
        records = [turns.get((row["id"], i)) for i in range(len(row["history"]))]
        if any(r is None for r in records):
            continue
        official = []
        for record in records:
            turn = {"role": "assistant", "message": record["generated_response"]}
            if record.get("error"):
                turn["error"] = record["error"]
            official.append(turn)
        answers.append({"id": row["id"], "task": row["task"], "method": row["method"],
                        "answer_id": base.stable_id(run_id, row["id"]), "model_id": args.model_id,
                        "choices": [{"index": 0, "turns": official}],
                        "tstamp": max(r.get("tstamp", 0.0) for r in records)})
    atomic_jsonl(args.output_dir / "answers.jsonl", answers)


def compact_turns(args, turns):
    path = args.output_dir / "turns.jsonl"
    records = base.load_jsonl(path) if path.exists() else []
    if len(records) > len(turns):
        # Retain failed/superseded attempts and their candidate costs before compaction.
        archive = args.output_dir / "provenance/turn_journals"
        archive.mkdir(parents=True, exist_ok=True)
        destination = archive / (base.file_sha256(path) + ".jsonl")
        if not destination.exists():
            shutil.copy2(path, destination)
    atomic_jsonl(path, [turns[key] for key in sorted(turns)])


def import_legacy(args, manifest, selected):
    """Caller holds target lock; source lock prevents importing a running job."""
    source = args.resume_from
    if not source.is_dir():
        raise ValueError("Legacy source directory does not exist")
    with single_writer(source):
        old = json.loads((source / "run_config.json").read_text())
        if old != legacy.manifest_for(args, selected):
            raise ValueError("Legacy configuration/source hashes differ from requested run")
        if any(p.name != ".generation.lock" for p in args.output_dir.iterdir()):
            raise ValueError("Legacy import requires a fresh destination")
        files = [p for p in source.iterdir() if p.is_file() and not p.name.startswith('.')]
        hashes = {p.name: base.file_sha256(p) for p in files}
        stage = Path(tempfile.mkdtemp(prefix=".smoothllm-import-", dir=args.output_dir.parent))
        archive = stage / "provenance/legacy"
        archive.mkdir(parents=True)
        for path in files:
            shutil.copy2(path, archive / path.name)
        for name, digest in old["implementation_sha256"].items():
            src = Path(__file__).parent / name
            if base.file_sha256(src) != digest:
                raise ValueError("Legacy source changed during import")
            dst = archive / "scripts" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        # Recover any interrupted append in the staging copy, never in source/archive.
        shutil.copy2(archive / "turns.jsonl", stage / "turns.jsonl")
        originals = read_journal(stage / "turns.jsonl")
        turns = checked_turns(originals, args, old, selected)
        new_id = base.stable_id(json.dumps(manifest, sort_keys=True))
        for key, original in list(turns.items()):
            record = copy.deepcopy(original)
            record.update(run_id=new_id, answer_id=base.stable_id(new_id, key[0]))
            turns[key] = record
        atomic_jsonl(stage / "turns.jsonl", [turns[key] for key in sorted(turns)])
        base.ensure_manifest(stage / "run_config.json", manifest)
        staged_args = copy.copy(args)
        staged_args.output_dir = stage
        export_answers(staged_args, manifest, selected, turns)
        report = {"source_directory": str(source.resolve()), "original_file_sha256": hashes,
                  "source_run_id": base.stable_id(json.dumps(old, sort_keys=True)),
                  "destination_run_id": new_id, "reused_successful_turns": sum(not r.get("error") for r in turns.values()),
                  "retained_error_turns": sum(bool(r.get("error")) for r in turns.values()),
                  "model_calls_during_import": 0, "only_turn_identifier_fields_changed": True}
        (stage / "continuation.json").write_text(json.dumps(report, indent=2) + "\n")
        for name, digest in hashes.items():
            if base.file_sha256(source / name) != digest or base.file_sha256(archive / name) != digest:
                raise ValueError("Legacy source/archive changed during import")
        # Publish manifest last. A failed partial import is never treated as a fresh run.
        for path in stage.iterdir():
            if path.name != "run_config.json":
                path.rename(args.output_dir / path.name)
        (stage / "run_config.json").rename(args.output_dir / "run_config.json")
        stage.rmdir()
        print(f"Imported {report['reused_successful_turns']} successful turns; retained {report['retained_error_turns']} errors", flush=True)


def prepare_output(args, manifest, selected):
    config = args.output_dir / "run_config.json"
    if config.exists():
        base.ensure_manifest(config, manifest)
    elif args.resume_from:
        import_legacy(args, manifest, selected)
    else:
        if any(p.name != ".generation.lock" for p in args.output_dir.iterdir()):
            raise ValueError("Existing output has no SmoothLLM manifest")
        base.ensure_manifest(config, manifest)


def generate_dialogues(args, manifest, selected, generate):
    run_id = base.stable_id(json.dumps(manifest, sort_keys=True))
    turns = load_turns(args, manifest, selected)
    export_answers(args, manifest, selected, turns)
    for row in pending_rows(selected, turns, args):
        for index, source in enumerate(row["history"]):
            key = (row["id"], index)
            saved = turns.get(key)
            if saved and (not saved.get("error") or not args.retry_errors):
                continue
            seed = base.turn_seed(args.seed, args.model_id, row["id"], 0, index)
            messages = base.gold_messages(row["history"], index)
            error = None
            try:
                result = generate_smoothed(messages, manifest["defense"], seed, generate)
            except CandidateFailure as exc:
                error = str(exc)
                result = {"message": "ERROR", "smoothllm": exc.audit,
                          "prompt_tokens": None, "original_prompt_tokens": None,
                          "completion_tokens": None, "latency_seconds": None, "input_truncated": None}
                print(f"ERROR dialogue={row['id']} turn={index}: {error}", file=sys.stderr)
            record = {
                "run_id": run_id, "answer_id": base.stable_id(run_id, row["id"]),
                "benchmark": "SafeDialBench", "protocol": base.PROTOCOL,
                "model": args.model, "model_id": args.model_id, "dialogue_id": row["id"],
                "task": row["task"], "method": row["method"], "scene": row["scene"],
                "dataset_model_type": row.get("model_type"), "choice_index": 0, "turn_index": index,
                "seed": seed, "prompt_history": messages, "user_message": source["user"],
                "reference_response": source["bot"], "generated_response": result["message"],
                "error": error, "tstamp": time.time(),
                **{key: value for key, value in result.items() if key != "message"},
            }
            base.append_jsonl(args.output_dir / "turns.jsonl", [record])
            turns[key] = record
            print(f"Saved dialogue={row['id']} turn={index} status={'error' if error else 'success'}", flush=True)
        export_answers(args, manifest, selected, turns)
    compact_turns(args, turns)
    return sum(bool(r.get("error")) for r in turns.values())


def run_locked(args, manifest, selected):
    prepare_output(args, manifest, selected)
    turns = load_turns(args, manifest, selected)
    export_answers(args, manifest, selected, turns)
    pending = pending_rows(selected, turns, args)
    print(f"Saved successful turns: {sum(not r.get('error') for r in turns.values())}; pending dialogues: {len(pending)}", flush=True)
    if not pending:
        compact_turns(args, turns)
        return 2 if any(r.get("error") for r in turns.values()) else 0
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; submit the prepared GPU batch manually")
    started = time.perf_counter()
    if args.device.startswith("cuda"):
        torch.cuda.init()
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

    errors = generate_dialogues(args, manifest, selected, generate)
    stats = {"elapsed_seconds": time.perf_counter() - started, "device": args.device,
             "gpu_name": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
             "torch_version": torch.__version__,
             "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(args.device) if args.device.startswith("cuda") else None,
             "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(args.device) if args.device.startswith("cuda") else None,
             "includes_model_loading": True, "scope": "current_invocation", "turn_errors": errors}
    (args.output_dir / "runtime_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    base.append_jsonl(args.output_dir / "invocations.jsonl", [stats])
    return 2 if errors else 0


def main(argv=None):
    args = parse_args(argv)
    rows = base.load_jsonl(args.dataset)
    base.validate_dataset(rows, args.dataset)
    selected = base.select_dialogues(rows, args)
    if not selected or len({row["id"] for row in selected}) != len(selected):
        raise ValueError("Selection must be nonempty with unique IDs")
    manifest = manifest_for(args, selected)
    count = sum(len(row["history"]) for row in selected)
    print(f"Validated {len(selected)} dialogues / {count} turns / {count * args.copies} candidate generations", flush=True)
    if args.validate_only:
        return 0
    with single_writer(args.output_dir):
        return run_locked(args, manifest, selected)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)
