#!/usr/bin/env python3
"""Run native SafeDialBench TPO with a frozen Zephyr actor and learned reward model.

Generation only. Completed events and turns resume deterministically; only fully
successful dialogues are exported as native answers. A process-level file lock
also protects direct CLI usage. No inference occurs in --validate-only mode.
"""

import argparse
import contextlib
import fcntl
import json
import os
import sys
import time
from pathlib import Path

import run_safedial_baseline as base
from safedial_generation_retry import reusable_empty_event
from safedial_tpo import TPOFailure, defense_config, generate_tpo, validate_audit
from safedial_tpo_artifacts import ACTOR, ACTOR_REVISION, DEFAULT_LOCK, validate_lock


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=base.DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/safedial_baseline/tpo_zephyr_smoke"))
    parser.add_argument("--artifact-lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--model-id", default="zephyr-7b-beta-tpo")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--max-iters", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--feedback-tokens", type=int, default=2048)
    parser.add_argument("--max-input-tokens", type=int)
    parser.add_argument("--seed", type=int, default=0)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--ids")
    selection.add_argument("--per-task", type=int)
    selection.add_argument("--limit", type=int)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--empty-generation-retries", type=int, choices=(0, 2), default=2)
    parser.add_argument("--validate-only", action="store_true")
    parser.set_defaults(model=ACTOR, revision=ACTOR_REVISION, temperature=0.7, top_p=0.95, num_choices=1)
    args = parser.parse_args(argv)
    base.validate_args(args)
    defense_config(args.sample_size, args.max_iters, args.max_new_tokens, args.feedback_tokens, args.empty_generation_retries)
    return args


def implementation_hashes():
    folder = Path(__file__).parent
    files = ["run_safedial_tpo.py", "safedial_tpo.py", "safedial_tpo_runtime.py", "safedial_tpo_artifacts.py", "safedial_generation_retry.py",
             "run_safedial_baseline.py", "tpo_vendor/llm_backward_prompts.py", "tpo_vendor/optimizer_prompts.py"]
    return {name: base.file_sha256(folder / name) for name in files}


def manifest_for(args, selected, lock):
    manifest = base.manifest_for(args, args.dataset, selected)
    manifest.update(defense=defense_config(args.sample_size, args.max_iters, args.max_new_tokens, args.feedback_tokens, args.empty_generation_retries),
                    artifacts=lock, implementation_sha256=implementation_hashes(),
                    token_accounting="actor_totals_include_candidates_loss_gradient; reward_prompt_tokens_separate",
                    backend="transformers_local_frozen_sdpa_top_k_0", device=args.device,
                    reward_encoding="native_chat_template_once_no_extra_special_tokens; raw_single_logit",
                    resume="per_event_and_turn; failed_calls_retained; complete_dialogues_only_in_answers")
    return manifest


@contextlib.contextmanager
def single_writer(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".generation.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another TPO writer is active") from exc
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
    """Recover only an interrupted final append, retaining its exact bytes."""
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
            print(f"Recovered interrupted final append in {path}; original tail retained", flush=True)
        else:
            with path.open("ab") as handle:
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
    return base.load_jsonl(path)


def load_state(args, manifest, selected):
    run_id = base.stable_id(json.dumps(manifest, sort_keys=True))
    valid_keys = {(r["id"], i) for r in selected for i in range(len(r["history"]))}
    turns, events = {}, {}
    for record in read_journal(args.output_dir / "turns.jsonl"):
        key = (record["dialogue_id"], record["turn_index"])
        if record["run_id"] != run_id or key not in valid_keys or record["model_id"] != args.model_id:
            raise ValueError("Foreign turn record in TPO output")
        turns[key] = record
    for row in read_journal(args.output_dir / "events.jsonl"):
        key = (row["dialogue_id"], row["turn_index"])
        if row["run_id"] != run_id or key not in valid_keys:
            raise ValueError("Foreign event record in TPO output")
        events[(*key, row["event"]["index"])] = row["event"]
    return run_id, turns, events


def export_answers(args, selected, turns, run_id):
    answers = []
    for row in selected:
        records = [turns.get((row["id"], i)) for i in range(len(row["history"]))]
        if any(r is None or r.get("error") for r in records):
            continue
        answers.append({"id": row["id"], "task": row["task"], "method": row["method"],
                        "model_id": args.model_id, "answer_id": base.stable_id(run_id, row["id"]),
                        "choices": [{"index": 0, "turns": [{"role": "assistant", "message": r["generated_response"]} for r in records]}],
                        "tstamp": records[-1]["tstamp"]})
    atomic_jsonl(args.output_dir / "answers.jsonl", answers)


def generate_selected(args, manifest, selected, execute):
    run_id, turns, events = load_state(args, manifest, selected)
    # Refuse to reuse altered success records, including altered gold histories.
    for row in selected:
        for index in range(len(row["history"])):
            previous = turns.get((row["id"], index))
            if previous and not previous.get("error"):
                if previous["prompt_history"] != base.gold_messages(row["history"], index):
                    raise ValueError("Saved TPO gold history changed")
                if previous["seed"] != base.turn_seed(args.seed, args.model_id, row["id"], 0, index):
                    raise ValueError("Saved TPO seed changed")
                validate_audit(previous, manifest["defense"])
    export_answers(args, selected, turns, run_id)
    for row in selected:
        for index, source in enumerate(row["history"]):
            key = (row["id"], index)
            previous = turns.get(key)
            if previous and not previous.get("error"):
                continue
            if previous and previous.get("error") and not args.retry_errors:
                print(f"Unresolved TPO error {key}; inspect audit before --retry-errors", flush=True)
                return 2
            cursor = 0

            def cached_execute(request):
                nonlocal cursor
                saved = events.get((*key, cursor))
                cursor += 1
                if saved and saved["request"] != request:
                    raise ValueError("Saved TPO event request changed")
                if saved and (not saved.get("error") or
                              (manifest["defense"].get("empty_generation_retry") and reusable_empty_event(saved))):
                    return saved["result"]
                return execute(request)

            def persist(event):
                event_key = (*key, event["index"])
                if events.get(event_key) == event:
                    return
                base.append_jsonl(args.output_dir / "events.jsonl", [{"run_id": run_id, "dialogue_id": key[0],
                                                                    "turn_index": key[1], "event": event}])
                events[event_key] = event
                if event.get("format_warnings"):
                    print(f"WARNING dialogue={key[0]} turn={key[1]} event={event['index']}: "
                          + "; ".join(event["format_warnings"]), flush=True)

            messages = base.gold_messages(row["history"], index)
            seed = base.turn_seed(args.seed, args.model_id, row["id"], 0, index)
            record = {"run_id": run_id, "answer_id": base.stable_id(run_id, row["id"]),
                      "benchmark": "SafeDialBench", "protocol": base.PROTOCOL,
                      "model": args.model, "model_id": args.model_id, "dialogue_id": row["id"],
                      "task": row["task"], "method": row["method"], "scene": row["scene"],
                      "dataset_model_type": row.get("model_type"), "choice_index": 0, "turn_index": index,
                      "seed": seed, "prompt_history": messages, "user_message": source["user"],
                      "reference_response": source["bot"], "error": None}
            try:
                result = generate_tpo(messages, manifest["defense"], seed, cached_execute, persist)
                record.update({k: v for k, v in result.items() if k != "message"}, generated_response=result["message"])
            except TPOFailure as exc:
                record.update(error=str(exc), generated_response=None, tpo=exc.audit)
            record["tstamp"] = time.time()
            base.append_jsonl(args.output_dir / "turns.jsonl", [record])
            turns[key] = record
            if record["error"]:
                export_answers(args, selected, turns, run_id)
                print(f"FAIL dialogue={key[0]} turn={key[1]}: {record['error']}", flush=True)
                return 2
            print(f"Saved dialogue={key[0]} turn={key[1]} candidates={len(record['tpo']['candidates'])}", flush=True)
        export_answers(args, selected, turns, run_id)
    atomic_jsonl(args.output_dir / "turns.jsonl", [turns[key] for key in sorted(turns)])
    return 0


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
    from safedial_tpo_runtime import LocalBackend, tokenizers_and_preflight
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

        def execute(request):
            nonlocal backend
            if backend is None:
                backend = LocalBackend(args, lock, tokenizers, limits)
                (args.output_dir / "model_loading.json").write_text(json.dumps(backend.loading, indent=2) + "\n")
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
                     "scope": "current_invocation_excludes_artifact_hashing_and_preflight", "actual_calls": actual_calls,
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
