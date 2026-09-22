#!/usr/bin/env python3
"""One predeclared fresh-seed attempt for SmoothLLM dialogue 344, turn index 4.

Supplemental recovery only: source results and primary judgments are read-only.
No seed search, candidate retries, paid judging, or automatic second attempt.
"""
import argparse
import contextlib
import copy
import fcntl
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import run_safedial_baseline as base
import run_safedial_smoothllm_turn_resume as resume
from safedial_smoothllm import CandidateFailure, candidate_inputs, generate_smoothed, validate_audit

POLICY = "smoothllm-single-fresh-seed-recovery-v1"
TARGET = (344, 4)
SOURCE = Path("outputs/safedial_baseline/smoothllm_zephyr_full_turn_resume")
OUTPUT = Path("outputs/safedial_baseline/smoothllm_zephyr_recovery_344_turn5_v1")
SOURCE_FILES = ("run_config.json", "turns.jsonl", "answers.jsonl")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@contextlib.contextmanager
def source_lock(source):
    # Open the existing lock read-only: never create or rewrite source artifacts.
    with (source / ".generation.lock").open("rb") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Source has an active writer; wait for it to finish") from exc
        yield


def fingerprints(source):
    return {name: base.file_sha256(source / name) for name in SOURCE_FILES}


def inspect_source(source):
    """Read-only audit of complete coverage, original seeds, history and votes."""
    hashes = fingerprints(source)
    manifest = json.loads((source / "run_config.json").read_text())
    args = SimpleNamespace(**manifest)
    args.dataset = Path(manifest["dataset"])
    args.copies = manifest["defense"]["num_copies"]
    args.perturbation_percent = manifest["defense"]["perturbation_percent"]
    rows = base.load_jsonl(args.dataset)
    base.validate_dataset(rows, args.dataset)
    by_id = {r["id"]: r for r in rows}
    selected = [by_id[i] for i in manifest["selected_ids"]]
    if resume.manifest_for(args, selected) != manifest:
        raise ValueError("Source configuration, dataset or pinned implementation changed")
    if (args.seed, args.copies, args.perturbation_percent, args.temperature,
            args.max_new_tokens) != (0, 8, 10.0, 0.0, 1024):
        raise ValueError("Expected the original seed-0, eight-copy, 10%, greedy, 1024-token run")
    turns = resume.checked_turns(base.load_jsonl(source / "turns.jsonl"), args, manifest, selected)
    expected = {(r["id"], i) for r in selected for i in range(len(r["history"]))}
    if set(turns) != expected:
        raise ValueError("Source is missing turns or has unexpected coverage")
    failed = {key for key, record in turns.items() if record.get("error")}
    if failed != {TARGET}:
        raise ValueError(f"Expected only failed turn {TARGET}; found {sorted(failed)}")
    if "Empty or failed candidate" not in turns[TARGET]["error"]:
        raise ValueError("Target does not have the expected empty-candidate failure")
    dialogue = [turns[TARGET[0], i] for i in range(len(by_id[TARGET[0]]["history"]))]
    original_seed = turns[TARGET]["seed"]
    # Deterministically declared BEFORE any output is generated; no tunable seed.
    seed = int(base.stable_id(original_seed, POLICY, 1, length=16), 16) % (2**31 - 1)
    if seed == original_seed:
        raise ValueError("Recovery seed unexpectedly equals original seed")
    old_prompts, _ = candidate_inputs(turns[TARGET]["prompt_history"], manifest["defense"], original_seed)
    new_prompts, _ = candidate_inputs(turns[TARGET]["prompt_history"], manifest["defense"], seed)
    if old_prompts[0] == new_prompts[0]:
        raise ValueError("Predeclared seed repeats the failed first prompt; do not search for a replacement")
    if hashes != fingerprints(source):
        raise ValueError("Source changed during inspection")
    return {"manifest": manifest, "hashes": hashes, "dialogue": dialogue,
            "successful_turns": len(turns) - 1, "recovery_seed": seed}


def recovery_manifest(source, state):
    return {"policy": POLICY, "reporting": "supplemental; not original fixed-seed protocol",
            "source_directory": str(source.resolve()), "source_sha256": state["hashes"],
            "source_manifest": state["manifest"], "dialogue_id": TARGET[0], "turn_index": TARGET[1],
            "original_seed": state["dialogue"][TARGET[1]]["seed"],
            "recovery_seed": state["recovery_seed"],
            "seed_derivation": "int(stable_id(original_seed, policy, 1, length=16),16) % (2**31-1)",
            "attempt_limit": 1, "candidate_retries": 0, "unchanged_successful_turns": state["successful_turns"],
            "seed_scope": "all eight perturbations, candidate seeds and existing majority selection RNG",
            "implementation_sha256": {Path(__file__).name: base.file_sha256(Path(__file__)),
                                      **state["manifest"]["implementation_sha256"]}}


def make_record(state, manifest, generate):
    record = copy.deepcopy(state["dialogue"][TARGET[1]])
    seed = state["recovery_seed"]
    try:
        result = generate_smoothed(record["prompt_history"], state["manifest"]["defense"], seed, generate)
        error = None
    except CandidateFailure as exc:
        error = str(exc)
        result = {"message": "ERROR", "smoothllm": exc.audit,
                  **{k: None for k in ("prompt_tokens", "original_prompt_tokens", "completion_tokens",
                                      "latency_seconds", "input_truncated")}}
    run_id = base.stable_id(json.dumps(manifest, sort_keys=True))
    record.update({k: v for k, v in result.items() if k != "message"})
    record.update(seed=seed, original_seed=manifest["original_seed"], recovery_policy=POLICY,
                  generated_response=result["message"], error=error, tstamp=time.time(),
                  run_id=run_id, answer_id=base.stable_id(run_id, TARGET[0]))
    return record


def validate_result(folder, state, manifest, require_gpu=False):
    """Audit successes with the original SmoothLLM vote/cost/perturbation validator."""
    if json.loads((folder / "run_config.json").read_text()) != manifest:
        raise ValueError("Recovery manifest mismatch")
    if json.loads((folder / "original_dialogue.json").read_text()) != state["dialogue"]:
        raise ValueError("Original dialogue snapshot mismatch")
    marker = json.loads((folder / "attempt_started.json").read_text())
    if marker["policy"] != POLICY or marker["seed"] != state["recovery_seed"]:
        raise ValueError("Attempt marker policy/seed mismatch")
    result = json.loads((folder / "result.json").read_text())
    record = result.get("record")
    if record is not None:
        original = state["dialogue"][TARGET[1]]
        changed = {"seed", "original_seed", "recovery_policy", "generated_response", "error", "tstamp",
                   "run_id", "answer_id", "smoothllm", "prompt_tokens", "original_prompt_tokens",
                   "completion_tokens", "latency_seconds", "input_truncated"}
        if ({k: v for k, v in record.items() if k not in changed}
                != {k: v for k, v in original.items() if k not in changed}):
            raise ValueError("Recovery changed target metadata or gold history")
        run_id = base.stable_id(json.dumps(manifest, sort_keys=True))
        if (record["seed"] != state["recovery_seed"] or record["original_seed"] != original["seed"]
                or record["recovery_policy"] != POLICY or record["run_id"] != run_id
                or record["answer_id"] != base.stable_id(run_id, TARGET[0])):
            raise ValueError("Recovery seed, policy or identifiers mismatch")
        if not record.get("error"):
            validate_audit(record, state["manifest"]["defense"])
        else:
            prompts, _ = candidate_inputs(record["prompt_history"], state["manifest"]["defense"], record["seed"])
            from safedial_smoothllm import copy_seed
            audit = record["smoothllm"]
            candidates = audit["candidates"]
            if not 1 <= len(candidates) <= 8 or audit["selected_index"] is not None:
                raise ValueError("Invalid failed-candidate count or unexpected vote")
            if audit["majority_heuristic_jailbroken"] is not None or not candidates[-1].get("error"):
                raise ValueError("Failed recovery must end at a candidate failure without voting")
            for i, candidate in enumerate(candidates):
                if (candidate["index"] != i or candidate["seed"] != copy_seed(record["seed"], i)
                        or candidate["perturbed_user_message"] != prompts[i][-1]["content"]):
                    raise ValueError("Failed-candidate seed or perturbation mismatch")
    success = record is not None and not record.get("error") and not result.get("infrastructure_error")
    if result["success"] != success:
        raise ValueError("Recovery success flag mismatch")
    if manifest["source_sha256"] != fingerprints(Path(manifest["source_directory"])):
        raise ValueError("Original results changed")
    if require_gpu:
        runtime = result["runtime"]
        if (not runtime.get("device", "").startswith("cuda") or not runtime.get("gpu_name")
                or runtime.get("peak_gpu_allocated_bytes", 0) <= 0):
            raise ValueError("Missing recovery GPU evidence")
    return {"audit_passed": True, "recovery_success": success, "policy": POLICY,
            "original_successful_turns_preserved": state["successful_turns"],
            "candidate_count": len(record["smoothllm"]["candidates"]) if record else 0,
            "failure": result.get("infrastructure_error") or (record.get("error") if record else None),
            "supplemental_dialogue_id": TARGET[0], "primary_results_modified": False}


def expected_exports(state, result):
    records, answers = [], []
    if result["success"]:
        records = copy.deepcopy(state["dialogue"])
        records[TARGET[1]] = result["record"]
        r = result["record"]
        answers = [{"id": TARGET[0], "task": r["task"], "method": r["method"],
                    "answer_id": r["answer_id"], "model_id": r["model_id"], "tstamp": r["tstamp"],
                    "choices": [{"index": 0, "turns": [
                        {"role": "assistant", "message": t["generated_response"]} for t in records]}]}]
    return records, answers


def validate_exports(folder, state):
    result = json.loads((folder / "result.json").read_text())
    records, answers = expected_exports(state, result)
    if (base.load_jsonl(folder / "dialogue_turns.jsonl") != records
            or base.load_jsonl(folder / "answers.jsonl") != answers):
        raise ValueError("Recovery dialogue/answer export differs from audited records")


def finish(folder, state, manifest):
    report = validate_result(folder, state, manifest)
    result = json.loads((folder / "result.json").read_text())
    records, answers = expected_exports(state, result)
    resume.atomic_jsonl(folder / "dialogue_turns.jsonl", records)
    resume.atomic_jsonl(folder / "answers.jsonl", answers)
    validate_exports(folder, state)
    write_json(folder / "validation.json", report)
    print(json.dumps(report), flush=True)
    return 0 if report["recovery_success"] else 2


def gpu_generator(manifest):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; submit the GPU launcher manually")
    args = SimpleNamespace(**manifest)
    args.device = "cuda:0"
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = tokenizer.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=base.dtype_from_name(args.dtype, torch),
        device_map=args.device, trust_remote_code=True).eval()
    limit = base.effective_context_limit(model, tokenizer, args.max_input_tokens, args.max_new_tokens)

    def generate(messages, seed):
        return base.generate_one(model=model, tokenizer=tokenizer, messages=messages, args=args,
                                 seed=seed, torch_module=torch, input_limit=limit)

    def stats():
        return {"device": args.device, "gpu_name": torch.cuda.get_device_name(args.device),
                "torch_version": torch.__version__,
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
                "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(args.device)}
    return generate, stats


def run_attempt(folder, state, manifest, factory=gpu_generator):
    """Caller holds output/source locks. A durable marker prevents automatic retries."""
    base.ensure_manifest(folder / "run_config.json", manifest)
    snapshot = folder / "original_dialogue.json"
    if snapshot.exists() and json.loads(snapshot.read_text()) != state["dialogue"]:
        raise ValueError("Original snapshot changed")
    if not snapshot.exists():
        write_json(snapshot, state["dialogue"])
    if (folder / "result.json").exists():
        return finish(folder, state, manifest)
    marker = folder / "attempt_started.json"
    if marker.exists():
        raise RuntimeError("Attempt already started but no final result exists; inspect logs. No automatic retry.")
    with marker.open("x") as handle:
        json.dump({"policy": POLICY, "seed": state["recovery_seed"], "started_at": time.time()}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    print(f"ONE supplemental attempt: dialogue={TARGET[0]} turn_index={TARGET[1]} "
          f"seed={state['recovery_seed']}; eight candidates, zero retries", flush=True)
    started = time.perf_counter()
    try:
        generate, stats = factory(state["manifest"])
        calls = 0

        def journaled(messages, seed):
            nonlocal calls
            calls += 1
            if calls > 8:
                raise RuntimeError("Recovery candidate budget exceeded")
            value = generate(messages, seed)
            base.append_jsonl(folder / "candidate_calls.jsonl", [{"index": calls - 1, "seed": seed,
                              "perturbed_user_message": messages[-1]["content"], "result": value}])
            return value

        record = make_record(state, manifest, journaled)
        result = {"success": not bool(record.get("error")), "record": record,
                  "runtime": {**stats(), "elapsed_seconds": time.perf_counter() - started},
                  "infrastructure_error": None}
    except Exception as exc:
        result = {"success": False, "record": None,
                  "infrastructure_error": f"{type(exc).__name__}: {exc}",
                  "runtime": {"elapsed_seconds": time.perf_counter() - started}}
    write_json(folder / "result.json", result)
    return finish(folder, state, manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--require-gpu", action="store_true", help="require GPU evidence when auditing")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-only", action="store_true", help="read-only source/policy preflight, no GPU")
    modes.add_argument("--audit-only", action="store_true", help="audit saved result, no generation")
    args = parser.parse_args(argv)
    source, output = args.source_dir.resolve(), args.output_dir.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Recovery requires a separate sibling output directory")
    with source_lock(source):
        state = inspect_source(source)
        manifest = recovery_manifest(source, state)
        if args.validate_only:
            print(json.dumps({"passed": True, "original_successful_turns": state["successful_turns"],
                              "target": TARGET, "original_seed": manifest["original_seed"],
                              "recovery_seed": state["recovery_seed"], "model_calls": 0,
                              "attempt_limit": 1, "output_directory": str(output)}))
            return 0
        if args.audit_only:
            report = validate_result(output, state, manifest, args.require_gpu)
            validate_exports(output, state)
            print(json.dumps(report))
            return 0 if report["recovery_success"] else 2
        with resume.single_writer(output):
            return run_attempt(output, state, manifest)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)
