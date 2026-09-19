#!/usr/bin/env python3
"""SafeDial main DCGS with last-token HL pooling and trained LL critic selection."""
import argparse
import fcntl
import json
import os
import shutil
import tempfile
from pathlib import Path
import time
import uuid

from run_safedial_baseline import (DEFAULT_DATASET, append_jsonl, file_sha256,
    gold_messages, load_jsonl, select_dialogues, stable_id, turn_seed, validate_dataset)
from safedial_dcgs_wildjailbreak import (ROOT, REFERENCE, LocalBackend, PolicyFailure,
    MAX_UNTRUNCATED_LEN, METHOD_SPEC, PREFIX_TEMPLATE, configuration, generate_turn, validate_turn)
from safedial_dcgs_run_state import (audit_state, context_summary, coverage, event_key,
                                     failure_record, read_records)

PROTOCOL = "safedial_main_wildjailbreak_v3"
LOCK = ROOT / "configs/safedial/dcgs_main.lock.json"


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def source_hashes():
    files = list((ROOT / "src").rglob("*.py")) + list((ROOT / "src/prompts").glob("*.jsonl"))
    files += [REFERENCE, ROOT / "models/MANIFEST.json"] + [ROOT / "scripts" / name for name in (
        "run_safedial_dcgs.py", "run_safedial_baseline.py", "judge_safedial.py",
        "validate_safedial_generation.py", "safedial_dcgs_wildjailbreak.py",
        "run_safedial_dcgs_wildjailbreak.py", "validate_safedial_dcgs_wildjailbreak.py",
        "safedial_dcgs_run_state.py")]
    return {path.relative_to(ROOT).as_posix(): file_sha256(path) for path in sorted(files)}


def load_artifact_lock(path=LOCK):
    lock = json.loads(Path(path).read_text())
    for name in ("actor", "high_level", "token_critic"):
        entry = lock[name]
        artifact = Path(entry["path"])
        entry["path"] = str(artifact if artifact.is_absolute() else ROOT / artifact)
    return lock


def verify_artifacts(lock):
    for name, digest in lock["actor"]["sha256"].items():
        if file_sha256(Path(lock["actor"]["path"]) / name) != digest:
            raise ValueError(f"Actor artifact mismatch: {name}")
    for name in ("high_level", "token_critic"):
        path = Path(lock[name]["path"])
        if file_sha256(path) != lock[name]["sha256"]:
            raise ValueError(f"{name} checkpoint mismatch: {path}. Materialize the pinned LFS weights before running.")


def manifest_for(args, selected, lock):
    config = configuration(args.method, args.device, lock)
    return {"benchmark": "SafeDialBench", "protocol": PROTOCOL, "method": args.method,
            "dataset": str(args.dataset.resolve()), "dataset_sha256": file_sha256(args.dataset),
            "selected_ids": [r["id"] for r in selected], "num_choices": 1, "seed": args.seed,
            "model_id": "zephyr-7b-beta-" + args.method + "-main-v3",
            "method_spec": METHOD_SPEC,
            "reference_config": str(REFERENCE), "effective_config": vars(config),
            "artifacts": {k: lock[k] for k in ("actor", "high_level", "token_critic")},
            "source_sha256": source_hashes(), "generation_only": True,
            "resume_unit": "complete turn; interrupted turn restarts from its original seed",
            "custom_empty_retries": 0, "critic_truncation": "original 1500-token cutoff",
            "execution_policy": {"on_turn_error": getattr(args, "on_turn_error", "stop"),
                "terminal_outputs": ["blank_final_response", "blank_ll_candidate", "beliefs_exhausted", "ll_context_overflow"],
                "infrastructure_recovery": "inspected fresh directory; original seed",
                "integrity_errors": "fatal"}}


def tokenizer_for(lock):
    from src.utils.llm_utils import get_tokenizer_instance
    return get_tokenizer_instance(lock["actor"]["path"])


def preflight(selected, config, tokenizer):
    # Synthetic responses expose the actual upstream prompts and generation budgets.
    # LL estimates reserve the entire pool budget for one response, since the list
    # parser does not enforce an equal token share across candidates.
    maximum = 0
    calls = 0
    ll_estimates = []
    current = {}

    def token_count(text):
        return len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])

    def probe(request):
        nonlocal maximum, calls
        if request["kind"] == "score":
            current["scoring_started"] = True
            return {"scores": [0.0] * len(request["observations"])}
        if request["kind"] == "score_ll":
            prefix = PREFIX_TEMPLATE.format(observation=request["observation"], selected_belief="")
            synthetic_prefix = PREFIX_TEMPLATE.format(observation=request["observation"],
                                                       selected_belief=request["selected_belief"])
            prefix_tokens = token_count(prefix)
            ll_estimates.append({"dialogue_id": current["dialogue_id"], "turn_index": current["turn_index"],
                "observation_prefix_tokens": prefix_tokens,
                "belief_generation_budget": current["hl_budget"],
                "response_generation_budget": current["ll_budget"],
                "estimated_input_tokens": prefix_tokens + current["hl_budget"] + current["ll_budget"],
                "synthetic_input_tokens": max(token_count(synthetic_prefix + a) for a in request["actions"])})
            return {"scores": {a: 0.0 for a in request["actions"]}, "objective": "shapley"}
        calls += 1
        budget_key = "ll_budget" if current["scoring_started"] else "hl_budget"
        current[budget_key] = max(current[budget_key], request["max_new_tokens"])
        for prompt in request["prompts"]:
            suffix = len(tokenizer(request["prefill_suffix"], add_special_tokens=False)["input_ids"]) if request.get("prefill_suffix") else 0
            maximum = max(maximum, token_count(prompt) + suffix + request["max_new_tokens"])
        text = "\n".join(f"{i}. Synthetic preflight instruction {i}." for i in range(1, 6))
        if request["max_new_tokens"] == 128 * config.n_ll_candidates:
            text = "\n".join(f"{i}. [RESPONSE]Synthetic response {i}.[/RESPONSE]" for i in range(1, config.n_ll_candidates + 1))
        return {"texts": [text if request["do_sample"] else "Synthetic preflight response."] * len(request["prompts"])}

    score_entries = []
    for row in selected:
        for turn in range(len(row["history"])):
            current = {"dialogue_id": row["id"], "turn_index": turn, "scoring_started": False,
                       "hl_budget": 0, "ll_budget": 0}
            try:
                audit = generate_turn(row, turn, 0, config, tokenizer, probe)["dcgs_original"]
            except PolicyFailure as exc:
                if exc.details["code"] != "ll_context_overflow":
                    raise
                # This is a declared terminal outcome; continue the dataset scan
                # so preflight lists all risks before any GPU work.
                audit = exc.audit
            score_entries.extend({"dialogue_id": row["id"], "turn_index": turn, **e}
                                 for e in audit["events"] if e["request"]["kind"] == "score")
    if maximum > 32768:
        raise ValueError("Original prompts exceed pinned actor context")
    risks = [entry for entry in ll_estimates
             if max(entry["estimated_input_tokens"], entry["synthetic_input_tokens"]) > MAX_UNTRUNCATED_LEN]
    return {"passed": True, "dialogues": len(selected), "turns": sum(len(r["history"]) for r in selected),
            "synthetic_prompt_calls": calls, "max_prompt_plus_generation_tokens": maximum,
            "model_calls": 0, "estimated_critic_context": context_summary(score_entries),
            "ll_critic_context": {
                "max_input_tokens": MAX_UNTRUNCATED_LEN,
                "max_estimated_input_tokens": max((e["estimated_input_tokens"] for e in ll_estimates), default=0),
                "max_synthetic_input_tokens": max((e["synthetic_input_tokens"] for e in ll_estimates), default=0),
                "potential_overflow_turns": len(risks), "potential_overflow_turn_details": risks,
                "synthetic_overflow_turns": sum(e["synthetic_input_tokens"] > MAX_UNTRUNCATED_LEN for e in ll_estimates),
                "overflow_policy": METHOD_SPEC["ll_context_overflow"],
                "note": "Estimates reserve upstream generation budgets; decoded-text retokenization can vary. Runtime checks are authoritative."},
            "note": "Preflight completion does not guarantee every turn fits. Review LL context risks; full runs record overflow failures and continue, smoke runs stop."}


def answers_for(selected, manifest, records):
    by_key = {(r["dialogue_id"], r["turn_index"]): r for r in records}
    answers = []
    for row in selected:
        turns = [by_key.get((row["id"], i)) for i in range(len(row["history"]))]
        if all(turns):
            answers.append({"id": row["id"], "task": row["task"], "method": row["method"],
                "model_id": manifest["model_id"], "answer_id": stable_id(manifest, row["id"]),
                "choices": [{"index": 0, "turns": [{"role": "assistant", "message": t["generated_response"]} for t in turns]}]})
    return answers


def export_answers(folder, selected, manifest, records):
    answers = answers_for(selected, manifest, records)
    path = folder / "answers.jsonl"
    content = "".join(json.dumps(answer) + "\n" for answer in answers)
    if not path.exists() or path.read_text() != content:
        temporary = path.with_suffix(".tmp")
        with temporary.open("w") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)


def success_record(row, turn, seed, invocation, manifest, result):
    return {"dialogue_id": row["id"], "turn_index": turn, "choice_index": 0,
            "seed": seed, "model_id": manifest["model_id"], "invocation": invocation,
            "prompt_history": gold_messages(row["history"], turn),
            "generated_response": result["message"], "dcgs_original": result["dcgs_original"]}


def run_selected(folder, selected, manifest, config, tokenizer, backend_factory):
    """Continue independent turns without giving terminal outputs new attempts."""
    state = audit_state(folder, selected, manifest, config, tokenizer)
    if state["blocker"]:
        raise ValueError("A policy failure or infrastructure failure is saved; use inspected recovery into a fresh directory")
    rows = {row["id"]: row for row in selected}
    mode = manifest.get("execution_policy", {}).get("on_turn_error", "stop")
    # Reconcile a crash between durable call completion and turn/failure commit.
    for (invocation, dialogue, turn), result, exc in state["uncommitted"]:
        row = rows[dialogue]
        seed = turn_seed(manifest["seed"], manifest["model_id"], dialogue, 0, turn)
        if exc is None:
            append_jsonl(folder / "turns.jsonl", [success_record(row, turn, seed, invocation, manifest, result)])
        else:
            failure = failure_record(exc, invocation, row, turn, seed)
            append_jsonl(folder / "failures.jsonl", [failure])
            if failure["category"] != "terminal_output" or mode == "stop":
                write_json(folder / "failure.json", failure)
                raise exc
    if state["uncommitted"]:
        state = audit_state(folder, selected, manifest, config, tokenizer)
    records = state["records"]
    export_answers(folder, selected, manifest, records)
    pending = [(row, i) for row in selected for i in range(len(row["history"]))
               if (row["id"], i) not in state["done"] | state["terminal"]]
    if not pending:
        return 0 if not state["terminal"] else 2
    invocation = uuid.uuid4().hex
    started = time.perf_counter()
    backend, row, turn, seed = None, None, None, None
    finished, model_work_started, failure_written = False, False, False
    try:
        backend = backend_factory()
        if hasattr(backend, "loading"):
            write_json(folder / "model_loading.json", backend.loading)
            append_jsonl(folder / "model_loading.jsonl", [{"invocation": invocation, **backend.loading}])
        for row, turn in pending:
            failure_written = False
            seed = turn_seed(manifest["seed"], manifest["model_id"], row["id"], 0, turn)
            def save_event(event):
                append_jsonl(folder / "events.jsonl", [{"invocation": invocation,
                    "dialogue_id": row["id"], "turn_index": turn, "seed": seed, **event}])
            call_index = 0
            def execute(request):
                nonlocal call_index, model_work_started
                append_jsonl(folder / "attempts.jsonl", [{"invocation": invocation,
                    "dialogue_id": row["id"], "turn_index": turn, "seed": seed,
                    "index": call_index, "request": request}])
                call_index += 1
                model_work_started = True
                return backend(request)
            try:
                result = generate_turn(row, turn, seed, config, tokenizer, execute, save_event)
            except PolicyFailure as exc:
                failure = failure_record(exc, invocation, row, turn, seed)
                append_jsonl(folder / "failures.jsonl", [failure])
                failure_written = True
                if failure["category"] == "terminal_output" and mode == "record-and-continue":
                    print(f"Terminal failure dialogue={row['id']} turn={turn}: {failure['code']}", flush=True)
                    continue
                write_json(folder / "failure.json", failure)
                raise
            record = success_record(row, turn, seed, invocation, manifest, result)
            append_jsonl(folder / "turns.jsonl", [record])
            records.append(record)
            export_answers(folder, selected, manifest, records)
            print(f"Saved dialogue {row['id']} turn {turn}", flush=True)
        state = audit_state(folder, selected, manifest, config, tokenizer)
        write_json(folder / "status.json", coverage(state))
        finished = True
        return 0 if not state["terminal"] else 2
    except Exception as exc:
        if not failure_written:
            failure = failure_record(exc, invocation, row, turn, seed)
            append_jsonl(folder / "failures.jsonl", [failure])
            write_json(folder / "failure.json", failure)
        raise
    finally:
        runtime = {"invocation": invocation, "execution_finished": finished,
                   "model_work_started": model_work_started, "elapsed_seconds": time.perf_counter() - started}
        if backend is not None and hasattr(backend, "torch") and config.device.startswith("cuda"):
            try:
                runtime.update(peak_gpu_allocated_bytes=backend.torch.cuda.max_memory_allocated(config.device),
                               peak_gpu_reserved_bytes=backend.torch.cuda.max_memory_reserved(config.device))
            except RuntimeError as exc:
                runtime["gpu_measurement_error"] = str(exc)
        append_jsonl(folder / "runtime.jsonl", [runtime])


def recover_run(source, target, manifest, selected, config, tokenizer, reason):
    """Verified same-policy continuation. Source stays intact; no inference."""
    source, target = Path(source).resolve(), Path(target).resolve()
    if target.exists() or source == target or source in target.parents or target in source.parents:
        raise ValueError("Recovery requires a fresh, separate output directory")
    if not reason or not reason.strip():
        raise ValueError("An inspection/recovery reason is required")
    with (source / ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = json.loads((source / "run_config.json").read_text())
        if old != manifest:
            raise ValueError("Recovery cannot change source hashes, seeds, method, artifacts or execution policy")
        state = audit_state(source, selected, manifest, config, tokenizer)
        if any(f["category"] == "integrity" for f in state["failures"]):
            raise ValueError("Integrity failures require investigation; recovery cannot bypass them")
        if any(exc and exc.details["category"] == "integrity" for _, _, exc in state["uncommitted"]):
            raise ValueError("Uncommitted integrity failure cannot be recovered")
        originals = {str(p.relative_to(source)): file_sha256(p) for p in source.rglob("*")
                     if p.is_file() and p.name != ".lock"}
        target.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".dcgs-recovery-", dir=target.parent))
        try:
            # Preserve a complete source snapshot, including prior recoveries.
            archived = stage / "provenance/source_run"
            shutil.copytree(source, archived, ignore=shutil.ignore_patterns(".lock"))
            for name in ("run_config.json", "turns.jsonl", "events.jsonl", "attempts.jsonl",
                         "failures.jsonl", "runtime.jsonl", "model_loading.jsonl", "model_loading.json", "answers.jsonl"):
                if (source / name).exists():
                    shutil.copy2(source / name, stage / name)
            reconciled_failures = []
            by_id = {r["id"]: r for r in selected}
            for (invocation, dialogue, turn), result, exc in state["uncommitted"]:
                seed = turn_seed(manifest["seed"], manifest["model_id"], dialogue, 0, turn)
                if exc is None:
                    append_jsonl(stage / "turns.jsonl", [success_record(by_id[dialogue], turn, seed, invocation, manifest, result)])
                else:
                    failure = failure_record(exc, invocation, by_id[dialogue], turn, seed)
                    append_jsonl(stage / "failures.jsonl", [failure])
                    reconciled_failures.append(failure)
            # Recovery acknowledges the blocker; it never deletes its ledger entry
            # or retries a terminal output. Reconciliation handles interrupted commits.
            for rel, digest in originals.items():
                if file_sha256(source / rel) != digest or file_sha256(archived / rel) != digest:
                    raise ValueError("Source changed during recovery")
            write_json(stage / "recovery.json", {"source_directory": str(source), "reason": reason,
                "source_file_sha256": originals, "source_manifest_sha256": file_sha256(source / "run_config.json"),
                "model_calls": 0, "policy_changed": False, "terminal_outputs_are_not_retried": True,
                "reconciled_failures": reconciled_failures,
                "acknowledged_failure_ids": [f["failure_id"] for f in state["failures"] + reconciled_failures]})
            recovered = audit_state(stage, selected, manifest, config, tokenizer)
            export_answers(stage, selected, manifest, recovered["records"])
            if target.exists():
                raise ValueError("Recovery destination appeared during preparation")
            stage.rename(target)
        except BaseException:
            # Keep failed staging evidence for inspection; never overwrite a destination.
            raise
    return {"prepared": str(target), "successful_turns_preserved": len(recovered["done"]),
            "terminal_failures_preserved": len(recovered["terminal"]), "model_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("vdcgs", "rdcgs"), required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--artifact-lock", type=Path, default=LOCK, help="Pinned actor, indexed HL critic, and trained LL critic")
    parser.add_argument("--seed", type=int, default=0)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--ids")
    selection.add_argument("--limit", type=int)
    selection.add_argument("--per-task", type=int)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--on-turn-error", choices=("stop", "record-and-continue"), default="stop")
    parser.add_argument("--recover-from", type=Path, help="Prepare a verified fresh continuation; does not run inference")
    parser.add_argument("--recovery-reason", help="What failed and why continuation is appropriate")
    args = parser.parse_args()
    rows = load_jsonl(args.dataset)
    validate_dataset(rows, args.dataset)
    selected = select_dialogues(rows, args)
    lock = load_artifact_lock(args.artifact_lock)
    config = configuration(args.method, args.device, lock)
    verify_artifacts(lock)
    tokenizer = tokenizer_for(lock)
    manifest = json.loads(json.dumps(manifest_for(args, selected, lock)))
    if args.recover_from:
        print(json.dumps(recover_run(args.recover_from, args.output_dir, manifest, selected, config,
                                     tokenizer, args.recovery_reason)))
        return 0
    if args.recovery_reason:
        raise ValueError("--recovery-reason requires --recover-from")
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
