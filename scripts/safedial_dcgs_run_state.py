"""Audit and execution-state helpers; no changes to the original DCGS policy."""
import json
from collections import Counter, defaultdict
from pathlib import Path

from run_safedial_baseline import file_sha256, gold_messages, load_jsonl, stable_id, turn_seed
from safedial_dcgs_wildjailbreak import PolicyFailure, TraceIncomplete, replay_trace, validate_turn


def read_records(path):
    return load_jsonl(path) if path.exists() else []


def normalized(value):
    return json.loads(json.dumps(value))


def event_key(event):
    return (event["invocation"], event["dialogue_id"], event["turn_index"], event["index"])


def failure_record(exc, invocation, row=None, turn=None, seed=None):
    from safedial_dcgs_wildjailbreak import exception_details
    details = exc.details if isinstance(exc, PolicyFailure) else exception_details(exc)
    audit = normalized(getattr(exc, "audit", None))
    # Journals sort JSON keys; identity must survive that serialization order.
    identity = json.dumps(details, sort_keys=True, separators=(",", ":"))
    return {"failure_id": stable_id(invocation, None if row is None else row["id"], turn, identity),
            "invocation": invocation, "dialogue_id": None if row is None else row["id"],
            "turn_index": turn, "seed": seed, "stage": getattr(exc, "stage", "initialization" if row is None else "execution"),
            **details, "event_index": audit["events"][-1]["index"] if audit and audit["events"] else None,
            "request": audit["events"][-1]["request"] if audit and audit["events"] else None,
            "audit": audit}


def context_summary(entries):
    """Separate distinct turns/contexts from repeated Q/Q_min/regret evaluations."""
    contexts, truncated_contexts, turns = set(), set(), set()
    calls, inputs, truncated_inputs = Counter(), Counter(), Counter()
    maximum, dropped, sides = 0, 0, set()
    for entry in entries:
        request, result = entry["request"], entry.get("result")
        if request["kind"] != "score" or entry.get("error") or result is None:
            continue
        info = result["critic_context"]  # Missing is an error, never zero truncation.
        head = request["function"]
        calls[head] += 1
        sides.add(info["truncation_side"])
        for i, (obs, belief) in enumerate(zip(request["observations"], request["high_level_actions"])):
            key = (entry["dialogue_id"], entry["turn_index"], obs, belief)
            contexts.add(key)
            inputs[head] += 1
            maximum = max(maximum, info["input_tokens"][i])
            dropped = max(dropped, info["dropped_tokens"][i])
            if info["input_truncated"][i]:
                truncated_contexts.add(key)
                turns.add(key[:2])
                truncated_inputs[head] += 1
    return {"critic_max_length": 1500, "critic_truncation_sides": sorted(sides),
            "critic_candidate_contexts": len(contexts), "critic_truncated_candidate_contexts": len(truncated_contexts),
            "critic_truncated_turns": len(turns), "critic_truncated_turn_keys": [list(k) for k in sorted(turns)],
            "critic_scoring_calls_by_head": dict(calls), "critic_scored_inputs_by_head": dict(inputs),
            "critic_truncated_inputs_by_head": dict(truncated_inputs),
            "critic_max_input_tokens": maximum, "critic_max_dropped_tokens": dropped,
            "critic_full_context": not bool(turns) if contexts else None}


def validate_recovery(folder, manifest):
    path = folder / "recovery.json"
    if not path.exists():
        return
    recovery = json.loads(path.read_text())
    archived = folder / "provenance/source_run"
    if recovery.get("policy_changed") is not False or not recovery.get("reason"):
        raise ValueError("Invalid recovery provenance")
    for rel, expected in recovery["source_file_sha256"].items():
        relative = Path(rel)
        if relative.is_absolute() or ".." in relative.parts or file_sha256(archived / relative) != expected:
            raise ValueError("Recovery source snapshot mismatch")
    if json.loads((archived / "run_config.json").read_text()) != manifest:
        raise ValueError("Recovery changed method or source identity")
    if file_sha256(archived / "run_config.json") != recovery["source_manifest_sha256"]:
        raise ValueError("Recovery source manifest mismatch")
    original_failures = read_records(archived / "failures.jsonl")
    reconciled = recovery.get("reconciled_failures", [])
    original_events = read_records(archived / "events.jsonl")
    for failure in reconciled:
        group = [e for e in original_events if (e["invocation"], e["dialogue_id"], e["turn_index"]) ==
                 (failure["invocation"], failure["dialogue_id"], failure["turn_index"])]
        bare = [{k: v for k, v in e.items() if k not in ("invocation", "dialogue_id", "turn_index", "seed")} for e in group]
        if not group or not failure.get("audit") or failure["audit"]["events"] != bare or failure["category"] == "integrity":
            raise ValueError("Recovery reconciliation lacks original call evidence")
        if failure not in read_records(folder / "failures.jsonl"):
            raise ValueError("Reconciled failure missing from ledger")
    if set(recovery.get("acknowledged_failure_ids", [])) != {f["failure_id"] for f in original_failures + reconciled}:
        raise ValueError("Recovery acknowledged an uninspected failure")
    for name in ("turns.jsonl", "events.jsonl", "attempts.jsonl", "failures.jsonl", "runtime.jsonl", "model_loading.jsonl"):
        if (archived / name).exists() and not (folder / name).read_bytes().startswith((archived / name).read_bytes()):
            raise ValueError("Recovery changed preserved records")



def audit_state(folder, selected, manifest, config, tokenizer):
    """Validate successful, failed, and interrupted traces before any new work."""
    validate_recovery(folder, manifest)
    rows = {r["id"]: r for r in selected}
    expected = {(r["id"], t) for r in selected for t in range(len(r["history"]))}
    records = read_records(folder / "turns.jsonl")
    failures = read_records(folder / "failures.jsonl")
    journal = read_records(folder / "events.jsonl")
    attempts = read_records(folder / "attempts.jsonl")

    def check_identity(item):
        key = (item["dialogue_id"], item["turn_index"])
        if key not in expected or item["seed"] != turn_seed(manifest["seed"], manifest["model_id"], key[0], 0, key[1]):
            raise ValueError("Unexpected turn identity or seed")
        if not isinstance(item["invocation"], str) or not item["invocation"]:
            raise ValueError("Missing invocation identity")
        return key

    indexed, groups, attempted = {}, defaultdict(list), {}
    for entry in attempts:
        check_identity(entry)
        key = event_key(entry)
        if key in attempted:
            raise ValueError("Duplicate call-start event")
        attempted[key] = entry
    for entry in journal:
        check_identity(entry)
        key = event_key(entry)
        if key in indexed or key not in attempted:
            raise ValueError("Duplicate event or missing call-start evidence")
        start = attempted[key]
        if start["request"] != entry["request"] or start["seed"] != entry["seed"]:
            raise ValueError("Attempt/result mismatch")
        indexed[key] = entry
        groups[key[:3]].append(entry)
    for key, group in groups.items():
        if [e["index"] for e in group] != list(range(len(group))):
            raise ValueError("Noncontiguous journal")
    starts_by_group = defaultdict(list)
    for key in attempted:
        starts_by_group[key[:3]].append(key[3])
    for key, indices in starts_by_group.items():
        if indices != list(range(len(indices))) or len(indices) - len(groups[key]) not in (0, 1):
            raise ValueError("Noncontiguous call-start journal")

    def bare(group):
        return [{k: v for k, v in e.items() if k not in ("invocation", "dialogue_id", "turn_index", "seed")} for e in group]

    done, terminal, committed, used = set(), set(), set(), set()
    for record in records:
        key = check_identity(record)
        if key in done or record.get("error") or record["choice_index"] != 0 or record["model_id"] != manifest["model_id"]:
            raise ValueError("Duplicate or invalid saved turn")
        if record["prompt_history"] != gold_messages(rows[key[0]]["history"], key[1]):
            raise ValueError("Gold-history mismatch")
        validate_turn(record, rows[key[0]], config, tokenizer)
        group_key = (record["invocation"], *key)
        if record["dcgs_original"]["events"] != bare(groups[group_key]):
            raise ValueError("Turn/journal mismatch")
        used.update(event_key(e) for e in groups[group_key])
        committed.add(group_key)
        done.add(key)
    failure_ids = set()
    for failure in failures:
        if failure["failure_id"] in failure_ids or failure["category"] not in ("terminal_output", "infrastructure", "integrity"):
            raise ValueError("Duplicate or invalid failure record")
        failure_ids.add(failure["failure_id"])
        if failure["dialogue_id"] is None:
            if failure["audit"] is not None or failure["stage"] != "initialization" or failure["category"] == "terminal_output":
                raise ValueError("Invalid initialization failure")
            continue
        key = check_identity(failure)
        group_key = (failure["invocation"], *key)
        if failure["audit"] is not None:
            if failure["audit"]["events"] != bare(groups[group_key]):
                raise ValueError("Failure/journal mismatch")
            try:
                replay_trace(rows[key[0]], key[1], failure["seed"], config, tokenizer, bare(groups[group_key]))
            except PolicyFailure as exc:
                if failure_record(exc, failure["invocation"], rows[key[0]], key[1], failure["seed"]) != failure:
                    raise ValueError("Failure replay mismatch")
            else:
                raise ValueError("Recorded failure replays successfully")
        elif failure["category"] == "terminal_output":
            raise ValueError("Terminal failure lacks policy evidence")
        committed.add(group_key)
        if failure["category"] == "terminal_output":
            if key in terminal or key in done:
                raise ValueError("Terminal output was retried or also marked successful")
            terminal.add(key)

    terminal_or_success_groups = {(r["invocation"], r["dialogue_id"], r["turn_index"]) for r in records}
    terminal_or_success_groups.update((f["invocation"], f["dialogue_id"], f["turn_index"])
                                     for f in failures if f["category"] == "terminal_output")
    positions = {event_key(e): i for i, e in enumerate(attempts)}
    last_turn_position = {(e["dialogue_id"], e["turn_index"]): i for i, e in enumerate(attempts)}
    for group_key in terminal_or_success_groups:
        end = max(positions[event_key(e)] for e in groups[group_key])
        if last_turn_position[group_key[1:]] > end:
            raise ValueError("Completed or terminal turn was attempted again")

    # Reconstruct outcomes whose call journal was fsynced but whose final record
    # was interrupted. A completed blank response must never receive a new draw.
    uncommitted = []
    for group_key, group in list(groups.items()):
        if group_key in committed or not group:
            continue
        invocation, dialogue, turn = group_key
        events = bare(group)
        try:
            result = replay_trace(rows[dialogue], turn, group[0]["seed"], config, tokenizer, events, True)
        except TraceIncomplete:
            continue
        except PolicyFailure as exc:
            if normalized(exc.audit["events"]) != events:
                raise ValueError("Interrupted failure trace does not replay")
            uncommitted.append((group_key, None, exc))
        else:
            if (dialogue, turn) in done or (dialogue, turn) in terminal:
                raise ValueError("Repeated completed policy execution")
            uncommitted.append((group_key, result, None))

    blocker = json.loads((folder / "failure.json").read_text()) if (folder / "failure.json").exists() else None
    if blocker is not None and blocker not in failures:
        raise ValueError("Blocking failure is not in the durable ledger")
    acknowledged = set()
    if (folder / "recovery.json").exists():
        acknowledged = set(json.loads((folder / "recovery.json").read_text())["acknowledged_failure_ids"])
    if blocker is None:
        mode = manifest.get("execution_policy", {}).get("on_turn_error", "stop")
        blocker = next((f for f in failures if f["failure_id"] not in acknowledged and
                        (f["category"] != "terminal_output" or mode == "stop")), None)
    return {"records": records, "failures": failures, "journal": journal, "attempts": attempts,
            "done": done, "terminal": terminal, "expected": expected, "used": used,
            "uncommitted": uncommitted, "blocker": blocker,
            "unknown_cost_calls": len(set(attempted) - set(indexed))}


def coverage(state):
    unattempted = state["expected"] - state["done"] - state["terminal"]
    return {"expected_turns": len(state["expected"]), "successful_turns": len(state["done"]),
            "terminal_failed_turns": len(state["terminal"]), "remaining_turns": len(unattempted),
            "terminal_failed_turn_keys": [list(k) for k in sorted(state["terminal"])],
            "execution_finished": not bool(unattempted) and state["blocker"] is None and not state["uncommitted"],
            "complete_without_failures": state["done"] == state["expected"] and state["blocker"] is None and not state["uncommitted"]}
