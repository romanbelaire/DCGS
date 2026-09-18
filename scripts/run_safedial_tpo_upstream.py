"""Durable run state for upstream TPO candidate handling; isolated from v8."""
import json
import time
from pathlib import Path

import run_safedial_baseline as base
import run_safedial_tpo as legacy
from run_safedial_tpo import single_writer, atomic_jsonl, read_journal, export_answers
from safedial_tpo_upstream import defense_config, generate_tpo, validate_audit


def parse_args(argv):
    argv = list(argv)
    if not any(a == "--empty-generation-retries" or a.startswith("--empty-generation-retries=") for a in argv):
        argv += ["--empty-generation-retries", "0"]
    args = legacy.parse_args(argv)
    if args.empty_generation_retries != 0 or args.retry_errors:
        raise ValueError("Upstream handling does not resample failures; do not use retry flags")
    return args


def implementation_hashes():
    folder = Path(__file__).parent
    names = ["run_safedial_tpo_upstream.py", "safedial_tpo_upstream.py", "run_safedial_tpo_full.py"]
    return {**legacy.implementation_hashes(), **{name: base.file_sha256(folder / name) for name in names}}


def manifest_for(args, selected, lock):
    manifest = legacy.manifest_for(args, selected, lock)
    manifest.update(defense=defense_config(args.sample_size, args.max_iters, args.max_new_tokens, args.feedback_tokens),
                    implementation_sha256=implementation_hashes(),
                    resume="immutable_events_and_terminal_outcomes; no_v8_import; no_failed_turn_resampling")
    return manifest


def failure_entries(run_id, turns, events):
    entries = []
    for (dialogue, index, event_index), event in sorted(events.items()):
        failure = event.get("candidate_failure")
        if failure:
            entries.append({"run_id": run_id, "dialogue_id": dialogue, "turn_index": index,
                            "event_index": event_index, "category": "candidate_parse", **failure})
        if event.get("error"):
            entries.append({"run_id": run_id, "dialogue_id": dialogue, "turn_index": index,
                            "event_index": event_index, "category": "fatal_execution",
                            "code": "execution_or_integrity_error", "action": "stop", "message": event["error"]})
    for (dialogue, index), record in sorted(turns.items()):
        if record.get("error"):
            entries.append({"run_id": run_id, "dialogue_id": dialogue, "turn_index": index,
                            "category": "terminal_output", "code": record["error"], "action": "continue_next_turn"})
    return entries


def load_state(args, manifest, selected, recover_tail=True):
    run_id = base.stable_id(json.dumps(manifest, sort_keys=True))
    by_id = {r["id"]: r for r in selected}
    expected = {(r["id"], i) for r in selected for i in range(len(r["history"]))}
    reader = read_journal if recover_tail else lambda p: base.load_jsonl(p) if p.exists() else []
    turns, events = {}, {}
    for record in reader(args.output_dir / "turns.jsonl"):
        key = (record["dialogue_id"], record["turn_index"])
        if key not in expected or key in turns or record["run_id"] != run_id or record["model_id"] != args.model_id:
            raise ValueError("Foreign or duplicate TPO turn")
        if record["choice_index"] != 0 or record["answer_id"] != base.stable_id(run_id, key[0]):
            raise ValueError("TPO answer identity mismatch")
        if record["prompt_history"] != base.gold_messages(by_id[key[0]]["history"], key[1]):
            raise ValueError("Saved TPO gold history changed")
        if record["seed"] != base.turn_seed(args.seed, args.model_id, key[0], 0, key[1]):
            raise ValueError("Saved TPO seed changed")
        validate_audit(record, manifest["defense"])
        turns[key] = record
    for row in reader(args.output_dir / "events.jsonl"):
        key = (row["dialogue_id"], row["turn_index"], row["event"]["index"])
        if key[:2] not in expected or key in events or row["run_id"] != run_id:
            raise ValueError("Foreign or duplicate TPO event")
        events[key] = row["event"]
    grouped = {}
    for key, event in sorted(events.items()):
        grouped.setdefault(key[:2], []).append(event)
    for key, record in turns.items():
        journal = grouped.get(key, [])
        if journal != record["tpo"]["events"]:
            raise ValueError("TPO turn/journal mismatch")
    for journal in grouped.values():
        indexes = [event["index"] for event in journal]
        if indexes != list(range(len(indexes))):
            raise ValueError("Noncontiguous TPO event journal")
    return run_id, turns, events


def coverage(selected, turns, events):
    expected = sum(len(r["history"]) for r in selected)
    failures = sum(bool(r.get("error")) for r in turns.values())
    fatal = sum(bool(e.get("error")) for e in events.values())
    successes = len(turns) - failures
    exported = sum(all((r["id"], i) in turns and not turns[r["id"], i].get("error")
                       for i in range(len(r["history"]))) for r in selected)
    return {"expected_turns": expected, "processed_turns": len(turns), "successful_turns": successes,
            "failed_turns": failures, "pending_turns": expected - len(turns),
            "processing_complete": len(turns) == expected,
            "complete_without_failures": successes == expected and not fatal,
            "exported_dialogues": exported,
            "skipped_candidates": sum(bool(e.get("candidate_failure")) for e in events.values()),
            "empty_candidates_scored": sum(not c["message"].strip() for r in turns.values() for c in r["tpo"]["candidates"]),
            "fatal_events": fatal}


def generate_selected(args, manifest, selected, execute):
    if (args.output_dir / "failure.json").exists():
        raise ValueError("Prior fatal failure requires inspection; do not delete the failure marker")
    run_id, turns, events = load_state(args, manifest, selected)
    saved_lengths = {}
    for key in events:
        saved_lengths[key[:2]] = max(saved_lengths.get(key[:2], 0), key[2] + 1)
    # Derived ledger is reconstructed after interrupted appends, without another model call.
    atomic_jsonl(args.output_dir / "failures.jsonl", failure_entries(run_id, turns, events))
    if any(e.get("error") for e in events.values()):
        raise ValueError("Fatal event in journal requires inspection")

    def export():
        export_answers(args, selected, turns, run_id)
        report = coverage(selected, turns, events)
        (args.output_dir / "coverage.json").write_text(json.dumps(report, indent=2) + "\n")

    export()
    try:
        for row in selected:
            for index, source in enumerate(row["history"]):
                key = (row["id"], index)
                if key in turns:  # Includes terminal empty-answer outcomes; never retry them.
                    continue
                cursor = 0

                def cached_execute(request):
                    nonlocal cursor
                    saved = events.get((*key, cursor))
                    cursor += 1
                    if saved:
                        if saved["request"] != request or saved.get("error"):
                            raise ValueError("Saved TPO request/error mismatch")
                        return saved["result"]
                    return execute(request)

                def persist(event):
                    event_key = (*key, event["index"])
                    if event_key in events:
                        if events[event_key] != event:
                            raise ValueError("Saved TPO event metadata changed")
                        return
                    base.append_jsonl(args.output_dir / "events.jsonl", [{"run_id": run_id,
                                      "dialogue_id": key[0], "turn_index": key[1], "event": event}])
                    events[event_key] = event
                    additions = failure_entries(run_id, {}, {event_key: event})
                    if additions:
                        base.append_jsonl(args.output_dir / "failures.jsonl", additions)
                        print(f"RECORDED dialogue={key[0]} turn={key[1]} event={event['index']}: {additions[0]['code']}", flush=True)
                    if event.get("format_warnings"):
                        print(f"WARNING dialogue={key[0]} turn={key[1]} event={event['index']}: {event['format_warnings']}", flush=True)

                messages = base.gold_messages(row["history"], index)
                seed = base.turn_seed(args.seed, args.model_id, row["id"], 0, index)
                result = generate_tpo(messages, manifest["defense"], seed, cached_execute, persist)
                if saved_lengths.get(key, 0) > cursor:
                    raise ValueError("Unused TPO journal events")
                record = {"run_id": run_id, "answer_id": base.stable_id(run_id, row["id"]),
                          "benchmark": "SafeDialBench", "protocol": base.PROTOCOL,
                          "model": args.model, "model_id": args.model_id, "dialogue_id": row["id"],
                          "task": row["task"], "method": row["method"], "scene": row["scene"],
                          "dataset_model_type": row.get("model_type"), "choice_index": 0, "turn_index": index,
                          "seed": seed, "prompt_history": messages, "user_message": source["user"],
                          "reference_response": source["bot"], "tstamp": time.time(),
                          "error": "empty_selected_answer" if not result["message"].strip() else None,
                          "generated_response": result["message"],
                          **{k: v for k, v in result.items() if k != "message"}}
                base.append_jsonl(args.output_dir / "turns.jsonl", [record])
                turns[key] = record
                if record["error"]:
                    base.append_jsonl(args.output_dir / "failures.jsonl", failure_entries(run_id, {key: record}, {}))
                print(f"Saved dialogue={key[0]} turn={key[1]} candidates={len(result['tpo']['candidates'])} error={record['error']}", flush=True)
            export()
    except Exception as exc:
        (args.output_dir / "failure.json").write_text(json.dumps({"category": "fatal_execution_or_integrity",
                      "type": type(exc).__name__, "message": str(exc)}, indent=2) + "\n")
        export()
        raise
    atomic_jsonl(args.output_dir / "failures.jsonl", failure_entries(run_id, turns, events))
    export()
    return 2 if any(r.get("error") for r in turns.values()) else 0
