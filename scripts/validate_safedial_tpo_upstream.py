#!/usr/bin/env python3
"""Audit upstream-handling TPO, separating processing from answer completeness."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import run_safedial_baseline as base
from run_safedial_tpo_upstream import load_state, coverage, failure_entries
from run_safedial_tpo_upstream_full import implementation_hashes, CONTEXT_POLICY, REWARD_CONTEXT_TOKENS
from safedial_tpo_upstream import defense_config


def validate_full(output_dir, require_gpu=False):
    folder = Path(output_dir)
    manifest = json.loads((folder / "run_config.json").read_text())
    if manifest["implementation_sha256"] != implementation_hashes():
        raise ValueError("TPO source hashes changed")
    config = manifest["defense"]
    if config != defense_config(config["sample_size"], config["max_iterations"], config["response_tokens"], config["feedback_tokens"]):
        raise ValueError("Unknown TPO configuration")
    if manifest.get("full_run_context") != CONTEXT_POLICY:
        raise ValueError("Unknown full-run context policy")
    if (manifest["num_choices"] != 1 or manifest["temperature"] != 0.7 or manifest["top_p"] != 0.95
            or manifest["max_new_tokens"] != config["response_tokens"]):
        raise ValueError("TPO decoding settings changed")
    dataset = Path(manifest["dataset"])
    if base.file_sha256(dataset) != manifest["dataset_sha256"]:
        raise ValueError("Dataset hash mismatch")
    rows = {r["id"]: r for r in base.load_jsonl(dataset)}
    selected = [rows[i] for i in manifest["selected_ids"]]
    if len(set(manifest["selected_ids"])) != len(selected):
        raise ValueError("Duplicate selected IDs")
    args = SimpleNamespace(output_dir=folder, model_id=manifest["model_id"], seed=manifest["seed"])
    run_id, turns, events = load_state(args, manifest, selected, recover_tail=False)
    expected_failures = failure_entries(run_id, turns, events)
    actual_failures = base.load_jsonl(folder / "failures.jsonl")
    # Order differs during a live append versus final derived export.
    canonical = lambda entries: sorted(json.dumps(e, sort_keys=True) for e in entries)
    if canonical(actual_failures) != canonical(expected_failures):
        raise ValueError("Failure ledger disagrees with audited outcomes")
    expected_answers = []
    for row in selected:
        records = [turns.get((row["id"], i)) for i in range(len(row["history"]))]
        if any(r is None or r.get("error") for r in records):
            continue
        expected_answers.append({"id": row["id"], "task": row["task"], "method": row["method"],
                                 "model_id": args.model_id, "answer_id": base.stable_id(run_id, row["id"]),
                                 "choices": [{"index": 0, "turns": [{"role": "assistant", "message": r["generated_response"]} for r in records]}],
                                 "tstamp": records[-1]["tstamp"]})
    if base.load_jsonl(folder / "answers.jsonl") != expected_answers:
        raise ValueError("Native answer export mismatch")
    report = coverage(selected, turns, events)
    used = {(k[0], k[1], e["index"]) for k, r in turns.items() for e in r["tpo"]["events"]}
    if report["processing_complete"] and used != set(events):
        raise ValueError("Unused completed-run journal events")
    marker = (folder / "failure.json").exists()
    report.update(audit_passed=not report["fatal_events"] and not marker and used == set(events),
                  passed=report["complete_without_failures"] and not marker,
                  partial_events=len(set(events) - used),
                  candidates=sum(len(r["tpo"]["candidates"]) for r in turns.values()),
                  actual_generation_calls=sum(e["request"]["kind"] == "generate" for e in events.values()),
                  actual_reward_calls=sum(e["request"]["kind"] == "reward" for e in events.values()),
                  extra_generation_retries=0, judge_calls=0)
    if require_gpu:
        runtime = json.loads((folder / "runtime_stats.json").read_text())
        loading = json.loads((folder / "model_loading.json").read_text())
        probe = json.loads((folder / "context_probe.json").read_text())
        expected_status = 0 if report["complete_without_failures"] else 2
        if (not runtime.get("cuda_ready") or not runtime.get("peak_gpu_allocated_bytes")
                or runtime.get("failure") or runtime.get("exit_code") != expected_status):
            raise ValueError("GPU runtime evidence missing or failed")
        if set(loading) != {"actor", "reward"} or any(not v.get("passed") or v.get("trainable_parameters") != 0 for v in loading.values()):
            raise ValueError("Frozen checkpoint loading not verified")
        if not probe.get("passed") or probe.get("sequence_tokens") != REWARD_CONTEXT_TOKENS:
            raise ValueError("Reward context probe failed")
        report["gpu_evidence_passed"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    report = validate_full(args.output_dir, args.require_gpu)
    (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
