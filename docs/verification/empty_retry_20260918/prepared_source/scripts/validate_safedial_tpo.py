#!/usr/bin/env python3
"""Offline coverage, native-history and full TPO event/cost validation."""

import argparse
import json
from pathlib import Path

import run_safedial_baseline as base
from run_safedial_tpo import implementation_hashes
from safedial_tpo import MISSING_END_WARNING, TRAILING_START_WARNING, TERMINAL_START_WARNING, DUPLICATE_BLOCK_WARNING, FIRST_IMPROVEMENT_WARNING, defense_config, validate_audit
from validate_safedial_generation import validate_run


def validate_tpo(output_dir, require_gpu=False, expected_implementation=None):
    folder = Path(output_dir)
    report = validate_run(folder)
    manifest = json.loads((folder / "run_config.json").read_text())
    config = manifest["defense"]
    if config != defense_config(config["sample_size"], config["max_iterations"], config["response_tokens"], config["feedback_tokens"], config.get("empty_generation_retry", {}).get("extra_attempts", 0)):
        raise ValueError("Unknown TPO configuration")
    expected_implementation = implementation_hashes() if expected_implementation is None else expected_implementation
    if manifest["implementation_sha256"] != expected_implementation:
        raise ValueError("TPO source hashes changed since generation")
    if manifest["num_choices"] != 1 or manifest["temperature"] != 0.7 or manifest["top_p"] != 0.95:
        raise ValueError("Unexpected TPO decoding/choice settings")
    if config["response_tokens"] != manifest["max_new_tokens"] or "adapter" in manifest:
        raise ValueError("TPO response budget/actor mismatch")
    run_id = base.stable_id(json.dumps(manifest, sort_keys=True))
    journal = {}
    for entry in base.load_jsonl(folder / "events.jsonl"):
        if entry["run_id"] != run_id:
            raise ValueError("Foreign TPO event run")
        journal[(entry["dialogue_id"], entry["turn_index"], entry["event"]["index"])] = entry["event"]
    used = set()
    capped = 0
    retry_generations = 0
    empty_attempts = 0
    actual_generations = 0
    missing_closing_tags = 0
    trailing_opening_tags = 0
    terminal_opening_tags = 0
    duplicate_blocks = 0
    first_improvements = 0
    for record in base.load_jsonl(folder / "turns.jsonl"):
        expected_seed = base.turn_seed(manifest["seed"], manifest["model_id"], record["dialogue_id"], 0, record["turn_index"])
        if record["seed"] != expected_seed or record["run_id"] != run_id:
            raise ValueError("TPO turn seed/run mismatch")
        validate_audit(record, config)
        for event in record["tpo"]["events"]:
            key = (record["dialogue_id"], record["turn_index"], event["index"])
            used.add(key)
            if journal.get(key) != event:
                raise ValueError("TPO event journal/turn audit mismatch")
            capped += bool(event["result"].get("hit_token_cap"))
            retry_generations += bool(event["request"].get("retry_attempt"))
            empty_attempts += bool(event.get("error"))
            actual_generations += event["request"]["kind"] == "generate"
            missing_closing_tags += MISSING_END_WARNING in event.get("format_warnings", [])
            trailing_opening_tags += TRAILING_START_WARNING in event.get("format_warnings", [])
            terminal_opening_tags += TERMINAL_START_WARNING in event.get("format_warnings", [])
            duplicate_blocks += DUPLICATE_BLOCK_WARNING in event.get("format_warnings", [])
            first_improvements += FIRST_IMPROVEMENT_WARNING in event.get("format_warnings", [])
    if used != set(journal):
        raise ValueError("Unexpected TPO journal event keys")
    if require_gpu:
        runtime = json.loads((folder / "runtime_stats.json").read_text())
        loading = json.loads((folder / "model_loading.json").read_text())
        if not runtime.get("peak_gpu_allocated_bytes") or runtime.get("exit_code") != 0:
            raise ValueError("Successful GPU runtime/memory measurement missing")
        if set(loading) != {"actor", "reward"} or any(not v.get("passed") or v.get("trainable_parameters") != 0 for v in loading.values()):
            raise ValueError("Frozen actor/reward checkpoint loading not verified")
    report.update(retry_generations=retry_generations, empty_generation_attempts=empty_attempts,
                  actual_generation_calls=actual_generations, tpo_audit_passed=True, candidates=report["turns"] * config["sample_size"] * (config["max_iterations"] + 1),
                  feedback_generations=report["turns"] * 2 * config["max_iterations"],
                  generations_at_token_cap=capped,
                  optimizer_missing_closing_tag_warnings=missing_closing_tags,
                  optimizer_trailing_opening_tag_warnings=trailing_opening_tags,
                  optimizer_terminal_opening_tag_warnings=terminal_opening_tags,
                  optimizer_duplicate_block_warnings=duplicate_blocks,
                  optimizer_first_improvement_warnings=first_improvements)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    report = validate_tpo(args.output_dir, args.require_gpu)
    (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
