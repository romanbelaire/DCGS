#!/usr/bin/env python3
"""Validate native answer/turn coverage and history without judge API calls."""

import argparse
import json
from pathlib import Path

from judge_safedial import validate_and_join
from run_safedial_baseline import file_sha256, gold_messages, load_jsonl


def validate_run(output_dir, require_adapter=False):
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "run_config.json").read_text())
    dataset_path = Path(manifest["dataset"])
    if file_sha256(dataset_path) != manifest["dataset_sha256"]:
        raise ValueError("Dataset hash mismatch")
    answers = load_jsonl(output_dir / "answers.jsonl")
    records = load_jsonl(output_dir / "turns.jsonl")
    if sorted(a["id"] for a in answers) != sorted(manifest["selected_ids"]):
        raise ValueError("Answer coverage does not match selected IDs")
    turns = {}
    for record in records:
        key = (record["dialogue_id"], record["choice_index"], record["turn_index"])
        if key in turns:
            raise ValueError(f"Duplicate turn record: {key}")
        turns[key] = record
    expected = set()
    for choice in range(manifest["num_choices"]):
        for question, answer in validate_and_join(answers, load_jsonl(dataset_path), choice):
            if len(answer["choices"]) != manifest["num_choices"] or answer["model_id"] != manifest["model_id"]:
                raise ValueError("Answer model or choice count mismatch")
            for index, generated in enumerate(answer["choices"][choice]["turns"]):
                key = (question["id"], choice, index)
                expected.add(key)
                record = turns.get(key)
                if not record or record.get("error") or not generated["message"].strip():
                    raise ValueError(f"Missing, empty or failed turn: {key}")
                if record["generated_response"] != generated["message"]:
                    raise ValueError(f"Answer/turn mismatch: {key}")
                if record["prompt_history"] != gold_messages(question["history"], index):
                    raise ValueError(f"Gold-history mismatch: {key}")
                if record["model_id"] != manifest["model_id"]:
                    raise ValueError(f"Turn model mismatch: {key}")
    if expected != set(turns):
        raise ValueError("Unexpected turn records")
    if require_adapter:
        if "adapter" not in manifest:
            raise ValueError("Adapter provenance missing")
        validation = json.loads((output_dir / "adapter_validation.json").read_text())
        if not validation.get("passed") or validation.get("max_abs_logit_difference", 0) <= 1e-6:
            raise ValueError("Adapter effect verification did not pass")
        runtime = json.loads((output_dir / "runtime_stats.json").read_text())
        if not runtime.get("peak_gpu_allocated_bytes"):
            raise ValueError("GPU memory measurement missing")
    report = {"passed": True, "dialogues": len(answers), "turns": len(turns),
              "input_truncated_turns": sum(bool(r.get("input_truncated")) for r in records),
              "adapter_required": require_adapter, "judge_calls": 0}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-adapter", action="store_true")
    args = parser.parse_args()
    report = validate_run(args.output_dir, args.require_adapter)
    (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
