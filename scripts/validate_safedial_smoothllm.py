#!/usr/bin/env python3
"""Offline native coverage plus SmoothLLM vote, perturbation and cost audit."""

import argparse
import json
from pathlib import Path

from run_safedial_baseline import load_jsonl, turn_seed
from safedial_smoothllm import defense_config, validate_audit
from validate_safedial_generation import validate_run


def validate_smoothllm(output_dir, require_gpu=False):
    output_dir = Path(output_dir)
    report = validate_run(output_dir)
    manifest = json.loads((output_dir / "run_config.json").read_text())
    config = manifest["defense"]
    if config != defense_config(config["num_copies"], config["perturbation_percent"]):
        raise ValueError("Unknown SmoothLLM configuration")
    if manifest["temperature"] != 0 or manifest["num_choices"] != 1 or "adapter" in manifest:
        raise ValueError("Expected greedy, unadapted base model with one selected answer")
    for record in load_jsonl(output_dir / "turns.jsonl"):
        expected_seed = turn_seed(manifest["seed"], manifest["model_id"], record["dialogue_id"], 0, record["turn_index"])
        if record["seed"] != expected_seed:
            raise ValueError("Turn seed mismatch")
        validate_audit(record, config)
    if require_gpu:
        runtime = json.loads((output_dir / "runtime_stats.json").read_text())
        if not runtime.get("peak_gpu_allocated_bytes") or runtime.get("turn_errors"):
            raise ValueError("Successful GPU runtime/memory measurement missing")
    report.update(smoothllm_audit_passed=True, candidates=report["turns"] * config["num_copies"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    report = validate_smoothllm(args.output_dir, args.require_gpu)
    (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
