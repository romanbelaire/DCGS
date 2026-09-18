#!/usr/bin/env python3
"""Validate full TPO outputs, their explicit 8K policy and capacity probe."""
import argparse
import json
from pathlib import Path

from run_safedial_tpo_full import CONTEXT_POLICY, REWARD_CONTEXT_TOKENS, implementation_hashes
from validate_safedial_tpo import validate_tpo


def validate_full(output_dir, require_gpu=False):
    folder = Path(output_dir)
    manifest = json.loads((folder / "run_config.json").read_text())
    if manifest.get("full_run_context") != CONTEXT_POLICY:
        raise ValueError("Unknown full-run reward context policy")
    report = validate_tpo(folder, require_gpu, expected_implementation=implementation_hashes())
    if require_gpu:
        probe = json.loads((folder / "context_probe.json").read_text())
        if not probe.get("passed") or probe.get("sequence_tokens") != REWARD_CONTEXT_TOKENS:
            raise ValueError("Full reward-context capacity probe did not pass")
    report.update(reward_context_tokens=REWARD_CONTEXT_TOKENS)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    report = validate_full(args.output_dir, args.require_gpu)
    (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
