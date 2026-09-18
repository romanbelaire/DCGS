#!/usr/bin/env python3
"""Validate completed turn-resumable SmoothLLM output and pinned sources."""
import argparse
import json
from pathlib import Path

import run_safedial_baseline as base
from run_safedial_smoothllm_turn_resume import RESUME_POLICY
from validate_safedial_smoothllm import validate_smoothllm


def validate_turn_resume(output_dir, require_gpu=False):
    folder = Path(output_dir)
    manifest = json.loads((folder / "run_config.json").read_text())
    names = ("run_safedial_baseline.py", "run_safedial_smoothllm.py", "safedial_smoothllm.py",
             "run_safedial_smoothllm_turn_resume.py")
    expected = {name: base.file_sha256(Path(__file__).parent / name) for name in names}
    if manifest.get("resume_policy") != RESUME_POLICY or manifest["implementation_sha256"] != expected:
        raise ValueError("SmoothLLM resume policy/source hashes changed")
    run_id = base.stable_id(json.dumps(manifest, sort_keys=True))
    for record in base.load_jsonl(folder / "turns.jsonl"):
        if record["run_id"] != run_id or record["answer_id"] != base.stable_id(run_id, record["dialogue_id"]):
            raise ValueError("SmoothLLM turn run identifier mismatch")
    for answer in base.load_jsonl(folder / "answers.jsonl"):
        if answer["answer_id"] != base.stable_id(run_id, answer["id"]):
            raise ValueError("SmoothLLM answer identifier mismatch")
    report = validate_smoothllm(folder, require_gpu)
    report["turn_resume_source_audit_passed"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    report = validate_turn_resume(args.output_dir, args.require_gpu)
    (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
