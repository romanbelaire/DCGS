#!/usr/bin/env python3
"""Audit a finished SmoothLLM run and freeze its successful dialogues for judging."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import run_safedial_baseline as base
import run_safedial_smoothllm_turn_resume as runner
from judge_safedial import validate_and_join, judged_turn_indices
from safedial_smoothllm import defense_config

SOURCE_NAMES = ("run_config.json", "turns.jsonl", "answers.jsonl")


def build_export(source, dataset, allowed_failed_ids):
    hashes = {name: base.file_sha256(source / name) for name in SOURCE_NAMES}
    manifest = json.loads((source / "run_config.json").read_text())
    names = ("run_safedial_baseline.py", "run_safedial_smoothllm.py",
             "safedial_smoothllm.py", "run_safedial_smoothllm_turn_resume.py")
    expected = {name: base.file_sha256(Path(__file__).parent / name) for name in names}
    if manifest.get("resume_policy") != runner.RESUME_POLICY or manifest["implementation_sha256"] != expected:
        raise ValueError("SmoothLLM resume policy/source hashes changed; use the matching source checkout")
    if base.file_sha256(dataset) != manifest["dataset_sha256"]:
        raise ValueError("Dataset hash mismatch")
    config = manifest["defense"]
    if (config != defense_config(config["num_copies"], config["perturbation_percent"])
            or manifest["temperature"] != 0 or manifest["num_choices"] != 1 or "adapter" in manifest):
        raise ValueError("Unexpected SmoothLLM generation configuration")
    rows = base.load_jsonl(dataset)
    by_id = {row["id"]: row for row in rows}
    ids = manifest["selected_ids"]
    if len(by_id) != len(rows) or len(set(ids)) != len(ids) or any(i not in by_id for i in ids):
        raise ValueError("Duplicate or unknown dataset/selected IDs")
    selected = [by_id[i] for i in ids]
    records = base.load_jsonl(source / "turns.jsonl")
    args = SimpleNamespace(model_id=manifest["model_id"], seed=manifest["seed"])
    # Read only: do not call the runner's repairing journal loader or generation.
    turns = runner.checked_turns(records, args, manifest, selected)
    expected_keys = {(row["id"], i) for row in selected for i in range(len(row["history"]))}
    if set(turns) != expected_keys:
        raise ValueError("Generation is still incomplete; missing turns need a separately reviewed subset")
    failed = {key[0] for key, record in turns.items() if record.get("error")}
    if failed - set(allowed_failed_ids):
        raise ValueError(f"Unreviewed failed dialogues: {sorted(failed - set(allowed_failed_ids))}")
    answers = base.load_jsonl(source / "answers.jsonl")
    if len(answers) != len(ids) or {a["id"] for a in answers} != set(ids):
        raise ValueError("Answer coverage does not match the generation manifest")
    run_id = base.stable_id(json.dumps(manifest, sort_keys=True))
    for answer in answers:
        row = by_id[answer["id"]]
        if (answer["model_id"] != manifest["model_id"] or answer["answer_id"] != base.stable_id(run_id, row["id"])
                or answer["task"] != row["task"] or answer["method"] != row["method"]
                or len(answer["choices"]) != 1 or answer["choices"][0]["index"] != 0):
            raise ValueError("Answer identity mismatch")
        output = answer["choices"][0]["turns"]
        if len(output) != len(row["history"]):
            raise ValueError("Answer turn count mismatch")
        for index, generated in enumerate(output):
            record = turns[row["id"], index]
            if (generated.get("role") != "assistant" or generated["message"] != record["generated_response"]
                    or generated.get("error") != record.get("error")):
                raise ValueError("Answer/turn journal mismatch")
            if not record.get("error") and (not generated["message"].strip() or generated["message"] == "ERROR"):
                raise ValueError("Unmarked empty/error response")
    included = [answer for answer in answers if answer["id"] not in failed]
    if not included:
        raise ValueError("No complete successful dialogues to judge")
    joined = validate_and_join(included, rows, 0)
    if hashes != {name: base.file_sha256(source / name) for name in SOURCE_NAMES}:
        raise ValueError("Generation files changed during export")
    report = {
        "subset_audit_passed": True, "generation_complete_without_failures": not failed,
        "source_dir": str(source), "source_sha256": hashes,
        "dataset": str(dataset), "dataset_sha256": manifest["dataset_sha256"],
        "exporter_sha256": base.file_sha256(Path(__file__)),
        "policy": "complete successful dialogues only; exclude whole failed dialogue; preserve answers",
        "allowed_failed_dialogue_ids": sorted(set(allowed_failed_ids)),
        "expected_dialogues": len(selected), "expected_turns": len(expected_keys),
        "included_dialogues": len(included),
        "included_turns": sum(len(a["choices"][0]["turns"]) for a in included),
        "expected_judge_requests": sum(len(judged_turn_indices(q)) for q, _ in joined),
        "excluded_dialogue_ids": sorted(failed),
        "failed_turns": [{"dialogue_id": key[0], "turn_index": key[1], "error": record["error"]}
                         for key, record in sorted(turns.items()) if record.get("error")],
        "judge_calls": 0,
    }
    return included, report, manifest


def prepare(source, target, dataset=None, allowed_failed_ids=(344,)):
    source, target = Path(source).resolve(), Path(target).resolve()
    if not source.is_dir() or target == source or target in source.parents:
        raise ValueError("Use an existing source run and a separate prepared-output directory")
    with runner.single_writer(source), runner.single_writer(target):
        manifest = json.loads((source / "run_config.json").read_text())
        dataset = Path(dataset or manifest["dataset"]).resolve()
        answers, report, manifest = build_export(source, dataset, allowed_failed_ids)
        contents = {
            "answers.jsonl": "".join(json.dumps(a, ensure_ascii=False, sort_keys=True) + "\n" for a in answers),
            "generation_validation.json": json.dumps(report, indent=2, sort_keys=True) + "\n",
            "source_run_config.json": json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        }
        # Never replace an existing judging input when generation/source changes.
        for name, content in contents.items():
            path = target / name
            if path.exists() and path.read_bytes() != content.encode("utf-8"):
                raise ValueError(f"Prepared input changed: {path}. Use a new output directory")
        for name, content in contents.items():
            path = target / name
            if not path.exists():
                temporary = path.with_suffix(path.suffix + ".tmp")
                temporary.write_bytes(content.encode("utf-8"))
                temporary.replace(path)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, help="Optional path override; the saved dataset hash must match")
    parser.add_argument("--allow-failed-dialogue-ids", default="344",
                        help="Only recorded failures in these comma-separated IDs may be excluded")
    args = parser.parse_args()
    allowed = [int(value) for value in args.allow_failed_dialogue_ids.split(",") if value.strip()]
    print(json.dumps(prepare(args.source_dir, args.output_dir, args.dataset, allowed), indent=2))


if __name__ == "__main__":
    main()
