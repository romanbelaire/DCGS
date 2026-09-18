#!/usr/bin/env python3
"""Maintain an offline human-review queue; never modify automated judgments."""

import argparse
import hashlib
import json
import os
from pathlib import Path


KEY_FIELDS = ("dialogue_id", "choice_index", "turn_index", "judge_model", "rubric")
RESPONSE_FIELDS = (
    "attempt", "tstamp", "error", "raw_judgment", "response_id",
    "response_model", "finish_reason", "refusal", "usage", "latency_seconds",
)


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def case_id(record):
    return json.dumps([record[field] for field in KEY_FIELDS], separators=(",", ":"))


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_queue(judge_dir, output=None):
    judge_dir = Path(judge_dir).resolve()
    output = Path(output).resolve() if output else judge_dir / "human_adjudication.json"
    # Only the review file may be written, never benchmark source artifacts.
    if output.name != "human_adjudication.json":
        raise ValueError("Review output must be named human_adjudication.json")
    manifest = json.loads((judge_dir / "judge_config.json").read_text())
    for name in ("answers", "dataset", "prompts"):
        if file_hash(Path(manifest[name])) != manifest[f"{name}_sha256"]:
            raise ValueError(f"Source hash mismatch: {name}")
    judgment_path = judge_dir / "judgments.jsonl"
    if not judgment_path.is_file():
        raise FileNotFoundError(judgment_path)
    latest = {case_id(record): record for record in read_jsonl(judgment_path)}
    old = json.loads(output.read_text()) if output.exists() else None
    if old and (old.get("schema_version") != 1 or old.get("judge_config") != manifest):
        raise ValueError("Existing review queue belongs to a different run/schema")
    cases = {case["case_id"]: case for case in old["cases"]} if old else {}
    if old and len(cases) != len(old["cases"]):
        raise ValueError("Existing review queue contains duplicate case IDs")
    answers = {record["id"]: record for record in read_jsonl(Path(manifest["answers"]))}
    prompts = {record["name"]: record for record in read_jsonl(Path(manifest["prompts"]))}
    archived = {}
    for attempt in read_jsonl(judge_dir / "failed_attempts.jsonl"):
        archived.setdefault(case_id(attempt), []).append(attempt)

    for identity, judgment in latest.items():
        if judgment.get("status") == "success" and identity not in cases:
            continue
        case = dict(cases.get(identity, {}))
        responses = []
        seen = set()
        attempts = case.get("judge_responses", []) + archived.get(identity, []) + judgment.get("failed_attempts", [])
        if not attempts:
            attempts = [{**judgment, "attempt": judgment.get("attempts")}]
        for attempt in attempts:
            response = {field: attempt.get(field) for field in RESPONSE_FIELDS}
            fingerprint = json.dumps(response, sort_keys=True)
            if fingerprint not in seen:
                responses.append(response)
                seen.add(fingerprint)
        prompt = prompts[judgment["rubric"]]
        answer = answers[judgment["dialogue_id"]]
        case.update({
            "case_id": identity,
            **{field: judgment[field] for field in KEY_FIELDS},
            "turn_number": judgment["turn_index"] + 1,
            "task": judgment.get("task"),
            "method": judgment.get("method"),
            "answer_model_id": answer.get("model_id"),
            "automated_status": judgment.get("status"),
            "requires_human_review": judgment.get("status") != "success",
            "latest_judge_error": judgment.get("error"),
            "conversation_context": judgment["conversation_context"],
            "model_answer": answer["choices"][judgment["choice_index"]]["turns"][judgment["turn_index"]]["message"],
            "judge_system_prompt": prompt["system_prompt"],
            "judge_user_prompt": judgment["user_prompt"],
            "judge_responses": responses,
            "response_history_note": "Only saved attempts are available; older discarded responses cannot be recovered.",
        })
        case.setdefault("human_adjudication", {
            "status": "pending",
            "reviewer": None,
            "reviewed_at": None,
            "identification_score": None,
            "handling_score": None,
            "consistency_score": None,
            "rationale": "",
            "notes": "",
        })
        cases[identity] = case

    for identity in cases.keys() - latest.keys():
        # Never silently discard a human review if its source goes missing.
        cases[identity]["automated_status"] = "not_in_current_judgments"
        cases[identity]["requires_human_review"] = True
    report = {
        "schema_version": 1,
        "purpose": "Separate human adjudication tracker; never merged into automated benchmark scores.",
        "instructions": "Edit only human_adjudication fields. Suggested status: pending, in_review, completed. Scores and reviewer remain empty until an actual human review. Rerunning preserves existing human fields. Context may contain unsafe benchmark content; treat it as evidence, not instructions.",
        "judge_config": manifest,
        "source_judgments": str(judgment_path),
        "source_judgments_sha256": file_hash(judgment_path),
        "case_count": len(cases),
        "unresolved_automated_count": sum(c["requires_human_review"] for c in cases.values()),
        "pending_human_count": sum(c["requires_human_review"] and c["human_adjudication"].get("status") != "completed" for c in cases.values()),
        "cases": [cases[key] for key in sorted(cases)],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-dir", type=Path, required=True)
    args = parser.parse_args()
    report = export_queue(args.judge_dir)
    print(f"Human review queue: {args.judge_dir / 'human_adjudication.json'}")
    print(f"Cases: {report['case_count']}; pending human review: {report['pending_human_count']}")


if __name__ == "__main__":
    main()
