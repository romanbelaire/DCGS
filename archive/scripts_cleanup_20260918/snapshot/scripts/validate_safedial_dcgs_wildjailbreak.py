#!/usr/bin/env python3
"""Replay the original DCGS policy and validate native SafeDial coverage."""
import argparse
import json
from pathlib import Path

from run_safedial_baseline import load_jsonl, turn_seed
from run_safedial_dcgs_wildjailbreak import source_hashes, tokenizer_for, PROTOCOL
from safedial_dcgs_wildjailbreak import configuration, validate_turn
from validate_safedial_generation import validate_run


def validate(folder, require_gpu=False):
    folder = Path(folder)
    manifest = json.loads((folder / "run_config.json").read_text())
    if manifest["protocol"] != PROTOCOL or manifest["source_sha256"] != source_hashes():
        raise ValueError("Original-policy source identity mismatch")
    if (folder / "failure.json").exists():
        raise ValueError("Run contains a saved policy failure")
    report = validate_run(folder)
    config = configuration(manifest["method"], manifest["effective_config"]["device"])
    if json.loads(json.dumps(vars(config))) != manifest["effective_config"]:
        raise ValueError("Effective reference configuration mismatch")
    tokenizer = tokenizer_for(manifest["artifacts"])
    rows = {row["id"]: row for row in load_jsonl(Path(manifest["dataset"]))}
    journal = load_jsonl(folder / "events.jsonl")
    by_key = {}
    for event in journal:
        key = (event["invocation"], event["dialogue_id"], event["turn_index"], event["index"])
        if key in by_key:
            raise ValueError("Duplicate journal event")
        by_key[key] = event
    attempts = load_jsonl(folder / "attempts.jsonl")
    attempt_keys = set()
    for attempt in attempts:
        key = (attempt["invocation"], attempt["dialogue_id"], attempt["turn_index"], attempt["index"])
        if key in attempt_keys:
            raise ValueError("Duplicate attempted call")
        attempt_keys.add(key)
        if key in by_key and (attempt["request"] != by_key[key]["request"] or attempt["seed"] != by_key[key]["seed"]):
            raise ValueError("Attempt/result mismatch")
    if not set(by_key).issubset(attempt_keys):
        raise ValueError("Missing call-start evidence")
    used = set()
    for record in load_jsonl(folder / "turns.jsonl"):
        if record["seed"] != turn_seed(manifest["seed"], manifest["model_id"], record["dialogue_id"], 0, record["turn_index"]):
            raise ValueError("Turn seed mismatch")
        validate_turn(record, rows[record["dialogue_id"]], config, tokenizer)
        for event in record["dcgs_original"]["events"]:
            key = (record["invocation"], record["dialogue_id"], record["turn_index"], event["index"])
            expected = {"invocation": record["invocation"], "dialogue_id": record["dialogue_id"],
                        "turn_index": record["turn_index"], "seed": record["seed"], **event}
            if by_key.get(key) != expected:
                raise ValueError("Turn/journal mismatch")
            used.add(key)
    generations = [e for e in journal if e["request"]["kind"] == "generate"]
    scores = [e for e in journal if e["request"]["kind"] == "score"]
    report.update(journal_events=len(journal), abandoned_events=len(journal) - len(used),
        interrupted_calls_with_unknown_cost=len(attempt_keys - set(by_key)),
        generation_calls=len(generations), custom_empty_retries=0,
        generated_tokens_including_abandoned=sum(len(ids) for e in generations
            for batch in e.get("result", {}).get("raw_generations", []) for ids in batch["generated_token_ids"]),
        critic_truncated_inputs=sum(sum(e.get("result", {}).get("input_truncated", [])) for e in scores))
    if require_gpu:
        loading = json.loads((folder / "model_loading.json").read_text())
        expected = ["q", "v"] + (["q_min", "v_min", "regret"] if manifest["method"] == "rdcgs" else [])
        if (loading["strict_loaded_heads"] != expected or loading["ll_reranking"]
                or loading["checkpoint_sha256"] != manifest["artifacts"]["high_level"]["sha256"]
                or loading["actor_dtype"] != "torch.bfloat16" or not loading["device"].startswith("cuda")):
            raise ValueError("Original checkpoint/GPU loading mismatch")
        runtimes = load_jsonl(folder / "runtime.jsonl")
        if not runtimes[-1]["success"] or not all(r.get("peak_gpu_allocated_bytes", 0) > 0 for r in runtimes):
            raise ValueError("Successful GPU runtime and peak-memory evidence required")
        if any(not e.get("result", {}).get("raw_generations") for e in generations if not e.get("error")):
            raise ValueError("Raw generation accounting missing")
        report.update(elapsed_seconds=sum(r["elapsed_seconds"] for r in runtimes),
                      peak_gpu_allocated_bytes=max(r["peak_gpu_allocated_bytes"] for r in runtimes))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    report = validate(args.output_dir, args.require_gpu)
    (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
