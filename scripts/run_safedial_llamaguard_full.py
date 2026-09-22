#!/usr/bin/env python3
"""Full-dataset LlamaGuard evaluation with a tokenizer-only context preflight.

Reuses the frozen-input evaluator without changing its source-pinned pilots.
No GPU/model weights are loaded by --prepare-only or --preflight-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import math
from pathlib import Path
import sys

import judge_safedial_llamaguard as guard


def full_coverage(snapshot, rows, dataset):
    dataset = Path(dataset)
    if guard.file_hash(dataset) != snapshot["dataset_sha256"]:
        raise ValueError("Full-run dataset checksum mismatch")
    data = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    ids = [row["id"] for row in data]
    if len(ids) != len(set(ids)) or set(snapshot["selected_ids"]) != set(ids):
        raise ValueError("Full run requires every dataset dialogue; a pilot/subset is not a full run")
    expected = {(row["id"], t) for row in data for t in range(len(row["history"]))}
    found = [(row["dialogue_id"], row["turn_index"]) for row in rows]
    if len(found) != len(expected) or set(found) != expected:
        raise ValueError("Frozen inputs do not cover every expected dataset turn")
    counts = dict(Counter(row["generation_status"] for row in rows))
    return {"expected_dialogues": len(ids), "expected_turns": len(expected),
            "generation_status_counts": counts,
            "generation_complete": counts.get("available", 0) == len(expected)}


def load_tokenizer(args):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    # No weights: gated access and exact context lengths are checked first.
    path = Path(snapshot_download(
        guard.MODEL, revision=args.revision, local_files_only=not args.allow_download,
        allow_patterns=["*.json", "*.jinja", "tokenizer.model", "*.tiktoken"],
        ignore_patterns=["original/*"]))
    config = json.loads((path / "config.json").read_text())
    model_limit = config["max_position_embeddings"]
    if type(model_limit) is not int or model_limit < 1:
        raise ValueError("Invalid model context limit")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise ValueError("Guard tokenizer has no chat template")
    return tokenizer, {"model_context_limit": model_limit,
                       "model_config_sha256": guard.file_hash(path / "config.json"),
                       "tokenizer_template_sha256": guard.digest(tokenizer.chat_template),
                       "tokenizer_versions": {n: importlib.metadata.version(n)
                                              for n in ("transformers", "tokenizers", "huggingface-hub")}}


def audit_context(rows, tokenizer, limit):
    """Count exactly the rendered guard input, with no character/token clipping."""
    records, lengths, overflows = [], [], []
    for row in rows:
        record = {k: row[k] for k in ("dialogue_id", "turn_index", "input_sha256", "generation_status")}
        if row["generation_status"] == "available":
            rendered = guard.render_guard_prompt(tokenizer, row["messages"])
            ids = tokenizer(rendered, add_special_tokens=False, truncation=False)["input_ids"]
            length = len(ids)
            lengths.append(length)
            record.update(input_tokens=length, overflow=length > limit)
            if length > limit:
                overflows.append({k: record[k] for k in ("dialogue_id", "turn_index", "input_tokens")})
        records.append(record)
    ordered = sorted(lengths)

    def percentile(p):
        return ordered[max(0, math.ceil(p * len(ordered)) - 1)] if ordered else None

    return records, {"available_turns_audited": len(lengths), "effective_context_limit": limit,
                     "input_tokens_min": min(lengths) if lengths else None,
                     "input_tokens_max": max(lengths) if lengths else None,
                     "input_tokens_p50": percentile(.50), "input_tokens_p95": percentile(.95),
                     "input_tokens_p99": percentile(.99), "overflow_turns": overflows,
                     "passed": bool(lengths) and not overflows,
                     "truncation": "none; counts include full guard template and conversation"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dataset", type=Path)
    p.add_argument("--input-format", choices=["auto", "turns", "answers"], default="auto")
    p.add_argument("--choice-index", type=int, default=0)
    p.add_argument("--revision", default=guard.REVISION)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--max-input-tokens", type=int, default=16384)
    p.add_argument("--allow-download", action="store_true")
    p.add_argument("--retry-errors", action="store_true")
    p.add_argument("--allow-incomplete-generation", action="store_true",
                   help="Judge available turns, retaining missing/failed turns and incomplete final status")
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--prepare-only", action="store_true", help="Freeze and validate full coverage; no tokenizer/model")
    modes.add_argument("--preflight-only", action="store_true", help="Also count all guard inputs; no model weights/GPU")
    return p.parse_args(argv)


def evaluator_arguments(args):
    result = ["--run-dir", str(args.run_dir), "--output-dir", str(args.output_dir),
              "--input-format", args.input_format, "--choice-index", str(args.choice_index),
              "--revision", args.revision, "--device", args.device, "--dtype", args.dtype,
              "--max-input-tokens", str(args.max_input_tokens)]
    if args.dataset:
        result += ["--dataset", str(args.dataset)]
    if args.allow_download:
        result.append("--allow-download")
    if args.retry_errors:
        result.append("--retry-errors")
    return result


def main(argv=None):
    args = parse_args(argv)
    base_args = evaluator_arguments(args)
    # The existing evaluator binds the unchanged scoring config/source hash.
    guard.main(base_args + ["--prepare-only"])
    folder = args.output_dir.resolve()
    with guard.lock_output(folder):
        snapshot, rows = guard.load_snapshot(folder)
        config = json.loads((folder / "judge_config.json").read_text())
        if config["inputs_sha256"] != snapshot["inputs_sha256"]:
            raise ValueError("Judge configuration does not match frozen inputs")
        dataset = args.dataset or Path(snapshot["generation_config"]["dataset"])
        dataset = dataset if dataset.is_absolute() else guard.ROOT / dataset
        coverage = full_coverage(snapshot, rows, dataset)
        manifest = {"wrapper_sha256": guard.file_hash(__file__),
                    "judge_config_sha256": guard.digest(config),
                    "allow_incomplete_generation": args.allow_incomplete_generation,
                    "scope": "entire pinned dataset, frozen source snapshot", **coverage}
        manifest_path = folder / "full_run_manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Full-run configuration/source changed; use a fresh output directory")
        guard.atomic_json(manifest_path, manifest)
        if not coverage["generation_complete"] and not args.allow_incomplete_generation:
            print("Full snapshot has missing/failed generation. Use a fresh snapshot after generation finishes, "
                  "or explicitly --allow-incomplete-generation in a fresh directory.", file=sys.stderr)
            return 2
        if args.prepare_only:
            print("Full-dataset preparation passed; tokenizer/model not loaded.")
            return 0
        latest = guard.read_judgments(folder, rows, guard.digest(config), repair_tail=True)
        result = guard.summarize(folder, snapshot, rows, latest)
        pending = [r for r in rows if r["generation_status"] == "available" and
                   (r["input_sha256"] not in latest or
                    (args.retry_errors and latest[r["input_sha256"]]["status"] == "error"))]
        if not pending and not args.preflight_only:
            print(json.dumps(result["overall"], indent=2))
            return 0 if result["complete"] else 2
        print("Checking gated tokenizer access and all formatted input lengths before loading weights.", flush=True)
        tokenizer, metadata = load_tokenizer(args)
        limit = min(args.max_input_tokens, metadata["model_context_limit"])
        records, audit = audit_context(rows, tokenizer, limit)
        audit.update(inputs_sha256=snapshot["inputs_sha256"], judge_config_sha256=guard.digest(config),
                     model=guard.MODEL, revision=args.revision, label_prefix=guard.LABEL_PREFIX, **metadata)
        guard.atomic_json(folder / "context_lengths.json", records)
        guard.atomic_json(folder / "context_preflight.json", audit)
        print(json.dumps(audit, indent=2), flush=True)
        if not audit["passed"]:
            print("Context preflight failed; no weights loaded or inputs truncated. Inspect context_preflight.json.",
                  file=sys.stderr)
            return 2
        if args.preflight_only:
            return 0
        # GPU loading and the existing sanity probes happen only after all lengths fit.
        try:
            scorer = guard.LlamaGuard(args, folder)
            runtime = json.loads((folder / "runtime.json").read_text())
            if scorer.limit != limit or any(runtime[k] != metadata[k] for k in
                    ("model_config_sha256", "tokenizer_template_sha256")):
                raise ValueError("Scoring runtime differs from the audited tokenizer/config")
            guard.evaluate(folder, rows, latest, guard.digest(config), scorer,
                           retry_errors=args.retry_errors,
                           checkpoint=lambda: guard.summarize(folder, snapshot, rows, latest))
        finally:
            result = guard.summarize(folder, snapshot, rows, latest)
        print(json.dumps(result["overall"], indent=2))
        return 0 if result["complete"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
