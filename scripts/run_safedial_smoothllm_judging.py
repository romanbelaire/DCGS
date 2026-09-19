#!/usr/bin/env python3
"""CPU/API SmoothLLM judging on Windows or Linux, from live output or a saved snapshot."""
import argparse
from contextlib import contextmanager
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

import judge_safedial as judge
import run_safedial_baseline as base
from safedial_smoothllm import defense_config

ROOT = Path(__file__).resolve().parents[1]
RUN = "smoothllm_zephyr_full_turn_resume"
DEFAULT_SNAPSHOT = ROOT / "results/safedialbench/2026-09-19"
DEFAULT_SOURCE = ROOT / "outputs/safedial_baseline" / RUN
DATASET_REL = "data/complete/datasets_en.jsonl"
PROMPTS_REL = "FastChat/fastchat/llm_judge/data/judge_prompts.jsonl"
BENCHMARK_REVISION = "e242e5f3fcbf87f0e11e99d4563155da1e2e5a23"
REFERENCE_CONFIG = "safedial_baseline/cat_zephyr_full/judgments_gpt-4o-mini/judge_config.json"
SOURCE_NAMES = ("run_safedial_baseline.py", "run_safedial_smoothllm.py",
                "safedial_smoothllm.py", "run_safedial_smoothllm_turn_resume.py")


def sha(data):
    return hashlib.sha256(data).hexdigest()


@contextmanager
def file_lock(path):
    """Use OS locks released on process exit; compatible with POSIX flock launchers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            if os.name == "nt":
                import msvcrt
                if path.stat().st_size == 0:
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Another process holds the lock: {path}") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def snapshot_bytes(snapshot, index, relative):
    entry = next((item for item in index["files"] if item["path"] == relative), None)
    if entry is None:
        raise ValueError(f"Snapshot inventory missing {relative}")
    stored = (snapshot / relative).read_bytes()
    # Git may convert plain JSON/JSONL to CRLF on Windows. Compressed bytes must match exactly.
    canonical = stored if relative.endswith(".gz") else stored.replace(b"\r\n", b"\n")
    if sha(stored) != entry["sha256"] and sha(canonical) != entry["sha256"]:
        raise ValueError(f"Snapshot checksum mismatch: {relative}")
    data = gzip.decompress(stored) if relative.endswith(".gz") else canonical
    if sha(data) != entry["uncompressed_sha256"]:
        raise ValueError(f"Snapshot content checksum mismatch: {relative}")
    return data


def benchmark_file(path, expected, relative, fetch=False):
    if not path.is_file():
        if not fetch:
            raise FileNotFoundError(f"Missing {path}. Supply the file or rerun with --fetch-benchmark")
        url = ("https://raw.githubusercontent.com/drivetosouth/SafeDialBench-Dataset/"
               + BENCHMARK_REVISION + "/" + relative)
        with urllib.request.urlopen(url, timeout=90) as response:
            data = response.read()
        if sha(data) != expected:
            raise ValueError(f"Downloaded benchmark hash mismatch: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".download")
        temporary.write_bytes(data)
        temporary.replace(path)
    if base.file_sha256(path) != expected:
        raise ValueError(f"Benchmark hash mismatch: {path}")


def prepare_snapshot(snapshot, target, dataset, prompts, fetch=False):
    """Validate saved answers/status; explicitly do not claim a full candidate audit."""
    snapshot, target = Path(snapshot).resolve(), Path(target).resolve()
    if target == snapshot or snapshot in target.parents or target in snapshot.parents:
        raise ValueError("Prepared output must be separate from the saved snapshot")
    with file_lock(target / ".prepare.lock"):
        inventory = (snapshot / "snapshot.json").read_bytes()
        index = json.loads(inventory)
        prefix = "safedial_baseline/" + RUN + "/"
        source_data = {name: snapshot_bytes(snapshot, index, prefix + name)
                       for name in ("run_config.json", "answers.jsonl.gz", "turn_status.jsonl.gz")}
        reference_data = snapshot_bytes(snapshot, index, REFERENCE_CONFIG)
        manifest = json.loads(source_data["run_config.json"])
        reference = json.loads(reference_data)
        if manifest["dataset_sha256"] != reference["dataset_sha256"]:
            raise ValueError("Snapshot generation and judge datasets differ")
        if (manifest["protocol"] != base.PROTOCOL or manifest["num_choices"] != 1
                or manifest["temperature"] != 0 or "adapter" in manifest
                or manifest["defense"] != defense_config(8, 10.0)):
            raise ValueError("Unexpected saved SmoothLLM method")
        expected_sources = manifest["implementation_sha256"]
        if set(expected_sources) != set(SOURCE_NAMES):
            raise ValueError("Unexpected saved generator source set")
        for name in SOURCE_NAMES:
            data = (ROOT / "scripts" / name).read_bytes()
            if expected_sources[name] not in (sha(data), sha(data.replace(b"\r\n", b"\n"))):
                raise ValueError(f"Saved generator source mismatch: {name}")
        benchmark_file(dataset, manifest["dataset_sha256"], DATASET_REL, fetch)
        benchmark_file(prompts, reference["prompts_sha256"], PROMPTS_REL, fetch)
        judge.load_prompts(prompts)
        rows = judge.load_jsonl(dataset)
        by_id = {row["id"]: row for row in rows}
        ids = manifest["selected_ids"]
        if len(by_id) != len(rows) or len(set(ids)) != len(ids) or any(i not in by_id for i in ids):
            raise ValueError("Duplicate or unknown dataset/selected IDs")
        answers = [json.loads(line) for line in source_data["answers.jsonl.gz"].splitlines() if line.strip()]
        statuses = [json.loads(line) for line in source_data["turn_status.jsonl.gz"].splitlines() if line.strip()]
        if len(answers) != len(ids) or {a["id"] for a in answers} != set(ids):
            raise ValueError("Snapshot answer coverage mismatch")
        expected = {(i, t) for i in ids for t in range(len(by_id[i]["history"]))}
        records = {}
        for record in statuses:
            key = (record["dialogue_id"], record["turn_index"])
            if (key in records or key not in expected or record["choice_index"] != 0
                    or record["seed"] != base.turn_seed(manifest["seed"], manifest["model_id"], key[0], 0, key[1])):
                raise ValueError("Invalid saved turn identity/seed or duplicate turn")
            records[key] = record
        if set(records) != expected:
            raise ValueError("Snapshot has missing turn statuses")
        run_id = base.stable_id(json.dumps(manifest, sort_keys=True))
        failed = set()
        for answer in answers:
            row = by_id[answer["id"]]
            if (answer["model_id"] != manifest["model_id"] or answer["answer_id"] != base.stable_id(run_id, row["id"])
                    or answer["task"] != row["task"] or answer["method"] != row["method"]
                    or len(answer["choices"]) != 1 or answer["choices"][0]["index"] != 0):
                raise ValueError("Snapshot answer identity mismatch")
            turns = answer["choices"][0]["turns"]
            if len(turns) != len(row["history"]):
                raise ValueError("Snapshot answer length mismatch")
            for number, turn in enumerate(turns):
                record = records[row["id"], number]
                message = turn.get("message")
                if (turn.get("role") != "assistant" or not isinstance(message, str)
                        or sha(message.encode("utf-8")) != record["generated_response_sha256"]
                        or turn.get("error") != record.get("error")):
                    raise ValueError("Snapshot answer/status mismatch")
                if record.get("error"):
                    failed.add(row["id"])
                elif not message.strip() or message == "ERROR":
                    raise ValueError("Unmarked failed response")
        if failed - {344}:
            raise ValueError(f"Unreviewed failed dialogues: {sorted(failed - {344})}")
        included = [a for a in answers if a["id"] not in failed]
        joined = judge.validate_and_join(included, rows, 0)
        if not joined:
            raise ValueError("No complete successful dialogues to judge")
        report = {
            "snapshot_answer_audit_passed": True, "full_generation_audit_passed": False,
            "audit_scope": "snapshot checksums, source/dataset identity, answer/status hashes, seeds and coverage",
            "limitation": "Full generation/candidate journals are absent; this does not establish generation audit parity",
            "source_snapshot": str(snapshot), "snapshot_captured": index["snapshot_finished"],
            "snapshot_index_sha256": sha(inventory.replace(b"\r\n", b"\n")),
            "source_sha256": {name: sha(data) for name, data in source_data.items()},
            "dataset_sha256": manifest["dataset_sha256"], "prompts_sha256": reference["prompts_sha256"],
            "included_dialogues": len(included), "included_turns": sum(len(a["choices"][0]["turns"]) for a in included),
            "expected_dialogues": len(ids), "expected_turns": len(expected),
            "expected_judge_requests": sum(len(judge.judged_turn_indices(q)) for q, _ in joined),
            "excluded_dialogue_ids": sorted(failed), "generation_complete_without_failures": False,
            "failed_turns": [{"dialogue_id": key[0], "turn_index": key[1], "error": r["error"]}
                             for key, r in sorted(records.items()) if r.get("error")],
            "judge_calls": 0,
        }
        contents = {"answers.jsonl": "".join(json.dumps(a, ensure_ascii=False, sort_keys=True) + "\n" for a in included),
                    "generation_validation.json": json.dumps(report, indent=2, sort_keys=True) + "\n",
                    "source_run_config.json": json.dumps(manifest, indent=2, sort_keys=True) + "\n"}
        for name, content in contents.items():
            path = target / name
            if path.exists() and path.read_bytes() != content.encode("utf-8"):
                raise ValueError(f"Prepared input changed: {path}; use a fresh output directory")
        for name, content in contents.items():
            path = target / name
            if not path.exists():
                temporary = path.with_suffix(path.suffix + ".tmp")
                temporary.write_bytes(content.encode("utf-8"))
                temporary.replace(path)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--snapshot", type=Path, help="Saved result snapshot root (includes snapshot.json)")
    mode.add_argument("--source-dir", type=Path, help="Live full-journal run; Linux/WSL only")
    parser.add_argument("--output-dir", type=Path, help="Separate frozen-input directory; judgments are a subdirectory")
    parser.add_argument("--dataset", type=Path, default=ROOT / judge.DEFAULT_DATASET)
    parser.add_argument("--prompts", type=Path, default=ROOT / judge.DEFAULT_PROMPTS)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--fetch-benchmark", action="store_true", help="Fetch missing public dataset/rubrics at a pinned revision, verifying saved hashes")
    parser.add_argument("--dry-run", action="store_true", help="Prepare/audit inputs only; never call the judge API")
    args = parser.parse_args(argv)
    if args.parallel < 1:
        parser.error("--parallel must be positive")
    if args.snapshot is None and args.source_dir is None:
        if os.name != "nt" and (DEFAULT_SOURCE / "turns.jsonl").is_file():
            args.source_dir = DEFAULT_SOURCE
        else:
            args.snapshot = DEFAULT_SNAPSHOT
    if args.source_dir and os.name == "nt":
        parser.error("Full-journal preparation uses Linux locks; use --snapshot on Windows or run this mode in WSL")
    if args.snapshot:
        default_target = ROOT / "outputs/safedial_judging" / (RUN + "_snapshot_" + args.snapshot.name)
    else:
        default_target = args.source_dir / "judging_complete_dialogues_v1"
    target = (args.output_dir or default_target).resolve()
    if args.snapshot:
        snapshot = args.snapshot.resolve()
        if target == snapshot or snapshot in target.parents or target in snapshot.parents:
            parser.error("Output must be separate from the saved snapshot")
    judge_dir = target / "judgments_gpt-4o-mini"
    with file_lock(judge_dir / ".judge.lock"):
        if args.snapshot:
            print(f"Using saved snapshot: {args.snapshot.resolve()}", flush=True)
            report = prepare_snapshot(args.snapshot, target, args.dataset.resolve(), args.prompts.resolve(), args.fetch_benchmark)
        else:
            from prepare_safedial_smoothllm_judging import prepare
            report = prepare(args.source_dir, target, args.dataset, (344,))
        print(json.dumps(report, indent=2), flush=True)
        command = [sys.executable, "-u", "-B", str(ROOT / "scripts/judge_safedial.py"),
                   "--answers", str(target / "answers.jsonl"), "--output-dir", str(judge_dir),
                   "--dataset", str(args.dataset.resolve()), "--prompts", str(args.prompts.resolve()),
                   "--env-file", str(args.env_file.resolve()), "--judge-model", "gpt-4o-mini",
                   "--temperature", "0.7", "--max-tokens", "2048", "--parallel", str(args.parallel)]
        subprocess.run(command + ["--dry-run"], check=True, cwd=ROOT)
        if args.dry_run:
            return 0
        if any(importlib.util.find_spec(name) is None for name in ("openai", "httpx")):
            raise RuntimeError("Install judging dependencies with: python -m pip install openai httpx")
        return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
