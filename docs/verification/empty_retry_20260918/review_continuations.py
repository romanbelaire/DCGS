"""Replay real checkpoints, then use injected fixture calls only in /tmp."""
import copy
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "tests")]
import run_safedial_baseline as base
from test_safedial_dcgs_methods import fake_execute as fake_dcgs
from test_safedial_tpo import fake_execute as fake_tpo


def review(method, directory):
    if method == "tpo":
        import run_safedial_tpo_full as entry
        import run_safedial_tpo as runner
        from validate_safedial_tpo_full import validate_full as validate
        fake = fake_tpo
    else:
        import run_safedial_dcgs_methods as entry
        runner = entry
        from validate_safedial_dcgs_methods import validate_dcgs as validate
        fake = fake_dcgs
    directory = ROOT / directory
    source_hashes = {n: base.file_sha256(directory / n) for n in ("run_config.json", "events.jsonl", "turns.jsonl", "answers.jsonl")}
    prepared = json.loads((directory / "run_config.json").read_text())
    continuation = json.loads((directory / "continuation.json").read_text())
    failed = continuation["failed_event_preserved"]
    rows = base.load_jsonl(Path(prepared["dataset"]))
    rows = [row for row in rows if row["id"] in prepared["selected_ids"]]
    end = next(i for i, row in enumerate(rows) if row["id"] == failed[0])
    rows = rows[:end + 1]
    fixture = Path(tempfile.mkdtemp(prefix=f"empty-retry-{method}-fixture-"))
    flags = ["--dataset", prepared["dataset"], "--output-dir", str(fixture),
             "--model-id", prepared["model_id"], "--device", prepared["device"],
             "--ids", ",".join(str(row["id"]) for row in rows)]
    if method != "tpo":
        flags += ["--method", method, "--ll-device", prepared["ll_device"]]
    args = entry.parse_args(flags)
    manifest = entry.manifest_for(args, rows, prepared["artifacts"])
    run_id = base.stable_id(json.dumps(manifest, sort_keys=True))
    base.ensure_manifest(fixture / "run_config.json", manifest)
    old_turns = base.load_jsonl(directory / "turns.jsonl")
    old_events = base.load_jsonl(directory / "events.jsonl")
    turns, events = copy.deepcopy(old_turns), copy.deepcopy(old_events)
    for record in turns:
        record.update(run_id=run_id, answer_id=base.stable_id(run_id, record["dialogue_id"]))
    for record in events:
        record["run_id"] = run_id
    runner.atomic_jsonl(fixture / "turns.jsonl", turns)
    runner.atomic_jsonl(fixture / "events.jsonl", events)
    calls = []

    def injected(request):
        calls.append(copy.deepcopy(request))
        return fake(request)

    assert runner.generate_selected(args, manifest, rows, injected) == 0
    first = {k: v for k, v in calls[0].items() if k not in ("prompt", "messages")}
    assert first == continuation["next_new_call"]
    expected_first = copy.deepcopy(events[-1]["event"]["request"])
    from safedial_generation_retry import retry_request
    assert calls[0] == retry_request(expected_first, 1)
    fresh_turns = base.load_jsonl(fixture / "turns.jsonl")
    assert fresh_turns[:len(turns)] == turns, "Previously completed turns changed"
    assert base.load_jsonl(fixture / "events.jsonl")[:len(events)] == events, "Saved calls changed"
    report = validate(fixture)
    assert report["empty_generation_attempts"] == 1 and report["retry_generations"] == 1
    before = {n: base.file_sha256(fixture / n) for n in source_hashes}
    def forbidden(request):
        raise AssertionError("Completed fixture attempted another model call")
    assert runner.generate_selected(args, manifest, rows, forbidden) == 0
    assert before == {n: base.file_sha256(fixture / n) for n in before}
    assert source_hashes == {n: base.file_sha256(directory / n) for n in source_hashes}
    # Check every preserved original/nested provenance file, not just the journals.
    original = Path(continuation["source_directory"])
    archive = directory / "provenance/original_run"
    for name, digest in continuation["source_file_sha256"].items():
        assert base.file_sha256(original / name) == digest
        assert base.file_sha256(archive / name) == digest
    assert manifest["implementation_sha256"] == prepared["implementation_sha256"]
    return {"method": method, "fixture_directory": str(fixture),
            "real_events_preserved": len(events), "real_successful_turns_preserved": len(turns),
            "injected_calls": len(calls), "first_injected_call": first,
            "validation": report, "byte_identical_noop": True,
            "original_and_nested_archive_hashes_match": True, "model_calls": 0}


if __name__ == "__main__":
    results = [review(method, folder) for method, folder in [
        ("vdcgs", "outputs/safedial_dcgs/zephyr_vdcgs_aug11_a5000_full_retry_v1"),
        ("rdcgs", "outputs/safedial_dcgs/zephyr_rdcgs_aug11_a5000_full_retry_v1"),
        ("tpo", "outputs/safedial_baseline/tpo_zephyr_full_v8")]]
    (Path(__file__).parent / "continuation_review.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
