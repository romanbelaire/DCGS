"""Offline, verified continuation import. Never loads a model or submits a job."""
import argparse
import copy
import json
import shutil
import tempfile
from pathlib import Path

import run_safedial_baseline as base
from safedial_generation_retry import reusable_empty_event, retry_request


class PendingCall(Exception):
    pass


def prepare(method, source, target, snapshot):
    if target.exists():
        raise ValueError("Continuation destination must be fresh")
    if method == "tpo":
        import run_safedial_tpo_full as entry
        import run_safedial_tpo as runner
        import safedial_tpo as algorithm
        audit_key = "tpo"
    else:
        import run_safedial_dcgs_methods as entry
        runner = entry
        import safedial_dcgs_methods as algorithm
        audit_key = "dcgs"
    with runner.single_writer(source):
        old = json.loads((source / "run_config.json").read_text())
        if old["defense"].get("empty_generation_retry"):
            raise ValueError("Source already has a retry policy")
        source_hashes = {str(p.relative_to(source)): base.file_sha256(p)
                         for p in source.rglob("*") if p.is_file() and p.name != ".generation.lock"}
        for name, digest in old["implementation_sha256"].items():
            name = name if name.startswith(("scripts/", "src/")) else "scripts/" + name
            if base.file_sha256(snapshot / name) != digest:
                raise ValueError(f"Original source snapshot mismatch: {name}")
        flags = ["--dataset", old["dataset"], "--output-dir", str(target),
                 "--model-id", old["model_id"], "--seed", str(old["seed"]),
                 "--device", old["device"], "--empty-generation-retries", "2",
                 "--ids", ",".join(str(i) for i in old["selected_ids"])]
        if method != "tpo":
            flags += ["--method", method, "--ll-device", old["ll_device"]]
        else:
            config = old["defense"]
            flags += ["--sample-size", str(config["sample_size"]), "--max-iters", str(config["max_iterations"]),
                      "--max-new-tokens", str(config["response_tokens"]), "--feedback-tokens", str(config["feedback_tokens"])]
        args = entry.parse_args(flags)
        rows = base.select_dialogues(base.load_jsonl(args.dataset), args)
        new = entry.manifest_for(args, rows, old["artifacts"])
        equivalent = copy.deepcopy(new)
        equivalent["defense"].pop("empty_generation_retry")
        equivalent["defense"]["implementation_version"] = old["defense"]["implementation_version"]
        equivalent["implementation_sha256"] = old["implementation_sha256"]
        if equivalent != old:
            raise ValueError("Import would change fields besides retry policy/source identity")
        old_id = base.stable_id(json.dumps(old, sort_keys=True))
        new_id = base.stable_id(json.dumps(new, sort_keys=True))
        old_args = copy.copy(args)
        old_args.output_dir = source
        _, latest_turns, latest_events = runner.load_state(old_args, old, rows)
        originals = base.load_jsonl(source / "events.jsonl")
        # Old duplicate failed attempts remain in the archive; active journal is canonical.
        events = [{"run_id": new_id, "dialogue_id": k[0], "turn_index": k[1], "event": copy.deepcopy(e)}
                  for k, e in sorted(latest_events.items())]
        failures = [(k, e) for k, e in latest_events.items() if e.get("error")]
        if len(failures) != 1 or not reusable_empty_event(failures[0][1]):
            raise ValueError("Expected exactly one known empty-generation failure")
        failed_key, failed_event = failures[0]
        by_id = {r["id"]: r for r in rows}
        turns = []
        for key, original in sorted(latest_turns.items()):
            if original["prompt_history"] != base.gold_messages(by_id[key[0]]["history"], key[1]):
                raise ValueError("Saved gold history mismatch")
            if original["seed"] != base.turn_seed(args.seed, args.model_id, key[0], 0, key[1]):
                raise ValueError("Saved turn seed mismatch")
            journal = [e for k, e in sorted(latest_events.items()) if k[:2] == key]
            if original[audit_key]["events"] != journal:
                raise ValueError("Source event/turn audit mismatch")
            if original.get("error"):
                if key != failed_key[:2] or original.get("generated_response") is not None:
                    raise ValueError("Unexpected failed source turn")
                continue  # Preserve failed attempts in the journal; resume this turn automatically.
            algorithm.validate_audit(original, old["defense"])
            algorithm.validate_audit(original, new["defense"])
            record = copy.deepcopy(original)
            record.update(run_id=new_id, answer_id=base.stable_id(new_id, key[0]))
            turns.append(record)
        target.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".empty-retry-import-", dir=target.parent))
        archive = stage / "provenance/original_run"
        shutil.copytree(source, archive, ignore=shutil.ignore_patterns(".generation.lock"))
        shutil.copytree(snapshot, stage / "provenance/original_source")
        base.ensure_manifest(stage / "run_config.json", new)
        runner.atomic_jsonl(stage / "events.jsonl", events)
        runner.atomic_jsonl(stage / "turns.jsonl", turns)
        runner.atomic_jsonl(stage / "answers.jsonl", [])
        invocations = base.load_jsonl(source / "invocations.jsonl")
        for invocation in invocations:
            invocation.setdefault("source_run_id", old_id)
        runner.atomic_jsonl(stage / "invocations.jsonl", invocations)
        args.output_dir = stage
        pending = []

        def stop(request):
            pending.append(request)
            raise PendingCall()

        try:
            runner.generate_selected(args, new, rows, stop)
        except PendingCall:
            pass
        expected = retry_request(failed_event["request"], 1)
        if pending != [expected]:
            raise ValueError("First missing call is not precisely the first new-seed retry")
        new_events = base.load_jsonl(stage / "events.jsonl")
        if new_events != events:
            raise ValueError("Import changed original attempts")
        old_answers = base.load_jsonl(source / "answers.jsonl")
        new_answers = base.load_jsonl(stage / "answers.jsonl")
        expected_answers = copy.deepcopy(old_answers)
        for answer in expected_answers:
            answer["answer_id"] = base.stable_id(new_id, answer["id"])
        if new_answers != expected_answers:
            raise ValueError("Import changed exported answers")
        report = {"source_directory": str(source.resolve()), "source_run_id": old_id,
                  "destination_run_id": new_id, "source_file_sha256": source_hashes,
                  "successful_turns_preserved": len(turns), "complete_dialogues_preserved": len(new_answers),
                  "original_journal_rows": len(originals), "canonical_events_preserved": len(events),
                  "failed_event_preserved": list(failed_key), "policy": new["defense"]["empty_generation_retry"],
                  "next_new_call": {k: v for k, v in expected.items() if k not in ("prompt", "messages")},
                  "model_calls_during_preparation": 0,
                  "runtime_evidence": "Original evidence archived; continuation must write its own GPU evidence"}
        for name, digest in source_hashes.items():
            if base.file_sha256(source / name) != digest or base.file_sha256(archive / name) != digest:
                raise ValueError(f"Source/archive preservation failure: {name}")
        (stage / "continuation.json").write_text(json.dumps(report, indent=2) + "\n")
        if target.exists():
            raise ValueError("Continuation destination appeared during import")
        stage.rename(target)
        return {k: v for k, v in report.items() if k != "source_file_sha256"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("vdcgs", "rdcgs", "tpo"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.method, args.source, args.output_dir, args.source_snapshot), indent=2))
