"""One-time, offline import of the exact 254463 full run into TPO format policy v7.

Run from DCGS with its Python environment. Refuses changed source inputs or an
existing destination. No model calls; the original run remains untouched.
"""
import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "scripts"))
import run_safedial_baseline as base
import run_safedial_tpo as runner
import run_safedial_tpo_full as full
import safedial_tpo as tpo

SOURCE = Path("outputs/safedial_baseline/tpo_zephyr_full_v6")
TARGET = Path("outputs/safedial_baseline/tpo_zephyr_full_v7")
EXPECTED = {'run_config.json': '2ad6389cb051c72b73085749a8e556558523a9f0821f8a441431db90a2a3cfd0', 'events.jsonl': 'ab906d78bf58be2552c6f92e622660e0b37800a7f0099fac527db792c9200690', 'turns.jsonl': 'e42b23a884d68de9e0efe4ee35dfeb0a87e25f54ebc6a7a0508b88915f712e5d', 'answers.jsonl': '1a8ea579709e3e9ea62193bb2776a3a932c1013e9a89865c90d30f05caec9a82', 'invocations.jsonl': 'babe6c9795c312e04cedd6bc582b90e89cd10182ab7bbf42eb925c22cd9f683e', 'continuation.json': '504f9ef75e36ec90133eb9192286c4ee9c7117c6482284b0794668b731c9e33a', 'offline_preparation_review.json': '67d429ea7ffa06c7a8454b1562cd46d5466952e75185881287a18fbfd08c8b2e', 'preflight.json': '8ee2178cd1264e3dbbfc4221520c46223b3df14be643aa5ccf581ae9b14dd154', 'model_loading.json': 'e7e40c077db2c404acd609990c8b537b18fd53bd28dcbd65a7687fcb863bd45f', 'context_probe.json': '22f18c513e0163215522d5a92012a558b3116bfc9a51626fea829c396c6f17b3', 'runtime_stats.json': 'b337928a4017ce32e7c3040548be9d3202d3ac7f93d9bc50f39a6fecd9e00772', 'failure_review_254463.json': '0faa8e4266c5bd894f2e9c1cf7331114cc7fc284c8c423f3df24f5ad1b986954'}
SNAPSHOT = Path(sys.argv[1]) if len(sys.argv) > 1 else None


class PendingCall(Exception):
    pass


def main():
    assert SNAPSHOT is not None, "Pass the verified original source snapshot directory"
    assert not TARGET.exists(), "Destination must be fresh"
    with runner.single_writer(SOURCE):
        for name, digest in EXPECTED.items():
            assert base.file_sha256(SOURCE / name) == digest, name
        old = json.loads((SOURCE / "run_config.json").read_text())
        args = full.parse_args(["--output-dir", str(TARGET)])
        rows = base.select_dialogues(base.load_jsonl(args.dataset), args)
        new = full.manifest_for(args, rows, old["artifacts"])
        by_id = {r["id"]: r for r in rows}
        equivalent = copy.deepcopy(new)
        equivalent["defense"]["implementation_version"] = old["defense"]["implementation_version"]
        equivalent["defense"]["optimizer_extraction"] = old["defense"]["optimizer_extraction"]
        equivalent["implementation_sha256"] = old["implementation_sha256"]
        assert equivalent == old, "Only the approved parser policy/source may change"
        for name, digest in old["implementation_sha256"].items():
            assert base.file_sha256(SNAPSHOT / "scripts" / name) == digest
            if name not in ("safedial_tpo.py",):
                assert new["implementation_sha256"][name] == digest
        old_id = base.stable_id(json.dumps(old, sort_keys=True))
        new_id = base.stable_id(json.dumps(new, sort_keys=True))
        original_events = base.load_jsonl(SOURCE / "events.jsonl")
        original_turns = base.load_jsonl(SOURCE / "turns.jsonl")
        events, turns, accepted = [], [], []
        for original in original_events:
            assert original["run_id"] == old_id
            entry = copy.deepcopy(original)
            entry["run_id"] = new_id
            event = entry["event"]
            warnings = tpo.valid_result(event["request"], event["result"])
            if event.get("error"):
                assert event["error"] == "ValueError: Optimizer output requires one opening tag and at most one ordered closing tag"
                assert warnings == [tpo.FIRST_IMPROVEMENT_WARNING]
                assert (entry["dialogue_id"], entry["turn_index"], event["index"]) == (26, 3, 18)
                event.pop("error")
                event["format_warnings"] = warnings
                accepted.append([entry["dialogue_id"], entry["turn_index"], event["index"]])
            else:
                assert event.get("format_warnings", []) == warnings
            events.append(entry)
        assert accepted == [[26, 3, 18]]
        for original in original_turns:
            assert original["run_id"] == old_id
            assert original["prompt_history"] == base.gold_messages(by_id[original["dialogue_id"]]["history"], original["turn_index"])
            assert original["seed"] == base.turn_seed(args.seed, args.model_id, original["dialogue_id"], 0, original["turn_index"])
            if original.get("error"):
                assert (original["dialogue_id"], original["turn_index"]) == (26, 3) and original["generated_response"] is None
                assert original["tpo"]["events"] == [e["event"] for e in original_events if (e["dialogue_id"], e["turn_index"]) == (26, 3)]
                continue  # Incomplete turn is represented by its reusable event journal.
            tpo.validate_audit(original, new["defense"])
            assert original["tpo"]["events"] == [e["event"] for e in events if (e["dialogue_id"], e["turn_index"]) == (original["dialogue_id"], original["turn_index"])]
            record = copy.deepcopy(original)
            record.update(run_id=new_id, answer_id=base.stable_id(new_id, record["dialogue_id"]))
            turns.append(record)
        assert len(turns) == 130 and len(events) == 4439
        stage = Path(tempfile.mkdtemp(prefix=".tpo-format-import-", dir=TARGET.parent))
        archive = stage / "provenance/254463"
        archive.mkdir(parents=True)
        for name in EXPECTED:
            shutil.copy2(SOURCE / name, archive / name)
        shutil.copytree(SNAPSHOT / "scripts", archive / "scripts")
        shutil.copytree(SOURCE / "provenance", archive / "provenance")
        shutil.copy2(__file__, archive / "prepare_format_resume.py")
        base.ensure_manifest(stage / "run_config.json", new)
        runner.atomic_jsonl(stage / "events.jsonl", events)
        runner.atomic_jsonl(stage / "turns.jsonl", turns)
        runner.atomic_jsonl(stage / "answers.jsonl", [])
        invocations = base.load_jsonl(SOURCE / "invocations.jsonl")
        for record in invocations:
            record.setdefault("source_job_id", 254463)
            record.setdefault("source_run_id", old_id)
        runner.atomic_jsonl(stage / "invocations.jsonl", invocations)
        args.output_dir = stage
        attempted = []

        def stop_before_model(request):
            attempted.append(request)
            raise PendingCall()

        try:
            runner.generate_selected(args, new, rows, stop_before_model)
        except PendingCall:
            pass
        assert len(attempted) == 1
        assert len(base.load_jsonl(stage / "answers.jsonl")) == 25
        request = attempted[0]
        assert {k: request[k] for k in ("kind", "stage", "iteration", "slot")} == {
            "kind": "reward", "stage": "score", "iteration": 0, "slot": 3}
        expected_text = tpo.extract_update(original_events[-1]["event"]["result"]["message"])
        assert request["messages"] == base.gold_messages(by_id[26]["history"], 3) + [{"role": "assistant", "content": expected_text}]
        report = {
            "source_job_id": 254463, "source_directory": str(SOURCE.resolve()),
            "source_run_id": old_id, "destination_run_id": new_id, "original_file_sha256": EXPECTED,
            "policy": new["defense"]["optimizer_extraction"], "reused_events": 4439,
            "reused_successful_turns": 130, "reclassified_event_keys": accepted,
            "raw_results_and_requests_unchanged": True, "model_calls_during_preparation": 0,
            "next_new_call": {k: v for k, v in request.items() if k != "messages"},
            "reused_actor_calls": sum(e["event"]["request"]["kind"] == "generate" for e in events),
            "reused_reward_calls": sum(e["event"]["request"]["kind"] == "reward" for e in events),
            "remaining_expected_actor_calls": 10029 * 19 - sum(e["event"]["request"]["kind"] == "generate" for e in events),
            "remaining_expected_reward_calls": 10029 * 15 - sum(e["event"]["request"]["kind"] == "reward" for e in events),
            "inherited_runtime": "Original invocations retained with source_job_id; new runtime_stats measures continuation only.",
        }
        (stage / "continuation.json").write_text(json.dumps(report, indent=2) + "\n")
        for name, digest in EXPECTED.items():
            assert base.file_sha256(SOURCE / name) == digest, "Source mutated"
            assert base.file_sha256(archive / name) == digest, "Archive mismatch"
        assert not TARGET.exists()
        shutil.copy2(SNAPSHOT / "scripts/validate_safedial_tpo.py", archive / "scripts/validate_safedial_tpo.py")
        shutil.copy2(SNAPSHOT / "scripts/validate_safedial_tpo_full.py", archive / "scripts/validate_safedial_tpo_full.py")
        (archive / "scripts/slurm").mkdir(exist_ok=True)
        shutil.copy2(SNAPSHOT / "scripts/slurm/run_safedial_tpo_full.sbatch", archive / "scripts/slurm/run_safedial_tpo_full.sbatch")
        stage.rename(TARGET)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
