"""One-time, offline import of the exact 250576 full run into TPO format policy v4.

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

SOURCE = Path("outputs/safedial_baseline/tpo_zephyr_full")
TARGET = Path("outputs/safedial_baseline/tpo_zephyr_full_v4")
EXPECTED = {'run_config.json': 'f639fbaa9dc32c11ee27d3e7bc4e5a2b62852ec12915a6c43fa6dcbb0d85f409', 'preflight.json': '8ee2178cd1264e3dbbfc4221520c46223b3df14be643aa5ccf581ae9b14dd154', 'answers.jsonl': '74f8c8b83dd5eedbc2e89b083ec30e8fd88c511569f6f023f339ce8910ed2a83', 'model_loading.json': 'e7e40c077db2c404acd609990c8b537b18fd53bd28dcbd65a7687fcb863bd45f', 'context_probe.json': 'd149c91924f85e94546dc31334cd4c7ceca07f2090591492dc0d02d837d3618f', 'events.jsonl': '989a35635aad0bb040e6b5ef16b4e775b9eda6de3f9907997a9c745918a743ad', 'turns.jsonl': '899d119ec82f1c53c822d93ce36d740ad1237f426b4d48cae6b589d00dc8e9cb', 'runtime_stats.json': '14fbcfac2ebb932a8966d46ea027cc70970a0f6ef179f50cbf971ac4dae76865', 'invocations.jsonl': 'ebacd48d571225a2309fa182f6a9241d4881ac518fea059317d501f5b8fe80cb'}
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
        equivalent["defense"]["implementation_version"] = 3
        equivalent["defense"]["optimizer_extraction"] = "one_opening_tag_before_optional_close; missing_close_extracts_to_end_with_warning; trailing_opening_tags_after_close_warn; nonempty; ambiguous_is_error"
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
                assert warnings == [tpo.TERMINAL_START_WARNING]
                assert (entry["dialogue_id"], entry["turn_index"], event["index"]) == (2, 4, 14)
                event.pop("error")
                event["format_warnings"] = warnings
                accepted.append([entry["dialogue_id"], entry["turn_index"], event["index"]])
            else:
                assert event.get("format_warnings", []) == warnings
            events.append(entry)
        assert accepted == [[2, 4, 14]]
        for original in original_turns:
            assert original["run_id"] == old_id
            assert original["prompt_history"] == base.gold_messages(by_id[original["dialogue_id"]]["history"], original["turn_index"])
            assert original["seed"] == base.turn_seed(args.seed, args.model_id, original["dialogue_id"], 0, original["turn_index"])
            if original.get("error"):
                assert (original["dialogue_id"], original["turn_index"]) == (2, 4) and original["generated_response"] is None
                assert original["tpo"]["events"] == [e["event"] for e in original_events if (e["dialogue_id"], e["turn_index"]) == (2, 4)]
                continue  # Incomplete turn is represented by its reusable event journal.
            tpo.validate_audit(original, new["defense"])
            assert original["tpo"]["events"] == [e["event"] for e in events if (e["dialogue_id"], e["turn_index"]) == (original["dialogue_id"], original["turn_index"])]
            record = copy.deepcopy(original)
            record.update(run_id=new_id, answer_id=base.stable_id(new_id, record["dialogue_id"]))
            turns.append(record)
        assert len(turns) == 9 and len(events) == 321
        stage = Path(tempfile.mkdtemp(prefix=".tpo-format-import-", dir=TARGET.parent))
        archive = stage / "provenance/250576"
        archive.mkdir(parents=True)
        for name in EXPECTED:
            shutil.copy2(SOURCE / name, archive / name)
        shutil.copytree(SNAPSHOT / "scripts", archive / "scripts")
        shutil.copy2(__file__, archive / "prepare_format_resume.py")
        base.ensure_manifest(stage / "run_config.json", new)
        runner.atomic_jsonl(stage / "events.jsonl", events)
        runner.atomic_jsonl(stage / "turns.jsonl", turns)
        runner.atomic_jsonl(stage / "answers.jsonl", [])
        invocations = base.load_jsonl(SOURCE / "invocations.jsonl")
        for record in invocations:
            record.setdefault("source_job_id", 250576)
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
        assert len(base.load_jsonl(stage / "answers.jsonl")) == 1
        request = attempted[0]
        assert {k: request[k] for k in ("kind", "stage", "iteration", "slot")} == {
            "kind": "reward", "stage": "score", "iteration": 0, "slot": 1}
        expected_text = tpo.extract_update(original_events[-1]["event"]["result"]["message"])
        assert request["messages"] == base.gold_messages(by_id[2]["history"], 4) + [{"role": "assistant", "content": expected_text}]
        report = {
            "source_job_id": 250576, "source_directory": str(SOURCE.resolve()),
            "source_run_id": old_id, "destination_run_id": new_id, "original_file_sha256": EXPECTED,
            "policy": new["defense"]["optimizer_extraction"], "reused_events": 321,
            "reused_successful_turns": 9, "reclassified_event_keys": accepted,
            "raw_results_and_requests_unchanged": True, "model_calls_during_preparation": 0,
            "next_new_call": {k: v for k, v in request.items() if k != "messages"},
            "remaining_expected_actor_calls": 10029 * 19 - 180, "remaining_expected_reward_calls": 10029 * 15 - 141,
            "inherited_runtime": "Original invocation retained with source_job_id; new runtime_stats measures continuation only.",
        }
        (stage / "continuation.json").write_text(json.dumps(report, indent=2) + "\n")
        for name, digest in EXPECTED.items():
            assert base.file_sha256(SOURCE / name) == digest, "Source mutated"
        assert not TARGET.exists()
        shutil.copy2(SNAPSHOT / "scripts/validate_safedial_tpo.py", archive / "scripts/validate_safedial_tpo.py")
        shutil.copy2(SNAPSHOT / "scripts/validate_safedial_tpo_full.py", archive / "scripts/validate_safedial_tpo_full.py")
        (archive / "scripts/slurm").mkdir(exist_ok=True)
        shutil.copy2(SNAPSHOT / "scripts/slurm/run_safedial_tpo_full.sbatch", archive / "scripts/slurm/run_safedial_tpo_full.sbatch")
        stage.rename(TARGET)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
