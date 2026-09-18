"""One-time, offline import of the exact 254257 full run into TPO format policy v6.

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

SOURCE = Path("outputs/safedial_baseline/tpo_zephyr_full_v5")
TARGET = Path("outputs/safedial_baseline/tpo_zephyr_full_v6")
EXPECTED = {'run_config.json': 'fe5e49e46f769e27f38c920dbac9e1a864a4319b52fdfcb15c5741f93eee229c', 'events.jsonl': '5bbe7300bda5b9cf79850656e689634b6368ed7a447cb8f10b8795eb5ecbd7a0', 'turns.jsonl': '14e42e84392ed226d7bc851b5498ec86dce5152c498cf8c50aa2d0fdd4122e82', 'answers.jsonl': 'dda1b644fe6be47795bd1254fcb1eda77b991268f664c3e84c8917eddf4459a3', 'invocations.jsonl': '06735f7b537f69a51e97fff5f88134322e8f5e1fd812c5994ffde2ac4751161a', 'continuation.json': '9bbde0a02a7cbe3b6e217bdc157c083c867a23afc2971a8629bb5d19c0788cef', 'offline_preparation_review.json': '8078db01044effaad5dba552dff3848c336da891cd6b26a521bfdb6fb4563f4c', 'preflight.json': '8ee2178cd1264e3dbbfc4221520c46223b3df14be643aa5ccf581ae9b14dd154', 'model_loading.json': 'e7e40c077db2c404acd609990c8b537b18fd53bd28dcbd65a7687fcb863bd45f', 'context_probe.json': 'aee0c121dcc3a47e6bdbb48ff13242da9b48ba79163a1f7b38ed196cafff4c76', 'runtime_stats.json': '24e1e0bb9d82c00ea79478b4a2713156c51e4dbcae40ba21b1beec828b3b9827', 'failure_review_254257.json': '4c0449465c9ab2d5384c58a2508dd4deff1be86ccc2f5efee2ce2b934267c92a'}
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
        equivalent["defense"]["implementation_version"] = 5
        equivalent["defense"]["optimizer_extraction"] = 'one_opening_tag_before_optional_close; missing_close_extracts_to_end_with_warning; trailing_opening_tags_after_close_warn; exactly_two_boundary_opening_tags_without_close_extract_between_with_warning; exactly_two_complete_nonnested_blocks_equal_after_boundary_whitespace_and_one_outer_brace_pair_extract_first_with_warning; nonempty; ambiguous_is_error'
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
                assert warnings == [tpo.DUPLICATE_BLOCK_WARNING, tpo.TRAILING_START_WARNING]
                assert (entry["dialogue_id"], entry["turn_index"], event["index"]) == (25, 4, 16)
                event.pop("error")
                event["format_warnings"] = warnings
                accepted.append([entry["dialogue_id"], entry["turn_index"], event["index"]])
            else:
                assert event.get("format_warnings", []) == warnings
            events.append(entry)
        assert accepted == [[25, 4, 16]]
        for original in original_turns:
            assert original["run_id"] == old_id
            assert original["prompt_history"] == base.gold_messages(by_id[original["dialogue_id"]]["history"], original["turn_index"])
            assert original["seed"] == base.turn_seed(args.seed, args.model_id, original["dialogue_id"], 0, original["turn_index"])
            if original.get("error"):
                assert (original["dialogue_id"], original["turn_index"]) == (25, 4) and original["generated_response"] is None
                assert original["tpo"]["events"] == [e["event"] for e in original_events if (e["dialogue_id"], e["turn_index"]) == (25, 4)]
                continue  # Incomplete turn is represented by its reusable event journal.
            tpo.validate_audit(original, new["defense"])
            assert original["tpo"]["events"] == [e["event"] for e in events if (e["dialogue_id"], e["turn_index"]) == (original["dialogue_id"], original["turn_index"])]
            record = copy.deepcopy(original)
            record.update(run_id=new_id, answer_id=base.stable_id(new_id, record["dialogue_id"]))
            turns.append(record)
        assert len(turns) == 126 and len(events) == 4301
        stage = Path(tempfile.mkdtemp(prefix=".tpo-format-import-", dir=TARGET.parent))
        archive = stage / "provenance/254257"
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
            record.setdefault("source_job_id", 254257)
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
        assert len(base.load_jsonl(stage / "answers.jsonl")) == 24
        request = attempted[0]
        assert {k: request[k] for k in ("kind", "stage", "iteration", "slot")} == {
            "kind": "reward", "stage": "score", "iteration": 0, "slot": 2}
        expected_text = tpo.extract_update(original_events[-1]["event"]["result"]["message"])
        assert request["messages"] == base.gold_messages(by_id[25]["history"], 4) + [{"role": "assistant", "content": expected_text}]
        report = {
            "source_job_id": 254257, "source_directory": str(SOURCE.resolve()),
            "source_run_id": old_id, "destination_run_id": new_id, "original_file_sha256": EXPECTED,
            "policy": new["defense"]["optimizer_extraction"], "reused_events": 4301,
            "reused_successful_turns": 126, "reclassified_event_keys": accepted,
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
