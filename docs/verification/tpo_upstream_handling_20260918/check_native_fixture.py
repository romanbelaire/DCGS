"""Offline native five-turn coverage/resume check; no model or API execution."""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import run_safedial_baseline as base
import run_safedial_tpo_upstream as runner
import run_safedial_tpo_upstream_full as full
from validate_safedial_tpo_upstream import validate_full


def main():
    report = {}
    for empty_first in (False, True):
        with tempfile.TemporaryDirectory() as tmp:
            args = full.parse_args(["--ids", "1", "--device", "cpu", "--output-dir", tmp])
            rows = base.select_dialogues(base.load_jsonl(args.dataset), args)
            lock = json.loads(args.artifact_lock.read_text())
            manifest = full.manifest_for(args, rows, lock)
            base.ensure_manifest(args.output_dir / "run_config.json", manifest)
            calls = []

            def execute(request):
                calls.append(request)
                result = {"prompt_tokens": 10, "original_prompt_tokens": 10,
                          "latency_seconds": 0.01, "input_truncated": False}
                if request["kind"] == "reward":
                    text = request["messages"][-1]["content"]
                    result["score"] = 100 if not text else 1
                else:
                    text = "fixture answer"
                    if empty_first and request["stage"] == "initial" and len(request["messages"]) == 1:
                        text = ""
                    if request["stage"] == "update":
                        text = "missing tags" if request["slot"] == 0 else "<IMPROVED_VARIABLE>fixture update</IMPROVED_VARIABLE>"
                    result.update(message=text, completion_tokens=5, hit_token_cap=False)
                return result

            status = runner.generate_selected(args, manifest, rows, execute)
            audited = validate_full(args.output_dir)
            assert audited["audit_passed"] and audited["processed_turns"] == 5
            assert audited["skipped_candidates"] == 10 and audited["candidates"] == 65
            assert status == (2 if empty_first else 0)
            assert audited["failed_turns"] == int(empty_first)
            assert audited["exported_dialogues"] == int(not empty_first)
            files = [args.output_dir / n for n in ("turns.jsonl", "events.jsonl", "answers.jsonl", "failures.jsonl")]
            before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]

            def forbidden(request):
                raise AssertionError("No-op resume invoked a model")

            assert runner.generate_selected(args, manifest, rows, forbidden) == status
            assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
            report["with_terminal_failure" if empty_first else "with_candidate_skips"] = {
                **audited, "noop_resume_byte_identical": True, "injected_calls": len(calls)}
    (Path(__file__).parent / "native_fixture.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
