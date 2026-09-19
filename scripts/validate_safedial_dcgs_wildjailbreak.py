#!/usr/bin/env python3
"""Audit policy parity, context loss, coverage and GPU evidence separately."""
import argparse
import json
from pathlib import Path

from run_safedial_baseline import file_sha256, load_jsonl
from run_safedial_dcgs_wildjailbreak import source_hashes, tokenizer_for, PROTOCOL, answers_for
from safedial_dcgs_wildjailbreak import MAX_UNTRUNCATED_LEN, METHOD_SPEC, OBJECTIVES, configuration
from safedial_dcgs_run_state import audit_state, context_summary, coverage, event_key, read_records, validate_recovery
from validate_safedial_generation import validate_run



def validate(folder, require_gpu=False, tokenizer=None):
    folder = Path(folder)
    manifest = json.loads((folder / "run_config.json").read_text())
    if (manifest["protocol"] != PROTOCOL or manifest["source_sha256"] != source_hashes()
            or manifest.get("method_spec") != METHOD_SPEC):
        raise ValueError("Main-method/source identity mismatch; legacy v2 runs require their original checkout")
    if file_sha256(Path(manifest["dataset"])) != manifest["dataset_sha256"]:
        raise ValueError("Dataset hash mismatch")
    validate_recovery(folder, manifest)
    config = configuration(manifest["method"], manifest["effective_config"]["device"], manifest["artifacts"])
    if json.loads(json.dumps(vars(config))) != manifest["effective_config"]:
        raise ValueError("Effective reference configuration mismatch")
    if manifest["execution_policy"]["on_turn_error"] not in ("stop", "record-and-continue"):
        raise ValueError("Invalid execution policy")
    tokenizer = tokenizer or tokenizer_for(manifest["artifacts"])
    rows = {row["id"]: row for row in load_jsonl(Path(manifest["dataset"]))}
    selected = [rows[i] for i in manifest["selected_ids"]]
    if len(set(manifest["selected_ids"])) != len(selected):
        raise ValueError("Duplicate selected IDs")
    state = audit_state(folder, selected, manifest, config, tokenizer)
    if read_records(folder / "answers.jsonl") != answers_for(selected, manifest, state["records"]):
        raise ValueError("Native answer export does not match completed dialogues")
    report = {"method_spec": METHOD_SPEC, "integrity_passed": True, "policy_parity_passed": True, **coverage(state)}
    if report["complete_without_failures"]:
        native = validate_run(folder)
        report.update(dialogues=native["dialogues"], turns=native["turns"])
    successful = [e for e in state["journal"] if event_key(e) in state["used"]]
    other = [e for e in state["journal"] if event_key(e) not in state["used"]]
    report.update(context_summary(successful))
    report["failed_or_interrupted_critic_context"] = context_summary(other)
    generations = [e for e in state["journal"] if e["request"]["kind"] == "generate"]
    report.update(journal_events=len(state["journal"]), non_successful_turn_events=len(other),
        interrupted_calls_with_unknown_cost=state["unknown_cost_calls"],
        generation_calls=len(generations), custom_empty_retries=0, judge_calls=0,
        generated_tokens_including_failed_and_interrupted=sum(len(ids) for e in generations
            for batch in e.get("result", {}).get("raw_generations", []) for ids in batch["generated_token_ids"]))
    ll_scores = [e for e in state["journal"] if e["request"]["kind"] == "score_ll" and not e.get("error")]
    report.update(ll_critic_calls=len(ll_scores),
                  ll_scored_candidates=sum(len(e["request"]["actions"]) for e in ll_scores),
                  ll_critic_objectives=sorted({e["result"]["objective"] for e in ll_scores}),
                  ll_critic_max_input_tokens=max((n for e in ll_scores for n in e["result"]["ll_critic_context"]["input_tokens"]), default=0))
    overflows = [f for f in state["failures"] if f["code"] == "ll_context_overflow"]
    report.update(ll_critic_context_limit=MAX_UNTRUNCATED_LEN,
                  ll_critic_overflow_turns=len(overflows),
                  ll_critic_overflow_max_input_tokens=max((f["input_tokens"] for f in overflows), default=0))
    runtimes = read_records(folder / "runtime.jsonl")
    by_invocation = {r["invocation"]: r for r in runtimes}
    if len(by_invocation) != len(runtimes):
        raise ValueError("Duplicate runtime invocation")
    # A failed startup with no model work legitimately has no CUDA peak.
    report["startup_failures_without_gpu_measurement"] = sum(not r["model_work_started"] and
        not r.get("peak_gpu_allocated_bytes") for r in runtimes)
    attempted_invocations = {e["invocation"] for e in state["attempts"]}
    report["model_invocations_without_gpu_measurement"] = sorted(i for i in attempted_invocations
        if not by_invocation.get(i, {}).get("peak_gpu_allocated_bytes"))
    report["elapsed_seconds_recorded"] = sum(r["elapsed_seconds"] for r in runtimes)
    if require_gpu:
        loadings = read_records(folder / "model_loading.jsonl")
        loaded = {r["invocation"]: r for r in loadings}
        if len(loaded) != len(loadings):
            raise ValueError("Duplicate model-loading invocation")
        used_invocations = {r["invocation"] for r in state["records"]}
        if not used_invocations:
            raise ValueError("No successful GPU execution to verify")
        expected = ["q", "v"] + (["q_min", "v_min", "regret"] if manifest["method"] == "rdcgs" else [])
        for invocation in used_invocations:
            loading = loaded.get(invocation, {})
            runtime = by_invocation.get(invocation, {})
            if (loading.get("strict_loaded_heads") != expected or loading.get("ll_reranking") is not True
                    or loading.get("method_spec") != METHOD_SPEC
                    or loading.get("token_critic_sha256") != manifest["artifacts"]["token_critic"]["sha256"]
                    or loading.get("token_critic_objective") not in OBJECTIVES
                    or any(e["result"]["objective"] != loading["token_critic_objective"] for e in ll_scores if e["invocation"] == invocation)
                    or loading.get("checkpoint_sha256") != manifest["artifacts"]["high_level"]["sha256"]
                    or loading.get("actor_dtype") != "torch.bfloat16" or not loading.get("device", "").startswith("cuda")):
                raise ValueError("Original checkpoint/GPU loading mismatch")
            if not runtime.get("model_work_started") or runtime.get("peak_gpu_allocated_bytes", 0) <= 0:
                raise ValueError("Successful model work lacks GPU memory evidence")
        if any(not e.get("result", {}).get("raw_generations") for e in generations if not e.get("error")):
            raise ValueError("Raw generation accounting missing")
        report["peak_gpu_allocated_bytes"] = max(r.get("peak_gpu_allocated_bytes", 0) for r in runtimes)
        report["gpu_evidence_passed"] = True
    report["passed"] = report["complete_without_failures"]
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true", help="Audit partial/failed runs without treating missing coverage as success")
    args = parser.parse_args()
    report = validate(args.output_dir, args.require_gpu)
    (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    raise SystemExit(0 if report["passed"] or args.allow_incomplete else 2)
