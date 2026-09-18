"""Upstream TPO multi-sample extraction, with explicit audit and no resampling.

Reuse frozen prompt construction from the v8 implementation. Keep this policy
separate so running jobs and their manifests are never changed underneath them.
"""
import copy
import math
import time

from run_safedial_baseline import stable_id
from safedial_generation_retry import validate_diagnostics
from safedial_tpo import (START, END, UPSTREAM_COMMIT, EVALUATION_SYSTEM,
                          feedback_query, gradient_messages, update_messages, pair)


def defense_config(sample_size=5, max_iters=2, response_tokens=1024, feedback_tokens=2048):
    for value, minimum in ((sample_size, 2), (max_iters, 1), (response_tokens, 1), (feedback_tokens, 1)):
        if type(value) is not int or value < minimum:
            raise ValueError("Invalid TPO budget")
    return {"name": "TPO", "implementation_version": "upstream-handling-v1",
            "upstream_commit": UPSTREAM_COMMIT, "sample_size": sample_size,
            "max_iterations": max_iters, "response_tokens": response_tokens,
            "feedback_tokens": feedback_tokens, "temperature": 0.7,
            "candidate_top_p": 0.95, "feedback_top_p": 0.99,
            "optimizer_extraction": "split(start)[1].split(end)[0].strip(); skip_IndexError; allow_empty",
            "empty_generation_retries": 0,
            "selection": "max_reward_cumulative_cache_first_index_tie",
            "duplicates": "retain_indexed_candidates; differs_from_upstream_string_cache",
            "history": "native_gold_messages; role_tagged_history_in_feedback",
            "turn_error_policy": "record_empty_selected_answer_and_continue; infrastructure_integrity_stop",
            "model_updates": False, "judge_calls": 0}


def extract_update(raw):
    # Exact expression from pinned upstream TextualGradientDescent.step's
    # multi-sample branch. Do not use split(..., 1): repeated openers matter.
    return raw.split(START)[1].split(END)[0].strip()


def validate_result(request, result):
    """Validate execution evidence, without imposing a nonempty text policy."""
    if result.get("error") or result.get("input_truncated") is not False:
        raise ValueError("Failed or truncated TPO execution")
    for field in ("prompt_tokens", "original_prompt_tokens"):
        if type(result.get(field)) is not int or result[field] < 1:
            raise ValueError(f"Invalid {field}")
    if result["prompt_tokens"] != result["original_prompt_tokens"]:
        raise ValueError("Input token accounting indicates truncation")
    latency = result.get("latency_seconds")
    if type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0:
        raise ValueError("Invalid execution latency")
    if request["kind"] == "reward":
        if type(result.get("score")) not in (int, float) or not math.isfinite(result["score"]):
            raise ValueError("Reward must be finite")
    else:
        validate_diagnostics(request, result, strip_output=True)
        if not isinstance(result.get("message"), str):
            raise ValueError("Generation must return text")
        if type(result.get("completion_tokens")) is not int or not 0 < result["completion_tokens"] <= request["max_new_tokens"]:
            raise ValueError("Invalid generated-token count")


def generate_tpo(messages, config, seed, execute, on_event=None):
    expected = defense_config(config["sample_size"], config["max_iterations"],
                              config["response_tokens"], config["feedback_tokens"])
    if config != expected or not messages or messages[-1]["role"] != "user":
        raise ValueError("Unknown TPO policy or invalid history")
    started = time.perf_counter()
    audit = {"events": [], "candidates": [], "rounds": [], "selected_id": None}

    def call(kind, stage, iteration, slot, prompt):
        request = {"kind": kind, "stage": stage, "iteration": iteration, "slot": slot,
                   "messages": copy.deepcopy(prompt)}
        if kind == "generate":
            feedback = stage in ("loss", "gradient")
            request.update(seed=int(stable_id(seed, "tpo", stage, iteration, slot, length=16), 16) % (2**31 - 1),
                           temperature=config["temperature"],
                           top_p=config["feedback_top_p"] if feedback else config["candidate_top_p"],
                           max_new_tokens=config["feedback_tokens"] if feedback else config["response_tokens"])
        event = {"index": len(audit["events"]), "request": request}
        try:
            result = execute(copy.deepcopy(request))
            event["result"] = copy.deepcopy(result)
            validate_result(request, result)
            if stage == "update":
                try:
                    value = extract_update(result["message"])
                except IndexError:
                    event["candidate_failure"] = {"code": "missing_opening_tag",
                                                  "action": "skip_candidate", "iteration": iteration, "slot": slot}
                else:
                    raw = result["message"]
                    warnings = []
                    if not value:
                        warnings.append("empty_extracted_candidate_scored_as_upstream")
                    if raw.count(START) != 1 or raw.count(END) != 1 or raw.find(END) < raw.find(START):
                        warnings.append("noncanonical_tags_upstream_extraction")
                    if warnings:
                        event["format_warnings"] = warnings
            elif kind == "generate" and not result["message"].strip():
                event["format_warnings"] = ["empty_generation_retained_as_upstream"]
        except Exception as exc:
            event["error"] = f"{type(exc).__name__}: {exc}"
            audit["events"].append(event)
            if on_event:
                on_event(event)
            raise  # Never turn CUDA, context, accounting or integrity errors into skipped candidates.
        audit["events"].append(event)
        if on_event:
            on_event(event)
        return result, event

    def candidates(stage, iteration, prompt):
        for slot in range(config["sample_size"]):
            result, generated = call("generate", stage, iteration, slot, prompt)
            if generated.get("candidate_failure"):
                continue
            text = extract_update(result["message"]) if stage == "update" else result["message"]
            scored, rewarded = call("reward", "score", iteration, slot,
                                    [dict(m) for m in messages] + [{"role": "assistant", "content": text}])
            audit["candidates"].append({"id": len(audit["candidates"]), "iteration": iteration,
                                        "slot": slot, "message": text, "reward": scored["score"],
                                        "generation_event": generated["index"], "reward_event": rewarded["index"]})

    candidates("initial", -1, messages)
    for iteration in range(config["max_iterations"]):
        chosen, rejected = pair(audit["candidates"])
        system = EVALUATION_SYSTEM.format(query=feedback_query(messages), rejected_response=rejected["message"])
        loss, loss_event = call("generate", "loss", iteration, 0,
                                [{"role": "system", "content": system}, {"role": "user", "content": chosen["message"]}])
        prompt, conversation = gradient_messages(system, chosen["message"], loss["message"])
        grad, grad_event = call("generate", "gradient", iteration, 0, prompt)
        audit["rounds"].append({"iteration": iteration, "chosen_id": chosen["id"], "rejected_id": rejected["id"],
                                "loss_event": loss_event["index"], "gradient_event": grad_event["index"]})
        candidates("update", iteration, update_messages(chosen["message"], grad["message"], conversation))
    selected, _ = pair(audit["candidates"])
    audit["selected_id"] = selected["id"]
    generated = [e["result"] for e in audit["events"] if e["request"]["kind"] == "generate"]
    rewards = [e["result"] for e in audit["events"] if e["request"]["kind"] == "reward"]
    return {"message": selected["message"], "tpo": audit, "input_truncated": False,
            "latency_seconds": time.perf_counter() - started,
            "prompt_tokens": sum(r["prompt_tokens"] for r in generated),
            "original_prompt_tokens": sum(r["original_prompt_tokens"] for r in generated),
            "completion_tokens": sum(r["completion_tokens"] for r in generated),
            "reward_prompt_tokens": sum(r["prompt_tokens"] for r in rewards),
            "recorded_execution_seconds": sum(e["result"]["latency_seconds"] for e in audit["events"])}


def validate_audit(record, config):
    events = record["tpo"]["events"]
    cursor = 0

    def replay(request):
        nonlocal cursor
        if cursor >= len(events):
            raise ValueError("Missing TPO event")
        event = events[cursor]
        if event["index"] != cursor or event["request"] != request or event.get("error"):
            raise ValueError("TPO event mismatch")
        cursor += 1
        return event["result"]

    result = generate_tpo(record["prompt_history"], config, record["seed"], replay)
    if cursor != len(events) or result["tpo"] != record["tpo"]:
        raise ValueError("TPO audit mismatch")
    if result["message"] != record["generated_response"]:
        raise ValueError("TPO selected answer mismatch")
    if record.get("error") != ("empty_selected_answer" if not result["message"].strip() else None):
        raise ValueError("TPO turn outcome mismatch")
    for key in ("prompt_tokens", "original_prompt_tokens", "completion_tokens", "reward_prompt_tokens",
                "recorded_execution_seconds", "input_truncated"):
        if record[key] != result[key]:
            raise ValueError(f"TPO cost mismatch: {key}")
