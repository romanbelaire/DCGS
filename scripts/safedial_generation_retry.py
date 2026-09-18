"""Bounded, journaled retries of empty sampled generations; no model dependencies."""
import copy
import math
import re

from run_safedial_baseline import stable_id


def retry_policy(extra_attempts):
    if type(extra_attempts) is not int or extra_attempts not in (0, 2):
        raise ValueError("Empty generation retries must be 0 or 2")
    return {"extra_attempts": extra_attempts, "version": 1,
            "scope": "empty_sampled_generation_only; first_valid_result",
            "seed": "sha256(original_seed,empty-generation-retry,attempt); distinct_mod_2147483647",
            "journal": "every_attempt_fsync; reuse_saved_empty_attempts; include_all_costs"}


def retry_request(original, attempt):
    request = copy.deepcopy(original)
    if attempt:
        used = {original["seed"]}
        for index in range(1, attempt + 1):
            seed = int(stable_id(original["seed"], "empty-generation-retry", index, length=16), 16) % (2**31 - 1)
            while seed in used:
                seed = (seed + 1) % (2**31 - 1)
            used.add(seed)
        request.update(seed=seed, retry_attempt=attempt, original_seed=original["seed"])
    return request


def empty_generation(request, result):
    """Do not retry infrastructure errors, invalid accounting, or malformed output."""
    if (request.get("kind") != "generate" or request.get("temperature", 0) <= 0
            or not isinstance(result, dict) or result.get("error")
            or result.get("input_truncated") is not False):
        return False
    message = result.get("message")
    if not isinstance(message, str):
        return False
    if request.get("stage") == "response":
        # A tag-only response is not usable content either.
        message = re.sub(r"\[/?RESPONSE\]", "", message, flags=re.IGNORECASE)
    if message.strip():
        return False
    tokens = result.get("completion_tokens")
    prompt = result.get("prompt_tokens")
    latency = result.get("latency_seconds")
    return (type(tokens) is int and 0 < tokens <= request["max_new_tokens"]
            and type(prompt) is int and prompt > 0 and prompt == result.get("original_prompt_tokens")
            and type(latency) in (int, float) and math.isfinite(latency) and latency >= 0)


def reusable_empty_event(event):
    return (event.get("error") in ("ValueError: Empty DCGS generation", "ValueError: Generation is empty or failed")
            and empty_generation(event["request"], event.get("result")))


def execute_with_retries(original, execute, validate, events, on_event, extra_attempts,
                         failure_class, failure_prefix):
    """Each attempt is a separate durable event; attempt zero stays byte-compatible."""
    retry_policy(extra_attempts)
    for attempt in range(extra_attempts + 1):
        request = retry_request(original, attempt)
        event = {"index": len(events), "request": request}
        try:
            result = execute(copy.deepcopy(request))
            event["result"] = copy.deepcopy(result)
            warnings = validate(request, result)
            if warnings:
                event["format_warnings"] = warnings
        except (RuntimeError, ValueError, TypeError, KeyError) as exc:
            event["error"] = f"{type(exc).__name__}: {exc}"
            events.append(event)
            if on_event:
                on_event(event)
            if attempt < extra_attempts and reusable_empty_event(event):
                continue
            raise failure_class(f"{failure_prefix}: {exc}", {"events": events}) from exc
        events.append(event)
        if on_event:
            on_event(event)
        return result, event["index"]


def generation_diagnostics(tokenizer, tokens, max_new_tokens):
    ids = tokens.tolist()
    decoded = tokenizer.decode(tokens, skip_special_tokens=True)
    reason = ("eos" if ids and ids[-1] == tokenizer.eos_token_id else
              "max_new_tokens" if len(ids) == max_new_tokens else "other")
    return {"generated_token_ids": ids, "raw_decoded_text": decoded, "stop_reason": reason,
            "eos_token_id": tokenizer.eos_token_id}


def validate_diagnostics(request, result, strip_output=False):
    """Legacy calls have no diagnostics; newly recorded diagnostics must be coherent."""
    fields = {"generated_token_ids", "raw_decoded_text", "stop_reason", "eos_token_id"}
    if not fields.intersection(result):
        return
    if not fields.issubset(result):
        raise ValueError("Incomplete generation diagnostics")
    ids = result["generated_token_ids"]
    raw = result["raw_decoded_text"]
    if (not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids)
            or len(ids) != result.get("completion_tokens") or not isinstance(raw, str)
            or (raw.strip() if strip_output else raw) != result.get("message")):
        raise ValueError("Inconsistent generation diagnostics")
    reason = ("eos" if ids[-1] == result["eos_token_id"] else
              "max_new_tokens" if len(ids) == request["max_new_tokens"] else "other")
    if result["stop_reason"] != reason:
        raise ValueError("Inconsistent generation stop reason")
