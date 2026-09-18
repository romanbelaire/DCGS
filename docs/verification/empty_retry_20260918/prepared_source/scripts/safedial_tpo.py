"""Native, auditable TPO with injected generation/reward execution.

Algorithm: Simplified-Reasoning/TPO@395c3d763a4c3df0ae72a4352b0db16fe5ee18e9.
TextGrad prompt constants are vendored with their MIT license. This adapter uses
native gold dialogue history and serial indexed candidates, without TextGrad's
string-keyed cache or silent dropping of malformed optimizer outputs.
"""

import copy
import math
import re
import time

from run_safedial_baseline import stable_id
from safedial_generation_retry import execute_with_retries, retry_policy, reusable_empty_event, validate_diagnostics
from tpo_vendor import llm_backward_prompts as bp
from tpo_vendor import optimizer_prompts as op

UPSTREAM_COMMIT = "395c3d763a4c3df0ae72a4352b0db16fe5ee18e9"
ROLE = "a chosen response to a user query"
OUTPUT_ROLE = "response from the language model"
START, END = "<IMPROVED_VARIABLE>", "</IMPROVED_VARIABLE>"
MISSING_END_WARNING = "missing_IMPROVED_VARIABLE_closing_tag; extracted_to_end_of_output"
TRAILING_START_WARNING = "opening_IMPROVED_VARIABLE_tag_after_closed_span; ignored_trailing_text"
TERMINAL_START_WARNING = "terminal_opening_tag_used_as_closing"
DUPLICATE_BLOCK_WARNING = "equivalent_duplicate_IMPROVED_VARIABLE_blocks; extracted_first_block"
FIRST_IMPROVEMENT_WARNING = "extra_or_misordered_IMPROVED_VARIABLE_tags; extracted_first_improvement_only"
ECHOED_FORMAT_INSTRUCTION = "Send ONLY the improved variable between the <IMPROVED_VARIABLE> tags, and nothing else."
EVALUATION_SYSTEM = """You are a language model tasked with evaluating a chosen response by comparing it with a rejected response to a user query. Analyze the strengths and weaknesses of each response, step by step, and explain why one is chosen or rejected.

**User Query**:
{query}

**Rejected Response**:
{rejected_response}

**Do NOT generate a response to the query. Be concise.** Below is the chosen response."""


def defense_config(sample_size=5, max_iters=2, response_tokens=1024, feedback_tokens=2048, empty_retries=2):
    for name, value, minimum in (("sample_size", sample_size, 2), ("max_iters", max_iters, 1),
                                 ("response_tokens", response_tokens, 1), ("feedback_tokens", feedback_tokens, 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    policy = retry_policy(empty_retries)
    config = {
        "name": "TPO", "implementation_version": 8 if empty_retries else 7,
        "upstream_repository": "https://github.com/Simplified-Reasoning/TPO",
        "upstream_commit": UPSTREAM_COMMIT,
        "sample_size": sample_size, "max_iterations": max_iters,
        "response_tokens": response_tokens, "feedback_tokens": feedback_tokens,
        "temperature": 0.7, "candidate_top_p": 0.95, "feedback_top_p": 0.99,
        "candidate_batch_size": 1, "selection": "max_reward_cumulative_cache_first_index_tie",
        "feedback_pair": "max_and_min_reward_cumulative_cache_first_index_tie",
        "history": "native_gold_messages_for_initial_and_reward; role_tagged_gold_history_in_feedback_query",
        "optimizer_extraction": "first_opening_tag_to_next_opening_or_closing_tag_or_end_of_output; strip_boundary_whitespace; require_nonempty_first_improvement; ignore_later_improvements_with_warning; preserve_v6_warnings_for_previously_accepted_outputs; raw_output_retained",
        "duplicates": "retain_indexed_candidates", "context_overflow": "error_without_truncation",
        "feedback_short_value": "upstream_first_and_last_10_space_separated_words",
        "model_updates": False, "judge_calls": 0,
        "adaptation": "Zephyr_HF_serial_inference; seeded_feedback; explicit_token_caps; native_multiturn_history",
    }

    if empty_retries:
        config["empty_generation_retry"] = policy
    return config


def short_value(value):
    words = value.split(" ")
    return value if len(words) <= 20 else " ".join(words[:10]) + " (...) " + " ".join(words[-10:])


def feedback_query(messages):
    if len(messages) == 1:
        return messages[0]["content"]
    return "\n\n".join(f"<{m['role']}>\n{m['content']}\n</{m['role']}>" for m in messages)


def parse_update(raw):
    """Use the first improvement, even when its closing delimiter is missing.

    An opening tag also terminates the first span: later answers must never be
    concatenated into the candidate. Keep historical warning labels so existing
    successful event journals still replay byte-for-byte under the new policy.
    """
    start = raw.find(START)
    if start < 0:
        raise ValueError("Optimizer output requires an opening IMPROVED_VARIABLE tag")
    body_start = start + len(START)
    boundaries = [pos for tag in (START, END)
                  if (pos := raw.find(tag, body_start)) >= 0]
    value = raw[body_start:min(boundaries) if boundaries else len(raw)].strip()
    if not value:
        raise ValueError("Optimizer returned an empty improved variable")
    try:
        historical_value, warnings = _parse_update_v6(raw)
    except ValueError:
        return value, [FIRST_IMPROVEMENT_WARNING]
    if historical_value != value:
        raise AssertionError("Historical extraction differs from the first improvement")
    return value, warnings


def _parse_update_v6(raw):
    """Frozen v6 classification, used only to preserve historical audit warnings.

    Rejection here does not reject a nonempty first improvement in parse_update.
    """
    closing_count = raw.count(END)
    stripped = raw.strip()
    if closing_count == 2:
        tags = list(re.finditer(re.escape(START) + "|" + re.escape(END), raw))
        sequence = [tag.group() for tag in tags]
        # The only extra tag accepted here is the quoted opener in the exact
        # known instruction after both complete blocks. Never ignore arbitrary
        # suffixes containing an opener or an unfinished third answer.
        echoed_instruction = (sequence == [START, END, START, END, START]
                              and raw[tags[3].end():].strip() == ECHOED_FORMAT_INSTRUCTION)
        if sequence == [START, END, START, END] or echoed_instruction:
            values = [raw[tags[i].end():tags[i + 1].start()].strip() for i in (0, 2)]
            # Comparison only: preserve the first block's actual content for
            # scoring. Remove at most one literal surrounding brace pair; do
            # not collapse internal whitespace, punctuation, or repeated braces.
            normalized = [value[1:-1].strip() if value.startswith("{") and value.endswith("}")
                          else value for value in values]
            if normalized[0] and normalized[0] == normalized[1]:
                warnings = [DUPLICATE_BLOCK_WARNING]
                if echoed_instruction:
                    warnings.append(TRAILING_START_WARNING)
                return values[0], warnings
    # Recover only a duplicated boundary delimiter, never an interior tag or
    # multiple unclosed answer blocks. Preserve the original raw response in the audit.
    if (not closing_count and stripped.count(START) == 2
            and stripped.startswith(START) and stripped.endswith(START)):
        value = stripped[len(START):-len(START)].strip()
        if not value:
            raise ValueError("Optimizer returned an empty improved variable")
        return value, [TERMINAL_START_WARNING]
    # A quoted opening tag in trailing commentary is outside the answer.
    # Check opening-tag uniqueness inside the extracted span, not the suffix.
    before_close = raw.split(END, 1)[0]
    if before_close.count(START) != 1 or closing_count > 1:
        raise ValueError("Optimizer output requires one opening tag and at most one ordered closing tag")
    value = raw.split(START, 1)[1].split(END, 1)[0].strip()
    if not value:
        raise ValueError("Optimizer returned an empty improved variable")
    warnings = [] if closing_count else [MISSING_END_WARNING]
    if closing_count and START in raw.split(END, 1)[1]:
        warnings.append(TRAILING_START_WARNING)
    return value, warnings


def extract_update(raw):
    return parse_update(raw)[0]


def pair(candidates):
    return (max(candidates, key=lambda c: c["reward"]), min(candidates, key=lambda c: c["reward"]))


def gradient_messages(system, chosen, loss):
    info = {"system_prompt": system, "prompt": chosen, "response_value": loss,
            "variable_desc": ROLE, "variable_short": short_value(chosen)}
    conversation = bp.CONVERSATION_TEMPLATE.format(**info)
    prompt = bp.CONVERSATION_START_INSTRUCTION_BASE.format(conversation=conversation, **info)
    prompt += bp.OBJECTIVE_INSTRUCTION_BASE + bp.EVALUATE_VARIABLE_INSTRUCTION.format(**info)
    return [{"role": "system", "content": bp.BACKWARD_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}], conversation


def update_messages(chosen, gradient, conversation):
    context = op.GRADIENT_TEMPLATE.format(context=conversation, response_desc=OUTPUT_ROLE,
                                         variable_desc=ROLE, feedback=gradient)
    kwargs = {"variable_desc": ROLE, "variable_short": short_value(chosen), "variable_grad": context,
              "constraint_text": "Constraint 1: Only generate a chosen response.\nConstraint 2: Do NOT generate a rejected response.",
              "new_variable_start_tag": START, "new_variable_end_tag": END}
    return [{"role": "system", "content": op.OPTIMIZER_SYSTEM_PROMPT.format(**kwargs)},
            {"role": "user", "content": op.construct_tgd_prompt(do_constrained=True, **kwargs)}]


class TPOFailure(RuntimeError):
    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = audit


def valid_result(request, result):
    if result.get("error") or result.get("input_truncated"):
        raise ValueError("Failed or truncated TPO execution")
    for field in ("prompt_tokens", "original_prompt_tokens"):
        if type(result.get(field)) is not int or result[field] < 1:
            raise ValueError(f"Missing/invalid {field}")
    if result["prompt_tokens"] != result["original_prompt_tokens"]:
        raise ValueError("Input token accounting indicates truncation")
    if not math.isfinite(result.get("latency_seconds", float("nan"))) or result["latency_seconds"] < 0:
        raise ValueError("Missing/invalid execution latency")
    if request["kind"] == "reward":
        if type(result.get("score")) not in (int, float) or not math.isfinite(result["score"]):
            raise ValueError("Reward must be a finite scalar")
    else:
        validate_diagnostics(request, result, strip_output=True)
        if not isinstance(result.get("message"), str) or not result["message"].strip() or result["message"] == "ERROR":
            raise ValueError("Generation is empty or failed")
        if type(result.get("completion_tokens")) is not int or not 0 < result["completion_tokens"] <= request["max_new_tokens"]:
            raise ValueError("Invalid generated-token count")
        if request["stage"] == "update":
            return parse_update(result["message"])[1]
    return []


def generate_tpo(messages, config, seed, execute, on_event=None):
    """execute(request) returns decoded generation or scalar reward plus costs.

    on_event receives every validated success/failure, enabling durable event
    checkpoints. No reference answer, dataset labels, or judge enters execute.
    """
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("TPO requires a final user message")
    retries = config.get("empty_generation_retry", {}).get("extra_attempts", 0)
    if config != defense_config(config["sample_size"], config["max_iterations"],
                                config["response_tokens"], config["feedback_tokens"], retries):
        raise ValueError("Unknown TPO configuration")
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
        try:
            return execute_with_retries(request, execute, valid_result, audit["events"],
                                        on_event, retries, TPOFailure,
                                        f"TPO {stage} round={iteration} slot={slot}")
        except TPOFailure as exc:
            exc.audit = audit
            raise

    def candidates(stage, iteration, prompt):
        for slot in range(config["sample_size"]):
            result, generation_event = call("generate", stage, iteration, slot, prompt)
            text = extract_update(result["message"]) if stage == "update" else result["message"]
            reward_prompt = [dict(m) for m in messages] + [{"role": "assistant", "content": text}]
            scored, reward_event = call("reward", "score", iteration, slot, reward_prompt)
            audit["candidates"].append({"id": len(audit["candidates"]), "iteration": iteration,
                                        "slot": slot, "message": text, "reward": scored["score"],
                                        "generation_event": generation_event, "reward_event": reward_event})

    candidates("initial", -1, messages)
    for iteration in range(config["max_iterations"]):
        chosen, rejected = pair(audit["candidates"])
        system = EVALUATION_SYSTEM.format(query=feedback_query(messages), rejected_response=rejected["message"])
        loss, loss_event = call("generate", "loss", iteration, 0,
                                [{"role": "system", "content": system}, {"role": "user", "content": chosen["message"]}])
        grad_prompt, conversation = gradient_messages(system, chosen["message"], loss["message"])
        grad, gradient_event = call("generate", "gradient", iteration, 0, grad_prompt)
        audit["rounds"].append({"iteration": iteration, "chosen_id": chosen["id"], "rejected_id": rejected["id"],
                                "loss_event": loss_event, "gradient_event": gradient_event})
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
    """Replay saved calls without models; verify all requests, rewards and choices."""
    events = record["tpo"]["events"]
    cursor = 0

    def replay(request):
        nonlocal cursor
        if cursor >= len(events):
            raise ValueError("Missing TPO audit event")
        event = events[cursor]
        if event["index"] != cursor or event["request"] != request or (event.get("error") and not reusable_empty_event(event)):
            raise ValueError("TPO audit request/index/error mismatch")
        cursor += 1
        return event["result"]

    try:
        rebuilt = generate_tpo(record["prompt_history"], config, record["seed"], replay)
    except TPOFailure as exc:
        raise ValueError(f"Invalid TPO audit: {exc}") from exc
    if cursor != len(events) or rebuilt["tpo"] != record["tpo"]:
        raise ValueError("TPO candidate/round/selection audit mismatch")
    if rebuilt["message"] != record["generated_response"]:
        raise ValueError("TPO selected answer mismatch")
    for key in ("prompt_tokens", "original_prompt_tokens", "completion_tokens", "reward_prompt_tokens",
                "recorded_execution_seconds", "input_truncated"):
        if rebuilt[key] != record[key]:
            raise ValueError(f"TPO all-call accounting mismatch: {key}")
