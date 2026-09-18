"""Separate full-benchmark VDCGS and RDCGS policies. All model calls are injected and replayable."""
import copy
import math
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import run_safedial_baseline as base
from run_safedial_dcgs import format_dcgs_observation
from src.agents.high_level_agent import HighLevelAgent
from src.agents.low_level_agent import LowLevelAgent
from src.prompts.prompt_manager import PromptManager


def defense_config(method="rdcgs"):
    method = method.lower()
    if method not in ("vdcgs", "rdcgs"):
        raise ValueError("Expected vdcgs or rdcgs")
    return {
        "name": method.upper(), "implementation_version": 2,
        "hl_training_origin": "shared_Aug11_WildJailbreak_regret_training_checkpoint",
        "use_regret": method == "rdcgs",
        "intent_sides": ["nominal", "adversarial"] if method == "rdcgs" else ["nominal"],
        "intent_candidates_per_side": 5, "response_candidates": 5,
        "intent_tokens_per_side": 512, "response_tokens": 1024,
        "intent_temperature": 0.8, "intent_top_p": 0.9,
        "response_temperature": 0.7, "response_top_p": 1.0, "top_k": 0,
        "intent_selection": "seeded_softmax_over_0.8Q_minus_0.2regret" if method == "rdcgs" else "seeded_softmax_over_Q",
        "response_selection": "argmax_token_score_first_index_tie",
        "response_proposals": "five_independent_serial_samples_with_distinct_seeds",
        "duplicates": "retain_indexed_candidates_in_both_stages",
        "history": "gold_history_and_selected_intent_for_response_and_token_scorer",
        "generation_encoding": "repository_raw_templates_plus_separately_tokenized_prefill",
        "intent_encoding": "Observation: {observation}\nHigh-Level Context: {belief}; mean_all_tokens_serial_no_padding",
        "hl_context_limit": 32768, "ll_context_limit": 32768,
        "hl_context_adaptation": "full_benchmark_native_32768_without_truncation",
        "ll_context_adaptation": "8192_guard_extended_to_native_32768; unchanged_precision_offsets_and_scoring",
        "input_overflow": "error_without_truncation",
        "actor_and_hl_dtype": "bfloat16", "ll_encoder_dtype": "float16",
        "ll_head_dtype": "float32", "ll_objective": "shapley",
        "ll_score": "mean_action_tokens_of_0.5_times_harm_plus_follow",
        "model_updates": False, "judge_calls": 0,
    }


class Prompts:
    def __init__(self):
        self.manager = PromptManager(str(ROOT / "src/prompts/templates.jsonl"),
                                     str(ROOT / "src/prompts/personas.jsonl"))
        self.hl = HighLevelAgent(None, None, self.manager)
        self.ll = LowLevelAgent(None, None, self.manager)

    def intent(self, history, adversarial):
        return self.manager.get_high_level_prompt("belief_generation", history, 5,
                                                  history[-1][1] if adversarial else None)

    def response(self, history, belief):
        return self.manager.get_low_level_prompt("action_generation_wildjailbreak", belief,
                                                 history, belief_only=False)


def history_from_messages(messages):
    if not messages or len(messages) % 2 != 1:
        raise ValueError("Expected gold alternating history ending in user")
    for index, message in enumerate(messages):
        if message["role"] != ("user" if index % 2 == 0 else "assistant"):
            raise ValueError("Unexpected gold history roles")
    return [("[NO_AGENT_ACTION]" if i == 0 else messages[i - 1]["content"], messages[i]["content"])
            for i in range(0, len(messages), 2)]


class DCGSFailure(RuntimeError):
    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = audit


def validate_result(request, result):
    if result.get("error") or result.get("input_truncated") is not False:
        raise ValueError("Failed or truncated DCGS call")
    if (type(result.get("prompt_tokens")) is not int or result["prompt_tokens"] < 1
            or result["prompt_tokens"] != result.get("original_prompt_tokens")):
        raise ValueError("Invalid DCGS input accounting")
    if not math.isfinite(result.get("latency_seconds", float("nan"))) or result["latency_seconds"] < 0:
        raise ValueError("Invalid DCGS latency")
    if request["kind"] == "generate":
        if not isinstance(result.get("message"), str) or not result["message"].strip():
            raise ValueError("Empty DCGS generation")
        if type(result.get("completion_tokens")) is not int or not 0 < result["completion_tokens"] <= request["max_new_tokens"]:
            raise ValueError("Invalid DCGS generation length")
    else:
        keys = (("q", "regret") if request["use_regret"] else ("q",)) if request["kind"] == "hl_score" else ("score",)
        for key in keys:
            if type(result.get(key)) not in (float, int) or not math.isfinite(result[key]):
                raise ValueError(f"Nonfinite/missing DCGS {key}")
        if request["kind"] == "ll_score" and not (0 <= result.get("action_start", -1)
                < result.get("action_end", -1) <= result["prompt_tokens"]):
            raise ValueError("Invalid LL action-token span")


def generate_dcgs(messages, config, seed, execute, on_event=None):
    if config != defense_config(config["name"]):
        raise ValueError("Unknown two-stage DCGS policy")
    prompts = Prompts()
    history = history_from_messages(messages)
    observation = format_dcgs_observation(history)
    audit = {"events": [], "intents": [], "responses": [], "observation": observation,
             "selected_intent_id": None, "selected_response_id": None}

    def call(kind, stage, slot, **kwargs):
        request = {"kind": kind, "stage": stage, "slot": slot, **kwargs}
        if kind == "generate":
            request["seed"] = int(base.stable_id(seed, "dcgs", stage, slot, length=16), 16) % (2**31 - 1)
            request["top_k"] = config["top_k"]
        event = {"index": len(audit["events"]), "request": request}
        try:
            result = execute(copy.deepcopy(request))
            event["result"] = copy.deepcopy(result)
            validate_result(request, result)
        except (RuntimeError, ValueError, TypeError, KeyError) as exc:
            event["error"] = f"{type(exc).__name__}: {exc}"
            audit["events"].append(event)
            if on_event:
                on_event(event)
            raise DCGSFailure(f"{stage} slot={slot}: {exc}", audit) from exc
        audit["events"].append(event)
        if on_event:
            on_event(event)
        return result, event["index"]

    try:
        for side in config["intent_sides"]:
            adversarial = side == "adversarial"
            prefill = "Here are 5 candidate beliefs about the user's desires:\n\n1. "
            result, source_event = call("generate", side, 0, prompt=prompts.intent(history, adversarial),
                prefill=prefill, max_new_tokens=config["intent_tokens_per_side"],
                temperature=config["intent_temperature"], top_p=config["intent_top_p"])
            candidates = prompts.hl._parse_belief_candidates(prefill + result["message"], 5)
            for text in candidates:
                item = {"id": len(audit["intents"]), "side": side, "text": text,
                        "generation_event": source_event, "valid": text != "[SKIP]" and bool(text.strip())}
                if item["valid"]:
                    score, event_id = call("hl_score", "intent_score", item["id"], observation=observation, belief=text, use_regret=config["use_regret"])
                    item.update(q=score["q"], score=score["q"], score_event=event_id)
                    if config["use_regret"]:
                        item.update(regret=score["regret"], score=0.8 * score["q"] - 0.2 * score["regret"])
                audit["intents"].append(item)
        valid = [c for c in audit["intents"] if c["valid"]]
        if not valid:
            raise ValueError("No valid intent proposals; inspect saved raw generations")
        maximum = max(c["score"] for c in valid)
        weights = [math.exp(c["score"] - maximum) for c in valid]
        for item, weight in zip(valid, weights):
            item["probability"] = weight / sum(weights)
        chosen = random.Random(seed ^ 0x5AFE_D1A1).choices(valid, weights=weights, k=1)[0]
        audit["selected_intent_id"] = chosen["id"]
        for slot in range(config["response_candidates"]):
            result, source_event = call("generate", "response", slot, prompt=prompts.response(history, chosen["text"]),
                prefill="[RESPONSE]", max_new_tokens=config["response_tokens"],
                temperature=config["response_temperature"], top_p=config["response_top_p"])
            text = prompts.ll._parse_response_from_tags("[RESPONSE]" + result["message"])
            if not text.strip() or text == "ERROR":
                raise ValueError("Empty/failed parsed response candidate")
            score, event_id = call("ll_score", "response_score", slot, observation=observation, belief=chosen["text"], action=text)
            audit["responses"].append({"id": slot, "text": text, "score": score["score"],
                "generation_event": source_event, "score_event": event_id})
        winner = max(audit["responses"], key=lambda c: c["score"])
        audit["selected_response_id"] = winner["id"]
    except DCGSFailure:
        raise
    except (ValueError, RuntimeError, KeyError, TypeError) as exc:
        raise DCGSFailure(str(exc), audit) from exc
    generated = [e["result"] for e in audit["events"] if e["request"]["kind"] == "generate"]
    return {"message": winner["text"], "dcgs": audit, "input_truncated": False,
            "prompt_tokens": sum(r["prompt_tokens"] for r in generated),
            "original_prompt_tokens": sum(r["original_prompt_tokens"] for r in generated),
            "completion_tokens": sum(r["completion_tokens"] for r in generated),
            "critic_prompt_tokens": sum(e["result"]["prompt_tokens"] for e in audit["events"] if e["request"]["kind"] != "generate"),
            "latency_seconds": sum(e["result"]["latency_seconds"] for e in audit["events"])}


def validate_audit(record, config):
    events = record["dcgs"]["events"]
    cursor = 0
    def replay(request):
        nonlocal cursor
        if cursor >= len(events):
            raise ValueError("Missing DCGS event")
        event = events[cursor]
        if event["index"] != cursor or event["request"] != request or event.get("error"):
            raise ValueError("DCGS event request/index/error mismatch")
        cursor += 1
        return event["result"]
    result = generate_dcgs(record["prompt_history"], config, record["seed"], replay)
    if cursor != len(events) or result["dcgs"] != record["dcgs"] or result["message"] != record["generated_response"]:
        raise ValueError("DCGS selection/audit mismatch")
    for key in ("prompt_tokens", "original_prompt_tokens", "completion_tokens", "critic_prompt_tokens", "latency_seconds", "input_truncated"):
        if result[key] != record[key]:
            raise ValueError(f"DCGS cost audit mismatch: {key}")
