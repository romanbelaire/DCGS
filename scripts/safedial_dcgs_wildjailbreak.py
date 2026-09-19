"""SafeDial boundary adapter for main DCGS with trained LL token-critic reranking."""
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None
import src.main as original
from src.agents import FreeformHighLevelAgent, LowLevelAgent
import src.agents.high_level_agent as hl_module
import src.agents.low_level_agent as ll_module
from src.belief import BeliefState
from src.configs import BaseConfig
from src.prompts.prompt_manager import PromptManager
from src.training.episode_state import EpisodeState
import run_safedial_dcgs as legacy
from src.value.value_function import HL_CRITIC_MAX_SEQ_LENGTH, hl_q_text
from src.value.ll_token_critic import (LLContextLengthError, LLTokenCritic, MAX_UNTRUNCATED_LEN,
                                       OBJECTIVES, PREFIX_TEMPLATE, tokenize_action_span)

METHOD_SPEC = {
    "upstream_commit": "576ae184b8f49586456a136f8e14a93039cfeb88",
    "critic_pooling": "last_nonpad",
    "regret_min_target_mode": "min_q_over_states",
    "ll_selection": "trained_token_critic_argmax",
    "ll_candidate_generation": "upstream_numbered_list_with_upstream_fallback",
    "ll_candidates": 5,
    "ll_critic_max_input_tokens": MAX_UNTRUNCATED_LEN,
    "ll_context_overflow": "terminal turn failure; no truncation or extra retry",
    "critic_training": "frozen indexed checkpoint; no training performed by this runner",
}

REFERENCE = ROOT / "safedial_dcgs.json"


def configuration(method, device="cpu", artifacts=None):
    config = BaseConfig.from_json(str(REFERENCE))
    if method not in ("vdcgs", "rdcgs"):
        raise ValueError("Unknown method")
    config.use_regret_critic = method == "rdcgs"
    config.device = device
    if artifacts is not None:
        config.checkpoint_path = artifacts["high_level"]["path"]
        config.ll_token_critic_path = artifacts["token_critic"]["path"]
    for field in ("checkpoint_path", "ll_token_critic_path"):
        path = Path(getattr(config, field))
        setattr(config, field, str(path if path.is_absolute() else ROOT / path))
    if (config.environment_type != "wildjailbreak" or not config.ll_candidate_rerank
            or config.n_ll_candidates != 5 or not config.ll_token_critic_path
            or not config.ll_action_belief_only or config.freeform_iterative_candidate_generation
            or config.epsilon != 0 or config.contrastive_ablation_mode != "none"
            or config.regret_min_target_mode != "min_q_over_states"
            or config.hl_greedy_q or config.static_belief_mode != "none"
            or config.ground_truth_belief_selection or config.raw_judge_ll_selection
            or config.raw_judge_belief_selection or config.random_belief_selection
            or config.use_gpt_for_agents or config.paper_dsr_reward or config.baseline_mode
            or not config.use_hierarchical_agent or config.high_level_policy_type != "freeform"
            or config.defender_backend != "standard"):
        raise ValueError("The main WildJailbreak method configuration changed; review parity")
    return config

def episode_for(row, turn):
    history = legacy.dcgs_gold_history(row["history"], turn)
    env = SimpleNamespace(env_state=SimpleNamespace(hl_agent_inputs=[], hl_agent_outputs=[]))
    episode = EpisodeState(dialogue_idx=row["id"], dialogue_id=str(row["id"]),
        # SafeDial has no WJB base-prompt field. Its current user text is the
        # adversarial-template anchor; no benchmark label/reference answer leaks.
        dialogue_data={"base_prompt": history[-1][1], "harmful_level": 0},
        ground_truth_goal="", env=env, initial_observation=history[-1][1], initial_env_info={},
        belief_state=BeliefState(history=history), turn=turn)
    episode.update_current_observation_context()
    return episode


def critic_diagnostics(request, tokenizer):
    """Measure exactly the original critic encoding without changing its inputs."""
    texts = [hl_q_text(obs, belief) for obs, belief in
             zip(request["observations"], request["high_level_actions"])]
    raw = tokenizer(texts, add_special_tokens=True, padding=False, truncation=False)["input_ids"]
    kept = tokenizer(texts, add_special_tokens=True, padding=False, truncation=True,
                     max_length=HL_CRITIC_MAX_SEQ_LENGTH)["input_ids"]
    before, after = [len(x) for x in raw], [len(x) for x in kept]
    return {"pooling": METHOD_SPEC["critic_pooling"], "max_length": HL_CRITIC_MAX_SEQ_LENGTH, "truncation_side": tokenizer.truncation_side,
            "input_tokens": before, "retained_tokens": after,
            "dropped_tokens": [a - b for a, b in zip(before, after)],
            "input_truncated": [a > b for a, b in zip(before, after)],
            "encoded_sha256": [hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in kept]}


def ll_critic_diagnostics(request, tokenizer):
    prefix = PREFIX_TEMPLATE.format(observation=request["observation"],
                                    selected_belief=request["selected_belief"])
    lengths, action_tokens, hashes = [], [], []
    for action in request["actions"]:
        encoded, start, end = tokenize_action_span(tokenizer, prefix, prefix + action)
        ids = encoded["input_ids"][0].tolist()
        lengths.append(len(ids))
        action_tokens.append(end - start)
        hashes.append(hashlib.sha256(json.dumps(ids).encode()).hexdigest())
    return {"input_tokens": lengths, "action_tokens": action_tokens,
            "encoded_sha256": hashes, "truncated": False}


def exception_details(exc):
    if isinstance(exc, RecordedBackendError):
        return exc.details
    if isinstance(exc, LLContextLengthError):
        return {"category": "terminal_output", "code": "ll_context_overflow",
                "exception_type": type(exc).__name__, "message": str(exc),
                "input_tokens": exc.input_tokens, "max_tokens": exc.max_tokens}
    infrastructure = isinstance(exc, (OSError, TimeoutError)) or (
        isinstance(exc, RuntimeError) and any(word in str(exc).lower()
        for word in ("cuda", "cublas", "cudnn", "out of memory")))
    return {"category": "infrastructure" if infrastructure else "integrity",
            "code": "backend_exception", "exception_type": type(exc).__name__,
            "message": str(exc)}


class RecordedBackendError(Exception):
    def __init__(self, details):
        self.details = details
        super().__init__(details["message"])


class TraceIncomplete(BaseException):
    """Replay reached the first uncompleted call; never a model/policy failure."""


class TerminalOutputError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


class PolicyFailure(RuntimeError):
    def __init__(self, message, audit, details=None, stage="policy"):
        super().__init__(message)
        self.audit = audit
        self.details = details or {"category": "integrity", "code": "policy_exception",
                                   "exception_type": "RuntimeError", "message": message}
        self.stage = stage


def generate_turn(row, turn, seed, config, tokenizer, execute, on_event=None):
    """Run original orchestration; inject only generation/scoring for audit/replay."""
    import torch
    from transformers import set_seed
    set_seed(seed)  # Once per independent counterfactual benchmark turn.
    config = copy.copy(config)  # Runtime critic boundary must not enter the saved dataclass config.
    episode = episode_for(row, turn)
    manager = PromptManager(str(ROOT / "src/prompts/templates.jsonl"), str(ROOT / "src/prompts/personas.jsonl"))
    hl = FreeformHighLevelAgent(None, tokenizer, manager, template_name=config.belief_gen_template,
                               enable_thinking=config.hl_enable_thinking)
    ll = LowLevelAgent(None, tokenizer, manager, enable_thinking=config.ll_enable_thinking)
    audit = {"events": [], "history": episode.belief_state.history, "selected_belief": None}
    stage = "belief_generation"

    def call(request):
        event = {"index": len(audit["events"]), "request": copy.deepcopy(request)}
        try:
            result = execute(copy.deepcopy(request))
            if request["kind"] == "generate":
                if not isinstance(result.get("texts"), list) or len(result["texts"]) != len(request["prompts"]):
                    raise ValueError("Original generation returned incorrect batch size")
                if any(not isinstance(text, str) for text in result["texts"]):
                    raise ValueError("Non-text generation result")
                # Do not insert a blank retry: original parsers/fallback decide.
            elif request["kind"] == "score_ll":
                scores = result.get("scores")
                if (not isinstance(scores, dict) or set(scores) != set(request["actions"])
                        or any(not math.isfinite(x) for x in scores.values())
                        or result.get("objective") not in OBJECTIVES):
                    raise ValueError("Invalid trained LL token-critic result")
                diagnostics = ll_critic_diagnostics(request, tokenizer)
                if "ll_critic_context" in result and result["ll_critic_context"] != diagnostics:
                    raise ValueError("LL critic context diagnostics mismatch")
                result = {**result, "ll_critic_context": diagnostics}
            else:
                if len(result["scores"]) != len(request["observations"]) or any(not math.isfinite(x) for x in result["scores"]):
                    raise ValueError("Invalid original critic result")
                diagnostics = critic_diagnostics(request, tokenizer)
                if "critic_context" in result and result["critic_context"] != diagnostics:
                    raise ValueError("Critic context diagnostics mismatch")
                result = {**result, "critic_context": diagnostics}
            event["result"] = copy.deepcopy(result)
        except Exception as exc:
            details = exception_details(exc)
            event["error"] = f"{details['exception_type']}: {details['message']}"
            event["failure_details"] = details
            audit["events"].append(event)
            if on_event:
                on_event(event)
            raise
        audit["events"].append(event)
        if on_event:
            on_event(event)
        return result

    def generation_boundary(**kwargs):
        request = {k: copy.deepcopy(v) for k, v in kwargs.items() if k not in ("model", "tokenizer", "logits_processor")}
        request.update(kind="generate", suppressed_token_ids=sorted(kwargs["logits_processor"].suppress_token_ids))
        return call(request)["texts"]

    class CriticBoundary:
        use_regret_critic = config.use_regret_critic
        device = "cpu"  # Scalar result assembly only; real scoring stays on its backend device.

        def __getattr__(self, name):
            if name not in ("predict_q_value", "predict_q_min_value", "predict_regret_value"):
                raise AttributeError(name)
            def score(**kwargs):
                request = {k: copy.deepcopy(v) for k, v in kwargs.items() if k != "tokenizer"}
                request.update(kind="score", function=name)
                return torch.tensor(call(request)["scores"])
            return score

    class LLCriticBoundary:
        objective = "recorded"

        def score_actions(self, observation, selected_belief, actions):
            if any(not action.strip() for action in actions):
                raise TerminalOutputError("blank_ll_candidate", "Upstream LL pool/fallback produced an empty candidate")
            request = {"kind": "score_ll", "observation": observation,
                       "selected_belief": selected_belief, "actions": list(actions)}
            result = call(request)
            self.objective = result["objective"]
            # Preserve main's candidate order and string-keyed duplicate handling.
            return {action: result["scores"][action] for action in actions}

    config._ll_token_critic = LLCriticBoundary()
    try:
        with patch.object(hl_module, "batch_generate", generation_boundary), patch.object(ll_module, "batch_generate", generation_boundary):
            original.batch_generate_beliefs_for_episodes([episode], hl, config)
            audit["belief_raw"] = episode.env.env_state.hl_agent_outputs
            audit["belief_candidates"] = [c.summary for c in episode.belief_state.candidates]
            if not episode.is_active or episode.done_from_env:
                raise TerminalOutputError("beliefs_exhausted", "Original policy exhausted all-SKIP belief attempts")
            stage = "critic_scoring"
            original.batch_compute_q_values_for_episodes([episode], CriticBoundary(), tokenizer, config)
            valid = [c.summary for c in episode.belief_state.candidates if c.summary != "[SKIP]"]
            scores = {k: v for k, v in episode._q_values.items() if k != "[SKIP]"}
            selected = original._select_high_level_belief(valid, scores, episode, config, config.epsilon)
            audit.update(q_values=episode._q_values, regret_values=getattr(episode, "_regret_values", {}),
                         q_min_values=getattr(episode, "_q_min_values", {}), selected_belief=selected)
            stage = "final_response"
            response = original._generate_or_select_ll_action(episode, selected, episode.belief_state.history,
                                                              ll, CriticBoundary(), tokenizer, config)
            if not response or not response.strip():
                raise TerminalOutputError("blank_final_response", "Original low-level policy returned an empty response; no custom retry")
    except Exception as exc:
        details = exception_details(exc)
        if isinstance(exc, TerminalOutputError):
            details.update(category="terminal_output", code=exc.code)
        raise PolicyFailure(str(exc), audit, details, stage) from exc
    return {"message": response, "dcgs_original": audit}


class LocalBackend:
    def __init__(self, config, lock):
        import torch
        local = copy.deepcopy(config)
        local.model_name = lock["actor"]["path"]
        self.torch, self.tokenizer, self.hl, self.ll, self.value = legacy.initialize_dcgs(local, Path(lock["high_level"]["path"]))
        self.model = self.hl.model
        self.config = config
        self.token_critic = LLTokenCritic.from_checkpoint(
            lock["token_critic"]["path"], device=config.device, backbone=self.model,
            tokenizer=copy.deepcopy(self.tokenizer))
        self.token_critic.harm_head.requires_grad_(False)
        self.token_critic.follow_head.requires_grad_(False)
        self.loading = {"actor_dtype": str(self.model.dtype), "device": str(self.model.device),
                        "source": "main ValueFunction and LLTokenCritic with shared pinned backbone",
                        "method_spec": METHOD_SPEC,
                        "token_critic_sha256": lock["token_critic"]["sha256"],
                        "token_critic_objective": self.token_critic.objective,
                        "critic_mlp_dims": self.value._resolved_hidden_dims(),
                        "ll_reranking": True, "regret_enabled": config.use_regret_critic,
                        "checkpoint_sha256": lock["high_level"]["sha256"]}
        checkpoint = torch.load(lock["high_level"]["path"], map_location="cpu", weights_only=True)
        names = ["q", "v"] + (["q_min", "v_min", "regret"] if config.use_regret_critic else [])
        for name in names:
            head = getattr(self.value, name + "_mlp_head")
            expected = checkpoint[name + "_mlp_head_state_dict"]
            if any(not torch.equal(value.detach().cpu(), expected[key].to(dtype=value.dtype)) for key, value in head.state_dict().items()):
                raise ValueError("Original critic checkpoint did not load exactly")
            head.eval().requires_grad_(False)
        self.loading["strict_loaded_heads"] = names

    def __call__(self, request):
        import time
        from src.utils.llm_utils import batch_generate, SuppressWordsLogitsProcessor
        from run_safedial_baseline import synchronize_if_cuda
        torch = self.torch
        synchronize_if_cuda(torch, self.config.device)
        started = time.perf_counter()
        if request["kind"] == "generate":
            kwargs = {k: v for k, v in request.items() if k not in ("kind", "suppressed_token_ids")}
            processor = SuppressWordsLogitsProcessor(self.tokenizer, ["Example", "example", "Examples", "examples"])
            if sorted(processor.suppress_token_ids) != request["suppressed_token_ids"]:
                raise ValueError("Original suppression mask changed")
            raw = []
            original_generate = self.model.generate
            def recorded_generate(**generation_kwargs):
                length = generation_kwargs["input_ids"].shape[1]
                if length + generation_kwargs["max_new_tokens"] > self.model.config.max_position_embeddings:
                    raise ValueError("Original generation exceeds actor context capacity")
                output = original_generate(**generation_kwargs)
                raw.append({"input_ids": generation_kwargs["input_ids"].tolist(),
                            "generated_token_ids": output[:, length:].tolist()})
                return output
            with patch.object(self.model, "generate", recorded_generate):
                texts = batch_generate(model=self.model, tokenizer=self.tokenizer, logits_processor=processor, **kwargs)
            result = {"texts": texts, "raw_generations": raw}
        elif request["kind"] == "score_ll":
            result = {"scores": self.token_critic.score_actions(request["observation"],
                       request["selected_belief"], request["actions"]),
                      "objective": self.token_critic.objective}
        else:
            kwargs = {k: v for k, v in request.items() if k not in ("kind", "function")}
            scores = getattr(self.value, request["function"])(tokenizer=self.tokenizer, **kwargs)
            result = {"scores": scores.detach().float().cpu().tolist()}
        synchronize_if_cuda(torch, self.config.device)
        return {**result, "latency_seconds": time.perf_counter() - started}


def replay_trace(row, turn, seed, config, tokenizer, events, allow_incomplete=False):
    cursor = 0
    def replay(request):
        nonlocal cursor
        if cursor >= len(events):
            if allow_incomplete:
                raise TraceIncomplete()
            raise ValueError("Missing original-policy event")
        event = events[cursor]
        if event["index"] != cursor or event["request"] != request:
            raise ValueError("Original-policy event mismatch")
        cursor += 1
        if event.get("error"):
            if event["failure_details"].get("code") == "ll_context_overflow":
                if request["kind"] != "score_ll":
                    raise ValueError("LL context overflow recorded outside LL scoring")
                try:
                    ll_critic_diagnostics(request, tokenizer)
                except LLContextLengthError as exc:
                    if exception_details(exc) != event["failure_details"]:
                        raise ValueError("LL context overflow metadata mismatch") from exc
                else:
                    raise ValueError("Recorded LL context overflow fits within the critic limit")
            raise RecordedBackendError(event["failure_details"])
        if request["kind"] == "score" and event["result"].get("critic_context") != critic_diagnostics(request, tokenizer):
            raise ValueError("Missing or altered critic context diagnostics")
        if request["kind"] == "score_ll" and event["result"].get("ll_critic_context") != ll_critic_diagnostics(request, tokenizer):
            raise ValueError("Missing or altered LL critic context diagnostics")
        return event["result"]
    try:
        rebuilt = generate_turn(row, turn, seed, config, tokenizer, replay)
    except PolicyFailure:
        if cursor != len(events):
            raise ValueError("Unused events after recorded failure")
        raise
    if cursor != len(events):
        raise ValueError("Unused original-policy events")
    return rebuilt


def validate_turn(record, row, config, tokenizer):
    rebuilt = replay_trace(row, record["turn_index"], record["seed"], config, tokenizer,
                           record["dcgs_original"]["events"])
    if json.loads(json.dumps(rebuilt["dcgs_original"])) != record["dcgs_original"] or rebuilt["message"] != record["generated_response"]:
        raise ValueError("Original-policy selection/audit mismatch")
