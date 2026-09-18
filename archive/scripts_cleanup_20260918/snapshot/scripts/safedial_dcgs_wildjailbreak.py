"""SafeDial boundary adapter for the unchanged original WildJailbreak policy."""
import copy
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

REFERENCE = ROOT / "safedial_dcgs.json"


def configuration(method, device="cpu"):
    config = BaseConfig.from_json(str(REFERENCE))
    if method not in ("vdcgs", "rdcgs"):
        raise ValueError("Unknown method")
    # VDCGS retains the chosen shared Q checkpoint, without regret/pool expansion.
    config.use_regret_critic = method == "rdcgs"
    config.device = device
    if (config.environment_type != "wildjailbreak" or config.ll_candidate_rerank
            or not config.ll_action_belief_only or config.freeform_iterative_candidate_generation
            or config.epsilon != 0 or config.contrastive_ablation_mode != "none"):
        raise ValueError("The pinned WildJailbreak reference branch changed; review parity")
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


class PolicyFailure(RuntimeError):
    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = audit


def generate_turn(row, turn, seed, config, tokenizer, execute, on_event=None):
    """Run original orchestration; inject only generation/scoring for audit/replay."""
    import torch
    from transformers import set_seed
    set_seed(seed)  # Once per independent counterfactual benchmark turn.
    episode = episode_for(row, turn)
    manager = PromptManager(str(ROOT / "src/prompts/templates.jsonl"), str(ROOT / "src/prompts/personas.jsonl"))
    hl = FreeformHighLevelAgent(None, tokenizer, manager, template_name=config.belief_gen_template,
                               enable_thinking=config.hl_enable_thinking)
    ll = LowLevelAgent(None, tokenizer, manager, enable_thinking=config.ll_enable_thinking)
    audit = {"events": [], "history": episode.belief_state.history, "selected_belief": None}

    def call(request):
        event = {"index": len(audit["events"]), "request": copy.deepcopy(request)}
        try:
            result = execute(copy.deepcopy(request))
            event["result"] = copy.deepcopy(result)
            if request["kind"] == "generate":
                if not isinstance(result.get("texts"), list) or len(result["texts"]) != len(request["prompts"]):
                    raise ValueError("Original generation returned incorrect batch size")
                if any(not isinstance(text, str) for text in result["texts"]):
                    raise ValueError("Non-text generation result")
                # Do not insert a blank retry: original parsers/fallback decide.
            else:
                if len(result["scores"]) != len(request["observations"]) or any(not math.isfinite(x) for x in result["scores"]):
                    raise ValueError("Invalid original critic result")
        except (RuntimeError, ValueError, TypeError, KeyError) as exc:
            event["error"] = f"{type(exc).__name__}: {exc}"
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

    try:
        with patch.object(hl_module, "batch_generate", generation_boundary), patch.object(ll_module, "batch_generate", generation_boundary):
            original.batch_generate_beliefs_for_episodes([episode], hl, config)
            audit["belief_raw"] = episode.env.env_state.hl_agent_outputs
            audit["belief_candidates"] = [c.summary for c in episode.belief_state.candidates]
            if not episode.is_active or episode.done_from_env:
                raise ValueError("Original policy exhausted all-SKIP belief attempts")
            original.batch_compute_q_values_for_episodes([episode], CriticBoundary(), tokenizer, config)
            valid = [c.summary for c in episode.belief_state.candidates if c.summary != "[SKIP]"]
            scores = {k: v for k, v in episode._q_values.items() if k != "[SKIP]"}
            selected = original._select_high_level_belief(valid, scores, episode, config, config.epsilon)
            audit.update(q_values=episode._q_values, regret_values=getattr(episode, "_regret_values", {}),
                         q_min_values=getattr(episode, "_q_min_values", {}), selected_belief=selected)
            response = original._generate_or_select_ll_action(episode, selected, episode.belief_state.history,
                                                              ll, CriticBoundary(), tokenizer, config)
            if not response or not response.strip():
                raise ValueError("Original low-level policy returned an empty response; no custom retry")
    except (RuntimeError, ValueError, TypeError, KeyError) as exc:
        raise PolicyFailure(str(exc), audit) from exc
    return {"message": response, "dcgs_original": audit}


class LocalBackend:
    def __init__(self, config, lock):
        import torch
        local = copy.deepcopy(config)
        local.model_name = lock["actor"]["path"]
        self.torch, self.tokenizer, self.hl, self.ll, self.value = legacy.initialize_dcgs(local, Path(lock["high_level"]["path"]))
        self.model = self.hl.model
        self.config = config
        self.loading = {"actor_dtype": str(self.model.dtype), "device": str(self.model.device),
                        "source": "original initialize_dcgs and ValueFunction.load_checkpoint",
                        "ll_reranking": False, "regret_enabled": config.use_regret_critic,
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
        else:
            kwargs = {k: v for k, v in request.items() if k not in ("kind", "function")}
            scores = getattr(self.value, request["function"])(tokenizer=self.tokenizer, **kwargs)
            # Diagnostic only: preserve the original critic's 1500-token cutoff.
            lengths = [len(self.tokenizer(f"Observation: {obs}\nHigh-Level Context: {belief}")["input_ids"])
                       for obs, belief in zip(request["observations"], request["high_level_actions"])]
            result = {"scores": scores.detach().float().cpu().tolist(),
                      "untruncated_input_tokens": lengths, "critic_max_length": 1500,
                      "input_truncated": [n > 1500 for n in lengths]}
        synchronize_if_cuda(torch, self.config.device)
        return {**result, "latency_seconds": time.perf_counter() - started}


def validate_turn(record, row, config, tokenizer):
    events = record["dcgs_original"]["events"]
    cursor = 0
    def replay(request):
        nonlocal cursor
        if cursor >= len(events):
            raise ValueError("Missing original-policy event")
        event = events[cursor]
        if event["index"] != cursor or event["request"] != request or event.get("error"):
            raise ValueError("Original-policy event mismatch")
        cursor += 1
        return event["result"]
    rebuilt = generate_turn(row, record["turn_index"], record["seed"], config, tokenizer, replay)
    # History tuples become lists on JSON serialization.
    import json
    if cursor != len(events) or json.loads(json.dumps(rebuilt["dcgs_original"])) != record["dcgs_original"] or rebuilt["message"] != record["generated_response"]:
        raise ValueError("Original-policy selection/audit mismatch")
