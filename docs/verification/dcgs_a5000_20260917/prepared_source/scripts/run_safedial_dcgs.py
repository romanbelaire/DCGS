#!/usr/bin/env python3
"""Generate SafeDialBench answers with the DCGS hierarchical policy.

SafeDialBench uses counterfactual turn evaluation: every turn is conditioned on
the benchmark's gold assistant history, never on an earlier generated answer.
This runner preserves that protocol while applying DCGS independently at every
turn:

1. reconstruct the DCGS dialogue state from gold history;
2. generate high-level intent/instruction candidates;
3. score candidates with the checkpoint's Q and (optionally) regret heads;
4. sample a candidate from the DCGS policy; and
5. generate the low-level assistant response.

Generation only is performed here.  The resulting ``answers.jsonl`` remains
compatible with ``scripts/judge_safedial.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

# Running a file under ``scripts/`` puts that directory, but not necessarily the
# repository root, on sys.path.  Add the root so the local ``src`` package is
# importable regardless of how Python was launched.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_safedial_baseline import (
    DEFAULT_DATASET,
    TASKS,
    answer_has_error,
    append_jsonl,
    compact_answers,
    compact_turns,
    ensure_manifest,
    file_sha256,
    gold_messages,
    load_jsonl,
    load_latest_answers,
    select_dialogues,
    stable_id,
    turn_seed,
    validate_dataset,
)


DEFAULT_CONFIG = Path("safedial_dcgs.json")
DEFAULT_OUTPUT_DIR = Path("outputs/safedial_dcgs/zephyr_7b_beta_wildjailbreak_critic_full")
PROTOCOL = "safedial_gold_assistant_history_dcgs_v1"
MAX_BELIEF_ATTEMPTS = 3
LL_TEMPLATE = "action_generation_wildjailbreak"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the DCGS hierarchical policy on SafeDialBench."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, help="Override checkpoint_path from the config.")
    parser.add_argument("--model-id", default="zephyr-7b-beta-dcgs-wildjailbreak-critic")
    parser.add_argument("--num-choices", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--ids", help="Comma-separated SafeDialBench dialogue IDs.")
    selection.add_argument("--per-task", type=int, help="Select the first N IDs in each task.")
    selection.add_argument("--limit", type=int, help="Select the first N rows in dataset order.")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Regenerate dialogues whose latest answer contains an error.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate dataset, selection, config, and checkpoint without loading the model.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_choices < 1:
        raise ValueError("--num-choices must be positive")


def dcgs_gold_history(
    history: list[dict[str, str]], turn_index: int
) -> list[tuple[str, str]]:
    """Return DCGS (previous assistant, current user) pairs for one gold-history turn."""
    pairs: list[tuple[str, str]] = []
    for index in range(turn_index + 1):
        previous_assistant = "[NO_AGENT_ACTION]" if index == 0 else history[index - 1]["bot"]
        pairs.append((previous_assistant, history[index]["user"]))
    return pairs


def format_dcgs_observation(history: list[tuple[str, str]]) -> str:
    """Match EpisodeState.format_dialogue_history for critic state encoding."""
    parts: list[str] = []
    for index, (assistant, user) in enumerate(history, 1):
        assistant_text = assistant.strip() if assistant and assistant.strip() else "[NO_AGENT_ACTION]"
        user_text = user.strip() if user and user.strip() else "[NO_USER_RESPONSE]"
        parts.extend((f"Turn {index}:", f"Agent: {assistant_text}", f"User: {user_text}"))
    return "\n".join(parts)


def _valid_candidates(candidates: Iterable[Any]) -> list[Any]:
    return [candidate for candidate in candidates if candidate.summary != "[SKIP]"]


def _generate_candidate_side(
    *,
    hl_agent: Any,
    history: list[tuple[str, str]],
    config: Any,
    adversarial: bool,
) -> tuple[list[Any], str]:
    base_prompts = [history[-1][1] if adversarial else None]
    if config.use_hierarchical_agent and config.high_level_policy_type == "freeform":
        result = hl_agent.generate_instructions_batch(
            histories=[history],
            n_instructions=config.freeform_n_instructions,
            temperature=config.freeform_temperature,
            max_new_tokens=config.freeform_max_new_tokens,
            iterative_candidate_generation=config.freeform_iterative_candidate_generation,
            per_instruction_max_new_tokens=config.freeform_per_instruction_max_new_tokens,
            max_attempts_per_candidate=config.freeform_iterative_max_attempts_per_candidate,
            chunk_size=config.batch_generation_chunk_size,
            base_prompts=base_prompts,
        )
        return result["candidates"][0], result["raw_outputs"][0]

    candidates, raw_outputs = hl_agent.generate_candidate_beliefs_batch(
        histories=[history],
        n_candidates=config.n_candidates,
        temperature=config.belief_gen_temperature,
        max_new_tokens=config.max_tokens,
        return_debug_info=True,
        chunk_size=config.batch_generation_chunk_size,
        base_prompts=base_prompts,
    )
    return candidates[0], raw_outputs[0]


def generate_candidate_pool(
    *, hl_agent: Any, history: list[tuple[str, str]], config: Any
) -> tuple[list[Any], dict[str, str]]:
    """Generate the same nominal/adversarial pool used by regret-critic DCGS."""
    for attempt in range(1, MAX_BELIEF_ATTEMPTS + 1):
        nominal, nominal_raw = _generate_candidate_side(
            hl_agent=hl_agent, history=history, config=config, adversarial=False
        )
        if config.use_regret_critic:
            adversarial, adversarial_raw = _generate_candidate_side(
                hl_agent=hl_agent, history=history, config=config, adversarial=True
            )
            candidates = nominal + adversarial
        else:
            adversarial_raw = ""
            candidates = nominal
        if _valid_candidates(candidates):
            probability = 1.0 / len(candidates)
            for candidate in candidates:
                candidate.probability = probability
            return candidates, {
                "nominal": nominal_raw,
                "adversarial": adversarial_raw,
                "attempts": str(attempt),
            }
        print(
            f"[WARN] all belief candidates parsed as [SKIP]; retry {attempt}/{MAX_BELIEF_ATTEMPTS}",
            file=sys.stderr,
        )
    raise RuntimeError(
        f"all belief candidates parsed as [SKIP] after {MAX_BELIEF_ATTEMPTS} attempts"
    )


def score_candidates(
    *,
    candidates: list[Any],
    observation: str,
    value_function: Any,
    tokenizer: Any,
    config: Any,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    valid = _valid_candidates(candidates)
    if not valid:
        raise RuntimeError("no valid belief candidates to score")
    summaries = [candidate.summary for candidate in valid]
    observations = [observation] * len(summaries)
    q_values: dict[str, float] = {}
    regret_values: dict[str, float] = {}
    chunk_size = max(1, int(config.q_value_chunk_size))
    for start in range(0, len(summaries), chunk_size):
        chunk = summaries[start : start + chunk_size]
        chunk_observations = observations[start : start + chunk_size]
        q_tensor = value_function.predict_q_value(
            chunk_observations, chunk, tokenizer, requires_grad=False
        )
        for summary, score in zip(chunk, q_tensor.detach().cpu().tolist()):
            q_values[summary] = float(score)
        if config.use_regret_critic:
            regret_tensor = value_function.predict_regret_value(
                chunk_observations, chunk, tokenizer, requires_grad=False
            )
            for summary, score in zip(chunk, regret_tensor.detach().cpu().tolist()):
                regret_values[summary] = float(score)

    if config.use_regret_critic:
        beta = float(config.regret_critic_beta)
        policy_scores = {
            summary: (1.0 - beta) * q_values[summary] - beta * regret_values[summary]
            for summary in q_values
        }
    else:
        policy_scores = dict(q_values)
    return q_values, regret_values, policy_scores


def softmax_select(scores: dict[str, float], rng: random.Random) -> tuple[str, dict[str, float]]:
    if not scores:
        raise RuntimeError("cannot select from an empty score mapping")
    maximum = max(scores.values())
    weights = {key: math.exp(value - maximum) for key, value in scores.items()}
    total = sum(weights.values())
    probabilities = {key: value / total for key, value in weights.items()}
    selected = rng.choices(list(probabilities), weights=list(probabilities.values()), k=1)[0]
    return selected, probabilities


def generate_dcgs_turn(
    *,
    row: dict[str, Any],
    turn_index: int,
    seed: int,
    config: Any,
    hl_agent: Any,
    ll_agent: Any,
    value_function: Any,
    tokenizer: Any,
    torch_module: Any,
) -> dict[str, Any]:
    from transformers import set_seed

    set_seed(seed)
    history = dcgs_gold_history(row["history"], turn_index)
    observation = format_dcgs_observation(history)
    if config.device.startswith("cuda"):
        torch_module.cuda.synchronize()
    started = time.perf_counter()
    candidates, raw_outputs = generate_candidate_pool(
        hl_agent=hl_agent, history=history, config=config
    )
    q_values, regret_values, policy_scores = score_candidates(
        candidates=candidates,
        observation=observation,
        value_function=value_function,
        tokenizer=tokenizer,
        config=config,
    )
    selected_belief, probabilities = softmax_select(
        policy_scores, random.Random(seed ^ 0x5AFE_D1A1)
    )

    if config.ll_candidate_rerank and config.n_ll_candidates > 1:
        ll_candidates = ll_agent.generate_ll_candidates(
            belief_context=selected_belief,
            history=history,
            n_candidates=config.n_ll_candidates,
            template_name=LL_TEMPLATE,
            temperature=0.7,
            belief_only=config.ll_action_belief_only,
            chunk_size=config.batch_generation_chunk_size,
        )
        ll_scores = value_function.predict_ll_candidate_scores(ll_candidates, tokenizer)
        message, ll_probabilities = softmax_select(
            ll_scores, random.Random(seed ^ 0x11CA_D1A1)
        )
    else:
        ll_candidates = []
        ll_scores = {}
        ll_probabilities = {}
        message = ll_agent.generate_action(
            belief_context=selected_belief,
            history=history,
            temperature=0.7,
            belief_only=config.ll_action_belief_only,
            template_name=LL_TEMPLATE,
        )

    if config.device.startswith("cuda"):
        torch_module.cuda.synchronize()
    latency = time.perf_counter() - started
    return {
        "message": message.strip(),
        "completion_tokens": len(tokenizer.encode(message, add_special_tokens=False)),
        "latency_seconds": latency,
        "dcgs_history": history,
        "critic_observation": observation,
        "candidate_beliefs": [candidate.summary for candidate in candidates],
        "q_values": q_values,
        "regret_values": regret_values,
        "policy_scores": policy_scores,
        "selection_probabilities": probabilities,
        "selected_belief": selected_belief,
        "belief_generation_raw": raw_outputs,
        "ll_candidates": ll_candidates,
        "ll_candidate_scores": ll_scores,
        "ll_selection_probabilities": ll_probabilities,
    }


def load_config_and_checkpoint(args: argparse.Namespace) -> tuple[Any, Path, Path]:
    import torch

    from src.configs import BaseConfig

    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"DCGS config not found: {config_path}")
    config = BaseConfig.from_json(str(config_path))
    if args.checkpoint:
        config.checkpoint_path = str(args.checkpoint.expanduser())
    if not config.checkpoint_path:
        raise ValueError("DCGS config must define checkpoint_path (or pass --checkpoint)")
    checkpoint_path = Path(config.checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"DCGS checkpoint not found: {checkpoint_path}")
    if config.baseline_mode or config.random_belief_selection or config.raw_judge_belief_selection:
        raise ValueError("SafeDial DCGS requires critic-based high-level selection")
    if config.raw_judge_ll_selection:
        raise ValueError("raw_judge_ll_selection requires an environment judge and is unsupported here")
    if config.use_hierarchical_agent and config.freeform_n_instructions != config.n_candidates:
        raise ValueError("freeform_n_instructions must equal n_candidates")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_hidden_size = int(checkpoint.get("hidden_size", 0))
    if checkpoint_hidden_size <= 0:
        raise ValueError("DCGS checkpoint is missing a valid hidden_size")
    q_state = checkpoint.get("q_mlp_head_state_dict")
    if not isinstance(q_state, dict) or "0.weight" not in q_state:
        raise ValueError("DCGS checkpoint is missing q_mlp_head_state_dict['0.weight']")
    q_shape = tuple(q_state["0.weight"].shape)
    expected_mid = max(1, int(checkpoint_hidden_size * config.mlp_width_mult) // 2)
    expected_shape = (expected_mid, checkpoint_hidden_size)
    if q_shape != expected_shape:
        raise ValueError(
            "Checkpoint/config MLP mismatch: "
            f"q head has shape {q_shape}, expected {expected_shape} for "
            f"hidden_size={checkpoint_hidden_size}, mlp_width_mult={config.mlp_width_mult}"
        )
    checkpoint_uses_regret = bool(checkpoint.get("use_regret_critic", False))
    if checkpoint_uses_regret != bool(config.use_regret_critic):
        raise ValueError(
            "Checkpoint/config regret mismatch: "
            f"checkpoint use_regret_critic={checkpoint_uses_regret}, "
            f"config use_regret_critic={config.use_regret_critic}"
        )
    if config.use_regret_critic:
        missing = {
            "q_min_mlp_head_state_dict",
            "v_min_mlp_head_state_dict",
            "regret_mlp_head_state_dict",
        } - checkpoint.keys()
        if missing:
            raise ValueError(f"Regret checkpoint is missing heads: {sorted(missing)}")
    return config, config_path, checkpoint_path


def manifest_for(
    args: argparse.Namespace,
    dataset: Path,
    selected: list[dict[str, Any]],
    config: Any,
    config_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    return {
        "benchmark": "SafeDialBench",
        "protocol": PROTOCOL,
        "dataset": str(dataset.resolve()),
        "dataset_sha256": file_sha256(dataset),
        "selected_ids": [row["id"] for row in selected],
        "model": config.model_name,
        "model_id": args.model_id,
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "use_bf16": config.use_bf16,
        "n_candidates_per_side": config.n_candidates,
        "use_regret_critic": config.use_regret_critic,
        "regret_critic_beta": config.regret_critic_beta,
        "ll_action_belief_only": config.ll_action_belief_only,
        "ll_candidate_rerank": config.ll_candidate_rerank,
        "n_ll_candidates": config.n_ll_candidates,
        "num_choices": args.num_choices,
        "seed": args.seed,
        "generation_only": True,
    }


def initialize_dcgs(config: Any, checkpoint_path: Path) -> tuple[Any, Any, Any, Any, Any]:
    import torch

    from src.agents import FreeformHighLevelAgent, HighLevelAgent, LowLevelAgent
    from src.prompts.prompt_manager import PromptManager
    from src.utils.llm_utils import get_model_instance, get_tokenizer_instance
    from src.value import ValueFunction

    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Run this inside a Slurm GPU allocation.")
    device = config.device if config.device != "cuda" else "cuda:0"
    model = get_model_instance(config.model_name, device=device, use_bf16=config.use_bf16)
    tokenizer = get_tokenizer_instance(config.model_name)
    prompt_manager = PromptManager()
    agent_type = (
        FreeformHighLevelAgent
        if config.use_hierarchical_agent and config.high_level_policy_type == "freeform"
        else HighLevelAgent
    )
    hl_agent = agent_type(
        model,
        tokenizer,
        prompt_manager,
        template_name=config.belief_gen_template,
        enable_thinking=config.hl_enable_thinking,
    )
    ll_agent = LowLevelAgent(
        model, tokenizer, prompt_manager, enable_thinking=config.ll_enable_thinking
    )
    dtype = torch.bfloat16 if config.use_bf16 else torch.float32
    value_function = ValueFunction(
        model=model,
        hidden_size=model.config.hidden_size,
        learning_rate=config.learning_rate,
        device=device,
        dtype=dtype,
        use_regret_critic=config.use_regret_critic,
        mlp_width_mult=config.mlp_width_mult,
        critic_target_tau=config.critic_target_tau,
        critic_lora_r=config.critic_lora_r,
        critic_lora_alpha=config.critic_lora_alpha,
        critic_lora_layers=config.critic_lora_layers,
        critic_lora_lr=config.critic_lora_lr,
        normalize_td_targets=config.normalize_td_targets,
        reward_norm_momentum=config.reward_norm_momentum,
        reward_norm_clip=config.reward_norm_clip,
        use_behavior_snapshot=config.use_behavior_snapshot,
    )
    value_function.load_checkpoint(str(checkpoint_path), strict=True, load_optimizer=False)
    return torch, tokenizer, hl_agent, ll_agent, value_function


def main() -> int:
    args = parse_args()
    validate_args(args)
    dataset = args.dataset.expanduser().resolve()
    if not dataset.is_file():
        raise FileNotFoundError(f"SafeDialBench dataset not found: {dataset}")
    rows = load_jsonl(dataset)
    validate_dataset(rows, dataset)
    selected = select_dialogues(rows, args)
    config, config_path, checkpoint_path = load_config_and_checkpoint(args)
    task_counts = {task: sum(row["task"] == task for row in selected) for task in TASKS}
    print(f"Validated {len(rows)} SafeDialBench rows; selected {len(selected)} dialogues")
    print("Selection by task: " + ", ".join(f"{task}={count}" for task, count in task_counts.items()))
    print(f"DCGS config: {config_path}")
    print(f"DCGS checkpoint: {checkpoint_path}")
    if args.validate_only:
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_for(
        args, dataset, selected, config, config_path, checkpoint_path
    )
    ensure_manifest(args.output_dir / "run_config.json", manifest)
    answers_path = args.output_dir / "answers.jsonl"
    turns_path = args.output_dir / "turns.jsonl"
    completed = load_latest_answers(answers_path, args.model_id)
    pending = [
        row
        for row in selected
        if row["id"] not in completed
        or (args.retry_errors and answer_has_error(completed[row["id"]]))
    ]
    print(f"Already complete: {len(selected) - len(pending)}; pending: {len(pending)}")
    if not pending:
        compact_answers(answers_path, args.model_id)
        compact_turns(turns_path)
        return 0

    torch, tokenizer, hl_agent, ll_agent, value_function = initialize_dcgs(
        config, checkpoint_path
    )
    run_id = stable_id(
        manifest["dataset_sha256"], manifest["checkpoint_sha256"], args.model_id,
        args.seed, manifest["selected_ids"]
    )
    for dialogue_number, row in enumerate(pending, 1):
        answer_id = stable_id(run_id, row["id"])
        choices: list[dict[str, Any]] = []
        turn_records: list[dict[str, Any]] = []
        for choice_index in range(args.num_choices):
            choice_turns: list[dict[str, Any]] = []
            for turn_index, source_turn in enumerate(row["history"]):
                seed = turn_seed(args.seed, args.model_id, row["id"], choice_index, turn_index)
                error: str | None = None
                try:
                    result = generate_dcgs_turn(
                        row=row,
                        turn_index=turn_index,
                        seed=seed,
                        config=config,
                        hl_agent=hl_agent,
                        ll_agent=ll_agent,
                        value_function=value_function,
                        tokenizer=tokenizer,
                        torch_module=torch,
                    )
                except (RuntimeError, ValueError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    result = {
                        "message": "ERROR",
                        "completion_tokens": 0,
                        "latency_seconds": None,
                    }
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(
                        f"ERROR dialogue={row['id']} choice={choice_index} turn={turn_index}: {error}",
                        file=sys.stderr,
                    )
                official_turn: dict[str, Any] = {
                    "role": "assistant",
                    "message": result["message"],
                }
                if error:
                    official_turn["error"] = error
                choice_turns.append(official_turn)
                turn_records.append(
                    {
                        "run_id": run_id,
                        "answer_id": answer_id,
                        "benchmark": "SafeDialBench",
                        "protocol": PROTOCOL,
                        "model": config.model_name,
                        "model_id": args.model_id,
                        "dialogue_id": row["id"],
                        "task": row["task"],
                        "method": row["method"],
                        "scene": row["scene"],
                        "dataset_model_type": row.get("model_type"),
                        "choice_index": choice_index,
                        "turn_index": turn_index,
                        "seed": seed,
                        "prompt_history": gold_messages(row["history"], turn_index),
                        "user_message": source_turn["user"],
                        "generated_response": result["message"],
                        "reference_response": source_turn["bot"],
                        "completion_tokens": result.get("completion_tokens"),
                        "latency_seconds": result.get("latency_seconds"),
                        "error": error,
                        "dcgs": {
                            key: value
                            for key, value in result.items()
                            if key not in {"message", "completion_tokens", "latency_seconds"}
                        },
                    }
                )
            choices.append({"index": choice_index, "turns": choice_turns})

        answer = {
            "id": row["id"],
            "task": row["task"],
            "answer_id": answer_id,
            "model_id": args.model_id,
            "method": row["method"],
            "choices": choices,
            "tstamp": time.time(),
        }
        append_jsonl(turns_path, turn_records)
        append_jsonl(answers_path, [answer])
        print(
            f"[{dialogue_number}/{len(pending)}] completed dialogue {row['id']} "
            f"({len(row['history'])} turns)",
            flush=True,
        )

    compact_answers(answers_path, args.model_id)
    compact_turns(turns_path)
    print(f"Done. Judge-compatible answers: {answers_path}")
    print(f"Auditable DCGS turn records: {turns_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)
