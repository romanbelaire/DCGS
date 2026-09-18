"""Compare multiple value-function checkpoints on a shared offline slice."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Sequence

import torch

from ..agents import HighLevelAgent
from ..configs import BaseConfig
from ..data import format_multiwoz_goal, load_multiwoz_dataset
from ..environments import MultiWOZEnvironment
from ..prompts.prompt_manager import PromptManager
from ..utils.llm_utils import get_model_instance, get_tokenizer_instance
from ..value import ValueFunction
from ..data import compute_belief_accuracy


def _format_history(history: Sequence[Sequence[str]]) -> str:
    lines: List[str] = []
    for idx, (agent_action, user_obs) in enumerate(history, start=1):
        agent_text = agent_action.strip() if agent_action and agent_action.strip() else "[NO_AGENT_ACTION]"
        user_text = user_obs.strip() if user_obs and user_obs.strip() else "[NO_USER_RESPONSE]"
        lines.append(f"Turn {idx}:")
        lines.append(f"Agent: {agent_text}")
        lines.append(f"User: {user_text}")
    return "\n".join(lines)


def _collect_eval_dataset(
    dialogues,
    sample_size: int,
    max_turns: int,
    hl_agent: HighLevelAgent,
    config: BaseConfig,
    rng: random.Random,
) -> List[Dict]:
    examples: List[Dict] = []
    prompt_manager = hl_agent.prompt_manager
    
    for dialogue in dialogues:
        if len(examples) >= sample_size:
            break
        env = MultiWOZEnvironment(dialogue, debug=False)
        observation, _ = env.reset()
        if not observation:
            continue
        history: List[List[str]] = [["", observation]]
        ground_truth_goal = format_multiwoz_goal(dialogue.goal)
        
        for turn_idx in range(max_turns):
            formatted_history = [(agent, user) for agent, user in history]
            candidates = hl_agent.generate_candidate_beliefs(
                history=formatted_history,
                n_candidates=config.n_candidates,
                temperature=config.belief_gen_temperature,
                max_new_tokens=config.max_tokens
            )
            candidate_texts = [c.summary for c in candidates if c.summary and c.summary != "[SKIP]"]
            if candidate_texts:
                examples.append(
                    {
                        "dialogue_id": dialogue.dialogue_id,
                        "turn": turn_idx,
                        "observation": _format_history(formatted_history),
                        "candidates": candidate_texts,
                        "ground_truth": ground_truth_goal,
                    }
                )
            if len(examples) >= sample_size:
                break
            
            step_result = env.step()
            dataset_action = step_result.agent_action or ""
            next_observation = step_result.observation
            done = step_result.done
            if not next_observation or done:
                break
            history.append([dataset_action or "", next_observation])
    
    if not examples:
        raise ValueError("No evaluation datapoints collected for checkpoint ablation.")
    rng.shuffle(examples)
    return examples[:sample_size]


def _evaluate_checkpoint(
    checkpoint_path: Path,
    value_function: ValueFunction,
    tokenizer,
    dataset: List[Dict],
) -> Dict:
    value_function.load_checkpoint(str(checkpoint_path), strict=False, load_optimizer=False)
    
    accuracies: List[float] = []
    q_spans: List[float] = []
    
    for entry in dataset:
        observation = entry["observation"]
        candidates = entry["candidates"]
        if not candidates:
            continue
        obs_batch = [observation] * len(candidates)
        q_values = value_function.predict_q_value(
            observations=obs_batch,
            high_level_actions=candidates,
            tokenizer=tokenizer,
            requires_grad=False
        )
        q_list = q_values.tolist()
        best_idx = max(range(len(q_list)), key=lambda idx: q_list[idx])
        best_belief = candidates[best_idx]
        accuracy = compute_belief_accuracy(best_belief, entry["ground_truth"])
        accuracies.append(accuracy)
        q_spans.append(max(q_list) - min(q_list) if len(q_list) > 1 else 0.0)
    
    if not accuracies:
        raise ValueError(f"No valid evaluations produced for checkpoint {checkpoint_path}")
    
    return {
        "checkpoint": str(checkpoint_path),
        "num_samples": len(accuracies),
        "avg_accuracy": mean(accuracies),
        "std_accuracy": pstdev(accuracies) if len(accuracies) > 1 else 0.0,
        "avg_q_span": mean(q_spans),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run checkpoint ablations on offline slices.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config_online.json"),
        help="Config file to load (must be compatible with BaseConfig).",
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        required=True,
        help="List of checkpoint files to compare.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=64,
        help="Number of (observation, candidate) pairs to evaluate.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=4,
        help="Maximum turns per dialogue to include in the evaluation slice.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/llm_context_belief/checkpoint_ablation_report.json"),
        help="Where to save the comparison report.",
    )
    args = parser.parse_args()
    
    if not args.checkpoints:
        raise ValueError("At least one checkpoint path must be provided.")
    
    if args.config.exists():
        config = BaseConfig.from_json(str(args.config))
    else:
        config = BaseConfig()
    
    rng = random.Random(args.seed)
    print(f"[checkpoint_ablation] Sampling {args.sample_size} examples (seed={args.seed})")
    
    model = get_model_instance(
        model_name=config.model_name,
        device=config.device if config.device != "cuda" else "cuda:0",
        use_bf16=config.use_bf16
    )
    tokenizer = get_tokenizer_instance(config.model_name)
    prompt_manager = PromptManager()
    hl_agent = HighLevelAgent(model, tokenizer, prompt_manager, template_name=config.belief_gen_template)
    value_function = ValueFunction(
        model=model,
        hidden_size=4096,
        learning_rate=config.learning_rate,
        device=config.device,
        dtype=torch.bfloat16 if config.use_bf16 else torch.float32,
    )
    
    dialogues = load_multiwoz_dataset(
        data_path=config.data_path,
        split="train",
        use_huggingface=True
    )
    rng.shuffle(dialogues)
    dataset = _collect_eval_dataset(
        dialogues=dialogues,
        sample_size=args.sample_size,
        max_turns=args.max_turns,
        hl_agent=hl_agent,
        config=config,
        rng=rng,
    )
    
    results = []
    for checkpoint in args.checkpoints:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        stats = _evaluate_checkpoint(checkpoint, value_function, tokenizer, dataset)
        print(
            f"[checkpoint_ablation] {checkpoint}: "
            f"avg_acc={stats['avg_accuracy']:.4f}, "
            f"std_acc={stats['std_accuracy']:.4f}, "
            f"avg_q_span={stats['avg_q_span']:.4f}"
        )
        results.append(stats)
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump({"results": results}, handle, indent=2)
    print(f"[checkpoint_ablation] Saved report to {args.output}")


if __name__ == "__main__":
    main()


