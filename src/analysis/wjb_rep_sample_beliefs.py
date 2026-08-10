"""
Sample HL beliefs for WildJailbreak rep experiment using dual critics:
nominal Q from ablation3_marginal, regret from regret_critic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..agents import HighLevelAgent
from ..prompts.prompt_manager import PromptManager
from .rep_belief_selection import (
    generate_belief_cache,
    load_dual_critic_value_function,
    load_eval_records,
    save_belief_cache,
)


def run(args: argparse.Namespace) -> None:
    records = load_eval_records(Path(args.data_path))
    device = args.device
    dtype = torch.bfloat16 if args.use_bf16 else torch.float16

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    value_function = load_dual_critic_value_function(
        model=model,
        device=device,
        dtype=dtype,
        q_checkpoint_path=args.q_checkpoint_path,
        regret_checkpoint_path=args.regret_checkpoint_path,
    )

    prompt_manager = PromptManager()
    hl_agent = HighLevelAgent(
        model,
        tokenizer,
        prompt_manager,
        template_name="belief_generation",
    )

    cache = generate_belief_cache(
        records=records,
        hl_agent=hl_agent,
        value_function=value_function,
        model=model,
        tokenizer=tokenizer,
        device=device,
        n_candidates=args.n_candidates,
        belief_gen_temperature=args.belief_gen_temperature,
        max_new_tokens=args.max_new_tokens,
        regret_beta=args.regret_critic_beta,
        seed=args.seed,
    )

    output_path = Path(args.output_path)
    save_belief_cache(cache, output_path)

    config = {
        "model_name": args.model_name,
        "q_checkpoint_path": args.q_checkpoint_path,
        "regret_checkpoint_path": args.regret_checkpoint_path,
        "data_path": args.data_path,
        "num_records": len(cache),
        "n_candidates": args.n_candidates,
        "regret_critic_beta": args.regret_critic_beta,
        "seed": args.seed,
    }
    with output_path.with_suffix(".config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    print(f"Wrote belief cache ({len(cache)} records) to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample WJB critic-guided/random/orthogonal beliefs.")
    parser.add_argument("--data_path", type=str, default="data/WJB-Rep/wjb_rep_eval.jsonl")
    parser.add_argument("--output_path", type=str, default="data/WJB-Rep/wjb_belief_cache.jsonl")
    parser.add_argument("--q_checkpoint_path", type=str, required=True)
    parser.add_argument("--regret_checkpoint_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="HuggingFaceH4/zephyr-7b-beta")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_candidates", type=int, default=5)
    parser.add_argument("--belief_gen_temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=250)
    parser.add_argument("--regret_critic_beta", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_bf16", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
