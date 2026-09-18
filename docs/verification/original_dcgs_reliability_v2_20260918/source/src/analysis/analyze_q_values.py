"""Analyze what Q-values are learning.

This script loads a trained checkpoint and analyzes:
1. Q-value distribution (mean, std, min, max, percentiles)
2. Correlations with various features:
   - Belief length
   - Observation length
   - P(a|b) for dataset action
   - Cosine similarity to ground truth
   - Whether belief contains certain keywords
3. Visualizations of correlations
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from src.agents import UserAgent
from src.utils.llm_utils import get_user_model_instance

import numpy as np
import torch

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents.high_level_agent import HighLevelAgent
from src.agents.low_level_agent import LowLevelAgent
from src.configs.base_config import BaseConfig
from src.data import format_multiwoz_goal, load_multiwoz_dataset
from src.main import (
    batch_generate_beliefs_for_episodes,
    create_episode_state,
    extract_goal_from_dialogue_state,
    infer_goal_from_dialogue,
    is_valid_multiwoz_goal,
    process_turn_for_episode,
)
from src.prompts.prompt_manager import PromptManager
from src.utils.belief_evaluation import evaluate_beliefs_against_ground_truth
from src.utils.llm_utils import get_model_instance, get_tokenizer_instance
from src.value.value_function import ValueFunction


def collect_q_value_data(
    checkpoint_path: Path,
    config: BaseConfig,
    dialogues: List,
    n_examples: int = 200,
    model=None,
    tokenizer=None,
    device: str = "cuda",
    standard_dtype: torch.dtype = torch.bfloat16,
) -> List[Dict]:
    """Collect Q-values and associated features from episodes."""
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    value_function = ValueFunction(
        model=model,
        hidden_size=4096,
        learning_rate=config.learning_rate,
        device=device,
        dtype=standard_dtype,
    )
    value_function.load_checkpoint(str(checkpoint_path), strict=False, load_optimizer=False)
    
    prompt_manager = PromptManager()
    hl_agent = HighLevelAgent(model, tokenizer, prompt_manager, template_name=config.belief_gen_template)
    ll_agent = LowLevelAgent(model, tokenizer, prompt_manager)
    
    # Initialize user agent for online mode
    user_agent = None
    if config.environment_type == "multiwoz_online":
        print("Loading user model for online mode...")
        user_model_name = config.user_model_name if config.user_model_name else config.model_name
        user_model = get_user_model_instance(
            user_model_name=user_model_name,
            main_model_name=config.model_name,
            device="cuda:1" if device == "cuda:0" else device,
            use_bf16=config.use_bf16,
            fallback_to_main=True
        )
        user_agent = UserAgent(user_model, tokenizer, prompt_manager)
        print("User agent initialized for online mode")
    
    # Sample dialogues
    if len(dialogues) < n_examples:
        eval_dialogues = dialogues
    else:
        eval_dialogues = random.sample(dialogues, n_examples)
    
    print(f"Creating episodes from {len(eval_dialogues)} dialogues...")
    
    all_data = []
    
    for dialogue_idx, dialogue in enumerate(eval_dialogues):
        # Impute goal if missing
        if not is_valid_multiwoz_goal(dialogue.goal):
            goal_state = extract_goal_from_dialogue_state(dialogue.dialogue_state)
            if goal_state:
                dialogue.goal = goal_state
            else:
                inferred_goal = infer_goal_from_dialogue(dialogue)
                if inferred_goal:
                    dialogue.goal = inferred_goal
        
        if not is_valid_multiwoz_goal(dialogue.goal):
            continue
        
        episode = create_episode_state(
            dialogue_idx=dialogue_idx,
            dialogue_data=dialogue,
            multiwoz_mode=True,
            config=config,
            debug=False,
            user_agent=user_agent if config.environment_type == "multiwoz_online" else None
        )
        
        if not episode:
            continue
        
        # Process a few turns
        max_turns = min(config.max_turns, 5)  # Limit to 5 turns for quick analysis
        for turn_idx in range(max_turns):
            if episode.done_from_env:
                break
            
            # Generate beliefs
            batch_generate_beliefs_for_episodes(
                [episode],
                hl_agent,
                config,
                ground_truth_pool=None,
                model=model,
                tokenizer=tokenizer,
                device=device
            )
            
            if not episode.belief_state.candidates:
                break
            
            # Compute Q-values
            candidate_summaries = [c.summary for c in episode.belief_state.candidates if c.summary != "[SKIP]"]
            if not candidate_summaries:
                break
            
            observations = [episode.current_obs_for_action] * len(candidate_summaries)
            
            q_values_tensor = value_function.predict_q_value(
                observations=observations,
                high_level_actions=candidate_summaries,
                tokenizer=tokenizer,
                requires_grad=False
            )
            
            q_values = {
                candidate: float(q_val.item())
                for candidate, q_val in zip(candidate_summaries, q_values_tensor)
            }
            
            # Set Q-values on episode so process_turn_for_episode can use them
            episode._q_values = q_values
            
            # Get dataset action
            dataset_action = episode.env_info.get("actions", [])
            if dataset_action:
                dataset_action = dataset_action[0] if isinstance(dataset_action, list) else dataset_action
            else:
                dataset_action = ""
            
            # Compute P(a|b) for each belief
            belief_only = getattr(config, 'll_action_belief_only', True)
            observation = episode.observation if not belief_only else ""
            
            p_ab_values = {}
            for belief in candidate_summaries:
                if dataset_action:
                    try:
                        p_ab = value_function.compute_likelihood(
                            observation=observation,
                            high_level_action=belief,
                            low_level_action=dataset_action,
                            ll_model=model,
                            ll_tokenizer=tokenizer,
                            belief_only=belief_only
                        )
                        p_ab_values[belief] = p_ab
                    except Exception:
                        p_ab_values[belief] = None
                else:
                    p_ab_values[belief] = None
            
            # Get ground truth goal
            ground_truth_goal = format_multiwoz_goal(dialogue.goal) if dialogue.goal else None
            
            # Compute cosine similarity for each belief
            cosine_sims = {}
            if ground_truth_goal and model and tokenizer:
                try:
                    metrics = evaluate_beliefs_against_ground_truth(
                        chosen_beliefs=candidate_summaries,
                        ground_truth_text=ground_truth_goal,
                        model=model,
                        tokenizer=tokenizer,
                        device=device
                    )
                    for belief, metric in zip(candidate_summaries, metrics):
                        cosine_sims[belief] = metric.get("cosine_similarity")
                except Exception:
                    pass
            
            # Collect data for each candidate
            for belief in candidate_summaries:
                data_point = {
                    "dialogue_id": dialogue.dialogue_id,
                    "turn": turn_idx,
                    "belief": belief,
                    "q_value": q_values.get(belief),
                    "p_ab": p_ab_values.get(belief),
                    "cosine_sim": cosine_sims.get(belief),
                    "belief_length": len(belief),
                    "observation_length": len(episode.current_obs_for_action),
                    "dataset_action": dataset_action,
                    "dataset_action_length": len(dataset_action) if dataset_action else 0,
                }
                all_data.append(data_point)
            
            # Process turn
            if not process_turn_for_episode(
                episode=episode,
                hl_agent=hl_agent,
                ll_agent=ll_agent,
                value_function=value_function,
                tokenizer=tokenizer,
                model=model,
                config=config,
                multiwoz_mode=True,
                standard_dtype=standard_dtype,
            ):
                break
    
    return all_data


def analyze_q_values(data: List[Dict]) -> Dict:
    """Analyze Q-values and their correlations."""
    
    if not data:
        return {}
    
    # Extract arrays
    q_values = np.array([d["q_value"] for d in data if d["q_value"] is not None])
    p_ab_values = np.array([d["p_ab"] for d in data if d["p_ab"] is not None and d["p_ab"] != float('-inf')])
    cosine_sims = np.array([d["cosine_sim"] for d in data if d["cosine_sim"] is not None])
    belief_lengths = np.array([d["belief_length"] for d in data])
    obs_lengths = np.array([d["observation_length"] for d in data])
    
    results = {
        "num_data_points": len(data),
    }
    
    # Q-value statistics (only if we have Q-values)
    if len(q_values) > 0:
        results["q_value_stats"] = {
            "mean": float(np.mean(q_values)),
            "std": float(np.std(q_values)),
            "min": float(np.min(q_values)),
            "max": float(np.max(q_values)),
            "percentiles": {
                "25th": float(np.percentile(q_values, 25)),
                "50th": float(np.percentile(q_values, 50)),
                "75th": float(np.percentile(q_values, 75)),
                "90th": float(np.percentile(q_values, 90)),
                "95th": float(np.percentile(q_values, 95)),
            }
        }
    else:
        results["q_value_stats"] = None
    
    # Correlations
    correlations = {}
    
    # Q-value vs P(a|b)
    if len(p_ab_values) > 1:
        q_pab_pairs = [
            (d["q_value"], d["p_ab"])
            for d in data
            if d["q_value"] is not None and d["p_ab"] is not None and d["p_ab"] != float('-inf')
        ]
        if len(q_pab_pairs) > 1:
            q_vals_pab, p_ab_vals = zip(*q_pab_pairs)
            if np.std(q_vals_pab) > 0 and np.std(p_ab_vals) > 0:
                corr = np.corrcoef(q_vals_pab, p_ab_vals)[0, 1]
                correlations["q_vs_pab"] = float(corr)
    
    # Q-value vs Cosine Similarity
    if len(cosine_sims) > 1:
        q_cos_pairs = [
            (d["q_value"], d["cosine_sim"])
            for d in data
            if d["q_value"] is not None and d["cosine_sim"] is not None
        ]
        if len(q_cos_pairs) > 1:
            q_vals_cos, cos_vals = zip(*q_cos_pairs)
            if np.std(q_vals_cos) > 0 and np.std(cos_vals) > 0:
                corr = np.corrcoef(q_vals_cos, cos_vals)[0, 1]
                correlations["q_vs_cosine"] = float(corr)
    
    # Q-value vs Belief Length
    if len(q_values) > 0 and len(belief_lengths) > 0 and np.std(belief_lengths) > 0:
        if len(q_values) == len(belief_lengths):
            corr = np.corrcoef(q_values, belief_lengths)[0, 1]
            correlations["q_vs_belief_length"] = float(corr)
    
    # Q-value vs Observation Length
    if len(q_values) > 0 and len(obs_lengths) > 0 and np.std(obs_lengths) > 0:
        if len(q_values) == len(obs_lengths):
            corr = np.corrcoef(q_values, obs_lengths)[0, 1]
            correlations["q_vs_obs_length"] = float(corr)
    
    results["correlations"] = correlations
    
    # Analyze by Q-value quantiles (only if we have Q-values)
    if len(q_values) > 0:
        q_sorted_indices = np.argsort(q_values)
        n = len(q_values)
        bottom_10 = q_sorted_indices[:n//10] if n >= 10 else q_sorted_indices
        top_10 = q_sorted_indices[-n//10:] if n >= 10 else q_sorted_indices
        
        results["quantile_analysis"] = {
            "bottom_10_percent": {
                "avg_q": float(np.mean(q_values[bottom_10])),
                "avg_belief_length": float(np.mean(belief_lengths[bottom_10])),
                "avg_obs_length": float(np.mean(obs_lengths[bottom_10])),
            },
            "top_10_percent": {
                "avg_q": float(np.mean(q_values[top_10])),
                "avg_belief_length": float(np.mean(belief_lengths[top_10])),
                "avg_obs_length": float(np.mean(obs_lengths[top_10])),
            }
        }
    else:
        results["quantile_analysis"] = None
    
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze what Q-values are learning")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--n-examples", type=int, default=200, help="Number of examples to analyze")
    parser.add_argument("--output", type=str, default="q_value_analysis.json", help="Output file path")
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    config = BaseConfig.from_json(args.config)
    
    print("Loading model and tokenizer...")
    model = get_model_instance(
        model_name=config.model_name,
        device=config.device,
        use_bf16=config.use_bf16,
    )
    tokenizer = get_tokenizer_instance(config.model_name)
    
    print("Loading dataset...")
    dialogues = load_multiwoz_dataset(
        data_path=config.data_path,
        split="train",
        use_huggingface=True
    )
    
    # Filter and impute valid dialogues (same logic as main training loop)
    print(f"Filtering {len(dialogues)} dialogues for valid goals...")
    from src.data.dialogue_formatter import extract_goal_from_dialogue_state, infer_goal_from_dialogue
    
    valid_dialogues = []
    invalid_count = 0
    imputed_count = 0
    
    for d in dialogues:
        # Try to impute goal if missing/invalid (same as main training loop)
        if not is_valid_multiwoz_goal(d.goal):
            # Extract goal from dialogue state fields (same as main.py lines 2168-2175)
            extracted_goal = extract_goal_from_dialogue_state(d)
            if extracted_goal:
                d.goal = extracted_goal
                imputed_count += 1
            else:
                # Fallback: use legacy inference method
                inferred_goal = infer_goal_from_dialogue(d)
                if inferred_goal:
                    d.goal = inferred_goal
                    imputed_count += 1
        
        # Check again after imputation
        if is_valid_multiwoz_goal(d.goal):
            valid_dialogues.append(d)
        else:
            invalid_count += 1
    
    print(f"Loaded {len(valid_dialogues)} valid dialogues (imputed {imputed_count}, skipped {invalid_count} with invalid goals)")
    
    print(f"Collecting Q-value data from {args.n_examples} examples...")
    data = collect_q_value_data(
        checkpoint_path=checkpoint_path,
        config=config,
        dialogues=valid_dialogues,
        n_examples=args.n_examples,
        model=model,
        tokenizer=tokenizer,
        device=config.device,
        standard_dtype=torch.bfloat16 if config.use_bf16 else torch.float32,
    )
    
    print(f"Collected {len(data)} data points")
    
    if not data:
        print("ERROR: No data collected! Cannot analyze Q-values.")
        print("This may indicate:")
        print("  - No valid dialogues found")
        print("  - No valid episodes created")
        print("  - No Q-values computed")
        return
    
    print("Analyzing Q-values...")
    results = analyze_q_values(data)
    
    if not results:
        print("ERROR: Analysis returned empty results!")
        return
    
    # Print summary
    print("\n" + "="*80)
    print("Q-VALUE ANALYSIS RESULTS")
    print("="*80)
    print(f"\nData Points: {results.get('num_data_points', 0)}")
    
    if "q_value_stats" not in results or results["q_value_stats"] is None:
        print("ERROR: No Q-value statistics available!")
        print("  This indicates no Q-values were collected from the episodes.")
        return
    
    print(f"\nQ-Value Statistics:")
    stats = results["q_value_stats"]
    print(f"  Mean: {stats['mean']:.4f}")
    print(f"  Std:  {stats['std']:.4f}")
    print(f"  Min:  {stats['min']:.4f}")
    print(f"  Max:  {stats['max']:.4f}")
    print(f"  Percentiles:")
    for pct, val in stats["percentiles"].items():
        print(f"    {pct}: {val:.4f}")
    
    print(f"\nCorrelations:")
    correlations = results.get("correlations", {})
    if not correlations:
        print("  No correlations computed (insufficient data)")
    else:
        for name, corr in correlations.items():
            print(f"  {name}: {corr:.4f}")
            if abs(corr) > 0.5:
                print(f"    → Strong correlation!")
            elif abs(corr) > 0.3:
                print(f"    → Moderate correlation")
            else:
                print(f"    → Weak correlation")
    
    print(f"\nQuantile Analysis:")
    quant = results.get("quantile_analysis")
    if not quant:
        print("  No quantile analysis available (insufficient data)")
    else:
        print(f"  Bottom 10%: avg_q={quant['bottom_10_percent']['avg_q']:.4f}, "
              f"avg_belief_len={quant['bottom_10_percent']['avg_belief_length']:.1f}")
        print(f"  Top 10%:    avg_q={quant['top_10_percent']['avg_q']:.4f}, "
              f"avg_belief_len={quant['top_10_percent']['avg_belief_length']:.1f}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump({"results": results, "raw_data_sample": data[:10]}, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

