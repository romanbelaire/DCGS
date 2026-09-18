"""Live evaluation script for ablated models.

This script loads checkpoints for different ablation modes and runs evaluation
on a small set of examples (N=10 by default) to quickly compare performance.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional

import torch

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents.high_level_agent import HighLevelAgent
from src.agents.low_level_agent import LowLevelAgent
from src.agents import UserAgent
from src.configs.base_config import BaseConfig
from src.data import format_multiwoz_goal, load_multiwoz_dataset
from src.data.data_utils import compute_belief_accuracy
from src.main import (
    batch_generate_beliefs_for_episodes,
    create_episode_state,
    is_noise_candidate,
    is_valid_multiwoz_goal,
    process_turn_for_episode,
)
from src.prompts.prompt_manager import PromptManager
from src.utils.belief_evaluation import evaluate_beliefs_against_ground_truth
from src.utils.llm_utils import get_model_instance, get_tokenizer_instance, get_user_model_instance
from src.value.value_function import ValueFunction


# Ablation configurations
ABLATION_CONFIGS = {
    "crossed_data": {
        "name": "Main Method (Crossed Data)",
        "contrastive_coef": 1.0,
        "contrastive_ablation_mode": "crossed_data",
    },
    "random_noise": {
        "name": "Ablation 1 (Random Noise)",
        "contrastive_coef": 1.0,
        "contrastive_ablation_mode": "random_noise",
    },
    "noise_in_candidates": {
        "name": "Ablation 2 (Noise in Candidates)",
        "contrastive_coef": 0.0,
        "contrastive_ablation_mode": "noise_in_candidates",
    },
    "crossed_data_in_candidates": {
        "name": "Ablation 3 (Crossed Data in Candidates)",
        "contrastive_coef": 0.0,
        "contrastive_ablation_mode": "crossed_data_in_candidates",
    },
    "none": {
        "name": "Baseline (No Contrastive Loss)",
        "contrastive_coef": 0.0,
        "contrastive_ablation_mode": "none",
    },
}


def evaluate_ablation(
    checkpoint_path: Path,
    ablation_key: str,
    ablation_config: Dict,
    dialogues: List,
    n_examples: int,
    config: BaseConfig,
    model,
    tokenizer,
    prompt_manager,
    device: str,
    standard_dtype: torch.dtype,
    user_agent=None,
) -> Dict:
    """Evaluate a single ablation checkpoint on N examples."""
    print(f"\n{'='*80}")
    print(f"Evaluating: {ablation_config['name']}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*80}")
    
    # Create agents with ablation-specific config
    hl_agent = HighLevelAgent(model, tokenizer, prompt_manager, template_name=config.belief_gen_template)
    ll_agent = LowLevelAgent(model, tokenizer, prompt_manager)
    
    # Create value function and load checkpoint
    value_function = ValueFunction(
        model=model,
        hidden_size=4096,
        learning_rate=config.learning_rate,
        device=device,
        dtype=standard_dtype,
    )
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    value_function.load_checkpoint(str(checkpoint_path), strict=False, load_optimizer=False)
    
    # Update config with ablation settings
    config.contrastive_coef = ablation_config["contrastive_coef"]
    config.contrastive_ablation_mode = ablation_config["contrastive_ablation_mode"]
    
    # Sample N dialogues
    if len(dialogues) < n_examples:
        print(f"Warning: Only {len(dialogues)} dialogues available, using all of them")
        eval_dialogues = dialogues
    else:
        eval_dialogues = random.sample(dialogues, n_examples)
    
    print(f"Creating episodes from {len(eval_dialogues)} dialogues...")
    
    # Create episodes - fail fast on any error
    episodes = []
    for dialogue_idx, dialogue in enumerate(eval_dialogues):
        episode = create_episode_state(
            dialogue_idx=dialogue_idx,
            dialogue_data=dialogue,
            multiwoz_mode=True,
            config=config,
            debug=False,
            user_agent=user_agent if config.environment_type == "multiwoz_online" else None,
        )
        episodes.append(episode)
    
    if not episodes:
        raise ValueError(f"Failed to create any episodes from {len(eval_dialogues)} dialogues")
    
    print(f"Running evaluation on {len(episodes)} episodes...")
    
    # Run episodes and collect metrics
    all_turn_metrics = []
    episode_count = 0
    
    for episode in episodes:
        episode_count += 1
        print(f"  Processing episode {episode_count}/{len(episodes)}: {episode.dialogue_id}")
        
        # Process turns until episode ends
        max_turns = min(config.max_turns, 10)  # Limit to 10 turns for quick eval
        while episode.turn < max_turns and not episode.done_from_env:
            # Generate beliefs for current turn
            # For "crossed_data_in_candidates" mode, we need ground_truth_pool but it's not available in eval
            # So we pass None - the mode will gracefully skip if pool is unavailable
            batch_generate_beliefs_for_episodes(
                [episode], 
                hl_agent, 
                config,
                ground_truth_pool=None,  # Not available in live eval
                model=model,
                tokenizer=tokenizer,
                device=device
            )
            
            # Compute Q-values for beliefs
            if episode.belief_state.candidates:
                candidate_summaries = [c.summary for c in episode.belief_state.candidates]
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
                episode._q_values = q_values
            
            # Process the turn
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
        
        # Collect metrics from turn_evaluation_data
        for turn_eval in episode.turn_evaluation_data:
            baseline_metrics = turn_eval.get("baseline_metrics")
            ground_truth_goal = turn_eval.get("ground_truth_goal")
            selected_belief = turn_eval.get("selected_belief")
            q_values = turn_eval.get("q_values", {})
            min_q_belief = turn_eval.get("min_q_baseline_belief", "")
            ll_action = turn_eval.get("ll_action", "")
            
            # Compute actual cosine similarity using embeddings (not word-overlap accuracy)
            cosine_sim = None
            if selected_belief and ground_truth_goal and model and tokenizer:
                try:
                    metrics = evaluate_beliefs_against_ground_truth(
                        chosen_beliefs=[selected_belief],
                        ground_truth_text=ground_truth_goal,
                        model=model,
                        tokenizer=tokenizer,
                        device=device
                    )
                    if metrics:
                        cosine_sim = metrics[0].get("cosine_similarity")
                except Exception as e:
                    # If cosine similarity computation fails, skip this turn
                    print(f"Warning: Failed to compute cosine similarity: {e}")
                    continue
            
            # Compute P(a|b) for max-Q and min-Q beliefs
            max_q_p_ab = None
            min_q_p_ab = None
            if q_values and ll_action and selected_belief and min_q_belief:
                # Filter out [SKIP] from Q-values
                valid_q_values = {k: v for k, v in q_values.items() if k != "[SKIP]"}
                if len(valid_q_values) >= 2:
                    max_q_belief_from_q = max(valid_q_values, key=valid_q_values.get)
                    min_q_belief_from_q = min(valid_q_values, key=valid_q_values.get)
                    
                    # Use the beliefs from Q-values (should match selected_belief and min_q_belief)
                    belief_only = getattr(config, 'll_action_belief_only', True)
                    observation = turn_eval.get("observation", "") if not belief_only else ""
                    
                    try:
                        # Compute P(a|max-Q belief)
                        max_q_p_ab = value_function.compute_likelihood(
                            observation=observation,
                            high_level_action=max_q_belief_from_q,
                            low_level_action=ll_action,
                            ll_model=model,
                            ll_tokenizer=tokenizer,
                            belief_only=belief_only
                        )
                        
                        # Compute P(a|min-Q belief)
                        min_q_p_ab = value_function.compute_likelihood(
                            observation=observation,
                            high_level_action=min_q_belief_from_q,
                            low_level_action=ll_action,
                            ll_model=model,
                            ll_tokenizer=tokenizer,
                            belief_only=belief_only
                        )
                    except Exception as e:
                        # If computation fails, skip P(a|b) for this turn
                        pass
            
            if baseline_metrics:
                all_turn_metrics.append({
                    "cosine_similarity": cosine_sim,
                    "min_q_accuracy": turn_eval.get("baseline_metrics", {}).get("min_q_accuracy"),
                    "random_accuracy": turn_eval.get("baseline_metrics", {}).get("random_accuracy"),
                    "top_prob_accuracy": turn_eval.get("baseline_metrics", {}).get("top_prob_accuracy"),
                    "max_q_p_ab": max_q_p_ab,
                    "min_q_p_ab": min_q_p_ab,
                    "max_q_value": q_values.get(selected_belief) if selected_belief and q_values else None,
                    "min_q_value": q_values.get(min_q_belief) if min_q_belief and q_values else None,
                })
    
    # Aggregate metrics
    if not all_turn_metrics:
        print(f"Warning: No metrics collected for {ablation_config['name']}")
        return {
            "ablation": ablation_key,
            "name": ablation_config["name"],
            "checkpoint": str(checkpoint_path),
            "num_episodes": len(episodes),
            "num_turns": 0,
            "metrics": {},
        }
    
    # Filter out None values and compute statistics
    cosine_sims = [m["cosine_similarity"] for m in all_turn_metrics if m["cosine_similarity"] is not None]
    min_q_accs = [m["min_q_accuracy"] for m in all_turn_metrics if m["min_q_accuracy"] is not None]
    random_accs = [m["random_accuracy"] for m in all_turn_metrics if m["random_accuracy"] is not None]
    top_prob_accs = [m["top_prob_accuracy"] for m in all_turn_metrics if m["top_prob_accuracy"] is not None]
    
    # P(a|b) diagnostics
    p_ab_diagnostics = [
        m for m in all_turn_metrics 
        if m.get("max_q_p_ab") is not None and m.get("min_q_p_ab") is not None
    ]
    
    # Compute P(a|b) statistics
    p_ab_stats = {}
    if p_ab_diagnostics:
        p_ab_matches = sum(1 for m in p_ab_diagnostics if m["max_q_p_ab"] > m["min_q_p_ab"])
        max_p_abs = [m["max_q_p_ab"] for m in p_ab_diagnostics]
        min_p_abs = [m["min_q_p_ab"] for m in p_ab_diagnostics]
        p_ab_stats = {
            "p_ab_match_rate": p_ab_matches / len(p_ab_diagnostics) if p_ab_diagnostics else 0.0,
            "max_q_p_ab": {
                "mean": mean(max_p_abs),
                "std": stdev(max_p_abs) if len(max_p_abs) > 1 else 0.0,
                "count": len(max_p_abs),
            },
            "min_q_p_ab": {
                "mean": mean(min_p_abs),
                "std": stdev(min_p_abs) if len(min_p_abs) > 1 else 0.0,
                "count": len(min_p_abs),
            },
        }
        
        # Correlation between Q-values and P(a|b)
        try:
            import numpy as np
            q_diffs = [
                m["max_q_value"] - m["min_q_value"] 
                for m in p_ab_diagnostics 
                if m.get("max_q_value") is not None and m.get("min_q_value") is not None
            ]
            p_ab_diffs = [
                m["max_q_p_ab"] - m["min_q_p_ab"] 
                for m in p_ab_diagnostics
            ]
            
            if len(q_diffs) > 1 and len(p_ab_diffs) > 1 and len(q_diffs) == len(p_ab_diffs):
                if np.std(q_diffs) > 0 and np.std(p_ab_diffs) > 0:
                    correlation = np.corrcoef(q_diffs, p_ab_diffs)[0, 1]
                    p_ab_stats["q_vs_pab_correlation"] = float(correlation)
        except Exception:
            pass
    
    results = {
        "ablation": ablation_key,
        "name": ablation_config["name"],
        "checkpoint": str(checkpoint_path),
        "num_episodes": len(episodes),
        "num_turns": len(all_turn_metrics),
        "metrics": {
            "cosine_similarity": {
                "mean": mean(cosine_sims) if cosine_sims else None,
                "std": stdev(cosine_sims) if len(cosine_sims) > 1 else 0.0,
                "count": len(cosine_sims),
            },
            "min_q_accuracy": {
                "mean": mean(min_q_accs) if min_q_accs else None,
                "std": stdev(min_q_accs) if len(min_q_accs) > 1 else 0.0,
                "count": len(min_q_accs),
            },
            "random_accuracy": {
                "mean": mean(random_accs) if random_accs else None,
                "std": stdev(random_accs) if len(random_accs) > 1 else 0.0,
                "count": len(random_accs),
            },
            "top_prob_accuracy": {
                "mean": mean(top_prob_accs) if top_prob_accs else None,
                "std": stdev(top_prob_accs) if len(top_prob_accs) > 1 else 0.0,
                "count": len(top_prob_accs),
            },
            "p_ab_analysis": p_ab_stats,
        },
    }
    
    # Print summary
    print(f"\nResults for {ablation_config['name']}:")
    print(f"  Episodes: {results['num_episodes']}, Turns: {results['num_turns']}")
    if cosine_sims:
        print(f"  Cosine Similarity: {results['metrics']['cosine_similarity']['mean']:.4f} ± {results['metrics']['cosine_similarity']['std']:.4f}")
    if min_q_accs:
        print(f"  Min-Q Accuracy: {results['metrics']['min_q_accuracy']['mean']:.4f} ± {results['metrics']['min_q_accuracy']['std']:.4f}")
    if random_accs:
        print(f"  Random Accuracy: {results['metrics']['random_accuracy']['mean']:.4f} ± {results['metrics']['random_accuracy']['std']:.4f}")
    if top_prob_accs:
        print(f"  Top-Prob Accuracy: {results['metrics']['top_prob_accuracy']['mean']:.4f} ± {results['metrics']['top_prob_accuracy']['std']:.4f}")
    
    # Print P(a|b) analysis
    if p_ab_stats:
        print(f"\n  P(a|b) Analysis:")
        print(f"    P(a|max-Q) > P(a|min-Q): {p_ab_stats['p_ab_match_rate']*100:.1f}% ({int(p_ab_stats['p_ab_match_rate'] * len(p_ab_diagnostics))}/{len(p_ab_diagnostics)} turns)")
        print(f"    Avg P(a|max-Q): {p_ab_stats['max_q_p_ab']['mean']:.4f} ± {p_ab_stats['max_q_p_ab']['std']:.4f}")
        print(f"    Avg P(a|min-Q): {p_ab_stats['min_q_p_ab']['mean']:.4f} ± {p_ab_stats['min_q_p_ab']['std']:.4f}")
        if p_ab_stats.get('q_vs_pab_correlation') is not None:
            corr = p_ab_stats['q_vs_pab_correlation']
            print(f"    Q-value vs P(a|b) correlation: {corr:.4f}")
            if abs(corr) > 0.5:
                print(f"      ✓ Strong correlation - Q-values ARE learning P(a|b)!")
            elif abs(corr) > 0.3:
                print(f"      Moderate correlation - Q-values partially learn P(a|b)")
            else:
                print(f"      [WARNING] Weak correlation - Q-values not strongly learning P(a|b)")
        if p_ab_stats['p_ab_match_rate'] > 0.7:
            print(f"    ✓ Q-function IS learning P(a|b) correctly!")
            print(f"    → Cosine similarity may not be the right metric - Q learns action probability, not semantic similarity")
    
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live evaluation of ablated models on N examples"
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        required=True,
        help="Comma-separated checkpoint paths. Can be 2 (crossed_data,crossed_data_in_candidates) or 4 (crossed_data,random_noise,crossed_data_in_candidates,none)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config JSON file",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=10,
        help="Number of examples to evaluate (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/live_evaluation_results.json"),
        help="Output path for results JSON (default: outputs/live_evaluation_results.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()
    
    # Parse checkpoint paths
    checkpoint_paths = [Path(p.strip()) for p in args.checkpoints.split(",")]
    if len(checkpoint_paths) == 2:
        # Two checkpoints: crossed_data and crossed_data_in_candidates
        checkpoint_map = {
            "crossed_data": checkpoint_paths[0],
            "crossed_data_in_candidates": checkpoint_paths[1],
        }
    elif len(checkpoint_paths) == 4:
        # Four checkpoints: all ablations
        checkpoint_map = {
            "crossed_data": checkpoint_paths[0],
            "random_noise": checkpoint_paths[1],
            "crossed_data_in_candidates": checkpoint_paths[2],
            "none": checkpoint_paths[3],
        }
    else:
        raise ValueError(
            f"Expected 2 or 4 checkpoint paths, got {len(checkpoint_paths)}. "
            "For 2: crossed_data,crossed_data_in_candidates. "
            "For 4: crossed_data,random_noise,crossed_data_in_candidates,none"
        )
    
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load config
    config = BaseConfig.from_json(str(args.config))
    # Note: Keep original environment_type from config (may be multiwoz_online or multiwoz_offline)
    config._evaluation_mode = True  # Disable gradient updates
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = get_model_instance(
        model_name=config.model_name,
        device=config.device,
        use_bf16=config.use_bf16,
    )
    tokenizer = get_tokenizer_instance(config.model_name)
    prompt_manager = PromptManager()
    
    # Initialize user agent for online mode
    user_agent = None
    if config.environment_type == "multiwoz_online":
        print("Loading user model for online mode...")
        user_model_name = config.user_model_name if config.user_model_name else config.model_name
        user_model = get_user_model_instance(
            user_model_name=user_model_name,
            main_model_name=config.model_name,
            device="cuda:1" if config.device == "cuda:0" else config.device,
            use_bf16=config.use_bf16,
            fallback_to_main=True
        )
        user_agent = UserAgent(user_model, tokenizer, prompt_manager)
        print("User agent initialized for online mode")
    
    standard_dtype = torch.bfloat16 if config.use_bf16 else torch.float32
    
    # Load dataset
    print("Loading dataset...")
    dialogues = load_multiwoz_dataset(
        data_path=config.data_path,
        split="train",
        use_huggingface=True,
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
    
    if len(valid_dialogues) == 0:
        raise ValueError(
            f"No valid dialogues found! All {len(dialogues)} dialogues were filtered out even after imputation. "
            "Check that dialogues have valid goal structures or dialogue state fields."
        )
    
    if len(valid_dialogues) < args.n_examples:
        print(f"Warning: Only {len(valid_dialogues)} valid dialogues available, using all of them")
        args.n_examples = len(valid_dialogues)
    
    # Evaluate each ablation
    all_results = []
    for ablation_key, checkpoint_path in checkpoint_map.items():
        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint not found: {checkpoint_path}, skipping {ablation_key}")
            continue
        
        ablation_config = ABLATION_CONFIGS[ablation_key]
        results = evaluate_ablation(
            checkpoint_path=checkpoint_path,
            ablation_key=ablation_key,
            ablation_config=ablation_config,
            dialogues=valid_dialogues,
            n_examples=args.n_examples,
            config=config,
            model=model,
            tokenizer=tokenizer,
            prompt_manager=prompt_manager,
            device=config.device,
            standard_dtype=standard_dtype,
            user_agent=user_agent,
        )
        all_results.append(results)
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(
            {
                "n_examples": args.n_examples,
                "seed": args.seed,
                "results": all_results,
            },
            f,
            indent=2,
        )
    
    print(f"\n{'='*80}")
    print("Evaluation Summary")
    print(f"{'='*80}")
    print(f"Results saved to: {args.output}")
    print("\nComparison Table:")
    print(f"{'Method':<40} {'Cosine Sim':<12} {'Min-Q Acc':<12} {'Random Acc':<12} {'Top-Prob Acc':<12}")
    print("-" * 88)
    for result in all_results:
        name = result["name"]
        metrics = result["metrics"]
        cos_sim = metrics["cosine_similarity"]["mean"]
        min_q = metrics["min_q_accuracy"]["mean"]
        random_acc = metrics["random_accuracy"]["mean"]
        top_prob = metrics["top_prob_accuracy"]["mean"]
        
        cos_str = f"{cos_sim:.4f}" if cos_sim is not None else "N/A"
        min_q_str = f"{min_q:.4f}" if min_q is not None else "N/A"
        random_str = f"{random_acc:.4f}" if random_acc is not None else "N/A"
        top_prob_str = f"{top_prob:.4f}" if top_prob is not None else "N/A"
        
        print(f"{name:<40} {cos_str:<12} {min_q_str:<12} {random_str:<12} {top_prob_str:<12}")


if __name__ == "__main__":
    main()

