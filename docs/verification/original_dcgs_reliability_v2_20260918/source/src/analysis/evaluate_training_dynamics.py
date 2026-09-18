"""Evaluate training dynamics during training.

This script can be called periodically during training to monitor:
1. Q-value distribution over time
2. Correlation with P(a|b) over time
3. Whether Q-values are converging to constants
4. Whether P(a|max-Q) > P(a|min-Q) is improving
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents.high_level_agent import HighLevelAgent
from src.agents.low_level_agent import LowLevelAgent
from src.agents import UserAgent
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
from src.utils.llm_utils import get_user_model_instance
from src.value.value_function import ValueFunction


def evaluate_training_dynamics(
    value_function: ValueFunction,
    config: BaseConfig,
    model,
    tokenizer,
    device: str,
    standard_dtype: torch.dtype,
    n_examples: int = 50,
    dialogues: Optional[List] = None,
    episode_number: int = 0,
    output_dir: str = "outputs"
) -> Dict:
    """
    Evaluate training dynamics on a small sample of examples.
    
    Args:
        value_function: Current value function
        config: Config object
        model: LLM model
        tokenizer: Tokenizer
        device: Device
        standard_dtype: Data type
        n_examples: Number of examples to evaluate
        dialogues: Optional list of dialogues (will load if None)
        episode_number: Current episode number (for tracking)
        output_dir: Output directory
    
    Returns:
        Dictionary with training dynamics metrics
    """
    
    # Load dialogues if not provided
    if dialogues is None:
        dialogues = load_multiwoz_dataset(
            data_path=config.data_path,
            split="train",
            use_huggingface=True
        )
        dialogues = [d for d in dialogues if is_valid_multiwoz_goal(d.goal)]
    
    # Sample a small set for quick evaluation
    if len(dialogues) < n_examples:
        eval_dialogues = dialogues
    else:
        import random
        eval_dialogues = random.sample(dialogues, n_examples)
    
    prompt_manager = PromptManager()
    hl_agent = HighLevelAgent(model, tokenizer, prompt_manager, template_name=config.belief_gen_template)
    ll_agent = LowLevelAgent(model, tokenizer, prompt_manager)
    
    # Initialize user agent for online mode
    user_agent = None
    if config.environment_type == "multiwoz_online":
        user_model_name = config.user_model_name if config.user_model_name else config.model_name
        user_model = get_user_model_instance(
            user_model_name=user_model_name,
            main_model_name=config.model_name,
            device="cuda:1" if device == "cuda:0" else device,
            use_bf16=standard_dtype == torch.bfloat16,
            fallback_to_main=True
        )
        user_agent = UserAgent(user_model, tokenizer, prompt_manager)
    
    all_q_values = []
    all_p_ab_values = []
    all_cosine_sims = []
    p_ab_matches = 0
    total_comparisons = 0
    
    for idx, dialogue in enumerate(eval_dialogues):
        # Handle both dict and object formats (UserBench uses dicts, MultiWOZ uses objects)
        if isinstance(dialogue, dict):
            # UserBench format - skip evaluation (not MultiWOZ)
            continue
        
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
            dialogue_idx=idx,
            dialogue_data=dialogue,
            multiwoz_mode=True,
            config=config,
            user_agent=user_agent if config.environment_type == "multiwoz_online" else None
        )
        
        if not episode:
            continue
        
        # Process a few turns
        max_turns = min(config.max_turns, 3)  # Limit to 3 turns for quick eval
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
            
            # Filter out [SKIP]
            candidate_summaries = [c.summary for c in episode.belief_state.candidates if c.summary != "[SKIP]"]
            if len(candidate_summaries) < 2:
                break
            
            # Compute Q-values
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
            
            # Store Q-values on episode so process_turn_for_episode can use them
            # Include [SKIP] with 0.0 if it exists in candidates
            episode._q_values = q_values.copy()
            if "[SKIP]" in [c.summary for c in episode.belief_state.candidates]:
                episode._q_values["[SKIP]"] = 0.0
            
            # Get dataset action
            dataset_action = episode.env_info.get("actions", [])
            if dataset_action:
                dataset_action = dataset_action[0] if isinstance(dataset_action, list) else dataset_action
            else:
                dataset_action = ""
            
            # Compute P(a|b) for each belief
            belief_only = getattr(config, 'll_action_belief_only', True)
            observation = episode.observation if not belief_only else ""
            
            # Batch compute P(a|b) for all beliefs
            p_ab_values = {}
            if dataset_action and candidate_summaries:
                try:
                    obs_list = [observation if not belief_only else ""] * len(candidate_summaries)
                    action_list = [dataset_action] * len(candidate_summaries)
                    batch_p_ab = value_function.compute_likelihood_batch(
                        observations=obs_list,
                        high_level_actions=candidate_summaries,
                        low_level_actions=action_list,
                        ll_model=model,
                        ll_tokenizer=tokenizer,
                        belief_only=belief_only
                    )
                    for belief, p_ab in zip(candidate_summaries, batch_p_ab):
                        if p_ab != float('-inf'):
                            p_ab_values[belief] = p_ab
                except Exception:
                    pass
            
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
            
            # Collect Q-values
            all_q_values.extend(q_values.values())
            
            # Collect P(a|b) values
            all_p_ab_values.extend([p_ab_values.get(b) for b in candidate_summaries if p_ab_values.get(b) is not None])
            
            # Collect cosine similarities
            all_cosine_sims.extend([cosine_sims.get(b) for b in candidate_summaries if cosine_sims.get(b) is not None])
            
            # Check P(a|max-Q) > P(a|min-Q)
            if len(q_values) >= 2 and len(p_ab_values) >= 2:
                max_q_belief = max(q_values, key=q_values.get)
                min_q_belief = min(q_values, key=q_values.get)
                
                max_q_p_ab = p_ab_values.get(max_q_belief)
                min_q_p_ab = p_ab_values.get(min_q_belief)
                
                if max_q_p_ab is not None and min_q_p_ab is not None:
                    total_comparisons += 1
                    if max_q_p_ab > min_q_p_ab:
                        p_ab_matches += 1
            
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
    
    # Compute statistics
    results = {
        "episode_number": episode_number,
        "num_data_points": len(all_q_values),
    }
    
    if all_q_values:
        q_array = np.array(all_q_values)
        results["q_value_stats"] = {
            "mean": float(np.mean(q_array)),
            "std": float(np.std(q_array)),
            "min": float(np.min(q_array)),
            "max": float(np.max(q_array)),
            "range": float(np.max(q_array) - np.min(q_array)),
        }
        
        # Check if Q-values are converging to constant (low std relative to range)
        if results["q_value_stats"]["range"] > 0:
            cv = results["q_value_stats"]["std"] / results["q_value_stats"]["range"]
            results["q_value_stats"]["coefficient_of_variation"] = float(cv)
            results["q_value_stats"]["is_converging"] = cv < 0.1  # Low variation = converging
        else:
            results["q_value_stats"]["coefficient_of_variation"] = 0.0
            results["q_value_stats"]["is_converging"] = True
    
    # Correlation with P(a|b)
    if len(all_p_ab_values) > 1 and len(all_q_values) > 1:
        # Match Q-values with P(a|b) values (they may not align perfectly)
        # For simplicity, use all available pairs
        q_pab_pairs = []
        # We collected them in order, so we can pair them if we have enough
        min_len = min(len(all_q_values), len(all_p_ab_values))
        if min_len > 1:
            q_vals = all_q_values[:min_len]
            p_ab_vals = all_p_ab_values[:min_len]
            if np.std(q_vals) > 0 and np.std(p_ab_vals) > 0:
                corr = np.corrcoef(q_vals, p_ab_vals)[0, 1]
                results["q_vs_pab_correlation"] = float(corr)
    
    # Correlation with cosine similarity
    if len(all_cosine_sims) > 1 and len(all_q_values) > 1:
        min_len = min(len(all_q_values), len(all_cosine_sims))
        if min_len > 1:
            q_vals = all_q_values[:min_len]
            cos_vals = all_cosine_sims[:min_len]
            if np.std(q_vals) > 0 and np.std(cos_vals) > 0:
                corr = np.corrcoef(q_vals, cos_vals)[0, 1]
                results["q_vs_cosine_correlation"] = float(corr)
    
    # P(a|max-Q) > P(a|min-Q) match rate
    if total_comparisons > 0:
        results["p_ab_match_rate"] = p_ab_matches / total_comparisons
        results["p_ab_matches"] = p_ab_matches
        results["p_ab_total"] = total_comparisons
    
    # Save results
    output_path = Path(output_dir) / "training_dynamics"
    output_path.mkdir(parents=True, exist_ok=True)
    
    dynamics_file = output_path / "training_dynamics.jsonl"
    with dynamics_file.open("a") as f:
        f.write(json.dumps(results) + "\n")
    
    return results


def print_training_dynamics_summary(results: Dict) -> None:
    """Print a summary of training dynamics results."""
    print(f"\n[Training Dynamics] Episode {results['episode_number']}:")
    
    if "q_value_stats" in results:
        stats = results["q_value_stats"]
        print(f"  Q-value: mean={stats['mean']:.4f}, std={stats['std']:.4f}, range={stats['range']:.4f}")
        if "is_converging" in stats:
            status = "CONVERGING" if stats["is_converging"] else "LEARNING"
            print(f"  Q-value status: {status} (CV={stats.get('coefficient_of_variation', 0):.4f})")
    
    if "q_vs_pab_correlation" in results:
        corr = results["q_vs_pab_correlation"]
        print(f"  Q vs P(a|b) correlation: {corr:.4f}")
        if abs(corr) > 0.5:
            print(f"    → Strong correlation - Q learning P(a|b)!")
        elif abs(corr) > 0.3:
            print(f"    → Moderate correlation")
        else:
            print(f"    → Weak correlation - Q NOT learning P(a|b)")
    
    if "q_vs_cosine_correlation" in results:
        corr = results["q_vs_cosine_correlation"]
        print(f"  Q vs Cosine similarity correlation: {corr:.4f}")
    
    if "p_ab_match_rate" in results:
        rate = results["p_ab_match_rate"]
        matches = results["p_ab_matches"]
        total = results["p_ab_total"]
        print(f"  P(a|max-Q) > P(a|min-Q): {rate*100:.1f}% ({matches}/{total})")
        if rate > 0.7:
            print(f"    → Q-function IS learning P(a|b) correctly!")
        elif rate < 0.5:
            print(f"    → [WARNING] Q-function NOT learning P(a|b) correctly!")


if __name__ == "__main__":
    # This can be called standalone for testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate training dynamics")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--episode", type=int, default=0, help="Episode number")
    parser.add_argument("--n-examples", type=int, default=50, help="Number of examples")
    
    args = parser.parse_args()
    
    # Load config and model (similar to analyze_q_values.py)
    from src.configs.base_config import BaseConfig
    from src.utils.llm_utils import get_model_instance, get_tokenizer_instance
    
    config = BaseConfig.from_json(args.config)
    model = get_model_instance(
        model_name=config.model_name,
        device=config.device,
        use_bf16=config.use_bf16,
    )
    tokenizer = get_tokenizer_instance(config.model_name)
    
    value_function = ValueFunction(
        model=model,
        hidden_size=4096,
        learning_rate=config.learning_rate,
        device=config.device,
        dtype=torch.bfloat16 if config.use_bf16 else torch.float32,
    )
    value_function.load_checkpoint(args.checkpoint, strict=False, load_optimizer=False)
    
    results = evaluate_training_dynamics(
        value_function=value_function,
        config=config,
        model=model,
        tokenizer=tokenizer,
        device=config.device,
        standard_dtype=torch.bfloat16 if config.use_bf16 else torch.float32,
        n_examples=args.n_examples,
        episode_number=args.episode,
        output_dir=config.output_dir
    )
    
    print_training_dynamics_summary(results)

