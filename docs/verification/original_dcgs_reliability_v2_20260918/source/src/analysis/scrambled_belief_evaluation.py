"""Scrambled belief evaluation for Q-function.

This evaluation tests whether the Q-function can distinguish correct beliefs
from incorrect (scrambled) ones, and measures actual task performance.

For each evaluation episode:
1. Take first step using 5 benchmark states in parallel
2. Generate candidates for each state
3. Scramble 4 candidates per state across other states, mark one non-shuffled as b*
4. Measure Q-values and observe P(Q(b*) > Q(b_other))
5. Step conversation using highest ranked belief
6. Compute reward: 0.99^(num_turns) if all goals fulfilled, else 0
7. Report mean score across episodes
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

import torch

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents import HighLevelAgent, LowLevelAgent, UserAgent
from src.configs.base_config import BaseConfig
from src.data import format_multiwoz_goal, load_multiwoz_dataset
from src.data.dialogue_formatter import extract_goal_from_dialogue_state, infer_goal_from_dialogue
from src.main import (
    batch_generate_beliefs_for_episodes,
    create_episode_state,
    is_valid_multiwoz_goal,
    process_turn_for_episode,
)
from src.prompts.prompt_manager import PromptManager
from src.utils.llm_utils import get_model_instance, get_tokenizer_instance, get_user_model_instance
from src.value.value_function import ValueFunction


def check_goals_fulfilled(episode) -> bool:
    """Check if all goals are fulfilled for an episode."""
    if hasattr(episode, 'goal_achieved'):
        return episode.goal_achieved >= 1.0
    if hasattr(episode, 'env_info') and episode.env_info:
        return episode.env_info.get('goal_achieved', False)
    if hasattr(episode, 'done_from_env'):
        return episode.done_from_env
    return False


def scramble_candidates(
    benchmark_states: List,
    n_scramble: int = 4
) -> List[Dict]:
    """
    Scramble candidates across benchmark states.
    
    For each state, keep one candidate as b* (non-shuffled), and replace
    n_scramble candidates with candidates from other states.
    
    Args:
        benchmark_states: List of benchmark states (episodes with generated candidates)
        n_scramble: Number of candidates to scramble per state
    
    Returns:
        List of dicts with keys: 'state_idx', 'original_candidates', 'scrambled_candidates', 'b_star_idx'
    """
    scrambled_configs = []
    
    # Collect all candidates from all states
    all_candidates_pool = []
    for state_idx, state in enumerate(benchmark_states):
        if state.belief_state.candidates:
            candidates = [c.summary for c in state.belief_state.candidates if c.summary != "[SKIP]"]
            all_candidates_pool.extend([(state_idx, cand) for cand in candidates])
    
    # For each state, scramble candidates
    for state_idx, state in enumerate(benchmark_states):
        if not state.belief_state.candidates:
            continue
        
        original_candidates = [c.summary for c in state.belief_state.candidates if c.summary != "[SKIP]"]
        if len(original_candidates) < n_scramble + 1:
            # Not enough candidates to scramble
            continue
        
        # Randomly select one candidate to keep as b*
        b_star_idx = random.randint(0, len(original_candidates) - 1)
        b_star = original_candidates[b_star_idx]
        
        # Get candidates from other states
        other_state_candidates = [
            (other_idx, cand) for other_idx, cand in all_candidates_pool
            if other_idx != state_idx
        ]
        
        if len(other_state_candidates) < n_scramble:
            # Not enough candidates from other states
            continue
        
        # Randomly sample n_scramble candidates from other states
        scrambled_candidates = random.sample(other_state_candidates, n_scramble)
        scrambled_candidate_texts = [cand for _, cand in scrambled_candidates]
        
        # Build scrambled candidate list: keep b* at its position, replace others
        scrambled_list = original_candidates.copy()
        replace_indices = [i for i in range(len(original_candidates)) if i != b_star_idx]
        replace_indices = random.sample(replace_indices, min(n_scramble, len(replace_indices)))
        
        for idx, scrambled_cand in zip(replace_indices, scrambled_candidate_texts):
            scrambled_list[idx] = scrambled_cand
        
        scrambled_configs.append({
            'state_idx': state_idx,
            'original_candidates': original_candidates,
            'scrambled_candidates': scrambled_list,
            'b_star': b_star,
            'b_star_idx': b_star_idx,
        })
    
    return scrambled_configs


def evaluate_with_scrambled_beliefs(
    checkpoint_path: Path,
    config: BaseConfig,
    dialogues: List,
    n_episodes: int,
    n_benchmark_states: int = 5,
    n_scramble: int = 4,
    model=None,
    tokenizer=None,
    device: str = "cuda",
    standard_dtype: torch.dtype = torch.bfloat16,
    user_agent=None,
) -> Dict:
    """
    Evaluate Q-function with scrambled belief test.
    
    Args:
        checkpoint_path: Path to checkpoint
        config: Config object
        dialogues: List of dialogues
        n_episodes: Number of evaluation episodes
        n_benchmark_states: Number of benchmark states to use in parallel
        n_scramble: Number of candidates to scramble per state
        model: Model instance
        tokenizer: Tokenizer instance
        device: Device
        standard_dtype: Data type
        user_agent: User agent (for online mode)
    
    Returns:
        Dictionary with evaluation results
    """
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
    
    # Sample evaluation episodes
    if len(dialogues) < n_episodes:
        eval_dialogues = dialogues
    else:
        eval_dialogues = random.sample(dialogues, n_episodes)
    
    print(f"Evaluating {len(eval_dialogues)} episodes with scrambled belief test...")
    
    all_episode_scores = []
    all_q_ranking_accuracies = []
    
    for episode_idx, dialogue in enumerate(eval_dialogues):
        print(f"\nEpisode {episode_idx + 1}/{len(eval_dialogues)}: {dialogue.dialogue_id}")
        
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
            print(f"  Skipping: invalid goal")
            continue
        
        # Create main evaluation episode
        main_episode = create_episode_state(
            dialogue_idx=episode_idx,
            dialogue_data=dialogue,
            multiwoz_mode=True,
            config=config,
            debug=False,
            user_agent=user_agent if config.environment_type == "multiwoz_online" else None,
        )
        
        if not main_episode:
            print(f"  Skipping: failed to create episode")
            continue
        
        # Initialize main episode to get base state
        main_episode.initialize()
        if hasattr(main_episode, 'env') and main_episode.env:
            obs, info = main_episode.env.reset()
            main_episode.initial_observation = obs
            main_episode.initial_env_info = info
            main_episode.observation = obs
            main_episode.current_obs_for_action = obs
        
        # Create n_benchmark_states benchmark states, all using the same base state
        # (same dialogue, same initial observation)
        benchmark_states = []
        base_observation = main_episode.current_obs_for_action
        base_env_info = main_episode.initial_env_info
        
        for bench_idx in range(n_benchmark_states):
            bench_episode = create_episode_state(
                dialogue_idx=episode_idx * 1000 + bench_idx,  # Unique index
                dialogue_data=dialogue,  # Use same dialogue for all benchmark states
                multiwoz_mode=True,
                config=config,
                debug=False,
                user_agent=user_agent if config.environment_type == "multiwoz_online" else None,
            )
            if bench_episode:
                bench_episode.initialize()
                # Set to same base state as main episode
                bench_episode.initial_observation = base_observation
                bench_episode.initial_env_info = base_env_info
                bench_episode.observation = base_observation
                bench_episode.current_obs_for_action = base_observation
                benchmark_states.append(bench_episode)
        
        if len(benchmark_states) < n_benchmark_states:
            print(f"  Skipping: only {len(benchmark_states)}/{n_benchmark_states} benchmark states created")
            continue
        
        # Generate candidates for all benchmark states in parallel
        print(f"  Generating candidates for {len(benchmark_states)} benchmark states...")
        batch_generate_beliefs_for_episodes(
            benchmark_states,
            hl_agent,
            config,
            ground_truth_pool=None,
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        
        # Scramble candidates
        scrambled_configs = scramble_candidates(benchmark_states, n_scramble=n_scramble)
        
        if not scrambled_configs:
            print(f"  Skipping: failed to scramble candidates")
            continue
        
        # For each scrambled config, compute Q-values and check ranking
        q_ranking_correct = 0
        q_ranking_total = 0
        
        for scrambled_config in scrambled_configs:
            state_idx = scrambled_config['state_idx']
            scrambled_candidates = scrambled_config['scrambled_candidates']
            b_star = scrambled_config['b_star']
            
            # Get observation from the benchmark state
            bench_state = benchmark_states[state_idx]
            observation = bench_state.current_obs_for_action or bench_state.observation
            
            # Compute Q-values for scrambled candidates
            observations = [observation] * len(scrambled_candidates)
            q_values_tensor = value_function.predict_q_value(
                observations=observations,
                high_level_actions=scrambled_candidates,
                tokenizer=tokenizer,
                requires_grad=False
            )
            
            q_values = {
                cand: float(q_val.item())
                for cand, q_val in zip(scrambled_candidates, q_values_tensor)
            }
            
            # Check if Q(b*) > Q(b_other) for all other candidates
            b_star_q = q_values.get(b_star)
            if b_star_q is not None:
                other_q_values = [q for cand, q in q_values.items() if cand != b_star]
                if other_q_values:
                    if b_star_q > max(other_q_values):
                        q_ranking_correct += 1
                    q_ranking_total += 1
        
        if q_ranking_total > 0:
            q_ranking_accuracy = q_ranking_correct / q_ranking_total
            all_q_ranking_accuracies.append(q_ranking_accuracy)
            print(f"  Q-ranking accuracy: {q_ranking_accuracy:.4f} ({q_ranking_correct}/{q_ranking_total})")
        
        # Now run the main episode using highest ranked belief
        # (main_episode is already initialized and reset above)
        
        # Run episode until completion or max turns
        num_turns = 0
        max_turns = config.max_turns
        
        while num_turns < max_turns and not main_episode.done_from_env:
            # Generate beliefs
            batch_generate_beliefs_for_episodes(
                [main_episode],
                hl_agent,
                config,
                ground_truth_pool=None,
                model=model,
                tokenizer=tokenizer,
                device=device
            )
            
            # Compute Q-values and select highest ranked belief
            if main_episode.belief_state.candidates:
                candidate_summaries = [c.summary for c in main_episode.belief_state.candidates if c.summary != "[SKIP]"]
                if candidate_summaries:
                    observations = [main_episode.current_obs_for_action] * len(candidate_summaries)
                    q_values_tensor = value_function.predict_q_value(
                        observations=observations,
                        high_level_actions=candidate_summaries,
                        tokenizer=tokenizer,
                        requires_grad=False
                    )
                    q_values = {
                        cand: float(q_val.item())
                        for cand, q_val in zip(candidate_summaries, q_values_tensor)
                    }
                    selected_belief = max(q_values, key=q_values.get)
                    main_episode._preselected_belief = selected_belief
            
            # Process turn
            if not process_turn_for_episode(
                episode=main_episode,
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
            
            num_turns += 1
            
            if main_episode.done_from_env:
                break
        
        # Compute reward: 0.99^(num_turns) if goals fulfilled, else 0
        goals_fulfilled = check_goals_fulfilled(main_episode)
        if goals_fulfilled:
            episode_score = 0.99 ** num_turns
        else:
            episode_score = 0.0
        
        all_episode_scores.append(episode_score)
        print(f"  Episode score: {episode_score:.4f} (turns: {num_turns}, goals fulfilled: {goals_fulfilled})")
    
    # Compute mean scores
    mean_score = mean(all_episode_scores) if all_episode_scores else 0.0
    mean_q_ranking_accuracy = mean(all_q_ranking_accuracies) if all_q_ranking_accuracies else 0.0
    
    results = {
        "n_episodes": len(all_episode_scores),
        "mean_score": mean_score,
        "mean_q_ranking_accuracy": mean_q_ranking_accuracy,
        "episode_scores": all_episode_scores,
        "q_ranking_accuracies": all_q_ranking_accuracies,
    }
    
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Q-function with scrambled belief test"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config JSON file",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes (default: 50)",
    )
    parser.add_argument(
        "--n-benchmark-states",
        type=int,
        default=5,
        help="Number of benchmark states to use in parallel (default: 5)",
    )
    parser.add_argument(
        "--n-scramble",
        type=int,
        default=4,
        help="Number of candidates to scramble per state (default: 4)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/scrambled_belief_evaluation.json"),
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load config
    config = BaseConfig.from_json(str(args.config))
    config._evaluation_mode = True  # Disable gradient updates
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = get_model_instance(
        model_name=config.model_name,
        device=config.device,
        use_bf16=config.use_bf16,
    )
    tokenizer = get_tokenizer_instance(config.model_name)
    
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
        user_agent = UserAgent(user_model, tokenizer, PromptManager())
        print("User agent initialized for online mode")
    
    standard_dtype = torch.bfloat16 if config.use_bf16 else torch.float32
    
    # Load dataset
    print("Loading dataset...")
    dialogues = load_multiwoz_dataset(
        data_path=config.data_path,
        split="train",
        use_huggingface=True,
    )
    
    # Filter valid dialogues
    print(f"Filtering {len(dialogues)} dialogues for valid goals...")
    valid_dialogues = []
    for d in dialogues:
        if not is_valid_multiwoz_goal(d.goal):
            goal_state = extract_goal_from_dialogue_state(d)
            if goal_state:
                d.goal = goal_state
            else:
                inferred_goal = infer_goal_from_dialogue(d)
                if inferred_goal:
                    d.goal = inferred_goal
        
        if is_valid_multiwoz_goal(d.goal):
            valid_dialogues.append(d)
    
    print(f"Loaded {len(valid_dialogues)} valid dialogues")
    
    # Run evaluation
    results = evaluate_with_scrambled_beliefs(
        checkpoint_path=args.checkpoint,
        config=config,
        dialogues=valid_dialogues,
        n_episodes=args.n_episodes,
        n_benchmark_states=args.n_benchmark_states,
        n_scramble=args.n_scramble,
        model=model,
        tokenizer=tokenizer,
        device=config.device,
        standard_dtype=standard_dtype,
        user_agent=user_agent,
    )
    
    # Print summary
    print("\n" + "="*80)
    print("SCRAMBLED BELIEF EVALUATION RESULTS")
    print("="*80)
    print(f"Episodes evaluated: {results['n_episodes']}")
    print(f"Mean score: {results['mean_score']:.4f}")
    print(f"Mean Q-ranking accuracy: {results['mean_q_ranking_accuracy']:.4f}")
    print("="*80)
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

