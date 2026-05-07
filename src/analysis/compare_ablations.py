"""Compare metrics across contrastive loss ablation experiments."""

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError(
        "matplotlib is required. Install it with `pip install matplotlib`."
    )


# Experiment configurations
EXPERIMENT_CONFIGS = {
    "crossed_data": {
        "name": "Main Method (Crossed Data)",
        "color": "#1f77b4",  # Blue
        "contrastive_coef": 1.0,
        "ablation_mode": "crossed_data"
    },
    "crossed_data_in_candidates": {
        "name": "Crossed Data in Candidates",
        "color": "#9467bd",  # Purple
        "contrastive_coef": 0.0,
        "ablation_mode": "crossed_data_in_candidates"
    },
    "random_noise": {
        "name": "Ablation 1 (Random Noise)",
        "color": "#ff7f0e",  # Orange
        "contrastive_coef": 1.0,
        "ablation_mode": "random_noise"
    },
    "noise_in_candidates": {
        "name": "Ablation 2 (Noise in Candidates)",
        "color": "#2ca02c",  # Green
        "contrastive_coef": 0.0,
        "ablation_mode": "noise_in_candidates"
    },
    "none": {
        "name": "Baseline (No Contrastive Loss)",
        "color": "#d62728",  # Red
        "contrastive_coef": 0.0,
        "ablation_mode": "none"
    }
}


def load_csv_metrics(csv_dir: Path, min_episode: int = 500) -> List[Dict]:
    """Load and aggregate metrics from belief_evaluation_episode_*.csv files.
    
    Handles duplicate episode numbers by keeping the most recently modified file.
    This prevents contamination from old data when CSV files are overwritten.
    """
    csv_files = list(csv_dir.glob("belief_evaluation_episode_*.csv"))
    if not csv_files:
        return []
    
    # Sort by modification time (most recent first) to handle duplicates
    # If same episode number appears in multiple files, keep the most recent one
    csv_files_with_episodes = []
    for csv_file in csv_files:
        match = re.search(r"episode_(\d+)\.csv", csv_file.name)
        if not match:
            continue
        episode_num = int(match.group(1))
        if episode_num < min_episode:
            continue
        csv_files_with_episodes.append((episode_num, csv_file.stat().st_mtime, csv_file))
    
    # Group by episode number to handle duplicates
    episodes_to_files: Dict[int, List[Tuple[float, Path]]] = {}
    for episode_num, mtime, csv_file in csv_files_with_episodes:
        if episode_num not in episodes_to_files:
            episodes_to_files[episode_num] = []
        episodes_to_files[episode_num].append((mtime, csv_file))
    
    # For each episode, keep only the most recent file
    episode_metrics: Dict[int, Dict] = {}
    duplicate_warnings = []
    
    for episode_num, files_list in sorted(episodes_to_files.items()):
        # Sort by modification time (most recent first)
        files_list.sort(key=lambda x: -x[0])
        
        # If there are duplicates, warn and keep only the most recent
        if len(files_list) > 1:
            kept_file = files_list[0][1]
            skipped_files = [f[1] for f in files_list[1:]]
            duplicate_warnings.append(
                f"Duplicate episode {episode_num}: keeping {kept_file.name} "
                f"(most recent, mtime: {files_list[0][0]:.0f}), "
                f"skipping {len(skipped_files)} older file(s)"
            )
        
        # Use the most recent file
        mtime, csv_file = files_list[0]
        
        cosine_sims = []
        min_q_cosine_sims = []
        random_cosine_sims = []
        contrastive_correct = []  # Max-Q Q-value > Random Q-value
        
        def is_noise_candidate(candidate: str) -> bool:
            """Check if a candidate is random noise."""
            return isinstance(candidate, str) and candidate.strip().startswith("[RANDOM_NOISE]")
        
        try:
            with csv_file.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    try:
                        cosine_sim = float(row.get("cosine_similarity", 0.0))
                        cosine_sims.append(cosine_sim)
                        
                        # Filter out noise candidates from min-Q computation
                        # (for noise_in_candidates ablation, min_q_belief might be noise)
                        min_q_belief_text = row.get("min_q_belief", "")
                        min_q_value = row.get("min_q_cosine_similarity")
                        if min_q_value not in (None, "", "nan") and not is_noise_candidate(min_q_belief_text):
                            min_q_cosine_sims.append(float(min_q_value))
                        
                        random_value = row.get("random_cosine_similarity")
                        if random_value not in (None, "", "nan"):
                            random_cosine_sims.append(float(random_value))
                        
                        # Contrastive accuracy: Max-Q Q-value should be > Random Q-value
                        max_q_val_str = row.get("max_q_value", "")
                        random_q_val_str = row.get("random_q_value", "")
                        if (max_q_val_str not in (None, "", "nan") and 
                            random_q_val_str not in (None, "", "nan")):
                            try:
                                max_q_val = float(max_q_val_str)
                                random_q_val = float(random_q_val_str)
                                contrastive_correct.append(1.0 if max_q_val > random_q_val else 0.0)
                            except (ValueError, TypeError):
                                pass
                    except (ValueError, KeyError):
                        continue
            
            if cosine_sims:
                episode_entry = {
                    "episode": episode_num,
                    "avg_cosine_similarity": mean(cosine_sims),
                    "std_cosine_similarity": stdev(cosine_sims) if len(cosine_sims) > 1 else 0.0,
                    "num_turns": len(cosine_sims),
                }
                
                if min_q_cosine_sims:
                    episode_entry["avg_min_q_cosine_similarity"] = mean(min_q_cosine_sims)
                if random_cosine_sims:
                    episode_entry["avg_random_cosine_similarity"] = mean(random_cosine_sims)
                if contrastive_correct:
                    episode_entry["contrastive_accuracy"] = mean(contrastive_correct)
                    episode_entry["num_contrastive_samples"] = len(contrastive_correct)
                
                episode_metrics[episode_num] = episode_entry
        except Exception as exc:
            print(f"Warning: Failed to load {csv_file}: {exc}")
            continue
    
    # Print warnings about duplicates
    if duplicate_warnings:
        print(f"Warning: Found {len(duplicate_warnings)} duplicate episode numbers in CSV files:")
        for warning in duplicate_warnings[:10]:  # Limit to first 10 warnings
            print(f"  {warning}")
        if len(duplicate_warnings) > 10:
            print(f"  ... and {len(duplicate_warnings) - 10} more duplicates")
        print("  Keeping most recently modified files. Consider cleaning old CSV files.")
    
    # Sort by episode number and check for gaps
    sorted_metrics = sorted(episode_metrics.values(), key=lambda x: x["episode"])
    
    # Check for non-sequential episodes (gaps)
    if len(sorted_metrics) > 1:
        episodes = [m["episode"] for m in sorted_metrics]
        gaps = []
        for i in range(len(episodes) - 1):
            if episodes[i+1] - episodes[i] > 1:
                gaps.append((episodes[i], episodes[i+1]))
        if gaps:
            print(f"Note: Found {len(gaps)} gaps in episode sequence (non-sequential episodes):")
            for start, end in gaps[:5]:  # Show first 5 gaps
                print(f"  Gap: {start} -> {end} (missing {end - start - 1} episodes)")
            if len(gaps) > 5:
                print(f"  ... and {len(gaps) - 5} more gaps")
    
    return sorted_metrics


def load_baseline_reports(report_dir: Path) -> List[Dict]:
    """Load baseline evaluation reports."""
    reports = []
    for report_path in sorted(report_dir.glob("baseline_eval_*.json")):
        try:
            with report_path.open("r") as handle:
                reports.append(json.load(handle))
        except Exception as exc:
            print(f"Warning: Failed to load {report_path}: {exc}")
            continue
    return reports


def load_training_metrics(metrics_file: Path) -> List[Dict]:
    """Load training metrics from JSONL file.
    
    Note: Training metrics (reward, value_loss, entropy, belief_accuracy) are logged
    during training via log_metrics() and written to all_metrics.jsonl. These are
    per-turn metrics, not the same as the evaluation metrics in CSV files.
    
    If this file doesn't exist, you need to re-run training to generate it.
    """
    if not metrics_file.exists():
        print(f"Warning: Training metrics file not found: {metrics_file}")
        print("  Training metrics are written during training via log_metrics().")
        print("  You need to re-run training to generate all_metrics.jsonl files.")
        print("  The CSV files (belief_evaluation_episode_*.csv) contain evaluation")
        print("  metrics (cosine similarity, Q-values) but not training metrics.")
        return []
    
    metrics = []
    with metrics_file.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                metrics.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    if not metrics:
        print(f"Warning: Training metrics file is empty: {metrics_file}")
    
    return metrics


def try_extract_episode_stats_from_rollouts(rollout_file: Path) -> List[Dict]:
    """Attempt to extract episode-level statistics from rollouts file.
    
    This is a fallback that tries to reconstruct some training metrics from
    all_rollouts.jsonl, but it will only have episode-level data (not per-turn).
    Note: Rollouts don't contain reward/value_loss, so this won't fully work.
    """
    if not rollout_file.exists():
        return []
    
    episode_stats = []
    try:
        with rollout_file.open("r") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    episode_idx = entry.get("episode_idx")
                    rollout = entry.get("rollout", [])
                    
                    if episode_idx is not None and rollout:
                        # Try to extract entropy from belief states if available
                        # This is very limited - we don't have reward or value_loss
                        entropies = []
                        for turn_data in rollout:
                            # Rollout structure varies, try to find entropy or belief probabilities
                            if isinstance(turn_data, dict):
                                # Check if there's entropy directly
                                if "entropy" in turn_data:
                                    entropies.append(turn_data["entropy"])
                                # Or try to compute from belief probabilities
                                elif "belief_probabilities" in turn_data:
                                    probs = turn_data["belief_probabilities"]
                                    if probs:
                                        from ..belief import compute_entropy
                                        entropies.append(compute_entropy(probs))
                        
                        if entropies:
                            avg_entropy = mean(entropies)
                            episode_stats.append({
                                "episode_idx": episode_idx,
                                "entropy": avg_entropy,
                                # Can't recover reward or value_loss from rollouts
                                "reward": None,
                                "value_loss": None,
                                "belief_accuracy": None
                            })
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
    except Exception as exc:
        print(f"Warning: Failed to extract stats from rollouts: {exc}")
    
    return episode_stats


def aggregate_training_metrics(metrics: List[Dict], window_size: int = 200) -> List[Dict]:
    """Aggregate training metrics with rolling average."""
    if not metrics:
        return []
    
    aggregated = []
    for i in range(len(metrics)):
        start_idx = max(0, i - window_size + 1)
        window = metrics[start_idx:i + 1]
        
        if window:
            # Helper to safely compute mean, filtering out None values
            # If key is missing, use default; if key exists but is None, filter it out
            def safe_mean(key: str, default: float = 0.0) -> float:
                values = []
                for m in window:
                    if key in m:
                        # Key exists - use its value if not None
                        val = m.get(key)
                        if val is not None:
                            values.append(val)
                    else:
                        # Key doesn't exist - use default
                        values.append(default)
                return mean(values) if values else default
            
            # For belief_accuracy, return None if no valid values exist
            belief_acc_values = [
                m.get("belief_accuracy") for m in window
                if m.get("belief_accuracy") is not None
            ]
            belief_accuracy = mean(belief_acc_values) if belief_acc_values else None
            
            aggregated.append({
                "step": metrics[i].get("episode_idx", i),
                "reward": safe_mean("reward", 0.0),
                "value_loss": safe_mean("value_loss", 0.0),
                "entropy": safe_mean("entropy", 0.0),
                "belief_accuracy": belief_accuracy
            })
    
    return aggregated


def compute_data_ylim(all_values: List[float], padding: float = 0.1, min_range: float = 0.01) -> Tuple[float, float]:
    """Compute Y-axis limits based on data distribution.
    
    Args:
        all_values: List of all valid data values across all experiments
        padding: Fraction of range to add as padding (default: 0.1 = 10%)
        min_range: Minimum range to use if data is too flat (default: 0.01)
    
    Returns:
        Tuple of (ymin, ymax)
    """
    valid_values = [v for v in all_values if not math.isnan(v) and v is not None]
    if not valid_values:
        return (0.0, 1.0)  # Default range if no data
    
    vmin = min(valid_values)
    vmax = max(valid_values)
    
    # If all values are the same, create a small range around that value
    if vmin == vmax:
        center = vmin
        if center == 0.0:
            ymin, ymax = -min_range / 2, min_range / 2
        else:
            ymin = center - abs(center) * min_range
            ymax = center + abs(center) * min_range
    else:
        # Add padding
        data_range = vmax - vmin
        padding_amount = data_range * padding
        ymin = vmin - padding_amount
        ymax = vmax + padding_amount
        
        # Ensure minimum range
        if ymax - ymin < min_range:
            center = (vmin + vmax) / 2
            ymin = center - min_range / 2
            ymax = center + min_range / 2
    
    # For metrics that should be >= 0, clamp bottom
    if ymin < 0.0:
        ymin = 0.0
    
    return (ymin, ymax)


def plot_cosine_similarity_comparison(
    experiment_data: Dict[str, Dict],
    output_dir: Path
) -> None:
    """Plot cosine similarity comparison across experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    cosine_series = [
        ("avg_cosine_similarity", "Max-Q vs Ground Truth", 0),
        ("avg_min_q_cosine_similarity", "Min-Q vs Ground Truth", 1),
        ("avg_random_cosine_similarity", "Random vs Ground Truth", 2),
    ]
    
    for key, label, ax_idx in cosine_series:
        ax = axes[ax_idx]
        all_values = []  # Collect all values for this metric
        
        for exp_key, exp_info in EXPERIMENT_CONFIGS.items():
            if exp_key not in experiment_data:
                continue
            
            csv_metrics = experiment_data[exp_key].get("csv_metrics", [])
            if not csv_metrics:
                continue
            
            episodes = [m["episode"] for m in csv_metrics]
            cosine_values = [
                m.get(key, math.nan)
                for m in csv_metrics
            ]
            
            # Filter out NaN values
            valid_data = [(ep, val) for ep, val in zip(episodes, cosine_values) if not math.isnan(val)]
            if valid_data:
                eps, vals = zip(*valid_data)
                all_values.extend(vals)  # Collect for Y-axis computation
                ax.plot(
                    eps, vals,
                    marker="o",
                    label=exp_info["name"],
                    color=exp_info["color"],
                    linewidth=2,
                    markersize=4,
                    alpha=0.8
                )
        
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Episode (Evaluation Checkpoint)", fontsize=10)
        ax.set_ylabel("Cosine Similarity", fontsize=10)
        # Set Y-axis limits based on data distribution
        ymin, ymax = compute_data_ylim(all_values, padding=0.1)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, linestyle="--", alpha=0.4)
        if ax_idx == 0:  # Only show legend on first subplot
            ax.legend(loc="best", fontsize=8)
    
    # Fourth subplot: All three together
    ax = axes[3]
    all_max_q_values = []  # Collect all values for Y-axis computation
    for exp_key, exp_info in EXPERIMENT_CONFIGS.items():
        if exp_key not in experiment_data:
            continue
        
        csv_metrics = experiment_data[exp_key].get("csv_metrics", [])
        if not csv_metrics:
            continue
        
        episodes = [m["episode"] for m in csv_metrics]
        max_q_values = [
            m.get("avg_cosine_similarity", math.nan)
            for m in csv_metrics
        ]
        
        valid_data = [(ep, val) for ep, val in zip(episodes, max_q_values) if not math.isnan(val)]
        if valid_data:
            eps, vals = zip(*valid_data)
            all_max_q_values.extend(vals)  # Collect for Y-axis computation
            ax.plot(
                eps, vals,
                marker="o",
                label=exp_info["name"],
                color=exp_info["color"],
                linewidth=2,
                markersize=4,
                alpha=0.8
            )
    
    ax.set_title("Max-Q Cosine Similarity (All Methods)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Episode (Evaluation Checkpoint)", fontsize=10)
    ax.set_ylabel("Cosine Similarity", fontsize=10)
    # Set Y-axis limits based on data distribution
    ymin, ymax = compute_data_ylim(all_max_q_values, padding=0.1)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=8)
    
    fig.suptitle("Cosine Similarity Comparison: Belief Selection vs Ground Truth", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "cosine_similarity_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved cosine similarity comparison to {output_dir / 'cosine_similarity_comparison.png'}")


def plot_contrastive_accuracy_comparison(
    experiment_data: Dict[str, Dict],
    output_dir: Path
) -> None:
    """Plot contrastive accuracy comparison across experiments."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    all_contrastive_accs = []  # Collect all values for Y-axis computation
    for exp_key, exp_info in EXPERIMENT_CONFIGS.items():
        if exp_key not in experiment_data:
            continue
        
        csv_metrics = experiment_data[exp_key].get("csv_metrics", [])
        if not csv_metrics:
            continue
        
        episodes = [m["episode"] for m in csv_metrics]
        contrastive_accs = [
            m.get("contrastive_accuracy", math.nan)
            for m in csv_metrics
        ]
        
        # Filter out NaN values
        valid_data = [(ep, val) for ep, val in zip(episodes, contrastive_accs) if not math.isnan(val)]
        if valid_data:
            eps, vals = zip(*valid_data)
            all_contrastive_accs.extend(vals)  # Collect for Y-axis computation
            ax.plot(
                eps, vals,
                marker="o",
                label=exp_info["name"],
                color=exp_info["color"],
                linewidth=2,
                markersize=4,
                alpha=0.8
            )
    
    ax.set_xlabel("Episode (Evaluation Checkpoint)", fontsize=12)
    ax.set_ylabel("Contrastive Accuracy", fontsize=12)
    ax.set_title("Contrastive Accuracy: Max-Q Q-value > Random Q-value", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=10)
    # Set Y-axis limits based on data distribution, but clamp to [0, 1] for accuracy
    ymin, ymax = compute_data_ylim(all_contrastive_accs, padding=0.1)
    ymin = max(0.0, ymin)  # Clamp to 0
    ymax = min(1.0, ymax)  # Clamp to 1
    ax.set_ylim(ymin, ymax)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Baseline (0.5)')
    
    fig.tight_layout()
    fig.savefig(output_dir / "contrastive_accuracy_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved contrastive accuracy comparison to {output_dir / 'contrastive_accuracy_comparison.png'}")


def plot_baseline_accuracy_comparison(
    experiment_data: Dict[str, Dict],
    output_dir: Path
) -> None:
    """Plot baseline accuracy comparison across experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    baseline_series = [
        ("max_q_accuracy", "Max-Q Accuracy", 0),
        ("min_q_accuracy", "Min-Q Accuracy", 1),
        ("random_accuracy", "Random Accuracy", 2),
        ("top_prob_accuracy", "Top-Prob Accuracy", 3),
    ]
    
    for key, label, ax_idx in baseline_series:
        ax = axes[ax_idx]
        all_values = []  # Collect all values for this metric
        
        for exp_key, exp_info in EXPERIMENT_CONFIGS.items():
            if exp_key not in experiment_data:
                continue
            
            baseline_reports = experiment_data[exp_key].get("baseline_reports", [])
            if not baseline_reports:
                continue
            
            episodes = [r["episodes_completed"] for r in baseline_reports]
            values = [
                r.get("averages", {}).get(key, math.nan)
                for r in baseline_reports
            ]
            
            valid_data = [(ep, val) for ep, val in zip(episodes, values) if not math.isnan(val)]
            if valid_data:
                eps, vals = zip(*valid_data)
                all_values.extend(vals)  # Collect for Y-axis computation
                ax.plot(
                    eps, vals,
                    marker="o",
                    label=exp_info["name"],
                    color=exp_info["color"],
                    linewidth=2,
                    markersize=4,
                    alpha=0.8
                )
        
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Episodes Completed", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        # Set Y-axis limits based on data distribution, but clamp to [0, 1] for accuracy
        ymin, ymax = compute_data_ylim(all_values, padding=0.1)
        ymin = max(0.0, ymin)  # Clamp to 0
        ymax = min(1.0, ymax)  # Clamp to 1
        ax.set_ylim(ymin, ymax)
        ax.grid(True, linestyle="--", alpha=0.4)
        if ax_idx == 0:  # Only show legend on first subplot
            ax.legend(loc="best", fontsize=8)
    
    fig.suptitle("Baseline Accuracy Comparison Across Ablations", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "baseline_accuracy_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved baseline accuracy comparison to {output_dir / 'baseline_accuracy_comparison.png'}")


def plot_training_metrics_comparison(
    experiment_data: Dict[str, Dict],
    output_dir: Path
) -> None:
    """Plot training metrics comparison across experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    training_series = [
        ("reward", "Average Reward", 0),
        ("value_loss", "Value Loss", 1),
        ("entropy", "Belief Entropy", 2),
        ("belief_accuracy", "Belief Accuracy", 3),
    ]
    
    # Track which experiments have data for debugging
    has_data_per_metric = {key: [] for key, _, _ in training_series}
    
    for key, label, ax_idx in training_series:
        ax = axes[ax_idx]
        all_values = []  # Collect all values for this metric
        
        for exp_key, exp_info in EXPERIMENT_CONFIGS.items():
            if exp_key not in experiment_data:
                print(f"Warning: {exp_key} not found in experiment_data")
                continue
            
            aggregated = experiment_data[exp_key].get("training_metrics", [])
            if not aggregated:
                print(f"Warning: No training metrics found for {exp_key}")
                continue
            
            steps = [m.get("step", i) for i, m in enumerate(aggregated)]
            values = [m.get(key) for m in aggregated]
            
            # Filter out None values
            valid_data = [(step, val) for step, val in zip(steps, values) if val is not None]
            if valid_data:
                steps_valid, vals_valid = zip(*valid_data)
                all_values.extend(vals_valid)  # Collect for Y-axis computation
                ax.plot(
                    steps_valid, vals_valid,
                    label=exp_info["name"],
                    color=exp_info["color"],
                    linewidth=1.5,
                    alpha=0.8
                )
                has_data_per_metric[key].append(exp_key)
            else:
                # Only warn for required metrics; belief_accuracy is optional (requires ground truth)
                if key != "belief_accuracy":
                print(f"Warning: No valid data points for {exp_key} metric {key} (all values are None or missing)")
        
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Episode", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        # Set Y-axis limits based on data distribution
        # For belief_accuracy, clamp to [0, 1] if it's an accuracy metric
        ymin, ymax = compute_data_ylim(all_values, padding=0.1)
        if key == "belief_accuracy":
            ymin = max(0.0, ymin)  # Clamp to 0
            ymax = min(1.0, ymax)  # Clamp to 1
        elif key in ["reward", "value_loss", "entropy"]:
            # For these metrics, ensure bottom is at least 0 if all values are >= 0
            if all(v >= 0 for v in all_values if not math.isnan(v)):
                ymin = max(0.0, ymin)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, linestyle="--", alpha=0.4)
        if ax_idx == 0:  # Only show legend on first subplot
            ax.legend(loc="best", fontsize=8)
    
    # Check if any plots have data
    total_plots_with_data = sum(len(exps) > 0 for exps in has_data_per_metric.values())
    if total_plots_with_data == 0:
        print("ERROR: No training metrics data found for any experiment! Plots will be empty.")
        print("  Check that all_metrics.jsonl files exist in experiment directories.")
        for exp_key in EXPERIMENT_CONFIGS.keys():
            if exp_key in experiment_data:
                metrics_count = len(experiment_data[exp_key].get("training_metrics", []))
                print(f"  {exp_key}: {metrics_count} aggregated training metrics")
    
    # Note about belief_accuracy if missing
    if len(has_data_per_metric["belief_accuracy"]) == 0:
        print("Note: belief_accuracy is missing for all experiments. This is expected if ground truth goals")
        print("  are not available during training. belief_accuracy is only computed when ground truth exists.")
    
    fig.suptitle("Training Metrics Comparison Across Ablations", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "training_metrics_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training metrics comparison to {output_dir / 'training_metrics_comparison.png'}")


def generate_summary_table(
    experiment_data: Dict[str, Dict],
    output_dir: Path
) -> None:
    """Generate a summary table comparing final metrics across experiments."""
    summary = {
        "experiments": {},
        "comparison": {}
    }
    
    for exp_key, exp_info in EXPERIMENT_CONFIGS.items():
        if exp_key not in experiment_data:
            continue
        
        exp_data = experiment_data[exp_key]
        exp_summary = {
            "name": exp_info["name"],
            "config": {
                "contrastive_coef": exp_info["contrastive_coef"],
                "ablation_mode": exp_info["ablation_mode"]
            }
        }
        
        # Final cosine similarity and contrastive accuracy
        csv_metrics = exp_data.get("csv_metrics", [])
        if csv_metrics:
            final_cosine = csv_metrics[-1].get("avg_cosine_similarity")
            exp_summary["final_cosine_similarity"] = final_cosine
            final_contrastive = csv_metrics[-1].get("contrastive_accuracy")
            if final_contrastive is not None:
                exp_summary["final_contrastive_accuracy"] = final_contrastive
        
        # Final baseline accuracies
        baseline_reports = exp_data.get("baseline_reports", [])
        if baseline_reports:
            final_report = baseline_reports[-1]
            exp_summary["final_baseline_accuracies"] = final_report.get("averages", {})
            exp_summary["episodes_completed"] = final_report.get("episodes_completed")
            exp_summary["turns_evaluated"] = final_report.get("turns_evaluated")
        
        # Training metrics summary
        training_metrics = exp_data.get("training_metrics", [])
        if training_metrics:
            final_training = training_metrics[-1]
            exp_summary["final_training_metrics"] = {
                "reward": final_training.get("reward"),
                "value_loss": final_training.get("value_loss"),
                "entropy": final_training.get("entropy"),
                "belief_accuracy": final_training.get("belief_accuracy")
            }
        
        summary["experiments"][exp_key] = exp_summary
    
    # Save summary
    summary_path = output_dir / "ablation_comparison_summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    
    print(f"Saved summary to {summary_path}")
    
    # Print summary table
    print("\n" + "=" * 100)
    print("ABLATION COMPARISON SUMMARY")
    print("=" * 100)
    print(f"{'Method':<35} {'Cosine Sim':<12} {'Contrastive Acc':<16} {'Max-Q Acc':<12} {'Min-Q Acc':<12} {'Random Acc':<12}")
    print("-" * 100)
    
    for exp_key, exp_info in EXPERIMENT_CONFIGS.items():
        if exp_key not in summary["experiments"]:
            continue
        
        exp_summary = summary["experiments"][exp_key]
        name = exp_info["name"]
        cosine = exp_summary.get("final_cosine_similarity", "N/A")
        contrastive = exp_summary.get("final_contrastive_accuracy", "N/A")
        accuracies = exp_summary.get("final_baseline_accuracies", {})
        
        max_q_acc = accuracies.get("max_q_accuracy", "N/A")
        min_q_acc = accuracies.get("min_q_accuracy", "N/A")
        random_acc = accuracies.get("random_accuracy", "N/A")
        
        if isinstance(cosine, float):
            cosine_str = f"{cosine:.4f}"
        else:
            cosine_str = str(cosine)
        
        if isinstance(contrastive, float):
            contrastive_str = f"{contrastive:.4f}"
        else:
            contrastive_str = str(contrastive)
        
        if isinstance(max_q_acc, float):
            max_q_acc = f"{max_q_acc:.4f}"
        else:
            max_q_acc = str(max_q_acc)
        
        if isinstance(min_q_acc, float):
            min_q_acc = f"{min_q_acc:.4f}"
        else:
            min_q_acc = str(min_q_acc)
        
        if isinstance(random_acc, float):
            random_acc = f"{random_acc:.4f}"
        else:
            random_acc = str(random_acc)
        
        print(f"{name:<35} {cosine_str:<12} {contrastive_str:<16} {max_q_acc:<12} {min_q_acc:<12} {random_acc:<12}")
    
    print("=" * 100 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare metrics across contrastive loss ablation experiments."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("outputs/llm_context_belief"),
        help="Base directory containing experiment subdirectories."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/llm_context_belief/comparison"),
        help="Directory to save comparison plots and summary."
    )
    parser.add_argument(
        "--min-episode",
        type=int,
        default=500,
        help="Minimum episode number to include from CSV files (default: 500)."
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=200,
        help="Rolling window size for training metrics (default: 200)."
    )
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data from all experiments
    experiment_data: Dict[str, Dict] = {}
    
    for exp_key in EXPERIMENT_CONFIGS.keys():
        exp_dir = args.base_dir / exp_key
        
        if not exp_dir.exists():
            print(f"Warning: Experiment directory not found: {exp_dir}")
            continue
        
        print(f"Loading data from {exp_key}...")
        
        # Load CSV metrics
        csv_metrics = load_csv_metrics(exp_dir, min_episode=args.min_episode)
        
        # Load baseline reports
        baseline_reports = load_baseline_reports(exp_dir)
        
        # Load training metrics
        metrics_file = exp_dir / "all_metrics.jsonl"
        training_metrics_raw = load_training_metrics(metrics_file)
        training_metrics = aggregate_training_metrics(training_metrics_raw, window_size=args.window_size)
        
        experiment_data[exp_key] = {
            "csv_metrics": csv_metrics,
            "baseline_reports": baseline_reports,
            "training_metrics": training_metrics
        }
        
        print(f"  - CSV metrics: {len(csv_metrics)} episodes")
        print(f"  - Baseline reports: {len(baseline_reports)} reports")
        print(f"  - Training metrics: {len(training_metrics)} aggregated points")
        if len(training_metrics) == 0 and not metrics_file.exists():
            print(f"    ⚠️  Missing all_metrics.jsonl - training metrics plots will be empty!")
            print(f"    ⚠️  Re-run training to generate this file.")
    
    if not experiment_data:
        print("Error: No experiment data found!")
        return
    
    # Generate plots
    print("\nGenerating comparison plots...")
    plot_cosine_similarity_comparison(experiment_data, args.output_dir)
    plot_contrastive_accuracy_comparison(experiment_data, args.output_dir)
    plot_baseline_accuracy_comparison(experiment_data, args.output_dir)
    plot_training_metrics_comparison(experiment_data, args.output_dir)
    
    # Generate summary
    print("\nGenerating summary...")
    generate_summary_table(experiment_data, args.output_dir)
    
    print(f"\nComparison complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()

