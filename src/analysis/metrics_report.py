"""Generate visual summaries from offline evaluation metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev, variance
from typing import Dict, List


def _load_metrics(metrics_path: Path) -> List[Dict]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    
    metrics: List[Dict] = []
    with metrics_path.open("r") as handle:
        for line_num, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse JSON on line {line_num} of {metrics_path}: {exc}"
                ) from exc
            metrics.append(record)
    
    if not metrics:
        raise ValueError(f"No metric entries found in {metrics_path}")
    
    return metrics


def _rolling_average(records: List[Dict], window: int) -> List[Dict]:
    if window < 1:
        raise ValueError("Window size must be >= 1")
    
    # Adaptive window: if we have fewer records than window, use smaller chunks
    if len(records) < window:
        window = max(1, len(records) // 10)  # Use 10% of data as window
    
    aggregated: List[Dict] = []
    for idx in range(0, len(records), max(1, window // 4)):  # Overlap windows for smoother plots
        chunk = records[idx : idx + window]
        if not chunk:
            continue
        def _safe_float(key: str) -> float:
            values = [
                float(entry.get(key, 0.0)) for entry in chunk if entry.get(key) is not None
            ]
            return mean(values) if values else 0.0
        
        # Use episode_idx from last entry, or fallback to index
        last_episode_idx = chunk[-1].get("episode_idx")
        if last_episode_idx is None:
            last_episode_idx = idx + len(chunk) - 1
        
        aggregated.append(
            {
                "step": last_episode_idx,
                "reward": _safe_float("reward"),
                "value_loss": _safe_float("value_loss"),
                "entropy": _safe_float("entropy"),
                "belief_accuracy": _safe_float("belief_accuracy"),
            }
        )
    return aggregated


def _load_csv_metrics(csv_dir: Path, min_episode: int = 500) -> List[Dict]:
    """Load and aggregate metrics from belief_evaluation_episode_*.csv files.
    
    Args:
        csv_dir: Directory containing CSV files
        min_episode: Minimum episode number to include (default: 500)
    """
    csv_files = sorted(csv_dir.glob("belief_evaluation_episode_*.csv"))
    if not csv_files:
        return []
    
    episode_metrics: Dict[int, Dict] = {}
    
    for csv_file in csv_files:
        # Extract episode number from filename
        match = re.search(r"episode_(\d+)\.csv", csv_file.name)
        if not match:
            continue
        episode_num = int(match.group(1))
        
        # Skip episodes before min_episode
        if episode_num < min_episode:
            continue
        
        cosine_sims = []
        min_q_cosine_sims = []
        random_cosine_sims = []
        
        try:
            with csv_file.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    try:
                        cosine_sim = float(row.get("cosine_similarity", 0.0))
                        cosine_sims.append(cosine_sim)
                        
                        min_q_value = row.get("min_q_cosine_similarity")
                        if min_q_value not in (None, "", "nan"):
                            min_q_cosine_sims.append(float(min_q_value))
                        
                        random_value = row.get("random_cosine_similarity")
                        if random_value not in (None, "", "nan"):
                            random_cosine_sims.append(float(random_value))
                    except (ValueError, KeyError):
                        continue
            
            if cosine_sims:
                # Calculate mean and coefficient of variation (CV = std/mean)
                # This gives a normalized measure that scales with the mean
                cosine_mean = mean(cosine_sims)
                
                # Calculate CV: std/mean (normalized measure)
                cosine_std = stdev(cosine_sims) if len(cosine_sims) > 1 else 0.0
                
                cosine_cv = cosine_std / cosine_mean if cosine_mean > 0 else 0.0
                
                episode_entry = {
                    "episode": episode_num,
                    "avg_cosine_similarity": cosine_mean,
                    "cv_cosine_similarity": cosine_cv,
                    "num_turns": len(cosine_sims),
                }
                
                if min_q_cosine_sims:
                    episode_entry["avg_min_q_cosine_similarity"] = mean(min_q_cosine_sims)
                if random_cosine_sims:
                    episode_entry["avg_random_cosine_similarity"] = mean(random_cosine_sims)
                
                episode_metrics[episode_num] = episode_entry
        except Exception as exc:
            print(f"Warning: Failed to load {csv_file}: {exc}")
            continue
    
    # Convert to sorted list
    return sorted(episode_metrics.values(), key=lambda x: x["episode"])


def _load_baseline_reports(report_dir: Path) -> List[Dict]:
    reports: List[Dict] = []
    for report_path in sorted(report_dir.glob("baseline_eval_*.json")):
        with report_path.open("r") as handle:
            reports.append(json.load(handle))
    return reports


def _plot_metrics(
    aggregated: List[Dict],
    baseline_reports: List[Dict],
    csv_metrics: List[Dict],
    output_dir: Path
) -> None:
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required to generate figures. Install it with `pip install matplotlib`."
        ) from exc
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Core training metrics from JSONL
    if aggregated:
        steps = [entry["step"] if entry["step"] is not None else idx for idx, entry in enumerate(aggregated)]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        axes = axes.flatten()
        
        axes[0].plot(steps, [entry["reward"] for entry in aggregated], label="Reward", color="#1f77b4", marker="o", markersize=3)
        axes[0].set_ylabel("Avg Reward")
        axes[0].set_title("Reward (rolling)")
        axes[0].grid(True, linestyle="--", alpha=0.4)
        
        axes[1].plot(steps, [entry["value_loss"] for entry in aggregated], label="Value Loss", color="#ff7f0e", marker="o", markersize=3)
        axes[1].set_ylabel("Avg Value Loss")
        axes[1].set_title("Value Loss (rolling)")
        axes[1].grid(True, linestyle="--", alpha=0.4)
        
        axes[2].plot(steps, [entry["entropy"] for entry in aggregated], label="Entropy", color="#2ca02c", marker="o", markersize=3)
        axes[2].set_ylabel("Avg Entropy")
        axes[2].set_xlabel("Episode Index")
        axes[2].set_title("Belief Entropy (rolling)")
        axes[2].grid(True, linestyle="--", alpha=0.4)
        
        axes[3].plot(steps, [entry["belief_accuracy"] for entry in aggregated], label="Belief Accuracy", color="#d62728", marker="o", markersize=3)
        axes[3].set_ylabel("Avg Belief Accuracy")
        axes[3].set_xlabel("Episode Index")
        axes[3].set_title("Belief Accuracy (rolling)")
        axes[3].grid(True, linestyle="--", alpha=0.4)
        
        fig.suptitle("Offline Training Metrics")
        fig.tight_layout()
        fig.savefig(output_dir / "offline_metrics_overview.png", dpi=200)
        plt.close(fig)
    else:
        print("Warning: No aggregated metrics to plot (JSONL data may be empty or window too large)")
    
    # Plot 2: Q-function improvement from CSV files (cosine similarity)
    if csv_metrics:
        csv_fig, csv_ax = plt.subplots(1, 1, figsize=(7, 5))
        
        episodes = [m["episode"] for m in csv_metrics]
        cosine_series = [
            ("avg_cosine_similarity", "Max-Q (policy)", "#1f77b4"),
            ("avg_min_q_cosine_similarity", "Min-Q", "#ff7f0e"),
            ("avg_random_cosine_similarity", "Random", "#2ca02c"),
        ]
        
        for key, label, color in cosine_series:
            values = [
                m.get(key, math.nan) if m.get(key) is not None else math.nan
                for m in csv_metrics
            ]
            if all(math.isnan(value) for value in values):
                continue
            csv_ax.plot(episodes, values, marker="o", color=color, label=label, linewidth=2, markersize=6)
        
        csv_ax.set_xlabel("Episode (Evaluation Checkpoint)")
        csv_ax.set_ylabel("Cosine Similarity")
        csv_ax.set_title("Q-Function Efficacy: Belief Selection vs Ground Truth\n(Cosine Similarity)")
        csv_ax.grid(True, linestyle="--", alpha=0.4)
        csv_ax.legend()
        
        csv_fig.tight_layout()
        csv_fig.savefig(output_dir / "q_function_improvement.png", dpi=200)
        plt.close(csv_fig)
    
    # Plot 3: Baseline comparison
    if baseline_reports:
        baseline_fig, baseline_ax = plt.subplots(figsize=(10, 4))
        report_steps = [report["episodes_completed"] for report in baseline_reports]
        baseline_series = [
            ("max_q_accuracy", "Max-Q", "#1f77b4"),
            ("min_q_accuracy", "Min-Q", "#ff7f0e"),
            ("random_accuracy", "Random", "#2ca02c"),
        ]
        for key, label, color in baseline_series:
            series = [
                report.get("averages", {}).get(key, math.nan) for report in baseline_reports
            ]
            if all(math.isnan(value) for value in series):
                continue
            baseline_ax.plot(report_steps, series, marker="o", label=label, color=color, linewidth=2)
        
        baseline_ax.set_title("Baseline Comparison: Max-Q vs Min-Q vs Random")
        baseline_ax.set_xlabel("Episodes Completed")
        baseline_ax.set_ylabel("Accuracy")
        baseline_ax.set_ylim(0.0, 1.0)
        baseline_ax.grid(True, linestyle="--", alpha=0.4)
        baseline_ax.legend()
        baseline_fig.tight_layout()
        baseline_fig.savefig(output_dir / "baseline_comparison.png", dpi=200)
        plt.close(baseline_fig)


def _summarize_to_json(aggregated: List[Dict], output_dir: Path) -> None:
    summary = {
        "num_points": len(aggregated),
        "reward": {
            "mean": mean(entry["reward"] for entry in aggregated),
        },
        "value_loss": {
            "mean": mean(entry["value_loss"] for entry in aggregated),
        },
        "entropy": {
            "mean": mean(entry["entropy"] for entry in aggregated),
        },
        "belief_accuracy": {
            "mean": mean(entry["belief_accuracy"] for entry in aggregated),
        },
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "metrics_summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots from offline metrics JSONL file.")
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=Path("outputs/llm_context_belief/all_metrics.jsonl"),
        help="Path to all_metrics.jsonl file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/llm_context_belief/figures"),
        help="Directory to store generated figures.",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=None,
        help="Directory containing belief_evaluation_episode_*.csv files. If not provided, uses metrics-file parent.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=200,
        help="Rolling window size for smoothing metrics (adaptive if data is sparse).",
    )
    parser.add_argument(
        "--min-episode",
        type=int,
        default=500,
        help="Minimum episode number to include from CSV evaluation files (default: 500).",
    )
    args = parser.parse_args()
    
    # Load JSONL metrics
    metrics = []
    if args.metrics_file.exists():
        metrics = _load_metrics(args.metrics_file)
    else:
        print(f"Warning: Metrics file not found: {args.metrics_file}")
    
    aggregated = _rolling_average(metrics, args.window_size) if metrics else []
    
    # Load CSV metrics from belief evaluation files
    csv_dir = args.csv_dir if args.csv_dir else args.metrics_file.parent
    csv_metrics = _load_csv_metrics(csv_dir, min_episode=args.min_episode)
    
    # Load baseline reports
    baseline_reports = _load_baseline_reports(args.metrics_file.parent)
    
    # Generate plots
    _plot_metrics(aggregated, baseline_reports, csv_metrics, args.output_dir)
    
    # Generate summary
    if aggregated:
        _summarize_to_json(aggregated, args.output_dir)
    
    print(
        f"[metrics_report] Generated figures in {args.output_dir} "
        f"using {len(metrics)} JSONL entries, {len(csv_metrics)} CSV checkpoints, "
        f"and {len(baseline_reports)} baseline reports."
    )


if __name__ == "__main__":
    main()


