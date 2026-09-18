"""Analyze cosine similarity patterns from belief evaluation CSVs to identify failure modes."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, List


def load_csv_metrics(csv_path: Path) -> List[Dict]:
    """Load metrics from a belief evaluation CSV file."""
    metrics = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                row['cosine_similarity'] = float(row.get('cosine_similarity', 0.0))
                row['turn'] = int(row.get('turn', 0))
                row['context_length'] = int(row.get('context_length', 0))
                metrics.append(row)
            except (ValueError, KeyError) as e:
                continue
    return metrics


def analyze_patterns(metrics: List[Dict], low_cosine_threshold: float = 0.1) -> Dict:
    """Analyze patterns in cosine similarity metrics."""
    if not metrics:
        return {}
    
    all_cosines = [m['cosine_similarity'] for m in metrics]
    low_cosine_cases = [m for m in metrics if m['cosine_similarity'] < low_cosine_threshold]
    
    # Group by turn number
    turn_groups: Dict[int, List[Dict]] = defaultdict(list)
    for m in metrics:
        turn_groups[m['turn']].append(m)
    
    # Group by context length
    context_groups: Dict[int, List[Dict]] = defaultdict(list)
    for m in metrics:
        ctx_len = m.get('context_length', 0)
        context_groups[ctx_len].append(m)
    
    # Group by episode
    episode_groups: Dict[str, List[Dict]] = defaultdict(list)
    for m in metrics:
        ep_id = f"{m.get('dialogue_id', 'unknown')}_{m.get('episode_id', 'unknown')}"
        episode_groups[ep_id].append(m)
    
    # Analyze [SKIP] beliefs
    skip_cases = [m for m in metrics if m.get('chosen_belief', '') == '[SKIP]']
    
    # Find worst cases
    worst_cases = sorted(metrics, key=lambda x: x['cosine_similarity'])[:10]
    
    analysis = {
        'summary': {
            'total_turns': len(metrics),
            'low_cosine_count': len(low_cosine_cases),
            'low_cosine_percentage': 100 * len(low_cosine_cases) / len(metrics) if metrics else 0,
            'mean_cosine': mean(all_cosines),
            'median_cosine': median(all_cosines),
            'min_cosine': min(all_cosines),
            'max_cosine': max(all_cosines),
            'std_cosine': stdev(all_cosines) if len(all_cosines) > 1 else 0.0,
            'skip_count': len(skip_cases),
            'skip_percentage': 100 * len(skip_cases) / len(metrics) if metrics else 0,
        },
        'by_turn': {},
        'by_context_length': {},
        'worst_cases': [],
        'low_cosine_episodes': []
    }
    
    # Analyze by turn
    for turn_num in sorted(turn_groups.keys()):
        turn_metrics = turn_groups[turn_num]
        turn_cosines = [m['cosine_similarity'] for m in turn_metrics]
        low_count = sum(1 for c in turn_cosines if c < low_cosine_threshold)
        analysis['by_turn'][turn_num] = {
            'count': len(turn_metrics),
            'mean_cosine': mean(turn_cosines),
            'median_cosine': median(turn_cosines),
            'low_cosine_count': low_count,
            'low_cosine_percentage': 100 * low_count / len(turn_metrics) if turn_metrics else 0,
        }
    
    # Analyze by context length
    for ctx_len in sorted(context_groups.keys()):
        ctx_metrics = context_groups[ctx_len]
        ctx_cosines = [m['cosine_similarity'] for m in ctx_metrics]
        low_count = sum(1 for c in ctx_cosines if c < low_cosine_threshold)
        analysis['by_context_length'][ctx_len] = {
            'count': len(ctx_metrics),
            'mean_cosine': mean(ctx_cosines),
            'median_cosine': median(ctx_cosines),
            'low_cosine_count': low_count,
            'low_cosine_percentage': 100 * low_count / len(ctx_metrics) if ctx_metrics else 0,
        }
    
    # Worst cases
    for case in worst_cases:
        analysis['worst_cases'].append({
            'dialogue_id': case.get('dialogue_id', 'unknown'),
            'episode_id': case.get('episode_id', 'unknown'),
            'turn': case['turn'],
            'context_length': case.get('context_length', 0),
            'cosine_similarity': case['cosine_similarity'],
            'chosen_belief': case.get('chosen_belief', '')[:100],
            'ground_truth': case.get('ground_truth_text', '')[:100],
            'is_skip': case.get('chosen_belief', '') == '[SKIP]',
        })
    
    # Episodes with high low-cosine rate
    for ep_id, ep_metrics in episode_groups.items():
        ep_cosines = [m['cosine_similarity'] for m in ep_metrics]
        low_count = sum(1 for c in ep_cosines if c < low_cosine_threshold)
        if low_count >= len(ep_metrics) * 0.5:  # More than 50% low cosine
            analysis['low_cosine_episodes'].append({
                'episode_id': ep_id,
                'total_turns': len(ep_metrics),
                'low_cosine_count': low_count,
                'mean_cosine': mean(ep_cosines),
                'dialogue_id': ep_metrics[0].get('dialogue_id', 'unknown'),
            })
    
    return analysis


def print_analysis_report(analysis: Dict, low_cosine_threshold: float = 0.1):
    """Print a human-readable analysis report."""
    summary = analysis['summary']
    
    print("\n" + "="*80)
    print("COSINE SIMILARITY PATTERN ANALYSIS")
    print("="*80)
    print(f"\nSummary Statistics:")
    print(f"  Total turns evaluated: {summary['total_turns']}")
    print(f"  Low cosine cases (cos < {low_cosine_threshold}): {summary['low_cosine_count']} ({summary['low_cosine_percentage']:.1f}%)")
    print(f"  [SKIP] beliefs: {summary['skip_count']} ({summary['skip_percentage']:.1f}%)")
    print(f"\nCosine Similarity Distribution:")
    print(f"  Mean: {summary['mean_cosine']:.4f}")
    print(f"  Median: {summary['median_cosine']:.4f}")
    print(f"  Min: {summary['min_cosine']:.4f}")
    print(f"  Max: {summary['max_cosine']:.4f}")
    print(f"  Std Dev: {summary['std_cosine']:.4f}")
    
    print(f"\nCosine Similarity by Turn Number:")
    for turn_num in sorted(analysis['by_turn'].keys()):
        turn_data = analysis['by_turn'][turn_num]
        print(f"  Turn {turn_num}: mean={turn_data['mean_cosine']:.4f}, "
              f"low_cos={turn_data['low_cosine_count']}/{turn_data['count']} "
              f"({turn_data['low_cosine_percentage']:.1f}%)")
    
    print(f"\nCosine Similarity by Context Length:")
    for ctx_len in sorted(analysis['by_context_length'].keys()):
        ctx_data = analysis['by_context_length'][ctx_len]
        print(f"  Context len {ctx_len}: mean={ctx_data['mean_cosine']:.4f}, "
              f"low_cos={ctx_data['low_cosine_count']}/{ctx_data['count']} "
              f"({ctx_data['low_cosine_percentage']:.1f}%)")
    
    if analysis['worst_cases']:
        print(f"\nWorst 10 Cases (lowest cosine similarity):")
        for idx, case in enumerate(analysis['worst_cases'], 1):
            print(f"\n  {idx}. Dialogue {case['dialogue_id']}, Turn {case['turn']} "
                  f"(context_len={case['context_length']}, cos={case['cosine_similarity']:.4f})")
            if case['is_skip']:
                print(f"     [SKIP] placeholder belief")
            else:
                belief_text = case['chosen_belief'][:100] + ('...' if len(case['chosen_belief']) > 100 else '')
                print(f"     Belief: '{belief_text}'")
            gt_text = case['ground_truth'][:100] + ('...' if len(case['ground_truth']) > 100 else '')
            print(f"     Ground Truth: '{gt_text}'")
    
    if analysis['low_cosine_episodes']:
        print(f"\nEpisodes with High Low-Cosine Rate (>=50%):")
        for ep in analysis['low_cosine_episodes'][:10]:
            print(f"  {ep['episode_id']}: {ep['low_cosine_count']}/{ep['total_turns']} turns "
                  f"(mean_cos={ep['mean_cosine']:.4f})")
    
    print("\n" + "="*80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze cosine similarity patterns from belief evaluation CSVs."
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        required=True,
        help="Directory containing belief_evaluation_episode_*.csv files.",
    )
    parser.add_argument(
        "--low-cosine-threshold",
        type=float,
        default=0.1,
        help="Threshold for low cosine similarity (default: 0.1).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional: Save analysis as JSON file.",
    )
    parser.add_argument(
        "--episode-filter",
        type=int,
        default=None,
        help="Optional: Analyze only a specific episode number.",
    )
    args = parser.parse_args()
    
    # Find CSV files
    if args.episode_filter:
        csv_files = [args.csv_dir / f"belief_evaluation_episode_{args.episode_filter}.csv"]
        csv_files = [f for f in csv_files if f.exists()]
    else:
        csv_files = sorted(args.csv_dir.glob("belief_evaluation_episode_*.csv"))
    
    if not csv_files:
        print(f"ERROR: No CSV files found in {args.csv_dir}")
        return
    
    print(f"Loading metrics from {len(csv_files)} CSV file(s)...")
    
    # Load all metrics
    all_metrics = []
    for csv_file in csv_files:
        metrics = load_csv_metrics(csv_file)
        all_metrics.extend(metrics)
        print(f"  Loaded {len(metrics)} turns from {csv_file.name}")
    
    if not all_metrics:
        print("ERROR: No metrics loaded from CSV files")
        return
    
    # Analyze patterns
    analysis = analyze_patterns(all_metrics, low_cosine_threshold=args.low_cosine_threshold)
    
    # Print report
    print_analysis_report(analysis, low_cosine_threshold=args.low_cosine_threshold)
    
    # Save JSON if requested
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as f:
            json.dump(analysis, f, indent=2)
        print(f"Analysis saved to {args.output_json}")


if __name__ == "__main__":
    main()

