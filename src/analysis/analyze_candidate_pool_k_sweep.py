"""Analyze candidate-pool K sweep metrics from all_metrics.jsonl logs."""

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List


SUPPORTED_DATASETS = ("wildjailbreak", "cares", "harmbench", "redbench")


def _infer_dataset_from_path(path: Path) -> str:
    path_s = str(path)
    for dataset in SUPPORTED_DATASETS:
        if dataset in path_s:
            return dataset
    raise ValueError(f"Unable to infer dataset from metrics path: {path}")


def _mean_confidence_interval_95(values: List[float]) -> Dict[str, float]:
    mu = mean(values)
    sigma = pstdev(values)
    stderr = sigma / (len(values) ** 0.5)
    half_width = 1.96 * stderr
    return {
        "mean": mu,
        "ci95_low": mu - half_width,
        "ci95_high": mu + half_width,
    }


def _load_turn_metrics(metrics_path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            turn_evaluation = payload["turn_evaluation"]
            candidate_pool_q_metrics = turn_evaluation["candidate_pool_q_metrics"]
            if not candidate_pool_q_metrics:
                continue

            rows.append(
                {
                    "dataset": _infer_dataset_from_path(metrics_path),
                    "k": int(candidate_pool_q_metrics["candidate_pool_size_total"]),
                    "q_min": float(candidate_pool_q_metrics["q_min"]),
                    "q_max": float(candidate_pool_q_metrics["q_max"]),
                    "q_gap": float(candidate_pool_q_metrics["q_gap"]),
                    "expected_q_baseline_pi": float(candidate_pool_q_metrics["expected_q_baseline_pi"]),
                    "expected_q_guided_pi_k": float(candidate_pool_q_metrics["expected_q_guided_pi_k"]),
                    "expected_q_delta": float(
                        candidate_pool_q_metrics["expected_q_delta_guided_minus_baseline"]
                    ),
                }
            )
    return rows


def _aggregate_by_dataset_and_k(turn_rows: List[Dict]) -> Dict[str, Dict[int, Dict]]:
    grouped: Dict[str, Dict[int, List[Dict]]] = {}
    for row in turn_rows:
        dataset_rows = grouped.setdefault(row["dataset"], {})
        k_rows = dataset_rows.setdefault(row["k"], [])
        k_rows.append(row)

    summary: Dict[str, Dict[int, Dict]] = {}
    for dataset, per_k_rows in grouped.items():
        summary[dataset] = {}
        for k, rows in sorted(per_k_rows.items()):
            q_min_values = [row["q_min"] for row in rows]
            q_max_values = [row["q_max"] for row in rows]
            q_gap_values = [row["q_gap"] for row in rows]
            baseline_values = [row["expected_q_baseline_pi"] for row in rows]
            guided_values = [row["expected_q_guided_pi_k"] for row in rows]
            delta_values = [row["expected_q_delta"] for row in rows]
            summary[dataset][k] = {
                "num_turns": len(rows),
                "q_min": _mean_confidence_interval_95(q_min_values),
                "q_max": _mean_confidence_interval_95(q_max_values),
                "q_gap": _mean_confidence_interval_95(q_gap_values),
                "expected_q_baseline_pi": _mean_confidence_interval_95(baseline_values),
                "expected_q_guided_pi_k": _mean_confidence_interval_95(guided_values),
                "expected_q_delta": _mean_confidence_interval_95(delta_values),
            }
    return summary


def _compute_monotonicity(summary: Dict[str, Dict[int, Dict]]) -> Dict[str, Dict]:
    monotonicity: Dict[str, Dict] = {}
    for dataset, per_k in summary.items():
        ks = sorted(per_k.keys())
        guided_means = [per_k[k]["expected_q_guided_pi_k"]["mean"] for k in ks]
        gap_means = [per_k[k]["q_gap"]["mean"] for k in ks]

        guided_monotonic_non_decreasing = all(
            guided_means[idx + 1] >= guided_means[idx] for idx in range(len(guided_means) - 1)
        )
        gap_monotonic_non_increasing = all(
            gap_means[idx + 1] <= gap_means[idx] for idx in range(len(gap_means) - 1)
        )
        monotonicity[dataset] = {
            "k_values": ks,
            "guided_means": guided_means,
            "q_gap_means": gap_means,
            "guided_monotonic_non_decreasing": guided_monotonic_non_decreasing,
            "q_gap_monotonic_non_increasing": gap_monotonic_non_increasing,
        }
    return monotonicity


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze candidate-pool K sweep outputs.")
    parser.add_argument(
        "--outputs-root",
        type=str,
        default="outputs",
        help="Root directory containing run outputs.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/candidate_pool_k_sweep_summary.json",
        help="Destination JSON for aggregated K-sweep metrics.",
    )
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    metrics_files = sorted(outputs_root.glob("**/all_metrics.jsonl"))
    if not metrics_files:
        raise FileNotFoundError(f"No all_metrics.jsonl files found under: {outputs_root}")

    all_rows: List[Dict] = []
    for metrics_file in metrics_files:
        all_rows.extend(_load_turn_metrics(metrics_file))
    if not all_rows:
        raise RuntimeError("No candidate_pool_q_metrics rows found in provided metrics logs.")

    summary = _aggregate_by_dataset_and_k(all_rows)
    monotonicity = _compute_monotonicity(summary)

    result = {
        "outputs_root": str(outputs_root),
        "num_metric_files": len(metrics_files),
        "num_turn_rows": len(all_rows),
        "summary_by_dataset_k": summary,
        "monotonicity_checks": monotonicity,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(f"Wrote K-sweep summary to {output_path}")
    print(f"Metric files scanned: {len(metrics_files)}")
    print(f"Turn rows analyzed: {len(all_rows)}")


if __name__ == "__main__":
    main()
