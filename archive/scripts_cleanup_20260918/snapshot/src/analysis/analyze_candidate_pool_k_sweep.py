"""Analyze candidate-pool K sweep metrics from all_metrics.jsonl logs."""

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional


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

            pool_total = int(candidate_pool_q_metrics["candidate_pool_size_total"])
            pool_valid = int(candidate_pool_q_metrics["candidate_pool_size_valid"])
            skip_rate = 1.0 - (pool_valid / pool_total)

            rows.append(
                {
                    "dataset": _infer_dataset_from_path(metrics_path),
                    "k": pool_total,
                    "pool_valid": pool_valid,
                    "skip_rate": skip_rate,
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
            skip_rate_values = [row["skip_rate"] for row in rows]
            pool_valid_values = [float(row["pool_valid"]) for row in rows]
            summary[dataset][k] = {
                "num_turns": len(rows),
                "candidate_pool_size_valid": _mean_confidence_interval_95(pool_valid_values),
                "skip_rate": _mean_confidence_interval_95(skip_rate_values),
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
        delta_means = [per_k[k]["expected_q_delta"]["mean"] for k in ks]
        skip_rate_means = [per_k[k]["skip_rate"]["mean"] for k in ks]

        guided_monotonic_non_decreasing = all(
            guided_means[idx + 1] >= guided_means[idx] for idx in range(len(guided_means) - 1)
        )
        gap_monotonic_non_decreasing = all(
            gap_means[idx + 1] >= gap_means[idx] for idx in range(len(gap_means) - 1)
        )
        delta_monotonic_non_decreasing = all(
            delta_means[idx + 1] >= delta_means[idx] for idx in range(len(delta_means) - 1)
        )
        skip_rate_monotonic_non_decreasing = all(
            skip_rate_means[idx + 1] >= skip_rate_means[idx]
            for idx in range(len(skip_rate_means) - 1)
        )

        confounded_k_pairs = []
        for idx in range(len(ks) - 1):
            k_lo, k_hi = ks[idx], ks[idx + 1]
            guided_up = guided_means[idx + 1] > guided_means[idx]
            skip_up = skip_rate_means[idx + 1] > skip_rate_means[idx]
            if guided_up and skip_up:
                confounded_k_pairs.append(
                    {
                        "k_lo": k_lo,
                        "k_hi": k_hi,
                        "guided_delta": guided_means[idx + 1] - guided_means[idx],
                        "skip_rate_delta": skip_rate_means[idx + 1] - skip_rate_means[idx],
                    }
                )

        monotonicity[dataset] = {
            "k_values": ks,
            "guided_means": guided_means,
            "q_gap_means": gap_means,
            "expected_q_delta_means": delta_means,
            "skip_rate_means": skip_rate_means,
            "guided_monotonic_non_decreasing": guided_monotonic_non_decreasing,
            "q_gap_monotonic_non_decreasing": gap_monotonic_non_decreasing,
            "expected_q_delta_monotonic_non_decreasing": delta_monotonic_non_decreasing,
            "skip_rate_monotonic_non_decreasing": skip_rate_monotonic_non_decreasing,
            "confounded_k_pairs_guided_up_and_skip_up": confounded_k_pairs,
        }
    return monotonicity


def _resolve_metrics_files(outputs_root: Path, metrics_glob: Optional[str]) -> List[Path]:
    if metrics_glob:
        return sorted(Path(".").glob(metrics_glob))
    return sorted(outputs_root.glob("**/all_metrics.jsonl"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze candidate-pool K sweep outputs.")
    parser.add_argument(
        "--outputs-root",
        type=str,
        default="outputs",
        help="Root directory containing run outputs (used when --metrics-glob is not set).",
    )
    parser.add_argument(
        "--metrics-glob",
        type=str,
        default=None,
        help="Glob for all_metrics.jsonl files (e.g. outputs/*_hierarchical_freeform_iterative_k_sweep/**/all_metrics.jsonl).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/candidate_pool_k_sweep_summary.json",
        help="Destination JSON for aggregated K-sweep metrics.",
    )
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    metrics_files = _resolve_metrics_files(outputs_root, args.metrics_glob)
    if not metrics_files:
        glob_desc = args.metrics_glob or f"{outputs_root}/**/all_metrics.jsonl"
        raise FileNotFoundError(f"No all_metrics.jsonl files found matching: {glob_desc}")

    all_rows: List[Dict] = []
    for metrics_file in metrics_files:
        all_rows.extend(_load_turn_metrics(metrics_file))
    if not all_rows:
        raise RuntimeError("No candidate_pool_q_metrics rows found in provided metrics logs.")

    summary = _aggregate_by_dataset_and_k(all_rows)
    monotonicity = _compute_monotonicity(summary)

    result = {
        "outputs_root": str(outputs_root),
        "metrics_glob": args.metrics_glob,
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
