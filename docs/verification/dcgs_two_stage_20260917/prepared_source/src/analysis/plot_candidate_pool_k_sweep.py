"""Bar-chart plots for candidate-pool K sweep using environment reward and ASR."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError(
        "matplotlib is required. Install it with `pip install matplotlib`."
    ) from exc


SWEEP_DIR_SUFFIX = "_hierarchical_freeform_iterative_k_sweep"
K_VALUES = (1, 3, 5, 10, 15)
MODE_ID = "critic_guided"


def _mean_confidence_interval_95(values: List[float]) -> Dict[str, float]:
    mu = mean(values)
    sigma = pstdev(values)
    stderr = sigma / (len(values) ** 0.5)
    half_width = 1.96 * stderr
    return {
        "mean": mu,
        "ci95_low": mu - half_width,
        "ci95_high": mu + half_width,
        "ci95_half_width": half_width,
    }


def _binomial_confidence_interval_95(successes: int, total: int) -> Dict[str, float]:
    rate = successes / total
    stderr = math.sqrt(rate * (1.0 - rate) / total)
    half_width = 1.96 * stderr
    return {
        "mean": rate,
        "ci95_low": max(0.0, rate - half_width),
        "ci95_high": min(1.0, rate + half_width),
        "ci95_half_width": half_width,
    }


def _parse_run_dir(run_dir: Path) -> Tuple[str, str, str, int]:
    parts = run_dir.parts
    sweep_dirs = [part for part in parts if part.endswith(SWEEP_DIR_SUFFIX)]
    if len(sweep_dirs) != 1:
        raise ValueError(f"Unable to parse sweep directory from run path: {run_dir}")

    sweep_dir = sweep_dirs[0]
    dataset = sweep_dir[: -len(SWEEP_DIR_SUFFIX)]
    sweep_idx = parts.index(sweep_dir)
    mode_id = parts[sweep_idx + 1]
    model_id = parts[sweep_idx + 2]
    k_token = parts[sweep_idx + 3]
    if not k_token.startswith("k_"):
        raise ValueError(f"Expected k_* directory in run path: {run_dir}")
    k = int(k_token[2:])
    return dataset, mode_id, model_id, k


def _load_episode_rewards(metrics_path: Path) -> List[float]:
    episode_rewards: Dict[int, float] = {}
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            episode_idx = payload["episode_idx"]
            episode_rewards[episode_idx] = episode_rewards.get(episode_idx, 0.0) + float(
                payload["reward"]
            )
    return list(episode_rewards.values())


def _load_adversarial_attack_outcomes(defense_path: Path) -> Tuple[int, int]:
    successes = 0
    total = 0
    with defense_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload["harmful_level"] <= 0:
                continue
            total += 1
            if float(payload["goal_achieved"]) < 0.5:
                successes += 1
    return successes, total


def _load_run_outcomes(run_dir: Path) -> Dict:
    dataset, mode_id, model_id, k = _parse_run_dir(run_dir)
    metrics_path = run_dir / "all_metrics.jsonl"
    defense_path = run_dir / "defense_episodes.jsonl"

    episode_rewards = _load_episode_rewards(metrics_path)
    attack_successes, adversarial_episodes = _load_adversarial_attack_outcomes(defense_path)

    return {
        "dataset": dataset,
        "mode_id": mode_id,
        "model_id": model_id,
        "k": k,
        "num_episodes": len(episode_rewards),
        "episode_rewards": episode_rewards,
        "attack_successes": attack_successes,
        "adversarial_episodes": adversarial_episodes,
    }


def _aggregate_runs(runs: List[Dict]) -> Dict[str, Dict[int, Dict]]:
    summary: Dict[str, Dict[int, Dict]] = {}
    for run in runs:
        dataset_summary = summary.setdefault(run["dataset"], {})
        dataset_summary[run["k"]] = {
            "model_id": run["model_id"],
            "num_episodes": run["num_episodes"],
            "mean_episode_reward": _mean_confidence_interval_95(run["episode_rewards"]),
            "attack_success_rate": _binomial_confidence_interval_95(
                run["attack_successes"],
                run["adversarial_episodes"],
            ),
            "adversarial_episodes": run["adversarial_episodes"],
            "attack_successes": run["attack_successes"],
        }
    return summary


def _resolve_run_dirs(metrics_glob: str) -> List[Path]:
    metrics_files = sorted(Path(".").glob(metrics_glob))
    if not metrics_files:
        raise FileNotFoundError(f"No all_metrics.jsonl files found matching: {metrics_glob}")
    return sorted({metrics_file.parent for metrics_file in metrics_files})


def _plot_metric_bars(
    dataset: str,
    per_k_summary: Dict[int, Dict],
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
    ylim: Tuple[float, float] | None = None,
) -> None:
    ks = [k for k in K_VALUES if k in per_k_summary]
    if not ks:
        raise RuntimeError(f"No K values in {K_VALUES} found for dataset={dataset}")

    model_id = per_k_summary[ks[0]]["model_id"]
    means = [per_k_summary[k][metric_key]["mean"] for k in ks]
    err_low = [
        per_k_summary[k][metric_key]["mean"] - per_k_summary[k][metric_key]["ci95_low"]
        for k in ks
    ]
    err_high = [
        per_k_summary[k][metric_key]["ci95_high"] - per_k_summary[k][metric_key]["mean"]
        for k in ks
    ]
    x_positions = list(range(len(ks)))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        x_positions,
        means,
        width=0.7,
        yerr=[err_low, err_high],
        capsize=4,
        color="tab:blue",
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_xticks(x_positions, [str(k) for k in ks])
    ax.set_xlabel("Candidate pool size K")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} (critic-guided, {model_id})")
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot candidate-pool K sweep bar charts from environment reward and ASR."
    )
    parser.add_argument(
        "--metrics-glob",
        type=str,
        default=(
            "outputs/*_hierarchical_freeform_iterative_k_sweep/"
            "critic_guided/zephyr/k_*/all_metrics.jsonl"
        ),
        help="Glob for critic-guided all_metrics.jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/candidate_pool_k_sweep_plots",
        help="Directory for plot PNGs and summary JSON.",
    )
    args = parser.parse_args()

    run_dirs = _resolve_run_dirs(args.metrics_glob)
    runs = []
    for run_dir in run_dirs:
        run = _load_run_outcomes(run_dir)
        if run["mode_id"] != MODE_ID:
            continue
        if run["k"] not in K_VALUES:
            continue
        runs.append(run)
    if not runs:
        raise RuntimeError(
            f"No critic-guided K-sweep runs found for K in {K_VALUES}."
        )

    summary = _aggregate_runs(runs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_paths: List[str] = []
    for dataset, per_k_summary in sorted(summary.items()):
        reward_plot = output_dir / f"{dataset}_mean_episode_reward_by_k.png"
        asr_plot = output_dir / f"{dataset}_asr_by_k.png"
        _plot_metric_bars(
            dataset=dataset,
            per_k_summary=per_k_summary,
            metric_key="mean_episode_reward",
            ylabel="Mean episode reward",
            title=f"{dataset}: mean episode reward vs K",
            output_path=reward_plot,
        )
        _plot_metric_bars(
            dataset=dataset,
            per_k_summary=per_k_summary,
            metric_key="attack_success_rate",
            ylabel="Attack success rate (adversarial episodes)",
            title=f"{dataset}: ASR vs K (goal_achieved < 0.5)",
            output_path=asr_plot,
            ylim=(0.0, 1.05),
        )
        plot_paths.extend([str(reward_plot), str(asr_plot)])

    summary_path = output_dir / "candidate_pool_k_sweep_plot_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "metrics_glob": args.metrics_glob,
                "num_runs": len(runs),
                "k_values": list(K_VALUES),
                "mode_id": MODE_ID,
                "summary_by_dataset_k": summary,
                "plot_paths": plot_paths,
                "asr_definition": "adversarial episodes with goal_achieved < 0.5",
                "reward_definition": "sum of per-turn environment rewards per episode",
            },
            handle,
            indent=2,
        )

    print(f"Wrote plot summary to {summary_path}")
    print(f"Runs scanned: {len(runs)}")
    for plot_path in plot_paths:
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
