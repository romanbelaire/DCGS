"""
Belief-swap intervention: last-token activations over the full templated input and 4x4 PCA grid.

Rows: empty / critic / random / orthogonal (templated LL input only).
Columns: layer quartiles n/4, n/2, 3n/4, n.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..prompts.prompt_manager import PromptManager
from .rep_belief_selection import (
    BELIEF_INTERVENTION_CONDITIONS,
    build_templated_input,
    belief_for_intervention,
    load_belief_cache,
    y_true_from_cache,
)
from .wjb_rep_layer_separation import cluster_metrics

def cosine_diff_between_class_centroids(
    activations: np.ndarray,
    harmful_mask: np.ndarray,
    seemingly_mask: np.ndarray,
) -> float:
    harmful_centroid = activations[harmful_mask].mean(axis=0)
    seemingly_centroid = activations[seemingly_mask].mean(axis=0)
    harmful_centroid = harmful_centroid / np.linalg.norm(harmful_centroid)
    seemingly_centroid = seemingly_centroid / np.linalg.norm(seemingly_centroid)
    return 1.0 - float(np.dot(harmful_centroid, seemingly_centroid))


def encode_intervention_prompt(tokenizer, text: str, max_length: int) -> Dict:
    # Llama-family tokenizers default to left truncation (keeps dialogue tail), which drops
    # the belief block at the prompt start. Force right truncation to keep the belief span.
    previous_truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "right"
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    tokenizer.truncation_side = previous_truncation_side
    return encoded


def target_layer_indices(num_hidden_layers: int) -> List[int]:
    n = num_hidden_layers
    return [n // 4, n // 2, 3 * n // 4, n - 1]


def layer_column_label(layer_idx: int, num_hidden_layers: int) -> str:
    del num_hidden_layers
    return f"L{layer_idx}"


def last_token_representation(
    hidden_states: Tuple[torch.Tensor, ...],
    target_layers: List[int],
    last_token_index: int,
) -> Dict[int, np.ndarray]:
    pooled = {}
    for layer_idx in target_layers:
        vec = hidden_states[layer_idx + 1][0, last_token_index, :].float().cpu().numpy()
        norm = np.linalg.norm(vec)
        pooled[layer_idx] = vec / norm
    return pooled


def extract_intervention_activations(
    cache: List[dict],
    conditions: Tuple[str, ...],
    model: AutoModelForCausalLM,
    tokenizer,
    prompt_manager: PromptManager,
    device: str,
    target_layers: List[int],
    max_length: int,
) -> Dict[str, Dict[int, np.ndarray]]:
    num_records = len(cache)
    hidden_size = model.config.hidden_size
    activations = {
        condition: {layer: np.zeros((num_records, hidden_size), dtype=np.float32) for layer in target_layers}
        for condition in conditions
    }

    model.eval()
    for condition in conditions:
        print(f"=== Intervention condition: {condition} ===")
        for record_idx, entry in enumerate(cache):
            belief = belief_for_intervention(entry, condition)
            text = build_templated_input(entry, belief, prompt_manager)
            encoded = encode_intervention_prompt(tokenizer, text, max_length)
            last_token_index = encoded["attention_mask"].sum(dim=1).item() - 1
            model_inputs = {
                "input_ids": encoded["input_ids"].to(device),
                "attention_mask": encoded["attention_mask"].to(device),
            }

            with torch.inference_mode():
                outputs = model(**model_inputs, output_hidden_states=True)

            pooled = last_token_representation(
                outputs.hidden_states,
                target_layers,
                last_token_index,
            )
            for layer_idx, vec in pooled.items():
                activations[condition][layer_idx][record_idx] = vec

            del outputs, model_inputs
            if (record_idx + 1) % 50 == 0:
                print(f"  {condition}: {record_idx + 1}/{num_records}")

    return activations


def compute_metric_grids(
    activations: Dict[str, Dict[int, np.ndarray]],
    conditions: Tuple[str, ...],
    target_layers: List[int],
    y_true: np.ndarray,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    ari_grid: Dict[str, Dict[str, float]] = {c: {} for c in conditions}
    sil_grid: Dict[str, Dict[str, float]] = {c: {} for c in conditions}
    n_layers = max(target_layers) + 1

    for condition in conditions:
        for layer_idx in target_layers:
            label = layer_column_label(layer_idx, n_layers)
            emb = activations[condition][layer_idx]
            ari, sil, _ = cluster_metrics(emb, y_true)
            ari_grid[condition][label] = ari
            sil_grid[condition][label] = sil

    return ari_grid, sil_grid


def write_metric_grid_csv(
    grid: Dict[str, Dict[str, float]],
    layer_labels: List[str],
    output_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition"] + layer_labels)
        writer.writeheader()
        for condition in BELIEF_INTERVENTION_CONDITIONS:
            row = {"condition": condition}
            row.update({label: grid[condition][label] for label in layer_labels})
            writer.writerow(row)


def zscore_pca_coordinates(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = coords.mean(axis=0)
    scale = coords.std(axis=0)
    if np.any(scale == 0):
        raise ValueError(f"Degenerate PCA axis for z-scoring: scale={scale}")
    return (coords - mean) / scale, mean, scale


def robust_equal_axis_limits(
    coords: np.ndarray,
    low_percentile: float = 5.0,
    high_percentile: float = 95.0,
    padding: float = 0.08,
) -> Tuple[float, float, float, float]:
    x_lo, x_hi = np.percentile(coords[:, 0], [low_percentile, high_percentile])
    y_lo, y_hi = np.percentile(coords[:, 1], [low_percentile, high_percentile])
    x_mid = 0.5 * (x_lo + x_hi)
    y_mid = 0.5 * (y_lo + y_hi)
    half_range = 0.5 * max(x_hi - x_lo, y_hi - y_lo)
    half_range *= 1.0 + padding
    return x_mid - half_range, x_mid + half_range, y_mid - half_range, y_mid + half_range


def display_window_mask(
    coords: np.ndarray,
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
) -> np.ndarray:
    return (
        (coords[:, 0] >= x_lo)
        & (coords[:, 0] <= x_hi)
        & (coords[:, 1] >= y_lo)
        & (coords[:, 1] <= y_hi)
    )


def scatter_class_split(
    ax,
    coords: np.ndarray,
    class_mask: np.ndarray,
    inlier_mask: np.ndarray,
    color: str,
    label: str,
) -> None:
    show = class_mask & inlier_mask
    if np.any(show):
        ax.scatter(
            coords[show, 0],
            coords[show, 1],
            c=color,
            label=label,
            alpha=0.35,
            s=8,
            edgecolors="none",
            rasterized=True,
        )
    outliers = class_mask & ~inlier_mask
    if np.any(outliers):
        ax.scatter(
            coords[outliers, 0],
            coords[outliers, 1],
            c=color,
            alpha=0.12,
            s=6,
            edgecolors="none",
            linewidths=0,
            rasterized=True,
        )


def add_covariance_ellipse(
    ax,
    coords: np.ndarray,
    color: str,
    n_std: float = 2.0,
    linestyle: str = "-",
    label: Optional[str] = None,
) -> None:
    if coords.shape[0] < 3:
        return
    mean = coords.mean(axis=0)
    cov = np.cov(coords.T)
    if np.any(np.isnan(cov)):
        return
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    if vals[0] <= 1e-12:
        ax.plot(mean[0], mean[1], "x", color=color, ms=7, mew=2, zorder=5, label=label)
        return
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width = 2 * n_std * np.sqrt(vals[0])
    height = 2 * n_std * np.sqrt(vals[1])
    ax.add_patch(
        Ellipse(
            xy=mean,
            width=width,
            height=height,
            angle=angle,
            fill=False,
            edgecolor=color,
            linewidth=2.0,
            linestyle=linestyle,
            zorder=4,
            label=label,
        )
    )
    ax.plot(mean[0], mean[1], "x", color=color, ms=7, mew=2, zorder=5)


def column_zscore_and_limits(
    layer_coords_by_condition: Dict[str, np.ndarray],
    low_percentile: float,
    high_percentile: float,
) -> Tuple[Dict[str, np.ndarray], Tuple[float, float, float, float], int, int]:
    stacked = np.vstack([layer_coords_by_condition[c] for c in layer_coords_by_condition])
    mean = stacked.mean(axis=0)
    scale = stacked.std(axis=0)
    if np.any(scale == 0):
        raise ValueError(f"Degenerate PCA axis for column z-scoring: scale={scale}")
    z_by_condition = {c: (layer_coords_by_condition[c] - mean) / scale for c in layer_coords_by_condition}
    z_stacked = np.vstack(list(z_by_condition.values()))
    limits = robust_equal_axis_limits(z_stacked, low_percentile, high_percentile)
    x_lo, x_hi, y_lo, y_hi = limits
    inlier_mask = display_window_mask(z_stacked, x_lo, x_hi, y_lo, y_hi)
    return z_by_condition, limits, int((~inlier_mask).sum()), z_stacked.shape[0]


def plot_belief_intervention_grid(
    activations: Dict[str, Dict[int, np.ndarray]],
    conditions: Tuple[str, ...],
    target_layers: List[int],
    y_true: np.ndarray,
    ari_grid: Dict[str, Dict[str, float]],
    output_path: Path,
    num_hidden_layers: int,
) -> None:
    layer_labels = [layer_column_label(layer_idx, num_hidden_layers) for layer_idx in target_layers]
    reference_layer = target_layers[-1]
    reference_emb = activations["critic"][reference_layer]
    pca = PCA(n_components=2, random_state=42)
    pca.fit(reference_emb)

    raw_coords_by_panel: Dict[Tuple[str, int], np.ndarray] = {}
    for condition in conditions:
        for layer_idx in target_layers:
            raw_coords_by_panel[(condition, layer_idx)] = pca.transform(activations[condition][layer_idx])

    seemingly_mask = y_true == 0
    harmful_mask = y_true == 1

    fig, axes = plt.subplots(len(conditions), len(target_layers), figsize=(16, 14), sharex=False, sharey=False)
    total_outliers = 0
    total_points = 0

    for col_idx, layer_idx in enumerate(target_layers):
        layer_raw = {c: raw_coords_by_panel[(c, layer_idx)] for c in conditions}
        z_by_condition, (x_lo, x_hi, y_lo, y_hi), n_out, n_pts = column_zscore_and_limits(
            layer_raw, low_percentile=10.0, high_percentile=90.0
        )
        total_outliers += n_out
        total_points += n_pts

        for row_idx, condition in enumerate(conditions):
            ax = axes[row_idx, col_idx]
            coords = z_by_condition[condition]
            inlier_mask = display_window_mask(coords, x_lo, x_hi, y_lo, y_hi)

            scatter_class_split(ax, coords, seemingly_mask, inlier_mask, "tab:blue", "seemingly-toxic")
            scatter_class_split(ax, coords, harmful_mask, inlier_mask, "tab:red", "harmful")
            show_ellipse_labels = row_idx == 0 and col_idx == 0
            add_covariance_ellipse(
                ax,
                coords[seemingly_mask & inlier_mask],
                "tab:blue",
                label="seemingly-toxic (2σ)" if show_ellipse_labels else None,
            )
            add_covariance_ellipse(
                ax,
                coords[harmful_mask & inlier_mask],
                "tab:red",
                label="harmful (2σ)" if show_ellipse_labels else None,
            )

            label = layer_labels[col_idx]
            ari = ari_grid[condition][label]
            ax.set_title(f"{label}  ARI={ari:.3f}", fontsize=10)
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)
            ax.set_aspect("equal", adjustable="box")
            ax.axhline(0.0, color="0.85", linewidth=0.6, zorder=0)
            ax.axvline(0.0, color="0.85", linewidth=0.6, zorder=0)
            if col_idx == 0:
                ax.set_ylabel(condition)
            if row_idx == len(conditions) - 1:
                ax.set_xlabel("PC (z, per layer)")
            if col_idx == 0 and row_idx == 0:
                ax.legend(loc="upper right", fontsize=7)

    fig.suptitle(
        "WJB belief intervention — shared PCA basis (critic @ L31); "
        "per-layer z-score + 10–90% zoom (shared across belief rows); "
        f"2σ ellipses; {total_outliers}/{total_points} faded outliers",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_class_separation_grid(
    activations: Dict[str, Dict[int, np.ndarray]],
    conditions: Tuple[str, ...],
    target_layers: List[int],
    y_true: np.ndarray,
    output_path: Path,
    num_hidden_layers: int,
) -> None:
    layer_labels = [layer_column_label(layer_idx, num_hidden_layers) for layer_idx in target_layers]
    seemingly_mask = y_true == 0
    harmful_mask = y_true == 1

    fig, axes = plt.subplots(len(conditions), len(target_layers), figsize=(16, 14), sharex=False, sharey=False)

    for col_idx, layer_idx in enumerate(target_layers):
        acts_by_condition = {c: activations[c][layer_idx] for c in conditions}
        stacked = np.vstack(list(acts_by_condition.values()))
        pca_col = PCA(n_components=2, random_state=42)
        pca_col.fit(stacked)
        raw_by_condition = {c: pca_col.transform(acts_by_condition[c]) for c in conditions}
        z_by_condition, (x_lo, x_hi, y_lo, y_hi), _, _ = column_zscore_and_limits(
            raw_by_condition, low_percentile=10.0, high_percentile=90.0
        )

        for row_idx, condition in enumerate(conditions):
            ax = axes[row_idx, col_idx]
            coords = z_by_condition[condition]
            inlier_mask = display_window_mask(coords, x_lo, x_hi, y_lo, y_hi)
            scatter_class_split(ax, coords, seemingly_mask, inlier_mask, "tab:blue", "seemingly-toxic")
            scatter_class_split(ax, coords, harmful_mask, inlier_mask, "tab:red", "harmful")
            add_covariance_ellipse(ax, coords[seemingly_mask & inlier_mask], "tab:blue")
            add_covariance_ellipse(ax, coords[harmful_mask & inlier_mask], "tab:red")
            cos_diff = cosine_diff_between_class_centroids(
                acts_by_condition[condition], harmful_mask, seemingly_mask
            )
            ax.set_title(f"{layer_labels[col_idx]}  cos diff={cos_diff:.3f}", fontsize=10)
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)
            ax.set_aspect("equal", adjustable="box")
            ax.axhline(0.0, color="0.85", linewidth=0.6, zorder=0)
            ax.axvline(0.0, color="0.85", linewidth=0.6, zorder=0)
            if col_idx == 0:
                ax.set_ylabel(condition)

    fig.suptitle(
        "Harmful vs seemingly-toxic separation (last-token reps); "
        "PCA fit per layer on pooled activations; per-layer z-score + 10–90% zoom",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_class_cosine_diff_summary(
    activations: Dict[str, Dict[int, np.ndarray]],
    conditions: Tuple[str, ...],
    target_layers: List[int],
    y_true: np.ndarray,
    output_path: Path,
    num_hidden_layers: int,
) -> None:
    layer_labels = [layer_column_label(layer_idx, num_hidden_layers) for layer_idx in target_layers]
    seemingly_mask = y_true == 0
    harmful_mask = y_true == 1

    fig, axes = plt.subplots(1, len(target_layers), figsize=(4 * len(target_layers), 4.5), sharey=True)
    if len(target_layers) == 1:
        axes = [axes]

    x_positions = np.arange(len(conditions))

    for col_idx, layer_idx in enumerate(target_layers):
        ax = axes[col_idx]
        cosine_diffs = []
        for condition in conditions:
            cosine_diffs.append(
                cosine_diff_between_class_centroids(
                    activations[condition][layer_idx],
                    harmful_mask,
                    seemingly_mask,
                )
            )
        ax.bar(x_positions, cosine_diffs, width=0.6, color="tab:purple")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(conditions)
        ax.set_title(layer_labels[col_idx])
        if col_idx == 0:
            ax.set_ylabel("cosine diff (1 − cos)")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Harmful ↔ seemingly-toxic centroid cosine diff per belief condition (last-token reps)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def replot_from_activations(
    activations_path: Path,
    output_dir: Path,
    num_hidden_layers: int,
) -> None:
    data = torch.load(activations_path, map_location="cpu", weights_only=False)
    activations = data["activations"]
    y_true = np.array(data["y_true"])
    target_layers = data["target_layers"]
    layer_labels = [layer_column_label(layer_idx, num_hidden_layers) for layer_idx in target_layers]

    ari_grid, sil_grid = compute_metric_grids(
        activations, BELIEF_INTERVENTION_CONDITIONS, target_layers, y_true
    )
    write_metric_grid_csv(ari_grid, layer_labels, output_dir / "ari_grid.csv")
    write_metric_grid_csv(sil_grid, layer_labels, output_dir / "silhouette_grid.csv")

    plot_belief_intervention_grid(
        activations=activations,
        conditions=BELIEF_INTERVENTION_CONDITIONS,
        target_layers=target_layers,
        y_true=y_true,
        ari_grid=ari_grid,
        output_path=output_dir / "belief_intervention_grid.png",
        num_hidden_layers=num_hidden_layers,
    )
    plot_class_separation_grid(
        activations=activations,
        conditions=BELIEF_INTERVENTION_CONDITIONS,
        target_layers=target_layers,
        y_true=y_true,
        output_path=output_dir / "belief_intervention_class_separation.png",
        num_hidden_layers=num_hidden_layers,
    )
    plot_class_cosine_diff_summary(
        activations=activations,
        conditions=BELIEF_INTERVENTION_CONDITIONS,
        target_layers=target_layers,
        y_true=y_true,
        output_path=output_dir / "belief_intervention_cosine_diff.png",
        num_hidden_layers=num_hidden_layers,
    )


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.replot_only:
        activations_path = output_dir / "activations.pt"
        if not activations_path.is_file():
            raise FileNotFoundError(f"--replot_only requires {activations_path}")
        data = torch.load(activations_path, map_location="cpu", weights_only=False)
        num_hidden_layers = max(data["target_layers"]) + 1
        replot_from_activations(activations_path, output_dir, num_hidden_layers)
        print(f"Replotted figures in {output_dir}")
        return

    cache = load_belief_cache(Path(args.beliefs_cache))
    if args.max_records is not None:
        cache = cache[: args.max_records]

    if len(cache) < 10:
        raise ValueError(
            f"Need at least 10 belief-cache records for clustering metrics, got {len(cache)}"
        )

    y_true = np.array(y_true_from_cache(cache), dtype=np.int64)
    prompt_manager = PromptManager()

    device = args.device
    dtype = torch.bfloat16 if args.use_bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_hidden_layers = model.config.num_hidden_layers
    target_layers = target_layer_indices(num_hidden_layers)
    layer_labels = [layer_column_label(layer_idx, num_hidden_layers) for layer_idx in target_layers]

    print(f"Model layers: {num_hidden_layers}, target layers: {target_layers}")

    activations = extract_intervention_activations(
        cache=cache,
        conditions=BELIEF_INTERVENTION_CONDITIONS,
        model=model,
        tokenizer=tokenizer,
        prompt_manager=prompt_manager,
        device=device,
        target_layers=target_layers,
        max_length=args.max_length,
    )

    torch.save(
        {
            "activations": activations,
            "y_true": y_true,
            "target_layers": target_layers,
            "conditions": list(BELIEF_INTERVENTION_CONDITIONS),
        },
        output_dir / "activations.pt",
    )

    ari_grid, sil_grid = compute_metric_grids(
        activations,
        BELIEF_INTERVENTION_CONDITIONS,
        target_layers,
        y_true,
    )
    write_metric_grid_csv(ari_grid, layer_labels, output_dir / "ari_grid.csv")
    write_metric_grid_csv(sil_grid, layer_labels, output_dir / "silhouette_grid.csv")

    plot_belief_intervention_grid(
        activations=activations,
        conditions=BELIEF_INTERVENTION_CONDITIONS,
        target_layers=target_layers,
        y_true=y_true,
        ari_grid=ari_grid,
        output_path=output_dir / "belief_intervention_grid.png",
        num_hidden_layers=num_hidden_layers,
    )
    plot_class_separation_grid(
        activations=activations,
        conditions=BELIEF_INTERVENTION_CONDITIONS,
        target_layers=target_layers,
        y_true=y_true,
        output_path=output_dir / "belief_intervention_class_separation.png",
        num_hidden_layers=num_hidden_layers,
    )
    plot_class_cosine_diff_summary(
        activations=activations,
        conditions=BELIEF_INTERVENTION_CONDITIONS,
        target_layers=target_layers,
        y_true=y_true,
        output_path=output_dir / "belief_intervention_cosine_diff.png",
        num_hidden_layers=num_hidden_layers,
    )

    config = {
        "model_name": args.model_name,
        "beliefs_cache": args.beliefs_cache,
        "num_records": len(cache),
        "conditions": list(BELIEF_INTERVENTION_CONDITIONS),
        "target_layers": target_layers,
        "layer_labels": layer_labels,
        "max_length": args.max_length,
        "representation": "full_input_last_token_l2_normalized",
        "pca_reference": "critic_last_layer",
        "class_separation_metric": "centroid_cosine_diff_1_minus_cos",
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    print(f"ARI grid ({layer_labels[-1]}): empty={ari_grid['empty'][layer_labels[-1]]:.4f} "
          f"critic={ari_grid['critic'][layer_labels[-1]]:.4f} "
          f"random={ari_grid['random'][layer_labels[-1]]:.4f} "
          f"orthogonal={ari_grid['orthogonal'][layer_labels[-1]]:.4f}")
    print(f"Wrote outputs to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WJB belief-intervention representation grid.")
    parser.add_argument("--beliefs_cache", type=str, default="data/WJB-Rep/wjb_belief_cache.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/wjb_belief_intervention")
    parser.add_argument("--model_name", type=str, default="HuggingFaceH4/zephyr-7b-beta")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument(
        "--replot_only",
        action="store_true",
        help="Regenerate figures from output_dir/activations.pt without re-running the model.",
    )
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
