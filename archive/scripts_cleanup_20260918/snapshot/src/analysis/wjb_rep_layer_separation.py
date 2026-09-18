"""
Per-layer activation clustering for WildJailbreak harmful vs seemingly-toxic prompts.

DEPRECATED: Use wjb_rep_belief_intervention.py for the belief-swap intervention grid.
This module compared question-only vs templated belief inputs (confounded).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.metrics import (
    adjusted_rand_score,
    confusion_matrix,
    silhouette_score,
    accuracy_score,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..prompts.prompt_manager import PromptManager
from .rep_belief_selection import (
    BELIEF_CONDITIONS,
    HARMFUL_CLUSTER_ID,
    SEEMINGLY_TOXIC_CLUSTER_ID,
    build_input_text_for_condition,
    load_belief_cache,
)

TARGET_LAYER = 15
KMEANS_RANDOM_STATE = 142857


def y_true_from_cache(cache: List[dict]) -> np.ndarray:
    return np.array([0 if r["cluster_id"] == SEEMINGLY_TOXIC_CLUSTER_ID else 1 for r in cache])


def extract_layer_activations(
    texts: List[str],
    model: AutoModelForCausalLM,
    tokenizer,
    device: str,
    batch_size: int,
    max_length: int,
) -> Dict[int, np.ndarray]:
    num_layers = model.config.num_hidden_layers
    layer_buffers: Dict[int, List[np.ndarray]] = {layer: [] for layer in range(num_layers)}

    model.eval()
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.inference_mode():
            outputs = model(**encoded, output_hidden_states=True)

        hidden_states = outputs.hidden_states[1:]
        for layer_idx, layer_hidden in enumerate(hidden_states):
            last_token_indices = encoded["attention_mask"].sum(dim=1) - 1
            batch_vecs = layer_hidden[torch.arange(layer_hidden.size(0)), last_token_indices]
            layer_buffers[layer_idx].append(batch_vecs.float().cpu().numpy())

        del outputs, encoded

    return {layer: np.concatenate(chunks, axis=0) for layer, chunks in layer_buffers.items()}


def align_accs(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, float]:
    n_classes = int(y_true.max()) + 1
    cm = np.zeros((2, 2), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[pred_label, true_label] += 1

    row_ind, col_ind = linear_sum_assignment(cm.max() - cm)
    mapping = {row: col for row, col in zip(row_ind, col_ind)}
    y_pred_aligned = np.array([mapping[p] for p in y_pred])

    cm_align = confusion_matrix(y_true, y_pred_aligned, labels=np.arange(n_classes))
    acc_vec = np.diag(cm_align) / cm_align.sum(axis=1)
    acc_total = accuracy_score(y_true, y_pred_aligned)
    return acc_vec, acc_total


def cluster_metrics(emb: np.ndarray, y_true: np.ndarray) -> Tuple[float, float, float]:
    kmeans = KMeans(n_clusters=2, n_init="auto", random_state=KMEANS_RANDOM_STATE)
    y_pred = kmeans.fit_predict(emb)

    ari = adjusted_rand_score(y_true, y_pred)
    sil = silhouette_score(emb, y_pred, metric="cosine")
    _, acc_total = align_accs(y_true, y_pred)
    return ari, sil, acc_total


def save_metric_table(metrics_by_layer: Dict[int, Dict[str, float]], output_path: Path, metric_name: str) -> None:
    rows = [{"layer": layer, metric_name: metrics_by_layer[layer][metric_name]} for layer in sorted(metrics_by_layer)]
    pd.DataFrame(rows).to_csv(output_path, index=False)


def plot_mds_layer15(embeddings: np.ndarray, y_true: np.ndarray, output_path: Path, title: str) -> None:
    coords = MDS(n_components=2, random_state=42, normalized_stress="auto").fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(6, 5))
    safe_mask = y_true == 0
    unsafe_mask = y_true == 1
    ax.scatter(coords[safe_mask, 0], coords[safe_mask, 1], c="tab:blue", label="seemingly-toxic", alpha=0.7, s=20)
    ax.scatter(coords[unsafe_mask, 0], coords[unsafe_mask, 1], c="tab:red", label="harmful", alpha=0.7, s=20)
    ax.set_title(title)
    ax.set_xlabel("MDS-1")
    ax.set_ylabel("MDS-2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ari_bar(summary_rows: List[dict], output_path: Path) -> None:
    labels = [row["condition"] for row in summary_rows]
    aris = [row["ari"] for row in summary_rows]
    colors = ["tab:gray", "tab:green", "tab:orange", "tab:purple"]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, aris, color=colors[: len(labels)])
    ax.set_ylabel("ARI at layer 15")
    ax.set_title("WJB layer-15 separation by belief selection")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = load_belief_cache(Path(args.beliefs_cache))
    y_true = y_true_from_cache(cache)
    prompt_manager = PromptManager()

    device = args.device
    dtype = torch.bfloat16 if args.use_bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = {
        "model_name": args.model_name,
        "beliefs_cache": args.beliefs_cache,
        "num_records": len(cache),
        "conditions": list(BELIEF_CONDITIONS),
        "target_layer": TARGET_LAYER,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    summary_rows = []
    for condition in BELIEF_CONDITIONS:
        print(f"=== Condition: {condition} ===")
        texts = [build_input_text_for_condition(entry, condition, prompt_manager) for entry in cache]
        activations = extract_layer_activations(
            texts=texts,
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

        condition_dir = output_dir / condition
        condition_dir.mkdir(parents=True, exist_ok=True)
        torch.save(activations, condition_dir / "llm_activations.pt")

        ari_by_layer: Dict[int, float] = {}
        sil_by_layer: Dict[int, float] = {}
        acc_by_layer: Dict[int, float] = {}

        for layer, emb in activations.items():
            ari, sil, acc_total = cluster_metrics(emb, y_true)
            ari_by_layer[layer] = ari
            sil_by_layer[layer] = sil
            acc_by_layer[layer] = acc_total

        save_metric_table({layer: {"ari": ari_by_layer[layer]} for layer in ari_by_layer}, condition_dir / "ari_by_layer.csv", "ari")
        save_metric_table({layer: {"silhouette": sil_by_layer[layer]} for layer in sil_by_layer}, condition_dir / "silhouette_by_layer.csv", "silhouette")
        save_metric_table({layer: {"accuracy": acc_by_layer[layer]} for layer in acc_by_layer}, condition_dir / "accuracy_by_layer.csv", "accuracy")

        summary_rows.append({
            "condition": condition,
            "layer": TARGET_LAYER,
            "ari": ari_by_layer[TARGET_LAYER],
            "silhouette": sil_by_layer[TARGET_LAYER],
            "accuracy": acc_by_layer[TARGET_LAYER],
        })
        print(
            f"  layer {TARGET_LAYER}: ARI={ari_by_layer[TARGET_LAYER]:.4f}, "
            f"silhouette={sil_by_layer[TARGET_LAYER]:.4f}, accuracy={acc_by_layer[TARGET_LAYER]:.4f}"
        )
        plot_mds_layer15(activations[TARGET_LAYER], y_true, condition_dir / "mds_layer15.png", f"{condition} (layer {TARGET_LAYER})")

    with (output_dir / "summary_layer15.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "layer", "ari", "silhouette", "accuracy"])
        writer.writeheader()
        writer.writerows(summary_rows)

    plot_ari_bar(summary_rows, output_dir / "ari_layer15_bar.png")
    print(f"Wrote outputs to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WJB per-layer activation separation experiment.")
    parser.add_argument("--beliefs_cache", type=str, default="data/WJB-Rep/wjb_belief_cache.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/wjb_layer_sep")
    parser.add_argument("--model_name", type=str, default="HuggingFaceH4/zephyr-7b-beta")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--use_bf16", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
