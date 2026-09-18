"""
Small-scale eNTK subset for WildJailbreak representation experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ENTK_CONDITIONS = (
    "question_only",
    "question_critic_belief",
    "question_random_belief",
    "question_dcr_model",
)

ENTK_CONDITION_LABELS = {
    "question_only": "empty belief",
    "question_critic_belief": "critic belief",
    "question_random_belief": "random belief",
    "question_dcr_model": "DCR model (raw prompt)",
}

DCR_ENTK_CONDITION = "question_dcr_model"

ENTK_CATEGORY_LABELS = ("S-Toxic", "Toxic", "Normal")


def normalize_prompt(text: str) -> str:
    return " ".join(text.strip().split())


def resolve_alpaca_column(column_names: List[str]) -> str:
    for candidate in ("instruction", "prompt", "text"):
        if candidate in column_names:
            return candidate
    raise ValueError(f"Alpaca prompt column not found. Available columns: {column_names}")


def select_alpaca_normal_subset(
    per_class: int,
    seed: int,
    alpaca_dataset: str,
    alpaca_split: str,
) -> List[dict]:
    from datasets import load_dataset

    from .rep_belief_selection import VANILLA_BENIGN_CLUSTER_ID, ground_truth_belief

    normal_belief = ground_truth_belief(VANILLA_BENIGN_CLUSTER_ID)

    alpaca = load_dataset(alpaca_dataset, split=alpaca_split)
    prompt_col = resolve_alpaca_column(list(alpaca.column_names))
    pool: List[str] = []
    seen = set()
    for row in alpaca:
        prompt = normalize_prompt(row[prompt_col])
        if prompt in seen:
            continue
        seen.add(prompt)
        pool.append(prompt)

    if len(pool) < per_class:
        raise ValueError(
            f"Need at least {per_class} Alpaca prompts, found {len(pool)} in {alpaca_dataset}:{alpaca_split}"
        )

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=per_class, replace=False)
    selected_prompts = [pool[i] for i in sorted(idx.tolist())]
    return [
        {
            "instruction": prompt,
            "cluster_id": VANILLA_BENIGN_CLUSTER_ID,
            "source": "alpaca",
            "beliefs": {
                "critic_guided": normal_belief,
                "random": normal_belief,
            },
        }
        for prompt in selected_prompts
    ]


def select_entk_subset(
    cache: List[dict],
    per_class: int,
    seed: int,
    alpaca_dataset: str,
    alpaca_split: str,
) -> List[dict]:
    from .rep_belief_selection import (
        SUBSET_CLASS_ORDER,
        HARMFUL_CLUSTER_ID,
        SEEMINGLY_TOXIC_CLUSTER_ID,
    )

    pools = {
        HARMFUL_CLUSTER_ID: [r for r in cache if r["cluster_id"] == HARMFUL_CLUSTER_ID],
        SEEMINGLY_TOXIC_CLUSTER_ID: [r for r in cache if r["cluster_id"] == SEEMINGLY_TOXIC_CLUSTER_ID],
    }
    for cluster_id in (HARMFUL_CLUSTER_ID, SEEMINGLY_TOXIC_CLUSTER_ID):
        if len(pools[cluster_id]) < per_class:
            raise ValueError(
                f"Need at least {per_class} belief-cache records for cluster_id={cluster_id}, "
                f"got {len(pools[cluster_id])}. Rebuild wjb_rep_eval.jsonl and wjb_belief_cache.jsonl."
            )

    rng = np.random.default_rng(seed)
    selected = {}
    for cluster_id in (HARMFUL_CLUSTER_ID, SEEMINGLY_TOXIC_CLUSTER_ID):
        idx = rng.choice(len(pools[cluster_id]), size=per_class, replace=False)
        selected[cluster_id] = [pools[cluster_id][i] for i in sorted(idx.tolist())]

    alpaca_subset = select_alpaca_normal_subset(
        per_class=per_class,
        seed=seed,
        alpaca_dataset=alpaca_dataset,
        alpaca_split=alpaca_split,
    )

    return (
        selected[HARMFUL_CLUSTER_ID]
        + selected[SEEMINGLY_TOXIC_CLUSTER_ID]
        + alpaca_subset
    )


def entk_input_texts_for_condition(subset: List[dict], condition: str) -> List[str]:
    from .rep_belief_selection import build_entk_input_text

    if condition == DCR_ENTK_CONDITION:
        return [entry["instruction"] for entry in subset]
    return [build_entk_input_text(entry, condition) for entry in subset]


def load_entk_model_and_tokenizer(
    model_name: str,
    device: str,
    dtype,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(model_name)
    adapter_config_path = model_path / "adapter_config.json"
    if adapter_config_path.is_file():
        from peft import PeftModel

        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        base_model_name = adapter_config["base_model_name_or_path"]
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype).to(device)
        model = PeftModel.from_pretrained(base_model, str(model_path))
        model.eval()
        print(f"Loaded LoRA eNTK model: base={base_model_name} adapter={model_path}", flush=True)
        return model, tokenizer, "lora-only"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()
    return model, tokenizer, None


def save_entk_subset_manifest(subset: List[dict], output_dir: Path) -> None:
    manifest_path = output_dir / "entk_subset_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for entry in subset:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")


def compute_knorm_matrix(
    texts: List[str],
    model_name: str,
    device: str,
    dtype,
    max_length: Optional[int],
    topk_o: int,
    topk_u: int,
    param_mode: str,
    last_n: int,
    jvp_eps: float,
    column_batch_size: int,
) -> np.ndarray:
    dcr_experiment_dir = Path(__file__).resolve().parents[2] / "dcr" / "DCR-main" / "experiment"
    sys.path.insert(0, str(dcr_experiment_dir))
    from KT_similarity_jvp import (  # noqa: E402
        build_topk_token_cache,
        compute_K_block_loaded,
        pick_params,
    )

    pos_list = [-1]
    model, tok, param_mode_override = load_entk_model_and_tokenizer(model_name, device, dtype)
    effective_param_mode = param_mode_override or param_mode
    params = pick_params(model, mode=effective_param_mode, last_n=last_n)
    trunc_label = "disabled" if max_length is None else str(max_length)
    print(f"eNTK encode: truncation={trunc_label} param_mode={effective_param_mode}", flush=True)

    cache_topk = max(topk_o, topk_u)
    print(f"Building top-k token cache for {len(texts)} prompts (topk={cache_topk})...", flush=True)
    token_cache = build_topk_token_cache(
        model=model,
        tok=tok,
        texts=texts,
        pos_list=pos_list,
        topk=cache_topk,
        device=device,
        max_length=max_length,
    )

    def token_ids_for_prompt(prompt_idx: int, topk: int) -> List[List[int]]:
        return [[tid for tid in pos_tids[:topk]] for pos_tids in token_cache[prompt_idx]]

    n = len(texts)
    matrix = np.zeros((n, n), dtype=np.float32)
    for j, text_o in enumerate(texts):
        for k, text_u in enumerate(texts):
            k_block = compute_K_block_loaded(
                model=model,
                tok=tok,
                params=params,
                text_o=text_o,
                text_u=text_u,
                pos_o=pos_list,
                pos_u=pos_list,
                token_ids_per_pos_o=token_ids_for_prompt(j, topk_o),
                token_ids_per_pos_u=token_ids_for_prompt(k, topk_u),
                device=device,
                max_length=max_length,
                jvp_eps=jvp_eps,
                column_batch_size=column_batch_size,
            )
            matrix[j, k] = float(k_block.norm().item())
            print(f"  pair [{j + 1}/{n}] x [{k + 1}/{n}] K_frob={matrix[j, k]:.6f}", flush=True)
    return matrix


def normalize_pairwise_kt_matrix(matrix: np.ndarray) -> np.ndarray:
    """Paper normalization: ||K^t(x',x)||_F / ||K^t(x',x')||_F (row x' = divide by diagonal)."""
    diag = np.diag(matrix).astype(np.float64)
    diag = np.maximum(diag, 1e-12)
    return (matrix.astype(np.float64) / diag[:, np.newaxis]).astype(np.float32)


def category_block_matrix(matrix: np.ndarray, class_slices: Dict[int, slice]) -> np.ndarray:
    """3x3 mean normalized ||K^t(x',x)||_F per category block; axes S-Toxic, Toxic, Normal."""
    from .rep_belief_selection import ENTK_CLASS_ORDER

    n = len(ENTK_CLASS_ORDER)
    block = np.zeros((n, n), dtype=np.float64)
    for row_idx, row_class in enumerate(ENTK_CLASS_ORDER):
        for col_idx, col_class in enumerate(ENTK_CLASS_ORDER):
            submatrix = matrix[class_slices[row_class], class_slices[col_class]]
            if row_class == col_class:
                n_class = submatrix.shape[0]
                offdiag = submatrix[np.triu_indices(n_class, k=1)]
                block[row_idx, col_idx] = offdiag.mean()
            else:
                block[row_idx, col_idx] = submatrix.mean()
    return block


def block_stats(matrix: np.ndarray, class_slices: Dict[int, slice]) -> Dict[str, float]:
    from .rep_belief_selection import HARMFUL_CLUSTER_ID, SEEMINGLY_TOXIC_CLUSTER_ID

    harmful_slice = class_slices[HARMFUL_CLUSTER_ID]
    seemingly_slice = class_slices[SEEMINGLY_TOXIC_CLUSTER_ID]
    n_harmful = harmful_slice.stop - harmful_slice.start
    n_seemingly = seemingly_slice.stop - seemingly_slice.start

    within_harmful = matrix[harmful_slice, harmful_slice]
    within_seemingly = matrix[seemingly_slice, seemingly_slice]
    cross = matrix[harmful_slice, seemingly_slice]

    harmful_offdiag = within_harmful[np.triu_indices(n_harmful, k=1)]
    seemingly_offdiag = within_seemingly[np.triu_indices(n_seemingly, k=1)]

    within_harmful_mean = float(harmful_offdiag.mean()) if harmful_offdiag.size else float("nan")
    within_seemingly_mean = float(seemingly_offdiag.mean()) if seemingly_offdiag.size else float("nan")
    cross_mean = float(cross.mean())
    within_mean = float(np.mean([within_harmful_mean, within_seemingly_mean]))
    cross_within_ratio = cross_mean / within_mean if within_mean > 0 else float("nan")

    return {
        "within_harmful_mean": within_harmful_mean,
        "within_seemingly_mean": within_seemingly_mean,
        "within_mean": within_mean,
        "cross_mean": cross_mean,
        "cross_within_ratio": cross_within_ratio,
    }


def write_block_stats_csv(output_dir: Path, summary_rows: List[dict]) -> None:
    fieldnames = [
        "condition",
        "within_harmful_mean",
        "within_seemingly_mean",
        "within_mean",
        "cross_mean",
        "cross_within_ratio",
    ]
    with (output_dir / "block_stats.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def load_block_stats_csv(output_dir: Path) -> List[dict]:
    rows: List[dict] = []
    with (output_dir / "block_stats.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "condition": row["condition"],
                    "within_harmful_mean": float(row["within_harmful_mean"]),
                    "within_seemingly_mean": float(row["within_seemingly_mean"]),
                    "within_mean": float(row["within_mean"]),
                    "cross_mean": float(row["cross_mean"]),
                    "cross_within_ratio": float(row["cross_within_ratio"]),
                }
            )
    return rows


def plot_block_stats_bars(summary_rows: List[dict], output_path: Path) -> None:
    labels = [ENTK_CONDITION_LABELS[row["condition"]] for row in summary_rows]
    x = np.arange(len(labels), dtype=np.float32)
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, [row["within_harmful_mean"] for row in summary_rows], width=width, label="within harmful")
    ax.bar(x, [row["within_seemingly_mean"] for row in summary_rows], width=width, label="within seemingly-toxic")
    ax.bar(x + width, [row["cross_mean"] for row in summary_rows], width=width, label="cross harmful↔seemingly")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean normalized ||K^t||_F")
    ax.set_title("WJB eNTK block means by belief condition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cross_within_ratio(summary_rows: List[dict], output_path: Path) -> None:
    labels = [ENTK_CONDITION_LABELS[row["condition"]] for row in summary_rows]
    ratios = [row["cross_within_ratio"] for row in summary_rows]
    colors = ["tab:gray", "tab:green", "tab:orange", "tab:purple"]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, ratios, color=colors[: len(labels)])
    ax.set_ylabel("cross / within mean")
    ax.set_title("WJB eNTK cross-within ratio by belief condition")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _annotate_category_block(ax, block: np.ndarray, vmin: float, vmax: float, title: str):
    n = block.shape[0]
    im = ax.imshow(block, cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(ENTK_CATEGORY_LABELS)
    ax.set_yticklabels(ENTK_CATEGORY_LABELS)
    ax.set_title(title)
    mid = 0.5 * (vmin + vmax)
    for row in range(n):
        for col in range(n):
            value = block[row, col]
            text_color = "white" if value > mid else "black"
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=10)
    return im


def load_category_blocks(
    output_dir: Path,
    conditions: List[str],
    class_slices: Dict[int, slice],
) -> Dict[str, np.ndarray]:
    blocks: Dict[str, np.ndarray] = {}
    for condition in conditions:
        raw = np.load(output_dir / f"knorm_matrix_{condition}.npy")
        normalized = normalize_pairwise_kt_matrix(raw)
        block = category_block_matrix(normalized, class_slices)
        np.save(output_dir / f"category_block_{condition}.npy", block)
        blocks[condition] = block
    return blocks


def plot_entk_category_blocks_grid(
    output_dir: Path,
    conditions: List[str],
    class_slices: Dict[int, slice],
) -> None:
    blocks = load_category_blocks(output_dir, conditions, class_slices)
    stacked = np.stack(list(blocks.values()))
    vmin = float(stacked.min())
    vmax = float(stacked.max())

    n = len(conditions)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()

    im0 = None
    panel_letters = "abcd"
    for panel_idx, (condition, block) in enumerate(blocks.items()):
        letter = panel_letters[panel_idx] if panel_idx < len(panel_letters) else str(panel_idx + 1)
        title = f"({letter}) {ENTK_CONDITION_LABELS[condition]}"
        im0 = _annotate_category_block(axes_flat[panel_idx], block, vmin, vmax, title)

    for ax in axes_flat[len(conditions) :]:
        ax.axis("off")

    fig.subplots_adjust(right=0.90, hspace=0.35, wspace=0.30)
    cbar = fig.colorbar(im0, ax=axes_flat[: len(conditions)], fraction=0.046, pad=0.04)
    cbar.set_label(r"Mean normalized $\|K^t(x', x)\|_F$")
    fig.suptitle(
        r"WJB eNTK — normalized $\|K^t\|_F$ (off-diag within class on diagonal blocks)",
        y=1.02,
        fontsize=11,
    )
    fig.savefig(output_dir / "entk_category_blocks_grid.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_plots(
    output_dir: Path,
    summary_rows: List[dict],
    class_slices: Dict[int, slice],
    conditions: List[str],
) -> None:
    plot_entk_category_blocks_grid(output_dir, conditions, class_slices)
    plot_block_stats_bars(summary_rows, output_dir / "entk_block_stats_bars.png")
    plot_cross_within_ratio(summary_rows, output_dir / "entk_cross_within_ratio.png")
    print(f"Wrote plots to {output_dir}")


def plot_existing_outputs(output_dir: Path) -> None:
    summary_rows = load_block_stats_csv(output_dir)
    with (output_dir / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    from .rep_belief_selection import (
        HARMFUL_CLUSTER_ID,
        SEEMINGLY_TOXIC_CLUSTER_ID,
        VANILLA_BENIGN_CLUSTER_ID,
        class_slices,
    )

    n_normal = config.get("n_normal", config.get("n_vanilla_benign", 0))
    if n_normal <= 0:
        raise ValueError(
            "eNTK outputs are from an old run without a normal/Alpaca class count in config.json. "
            "Re-run the eNTK job with the updated pipeline."
        )
    class_counts = {
        HARMFUL_CLUSTER_ID: config["n_harmful"],
        SEEMINGLY_TOXIC_CLUSTER_ID: config["n_seemingly"],
        VANILLA_BENIGN_CLUSTER_ID: n_normal,
    }
    write_plots(
        output_dir,
        summary_rows,
        class_slices(class_counts),
        config["conditions"],
    )


def run(args: argparse.Namespace) -> None:
    import torch

    from .rep_belief_selection import (
        HARMFUL_CLUSTER_ID,
        SEEMINGLY_TOXIC_CLUSTER_ID,
        VANILLA_BENIGN_CLUSTER_ID,
        class_counts_in_subset,
        class_slices,
        load_belief_cache,
    )

    output_dir = Path(args.output_dir) / "entk_subset"
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = load_belief_cache(Path(args.beliefs_cache))
    subset = select_entk_subset(
        cache,
        args.per_class,
        args.seed,
        alpaca_dataset=args.alpaca_dataset,
        alpaca_split=args.alpaca_split,
    )
    save_entk_subset_manifest(subset, output_dir)
    counts = class_counts_in_subset(subset)
    slices = class_slices(counts)
    n_harmful = counts[HARMFUL_CLUSTER_ID]
    n_seemingly = counts[SEEMINGLY_TOXIC_CLUSTER_ID]
    n_normal = counts[VANILLA_BENIGN_CLUSTER_ID]

    device = args.device
    dtype = torch.bfloat16 if args.use_bf16 else torch.float16

    summary_rows = []
    for condition in ENTK_CONDITIONS:
        print(f"=== eNTK condition: {condition} ===")
        texts = entk_input_texts_for_condition(subset, condition)
        model_name = args.dcr_model_name if condition == DCR_ENTK_CONDITION else args.model_name
        raw_matrix = compute_knorm_matrix(
            texts=texts,
            model_name=model_name,
            device=device,
            dtype=dtype,
            max_length=args.max_length,
            topk_o=args.topk_o,
            topk_u=args.topk_u,
            param_mode=args.param_mode,
            last_n=args.last_n,
            jvp_eps=args.jvp_eps,
            column_batch_size=args.column_batch_size,
        )
        np.save(output_dir / f"knorm_matrix_{condition}.npy", raw_matrix)
        matrix = normalize_pairwise_kt_matrix(raw_matrix)
        stats = block_stats(matrix, slices)
        stats["condition"] = condition
        summary_rows.append(stats)
        print(
            f"  model={model_name} within_mean={stats['within_mean']:.6f}, "
            f"cross_mean={stats['cross_mean']:.6f}, ratio={stats['cross_within_ratio']:.6f}"
        )

    write_block_stats_csv(output_dir, summary_rows)

    config = {
        "model_name": args.model_name,
        "dcr_model_name": args.dcr_model_name,
        "beliefs_cache": args.beliefs_cache,
        "alpaca_dataset": args.alpaca_dataset,
        "alpaca_split": args.alpaca_split,
        "input_format": "belief_first_prompt_second",
        "truncation": args.max_length is not None,
        "per_class": args.per_class,
        "seed": args.seed,
        "n_harmful": n_harmful,
        "n_seemingly": n_seemingly,
        "n_normal": n_normal,
        "n_vanilla_benign": n_normal,
        "class_order": ["S-Toxic", "Toxic", "Normal"],
        "normal_source": "alpaca",
        "max_length": args.max_length,
        "topk_o": args.topk_o,
        "topk_u": args.topk_u,
        "param_mode": args.param_mode,
        "last_n": args.last_n,
        "jvp_eps": args.jvp_eps,
        "column_batch_size": args.column_batch_size,
        "pairwise_metric": "frob",
        "pairwise_normalize": "frob_div_self_x_prime",
        "conditions": list(ENTK_CONDITIONS),
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    write_plots(output_dir, summary_rows, slices, list(ENTK_CONDITIONS))
    print(f"Wrote eNTK outputs to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WJB small-scale eNTK subset experiment.")
    parser.add_argument("--beliefs_cache", type=str, default="data/WJB-Rep/wjb_belief_cache.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/wjb_layer_sep")
    parser.add_argument("--model_name", type=str, default="HuggingFaceH4/zephyr-7b-beta")
    parser.add_argument(
        "--dcr_model_name",
        type=str,
        default=None,
        help="DCR-trained checkpoint for the question_dcr_model condition (raw prompts, no belief).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--per_class", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Optional token cap. If omitted, full inputs are encoded with no truncation.",
    )
    parser.add_argument("--alpaca_dataset", type=str, default="tatsu-lab/alpaca")
    parser.add_argument("--alpaca_split", type=str, default="train")
    parser.add_argument("--topk_o", type=int, default=10)
    parser.add_argument("--topk_u", type=int, default=10)
    parser.add_argument("--param_mode", type=str, default="lastN+head", choices=["all", "lora-only", "lastN+head"])
    parser.add_argument("--last_n", type=int, default=2)
    parser.add_argument("--jvp_eps", type=float, default=1e-3)
    parser.add_argument(
        "--column_batch_size",
        type=int,
        default=10,
        help="Unused (kept for CLI compatibility). Each eNTK column uses its own forward pass.",
    )
    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Regenerate plots from existing entk_subset outputs (block_stats.csv + knorm matrices).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plot_only:
        plot_existing_outputs(Path(args.output_dir) / "entk_subset")
        return
    if not args.dcr_model_name:
        raise ValueError("--dcr_model_name is required unless --plot_only")
    run(args)


if __name__ == "__main__":
    main()
