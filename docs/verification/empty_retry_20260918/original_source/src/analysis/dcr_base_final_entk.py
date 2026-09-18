#!/usr/bin/env python3
"""
Base vs final eNTK similarity experiment for DCR.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
DCR_EXPERIMENT_DIR = REPO_ROOT / "dcr" / "DCR-main" / "experiment"
sys.path.insert(0, str(DCR_EXPERIMENT_DIR))
from KT_similarity_jvp import compute_single_column_vjp_jvp_fd, pick_params, select_tokens_from_logits  # noqa: E402


CATEGORIES = ("seemingly_toxic", "toxic", "general")
PAIR_ORDER = (
    "seemingly_toxic-seemingly_toxic",
    "seemingly_toxic-toxic",
    "seemingly_toxic-general",
    "toxic-seemingly_toxic",
    "toxic-toxic",
    "toxic-general",
    "general-seemingly_toxic",
    "general-toxic",
    "general-general",
)


def normalize_prompt(text: str) -> str:
    return " ".join(text.strip().split())


def resolve_prompt_column(column_names: List[str]) -> str:
    for candidate in ("prompt", "instruction", "query", "text"):
        if candidate in column_names:
            return candidate
    raise ValueError(f"Prompt column not found. Available columns: {column_names}")


def resolve_alpaca_column(column_names: List[str]) -> str:
    for candidate in ("instruction", "prompt", "text"):
        if candidate in column_names:
            return candidate
    raise ValueError(f"Alpaca prompt column not found. Available columns: {column_names}")


def build_prompt_manifest(
    output_path: Path,
    xstest_dataset: str,
    xstest_split: str,
    alpaca_dataset: str,
    alpaca_split: str,
    per_class: int,
    seed: int,
) -> List[dict]:
    rng = random.Random(seed)
    entries: List[dict] = []

    xstest = load_dataset(xstest_dataset, split=xstest_split)
    prompt_col = resolve_prompt_column(list(xstest.column_names))
    if "label" not in xstest.column_names:
        raise ValueError(f"XSTest label column missing. Available columns: {xstest.column_names}")

    seemingly_pool: List[str] = []
    toxic_pool: List[str] = []
    seen_xstest = set()
    for row in xstest:
        prompt = normalize_prompt(row[prompt_col])
        label = row["label"].strip().lower()
        if prompt in seen_xstest:
            continue
        seen_xstest.add(prompt)
        if label == "safe":
            seemingly_pool.append(prompt)
        elif label == "unsafe":
            toxic_pool.append(prompt)

    if len(seemingly_pool) < per_class:
        raise ValueError(f"Not enough seemingly-toxic prompts: need {per_class}, found {len(seemingly_pool)}")
    if len(toxic_pool) < per_class:
        raise ValueError(f"Not enough toxic prompts: need {per_class}, found {len(toxic_pool)}")

    selected_seemingly = rng.sample(seemingly_pool, per_class)
    selected_toxic = rng.sample(toxic_pool, per_class)

    alpaca = load_dataset(alpaca_dataset, split=alpaca_split)
    alpaca_col = resolve_alpaca_column(list(alpaca.column_names))
    general_pool: List[str] = []
    seen_general = set()
    for row in alpaca:
        prompt = normalize_prompt(row[alpaca_col])
        if prompt in seen_general:
            continue
        seen_general.add(prompt)
        general_pool.append(prompt)

    if len(general_pool) < per_class:
        raise ValueError(f"Not enough general prompts: need {per_class}, found {len(general_pool)}")
    selected_general = rng.sample(general_pool, per_class)

    for idx, prompt in enumerate(selected_seemingly):
        entries.append(
            {
                "prompt_id": f"seemingly_toxic_{idx:03d}",
                "category": "seemingly_toxic",
                "prompt": prompt,
                "source_dataset": xstest_dataset,
                "source_label": "safe",
            }
        )
    for idx, prompt in enumerate(selected_toxic):
        entries.append(
            {
                "prompt_id": f"toxic_{idx:03d}",
                "category": "toxic",
                "prompt": prompt,
                "source_dataset": xstest_dataset,
                "source_label": "unsafe",
            }
        )
    for idx, prompt in enumerate(selected_general):
        entries.append(
            {
                "prompt_id": f"general_{idx:03d}",
                "category": "general",
                "prompt": prompt,
                "source_dataset": alpaca_dataset,
                "source_label": "general",
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in entries:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return entries


def load_manifest(manifest_path: Path, per_class: int) -> List[dict]:
    rows: List[dict] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    counts = {category: 0 for category in CATEGORIES}
    for row in rows:
        counts[row["category"]] += 1
    for category in CATEGORIES:
        if counts[category] != per_class:
            raise ValueError(
                f"Manifest category count mismatch for {category}: expected {per_class}, found {counts[category]}"
            )
    return rows


def compute_k_block_loaded(
    model,
    tokenizer,
    params: List[torch.nn.Parameter],
    text_o: str,
    text_u: str,
    pos_o: List[int],
    pos_u: List[int],
    topk_o: int,
    topk_u: int,
    max_length: int,
    jvp_eps: float,
) -> torch.Tensor:
    with torch.no_grad():
        enc_o = tokenizer(text_o, return_tensors="pt", truncation=True, max_length=max_length).to("cpu")
        logits_o = model(**enc_o).logits
        enc_u = tokenizer(text_u, return_tensors="pt", truncation=True, max_length=max_length).to("cpu")
        logits_u = model(**enc_u).logits

    token_ids_per_pos_o = [select_tokens_from_logits(logits_o, p, topk=topk_o) for p in pos_o]
    token_ids_per_pos_u = [select_tokens_from_logits(logits_u, p, topk=topk_u) for p in pos_u]

    col_specs: List[Tuple[int, int]] = []
    for p, tids in zip(pos_u, token_ids_per_pos_u):
        t_u = logits_u.size(1)
        pos_eff = t_u + p if p < 0 else p
        pos_eff = max(0, min(pos_eff, t_u - 1))
        for tid in tids:
            col_specs.append((pos_eff, tid))

    num_rows = sum(len(tids) for tids in token_ids_per_pos_o)
    num_cols = len(col_specs)
    k_block = torch.empty((num_rows, num_cols), dtype=logits_o.dtype, device="cpu")

    col_ptr = 0
    for pos_u_eff, tid_u in col_specs:
        cols_per_pos = compute_single_column_vjp_jvp_fd(
            model=model,
            tok=tokenizer,
            params=params,
            text_u=text_u,
            pos_u=pos_u_eff,
            tid_u=tid_u,
            text_o=text_o,
            pos_o_list=pos_o,
            eps=jvp_eps,
            device="cpu",
            max_length=max_length,
        )
        row_ptr = 0
        for tids, col_vec in zip(token_ids_per_pos_o, cols_per_pos):
            width = len(tids)
            k_block[row_ptr : row_ptr + width, col_ptr] = col_vec[tids]
            row_ptr += width
        col_ptr += 1
    return k_block


def compute_knorm_matrix(
    model_name_or_path: str,
    prompts: List[str],
    topk_o: int,
    topk_u: int,
    max_length: int,
    jvp_eps: float,
    param_mode: str,
    last_n: int,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float32).to("cpu")
    model.eval()
    params = pick_params(model, mode=param_mode, last_n=last_n)
    if len(params) == 0:
        raise ValueError("No parameters selected for eNTK computation.")

    n = len(prompts)
    matrix = np.zeros((n, n), dtype=np.float32)
    for i, text_o in enumerate(prompts):
        for j, text_u in enumerate(prompts):
            k_block = compute_k_block_loaded(
                model=model,
                tokenizer=tokenizer,
                params=params,
                text_o=text_o,
                text_u=text_u,
                pos_o=[-1],
                pos_u=[-1],
                topk_o=topk_o,
                topk_u=topk_u,
                max_length=max_length,
                jvp_eps=jvp_eps,
            )
            frob = float(k_block.norm().item())
            matrix[i, j] = frob
            print(f"[{model_name_or_path}] pair ({i + 1}/{n}, {j + 1}/{n}) -> {frob:.6f}", flush=True)
    return matrix


def category_indices(rows: List[dict]) -> Dict[str, List[int]]:
    idx: Dict[str, List[int]] = {category: [] for category in CATEGORIES}
    for i, row in enumerate(rows):
        idx[row["category"]].append(i)
    return idx


def block_means(matrix: np.ndarray, idx: Dict[str, List[int]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for a in CATEGORIES:
        for b in CATEGORIES:
            a_idx = np.array(idx[a], dtype=np.int64)
            b_idx = np.array(idx[b], dtype=np.int64)
            block = matrix[np.ix_(a_idx, b_idx)]
            out[f"{a}-{b}"] = float(block.mean())
    return out


def write_block_csv(path: Path, base_stats: Dict[str, float], final_stats: Dict[str, float]) -> List[dict]:
    rows: List[dict] = []
    for pair in PAIR_ORDER:
        base_val = base_stats[pair]
        final_val = final_stats[pair]
        abs_delta = final_val - base_val
        rel_delta = abs_delta / base_val
        rows.append(
            {
                "pair": pair,
                "base_mean": base_val,
                "final_mean": final_val,
                "abs_delta": abs_delta,
                "rel_delta": rel_delta,
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair", "base_mean", "final_mean", "abs_delta", "rel_delta"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_plot(path: Path, rows: List[dict]) -> None:
    labels = [r["pair"] for r in rows]
    base_vals = [r["base_mean"] for r in rows]
    final_vals = [r["final_mean"] for r in rows]
    x = np.arange(len(labels), dtype=np.float32)
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width / 2, base_vals, width=width, label="base")
    ax.bar(x + width / 2, final_vals, width=width, label="final")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Averaged ||K_t(x',x)||_F")
    ax.set_title("Base vs Final eNTK Block Means")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Base vs final eNTK similarity experiment.")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--final_model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/dcr_base_final_entk")
    parser.add_argument("--manifest_path", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per_class", type=int, default=25)
    parser.add_argument("--xstest_dataset", type=str, default="walledai/XSTest")
    parser.add_argument("--xstest_split", type=str, default="train")
    parser.add_argument("--alpaca_dataset", type=str, default="tatsu-lab/alpaca")
    parser.add_argument("--alpaca_split", type=str, default="train")
    parser.add_argument("--topk_o", type=int, default=10)
    parser.add_argument("--topk_u", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--jvp_eps", type=float, default=1e-3)
    parser.add_argument("--param_mode", type=str, default="lastN+head", choices=["all", "lora-only", "lastN+head"])
    parser.add_argument("--last_n", type=int, default=2)
    args = parser.parse_args()

    if torch.cuda.is_available():
        raise RuntimeError("GPU is disabled for this experiment. Run with CUDA_VISIBLE_DEVICES=''")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest_path) if args.manifest_path else (output_dir / "prompt_manifest_25_25_25.jsonl")

    if manifest_path.exists():
        manifest_rows = load_manifest(manifest_path, args.per_class)
    else:
        manifest_rows = build_prompt_manifest(
            output_path=manifest_path,
            xstest_dataset=args.xstest_dataset,
            xstest_split=args.xstest_split,
            alpaca_dataset=args.alpaca_dataset,
            alpaca_split=args.alpaca_split,
            per_class=args.per_class,
            seed=args.seed,
        )

    prompts = [row["prompt"] for row in manifest_rows]
    idx = category_indices(manifest_rows)

    base_matrix = compute_knorm_matrix(
        model_name_or_path=args.base_model,
        prompts=prompts,
        topk_o=args.topk_o,
        topk_u=args.topk_u,
        max_length=args.max_length,
        jvp_eps=args.jvp_eps,
        param_mode=args.param_mode,
        last_n=args.last_n,
    )
    final_matrix = compute_knorm_matrix(
        model_name_or_path=args.final_model,
        prompts=prompts,
        topk_o=args.topk_o,
        topk_u=args.topk_u,
        max_length=args.max_length,
        jvp_eps=args.jvp_eps,
        param_mode=args.param_mode,
        last_n=args.last_n,
    )

    np.save(output_dir / "pairwise_knorm_base.npy", base_matrix)
    np.save(output_dir / "pairwise_knorm_final.npy", final_matrix)

    base_stats = block_means(base_matrix, idx)
    final_stats = block_means(final_matrix, idx)
    rows = write_block_csv(output_dir / "block_means.csv", base_stats, final_stats)
    write_plot(output_dir / "base_final_block_means.png", rows)

    drop_pair = min(rows, key=lambda r: r["abs_delta"])
    summary = {
        "base_model": args.base_model,
        "final_model": args.final_model,
        "seed": args.seed,
        "per_class": args.per_class,
        "manifest_path": str(manifest_path),
        "settings": {
            "pos_o": [-1],
            "pos_u": [-1],
            "topk_o": args.topk_o,
            "topk_u": args.topk_u,
            "max_length": args.max_length,
            "jvp_eps": args.jvp_eps,
            "param_mode": args.param_mode,
            "last_n": args.last_n,
            "device": "cpu",
        },
        "key_readout": {
            "base_seemingly_toxic": base_stats["seemingly_toxic-toxic"],
            "final_seemingly_toxic": final_stats["seemingly_toxic-toxic"],
            "abs_delta_seemingly_toxic": final_stats["seemingly_toxic-toxic"] - base_stats["seemingly_toxic-toxic"],
            "rel_delta_seemingly_toxic": (final_stats["seemingly_toxic-toxic"] - base_stats["seemingly_toxic-toxic"])
            / base_stats["seemingly_toxic-toxic"],
            "largest_drop_pair": drop_pair["pair"],
            "largest_drop_abs_delta": drop_pair["abs_delta"],
            "base_seemingly_general": base_stats["seemingly_toxic-general"],
            "base_general_general": base_stats["general-general"],
            "final_seemingly_general": final_stats["seemingly_toxic-general"],
            "final_general_general": final_stats["general-general"],
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote matrices: {output_dir / 'pairwise_knorm_base.npy'} and {output_dir / 'pairwise_knorm_final.npy'}")
    print(f"Wrote block means: {output_dir / 'block_means.csv'}")
    print(f"Wrote plot: {output_dir / 'base_final_block_means.png'}")
    print(f"Wrote summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
