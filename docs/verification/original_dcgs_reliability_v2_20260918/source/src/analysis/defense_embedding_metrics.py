"""
Offline metrics on defense_episodes.jsonl: mean pairwise cosine distance (1 - similarity).

Interprets higher mean pairwise distance as more spread in embedding space (diversity).
Reports separately for successful vs failed episodes (goal_achieved vs threshold).
Optional per-attack_prompt_sha256 breakdown when multiple successes exist per key.

Usage:
  python -m src.analysis.defense_embedding_metrics --run_dirs outputs/exp1 outputs/exp2 \\
      --model_name HuggingFaceH4/zephyr-7b-beta --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.belief_evaluation import get_text_embeddings
from ..utils.logging_utils import DEFENSE_EPISODES_FILENAME

SUMMARY_FILENAME = "defense_embedding_summary.json"


def load_defense_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing defense log: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def mean_pairwise_cosine_distance(embeddings: torch.Tensor) -> float:
    """
    Mean of (1 - cosine_similarity) over all unordered pairs i < j.
    embeddings: [n, h] unnormalized (normalized inside).
    """
    n = embeddings.shape[0]
    if n < 2:
        raise ValueError(f"mean_pairwise_cosine_distance requires n >= 2, got {n}")
    emb = F.normalize(embeddings, p=2, dim=1)
    sim = emb @ emb.T
    idx = torch.triu_indices(n, n, offset=1, device=emb.device)
    pairwise_sim = sim[idx[0], idx[1]]
    return float((1.0 - pairwise_sim).mean().item())


def _embed_texts_batched(
    texts: List[str],
    model: AutoModelForCausalLM,
    tokenizer,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        chunks.append(get_text_embeddings(batch, model, tokenizer, device=device))
    if not chunks:
        return torch.empty(0, device=device)
    return torch.cat(chunks, dim=0)


def _subset_metrics(
    records: List[Dict[str, Any]],
    text_field: str,
    model: AutoModelForCausalLM,
    tokenizer,
    device: str,
    embed_batch_size: int,
) -> Dict[str, Any]:
    texts = [str(r.get(text_field, "") or "").strip() for r in records]
    texts = [t for t in texts if t]
    if len(texts) < 2:
        return {
            "count_raw": len(records),
            "count_embedded": len(texts),
            "mean_pairwise_cosine_distance": None,
        }
    emb = _embed_texts_batched(texts, model, tokenizer, device, embed_batch_size)
    return {
        "count_raw": len(records),
        "count_embedded": len(texts),
        "mean_pairwise_cosine_distance": mean_pairwise_cosine_distance(emb),
    }


def _per_attack_metrics(
    records: List[Dict[str, Any]],
    text_field: str,
    success_threshold: float,
    model: AutoModelForCausalLM,
    tokenizer,
    device: str,
    embed_batch_size: int,
) -> Dict[str, Any]:
    by_attack: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        key = r.get("attack_prompt_sha256")
        if not key:
            continue
        by_attack[str(key)].append(r)

    per_attack_out: Dict[str, Any] = {}
    for key, group in sorted(by_attack.items()):
        succ = [r for r in group if float(r["goal_achieved"]) >= success_threshold]
        fail = [r for r in group if float(r["goal_achieved"]) < success_threshold]
        entry: Dict[str, Any] = {
            "n_total": len(group),
            "n_success": len(succ),
            "n_failure": len(fail),
        }
        if len(succ) >= 2:
            m = _subset_metrics(succ, text_field, model, tokenizer, device, embed_batch_size)
            entry["success_mean_pairwise_cosine_distance"] = m["mean_pairwise_cosine_distance"]
        else:
            entry["success_mean_pairwise_cosine_distance"] = None
        if len(fail) >= 2:
            m = _subset_metrics(fail, text_field, model, tokenizer, device, embed_batch_size)
            entry["failure_mean_pairwise_cosine_distance"] = m["mean_pairwise_cosine_distance"]
        else:
            entry["failure_mean_pairwise_cosine_distance"] = None
        per_attack_out[key] = entry

    return per_attack_out


def compute_run_summary(
    records: List[Dict[str, Any]],
    text_field: str,
    adversarial_only: bool,
    success_threshold: float,
    model: AutoModelForCausalLM,
    tokenizer,
    device: str,
    embed_batch_size: int,
    per_attack: bool,
) -> Dict[str, Any]:
    if adversarial_only:
        filtered = [r for r in records if int(r.get("harmful_level", 0) or 0) > 0]
    else:
        filtered = list(records)

    succ = [r for r in filtered if float(r["goal_achieved"]) >= success_threshold]
    fail = [r for r in filtered if float(r["goal_achieved"]) < success_threshold]

    summary: Dict[str, Any] = {
        "text_field": text_field,
        "adversarial_only": adversarial_only,
        "success_threshold": success_threshold,
        "counts": {
            "total_loaded": len(records),
            "after_adversarial_filter": len(filtered),
            "success": len(succ),
            "failure": len(fail),
        },
        "success_subset": _subset_metrics(succ, text_field, model, tokenizer, device, embed_batch_size),
        "failure_subset": _subset_metrics(fail, text_field, model, tokenizer, device, embed_batch_size),
    }
    if per_attack:
        summary["per_attack_prompt_sha256"] = _per_attack_metrics(
            filtered,
            text_field,
            success_threshold,
            model,
            tokenizer,
            device,
            embed_batch_size,
        )
    return summary


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Defense embedding diversity metrics from defense_episodes.jsonl")
    p.add_argument(
        "--run_dirs",
        nargs="+",
        type=Path,
        required=True,
        help="Experiment output directories containing defense_episodes.jsonl",
    )
    p.add_argument("--model_name", type=str, default="HuggingFaceH4/zephyr-7b-beta")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--use_bf16", action="store_true", default=True)
    p.add_argument("--no_bf16", action="store_true", help="Disable bf16 (e.g. CPU)")
    p.add_argument(
        "--text_field",
        type=str,
        default="defense_last_ll_action",
        choices=("defense_last_ll_action", "defense_transcript", "defense_last_turn"),
    )
    p.add_argument("--adversarial_only", action="store_true", default=True)
    p.add_argument("--include_benign", action="store_true", help="Set adversarial_only=False")
    p.add_argument("--success_threshold", type=float, default=1.0)
    p.add_argument("--embed_batch_size", type=int, default=16)
    p.add_argument("--per_attack", action="store_true", help="Include per attack_prompt_sha256 stats")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    adversarial_only = False if args.include_benign else args.adversarial_only
    use_bf16 = not args.no_bf16

    if args.device == "cpu":
        use_bf16 = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        device_map=None,
    )
    model.eval()
    model.to(args.device)

    all_summaries: Dict[str, Any] = {"runs": {}}
    for run_dir in args.run_dirs:
        run_dir = run_dir.resolve()
        log_path = run_dir / DEFENSE_EPISODES_FILENAME
        records = load_defense_jsonl(log_path)
        summary = compute_run_summary(
            records=records,
            text_field=args.text_field,
            adversarial_only=adversarial_only,
            success_threshold=args.success_threshold,
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            embed_batch_size=args.embed_batch_size,
            per_attack=args.per_attack,
        )
        summary["run_dir"] = str(run_dir)
        out_path = run_dir / SUMMARY_FILENAME
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Wrote {out_path}", file=sys.stderr)
        all_summaries["runs"][str(run_dir)] = summary

    combined_path = Path("defense_embedding_summary_multi.json")
    with combined_path.open("w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    print(f"Wrote {combined_path.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
