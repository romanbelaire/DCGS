"""Load defense_embedding_summary.json from multiple runs and print a comparison table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SUMMARY_FILENAME = "defense_embedding_summary.json"


def load_summary(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing summary: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_dist(v: Any) -> str:
    if v is None:
        return "n/a"
    return f"{v:.6f}"


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Compare defense embedding metrics across runs")
    p.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="Directories containing defense_embedding_summary.json",
    )
    p.add_argument(
        "--label",
        nargs="*",
        default=None,
        help="Optional labels for each run (same length as run_dirs)",
    )
    args = p.parse_args(argv)

    labels = args.label
    if labels is not None and len(labels) != len(args.run_dirs):
        raise ValueError("--label count must match run_dirs count")

    rows = []
    for i, run_dir in enumerate(args.run_dirs):
        run_dir = run_dir.resolve()
        summ = load_summary(run_dir / SUMMARY_FILENAME)
        counts = summ["counts"]
        succ = summ["success_subset"]
        fail = summ["failure_subset"]
        label = labels[i] if labels else run_dir.name
        rows.append(
            {
                "label": label,
                "run_dir": str(run_dir),
                "n_adv": counts["after_adversarial_filter"],
                "n_succ": counts["success"],
                "n_fail": counts["failure"],
                "dist_succ": succ["mean_pairwise_cosine_distance"],
                "dist_fail": fail["mean_pairwise_cosine_distance"],
            }
        )

    headers = [
        "label",
        "n_adv",
        "n_succ",
        "n_fail",
        "mean_pairwise_dist_success",
        "mean_pairwise_dist_failure",
    ]
    col_w = [18, 8, 8, 8, 28, 28]
    header_line = "  ".join(h.ljust(col_w[j]) for j, h in enumerate(headers))
    print(header_line)
    print("-" * len(header_line))
    for r in rows:
        line = (
            f"{r['label'][:16]:<18}  "
            f"{r['n_adv']:<8}  "
            f"{r['n_succ']:<8}  "
            f"{r['n_fail']:<8}  "
            f"{format_dist(r['dist_succ']):<28}  "
            f"{format_dist(r['dist_fail']):<28}"
        )
        print(line)

    out = Path("compare_defense_ablations.json")
    with out.open("w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
