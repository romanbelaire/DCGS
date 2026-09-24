"""Read frozen smoke outputs and write the matched four-format comparison."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / "outputs/safedial_baseline"
ARMS = {
    "training": ["dcr_qwen_1.5b_smoke_v1", "dcr_qwen_1.5b_prompt_training_ids2-3_smoke_v1"],
    "raw": ["dcr_qwen_1.5b_prompt_raw_ids1-2-3_smoke_v1"],
    "zephyr": ["dcr_qwen_1.5b_prompt_zephyr_ids1-2-3_smoke_v1"],
    "qwen_standard": ["dcr_qwen_1.5b_prompt_qwen_standard_ids1-2-3_smoke_v1"],
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    manifests, records, sources = {}, {}, {}
    for arm, directories in ARMS.items():
        records[arm], sources[arm] = {}, []
        for directory in directories:
            path = BASE / directory
            m = json.loads((path / "run_config.json").read_text())
            manifests[directory] = m
            for filename, digest in m["source_sha256"].items():
                assert sha(ROOT / "scripts" / filename) == digest, (directory, filename)
            for r in jsonl(path / "turns.jsonl"):
                key = (r["dialogue_id"], r["turn_index"])
                assert key not in records[arm]
                records[arm][key] = r
            sources[arm].append({"directory": str(path),
                "files_sha256": {p.name: sha(p) for p in path.iterdir() if p.is_file() and p.suffix in (".json", ".jsonl")},
                "runtime": json.loads((path / "runtime_stats.json").read_text()),
                "validation": json.loads((path / "validation.json").read_text()) if (path / "validation.json").exists() else None})
    first = next(iter(manifests.values()))
    for m in manifests.values():
        for key in ["dataset_sha256", "model", "revision", "dtype", "seed", "temperature", "top_p", "num_choices", "max_new_tokens"]:
            assert m[key] == first[key], key
        for key in ["base_sha256", "sha256"]:
            assert m["adapter"][key] == first["adapter"][key], key
    expected = {(dialogue, turn) for dialogue in (1, 2, 3) for turn in range(5)}
    for arm, rows in records.items():
        assert set(rows) == expected, (arm, len(rows))
        for key, r in rows.items():
            reference = records["training"][key]
            for field in ["seed", "prompt_history", "user_message", "reference_response"]:
                assert r[field] == reference[field], (arm, key, field)

    raw_path = BASE / ARMS["qwen_standard"][0] / "raw_generations.jsonl"
    raw = {r["seed"]: r for r in jsonl(raw_path)}
    assert len(raw) == 15
    arms = {}
    responses = ["# Matched DCR smoke responses", "", "All 60 recorded outputs, unchanged. Turns below are one-based.", ""]
    for key in sorted(expected):
        responses.extend([f"## Dialogue {key[0]}, turn {key[1] + 1}", ""])
        for arm in ARMS:
            r = records[arm][key]
            responses.extend([f"### {arm}", "", r["generated_response"], ""])
    for arm, rows in records.items():
        metrics = []
        for key, r in sorted(rows.items()):
            text = r["generated_response"]
            ids = r.get("completion_token_ids")
            if arm == "qwen_standard":
                actual = raw[r["seed"]]
                assert actual["rendered_prompt_sha256"] == r["rendered_prompt_sha256"]
                if not r["error"]:
                    assert actual["completion_token_ids"] == ids
                    assert actual["message"] == text
                ids = actual["completion_token_ids"]
            role_lines = len(re.findall(r"(?m)^(?:USER|ASSISTANT|SYSTEM):", text))
            zephyr_roles = len(re.findall(r"<\|(?:user|assistant|system)\|>", text))
            qwen_starts = ids.count(151644) if ids is not None else None
            lines = Counter(line.strip() for line in text.splitlines() if line.strip())
            words = text.split()
            ngrams = Counter(tuple(words[i:i + 8]) for i in range(max(0, len(words) - 7)))
            metrics.append({"dialogue_id": key[0], "turn_index": key[1], "error": r["error"],
                "completion_tokens": len(ids) if ids is not None else (None if r["error"] else r["completion_tokens"]),
                "cap_hit": len(ids) == 1024 if ids is not None else (None if r["error"] else r["completion_tokens"] == 1024),
                "last_token_is_eos": ids[-1] == 151643 if ids else None,
                "role_lines": role_lines, "zephyr_role_markers": zephyr_roles,
                "qwen_message_starts": qwen_starts,
                "qwen_message_ends": ids.count(151645) if ids is not None else None,
                "has_extra_role_marker": bool(role_lines or zephyr_roles or qwen_starts),
                "cjk_characters": len(re.findall(r"[\u4e00-\u9fff]", text)),
                "arabic_characters": len(re.findall(r"[\u0600-\u06ff]", text)),
                "max_identical_line_occurrences": max(lines.values(), default=0),
                "max_identical_8word_span_occurrences": max(ngrams.values(), default=0)})
        arms[arm] = {"summary": {
            "attempted": len(metrics), "nonempty": sum(not m["error"] for m in metrics),
            "errors": sum(bool(m["error"]) for m in metrics),
            "cap_hits": sum(m["cap_hit"] is True for m in metrics),
            "extra_role_replies": sum(m["has_extra_role_marker"] for m in metrics),
            "cjk_replies": sum(m["cjk_characters"] > 0 for m in metrics),
            "arabic_replies": sum(m["arabic_characters"] > 0 for m in metrics),
            "confirmed_eos_endings": sum(m["last_token_is_eos"] is True for m in metrics),
            "unknown_completion_lengths": sum(m["completion_tokens"] is None for m in metrics),
            "unknown_eos_endings": sum(m["last_token_is_eos"] is None for m in metrics)},
            "turns": metrics, "sources": sources[arm]}
    result = {"matched_artifacts_decoding_histories_seeds_verified": True, "arms": arms,
        "limitations": ["15 selected turns; no quality or safety judging.",
            "Raw arm lost completion IDs for five empty replies; their cap/stop status is unknown.",
            "Original training ID1 lacks raw completion IDs; EOS status is unknown.",
            "Role markers do not measure all repetition or quality.",
            "Qwen standard arm includes default system text and retains endoftext-only stopping."]}
    (OUT / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / "responses.md").write_text("\n".join(responses))
    print(json.dumps({arm: data["summary"] for arm, data in arms.items()}, indent=2))


if __name__ == "__main__":
    main()
