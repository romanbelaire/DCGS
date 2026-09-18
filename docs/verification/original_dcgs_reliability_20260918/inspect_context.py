"""Tokenizer-only inspection of original critic inputs; no model inference."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from run_safedial_dcgs_wildjailbreak import LOCK, tokenizer_for
from run_safedial_baseline import DEFAULT_DATASET, load_jsonl, file_sha256
from safedial_dcgs_wildjailbreak import episode_for


def main():
    tokenizer = tokenizer_for(json.loads(LOCK.read_text()))
    rows = load_jsonl(DEFAULT_DATASET)
    beliefs = ["Explain the risks and offer safe alternatives.",
               "Ask the user to clarify their underlying goal."]
    records = []
    texts, keys = [], []
    for row in rows:
        for turn in range(len(row["history"])):
            observation = episode_for(row, turn).current_obs_for_action
            texts.extend(f"Observation: {observation}\nHigh-Level Context: {belief}" for belief in beliefs)
            keys.append((row["id"], turn))
    for start in range(0, len(keys), 128):
        batch = texts[start * 2:(start + 128) * 2]
        raw = tokenizer(batch, add_special_tokens=True, padding=False, truncation=False)["input_ids"]
        encoded = tokenizer(batch, add_special_tokens=True, max_length=1500, padding=False,
                            truncation=True)["input_ids"]
        for offset, (dialogue, turn) in enumerate(keys[start:start + 128]):
            i = offset * 2
            records.append({"dialogue_id": dialogue, "turn_index": turn,
                "untruncated_lengths": [len(raw[i]), len(raw[i + 1])],
                "truncated": any(len(raw[j]) > len(encoded[j]) for j in (i, i + 1)),
                "distinct_beliefs_have_identical_critic_tokens": encoded[i] == encoded[i + 1]})
    report = {"dataset_sha256": file_sha256(DEFAULT_DATASET), "dialogues": len(rows),
        "turns": len(records), "tokenizer_padding_side": tokenizer.padding_side,
        "tokenizer_truncation_side": tokenizer.truncation_side,
        "critic_max_length": 1500, "probe_beliefs": beliefs,
        "turns_with_truncation_for_probe_beliefs": sum(r["truncated"] for r in records),
        "turns_with_identical_critic_tokens_for_distinct_probe_beliefs": sum(r["distinct_beliefs_have_identical_critic_tokens"] for r in records),
        "maximum_probe_input_tokens": max(max(r["untruncated_lengths"]) for r in records),
        "dialogue_1": [r for r in records if r["dialogue_id"] == 1],
        "note": "Synthetic beliefs, real gold histories and actual tokenizer encoding. Counts are a diagnostic, not measured generated-candidate statistics.",
        "model_calls": 0}
    output = Path(__file__).with_name("critic_context_review.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
