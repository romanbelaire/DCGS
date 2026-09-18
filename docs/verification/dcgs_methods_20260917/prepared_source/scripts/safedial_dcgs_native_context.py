"""Full-benchmark offset encoding. Same upstream span rule, native 32768 limit.

Kept separate so historical smoke and upstream critic sources remain immutable.
"""
import torch

MAX_UNTRUNCATED_LEN = 32768


def tokenize_action_span(tokenizer, prefix: str, full: str) -> tuple:
    if not full.startswith(prefix):
        raise RuntimeError("full text is not prefix + action")
    if len(full) == len(prefix):
        raise RuntimeError("Action character span is empty")
    action_char_start = len(prefix)
    encoded = tokenizer(
        full,
        add_special_tokens=True,
        truncation=False,
        return_offsets_mapping=True,
        return_attention_mask=True,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    offsets = encoded["offset_mapping"]
    full_len = len(input_ids)
    if full_len > MAX_UNTRUNCATED_LEN:
        raise RuntimeError(
            f"Sequence length {full_len} exceeds {MAX_UNTRUNCATED_LEN} "
            "(refusing to truncate action tokens)"
        )
    action_positions = []
    for i, (start, end) in enumerate(offsets):
        if start == end:
            continue
        if end > action_char_start:
            action_positions.append(i)
    if not action_positions:
        raise RuntimeError("No action tokens in offset mapping")
    prefix_len = action_positions[0]
    action_end = action_positions[-1] + 1
    expected = list(range(prefix_len, action_end))
    if action_positions != expected:
        raise RuntimeError(
            f"Action token indices are not contiguous: {action_positions}"
        )
    if action_end > full_len:
        raise RuntimeError(f"Action end {action_end} > sequence length {full_len}")
    full_enc = {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
    }
    return full_enc, prefix_len, action_end

