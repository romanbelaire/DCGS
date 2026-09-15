"""Two-head low-level token critic for LL candidate rerank.

Loads the offline Shapley/TD/span checkpoint (`harm_head` + `follow_head`) and
scores a candidate utterance from last-layer Zephyr residuals on the action-token
span. Encoding matches `scripts/encode_ll_action_token_hiddens.py` (float16
backbone, Observation / High-Level Context / Low-Level Action prefix, BPE
straddles assigned to the action). Student weights: ../../models/README.md.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

HIDDEN_SIZE = 4096
MLP_WIDTH_MULT = 2.0
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
MAX_UNTRUNCATED_LEN = 8192
OBJECTIVES = ("shapley", "td", "span")
PREFIX_TEMPLATE = (
    "Observation: {observation}\n"
    "High-Level Context: {selected_belief}\n"
    "Low-Level Action: "
)


def even_spans(n: int, k: int) -> list:
    spans = []
    for i in range(k):
        start = (i * n) // k
        end = ((i + 1) * n) // k
        if end <= start:
            raise RuntimeError(f"Empty even span {i} for n={n} k={k}")
        spans.append(list(range(start, end)))
    return spans


def make_head(hidden_size: int, width_mult: float) -> nn.Sequential:
    mid = max(1, int(hidden_size * width_mult) // 2)
    return nn.Sequential(
        nn.Linear(hidden_size, mid),
        nn.ReLU(),
        nn.Linear(mid, 1),
    )


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


class LLTokenCritic:
    def __init__(
        self,
        harm_head: nn.Module,
        follow_head: nn.Module,
        encoder,
        tokenizer,
        device: str,
        objective: str,
        span_k: int,
    ):
        if objective not in OBJECTIVES:
            raise RuntimeError(f"Unknown LL critic objective {objective!r}")
        self.harm_head = harm_head
        self.follow_head = follow_head
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.device = device
        self.objective = objective
        self.span_k = span_k

    @classmethod
    def from_checkpoint(cls, path: str, device: str, backbone=None) -> "LLTokenCritic":
        ckpt_path = Path(path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Missing LL token critic checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        hidden_size = ckpt["hidden_size"]
        width_mult = ckpt["mlp_width_mult"]
        if hidden_size != HIDDEN_SIZE:
            raise RuntimeError(f"hidden_size {hidden_size} != {HIDDEN_SIZE}")
        if width_mult != MLP_WIDTH_MULT:
            raise RuntimeError(f"mlp_width_mult {width_mult} != {MLP_WIDTH_MULT}")
        objective = ckpt["objective"]
        if objective not in OBJECTIVES:
            raise RuntimeError(f"Unknown LL critic objective {objective!r} in {ckpt_path}")
        if "k" not in ckpt:
            raise RuntimeError(f"LL critic checkpoint missing k: {ckpt_path}")
        span_k = int(ckpt["k"])
        if objective == "span" and span_k < 1:
            raise RuntimeError(f"span critic k must be >= 1, got {span_k}")
        harm_head = make_head(hidden_size, width_mult)
        follow_head = make_head(hidden_size, width_mult)
        harm_head.load_state_dict(ckpt["harm_head"])
        follow_head.load_state_dict(ckpt["follow_head"])
        harm_head.to(device)
        follow_head.to(device)
        harm_head.eval()
        follow_head.eval()
        if backbone is None:
            print(f"Loading LL token critic backbone on {device}: {MODEL_NAME} float16")
            backbone = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True,
            )
            backbone.eval()
        else:
            print(f"Reusing shared backbone for LL token critic on {device}")
        encoder = backbone.model
        encoder.eval()
        print("LL critic encoder is CausalLM.model (last_hidden_state, matches action-token dump)")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        print(
            f"Loaded LL token critic from {ckpt_path} "
            f"(objective={objective}, n_examples={ckpt['n_examples']})"
        )
        return cls(
            harm_head,
            follow_head,
            encoder,
            tokenizer,
            device,
            objective,
            span_k,
        )

    def score_action(self, observation: str, selected_belief: str, ll_action: str) -> float:
        return self.score_actions(observation, selected_belief, [ll_action])[ll_action]

    def _score_action_hiddens(self, action_h: torch.Tensor) -> torch.Tensor:
        n = action_h.shape[0]
        if n < 1:
            raise RuntimeError("Empty action-token residual for LL critic")
        if self.objective == "shapley":
            harm = self.harm_head(action_h).squeeze(-1)
            follow = self.follow_head(action_h).squeeze(-1)
            return (0.5 * (harm + follow)).mean()
        if self.objective == "td":
            harm = self.harm_head(action_h[-1]).squeeze()
            follow = self.follow_head(action_h[-1]).squeeze()
            return 0.5 * (harm + follow)
        if self.objective != "span":
            raise RuntimeError(f"Unscored LL critic objective {self.objective!r}")
        pooled = []
        for span in even_spans(n, self.span_k):
            pooled.append(action_h[span].mean(dim=0))
        span_h = torch.stack(pooled, dim=0)
        harm = self.harm_head(span_h).squeeze(-1)
        follow = self.follow_head(span_h).squeeze(-1)
        return (0.5 * (harm + follow)).sum()

    def score_actions(
        self,
        observation: str,
        selected_belief: str,
        actions: list,
    ) -> dict:
        if not actions:
            raise RuntimeError("No LL candidates to score")
        if not observation.strip():
            raise RuntimeError("Empty observation for LL token critic")
        if not selected_belief.strip():
            raise RuntimeError("Empty selected belief for LL token critic")
        prefix = PREFIX_TEMPLATE.format(
            observation=observation,
            selected_belief=selected_belief,
        )
        rows = []
        spans = []
        for action in actions:
            if not action.strip():
                raise RuntimeError("Empty LL action for LL token critic")
            full = prefix + action
            full_enc, prefix_len, action_end = tokenize_action_span(
                self.tokenizer, prefix, full
            )
            rows.append((full_enc["input_ids"][0], full_enc["attention_mask"][0]))
            spans.append((prefix_len, action_end))
        max_len = max(ids.shape[0] for ids, _mask in rows)
        pad_id = self.tokenizer.pad_token_id
        padded_ids = []
        padded_mask = []
        for ids, mask in rows:
            pad = max_len - ids.shape[0]
            if pad:
                ids = torch.cat([ids, ids.new_full((pad,), pad_id)])
                mask = torch.cat([mask, mask.new_zeros((pad,))])
            padded_ids.append(ids)
            padded_mask.append(mask)
        input_ids = torch.stack(padded_ids, dim=0).to(self.device)
        attention_mask = torch.stack(padded_mask, dim=0).to(self.device)
        scores = {}
        with torch.inference_mode():
            hidden = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).last_hidden_state
            if hidden.shape[-1] != HIDDEN_SIZE:
                raise RuntimeError(
                    f"last_hidden_state width {hidden.shape[-1]} != {HIDDEN_SIZE}"
                )
            for action, (start, end), row_h in zip(actions, spans, hidden):
                action_h = row_h[start:end].float()
                scores[action] = float(self._score_action_hiddens(action_h).item())
        return scores
