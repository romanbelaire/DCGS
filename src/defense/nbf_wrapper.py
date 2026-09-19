"""Neural Barrier Function (Hu, Robey, Liu; TMLR 2026) inference-time filter.

Architecture and scoring match https://github.com/HanjiangHu/NBF-LLM:
sentence-transformer user embeddings drive a neural state-space model; a
five-class barrier network scores the latent state. The query is treated as
unsafe when the last-class probability minus the max of the other classes is
at least -threshold (paper default threshold=0.001).

This module is a defense baseline: it does not sample high-level beliefs.
Unsafe turns return a refusal; safe turns fall through to ordinary generation.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

NBF_STATE_DIM = 768
NBF_INPUT_DIM = 768
NBF_OUTPUT_DIM = 768
NBF_SSM_HIDDEN = 512
NBF_HIDDEN = 32
NBF_CLASS_NUM = 5
NBF_EMBEDDER_NAME = "sentence-transformers/all-mpnet-base-v2"
NBF_REFUSAL = "I'm sorry, I cannot assist with that request."


class NeuralStateSpaceModel(nn.Module):
    """Dialogue dynamics f(x, u) → (x', y). Weight layout matches NBF-LLM train.py."""

    def __init__(self, state_dim: int, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.state_transition = nn.Sequential(
            nn.Linear(state_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.observation_model = nn.Sequential(
            nn.Linear(state_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x_t_pre: torch.Tensor, u_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xu_t_pre = torch.cat([x_t_pre, u_t], dim=-1)
        x_t = self.state_transition(xu_t_pre)
        xu_t = torch.cat([x_t, u_t], dim=-1)
        y_t = self.observation_model(xu_t)
        return x_t, y_t


class NeuralBarrierFunction(nn.Module):
    """Barrier classifier B(x, u) → 5-class logits. Weight layout matches NBF-LLM train.py."""

    def __init__(self, state_dim: int, input_dim: int, hidden_dim: int, class_num: int = NBF_CLASS_NUM):
        super().__init__()
        self.nbf = nn.Sequential(
            nn.Linear(state_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, class_num),
        )

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.nbf(torch.cat([x, u], dim=-1))


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand_as(last_hidden).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return F.normalize(summed / counts, p=2, dim=1)


class NBFWrapper:
    """Score the incoming user utterance; refuse when the barrier says the turn is unsafe."""

    def __init__(
        self,
        model_path: str,
        threshold: float,
        embedder_name: str = NBF_EMBEDDER_NAME,
        device: str = "cpu",
    ):
        if not model_path:
            raise ValueError("nbf_model_path is required when defender_backend='nbf'.")
        if threshold < 0:
            raise ValueError(f"nbf_threshold must be >= 0, got {threshold}")
        self.threshold = float(threshold)
        self.device = torch.device(device)
        self.embedder_name = embedder_name
        self.embedder_tokenizer = AutoTokenizer.from_pretrained(embedder_name)
        self.embedder = AutoModel.from_pretrained(embedder_name).to(self.device)
        self.embedder.eval()
        for param in self.embedder.parameters():
            param.requires_grad = False

        self.ssm = NeuralStateSpaceModel(NBF_STATE_DIM, NBF_INPUT_DIM, NBF_OUTPUT_DIM, NBF_SSM_HIDDEN)
        self.nbf = NeuralBarrierFunction(NBF_STATE_DIM, NBF_INPUT_DIM, NBF_HIDDEN, class_num=NBF_CLASS_NUM)
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
        self.ssm.load_state_dict(payload["ssm"])
        self.nbf.load_state_dict(payload["nbf"])
        self.ssm.to(self.device).eval()
        self.nbf.to(self.device).eval()
        for param in list(self.ssm.parameters()) + list(self.nbf.parameters()):
            param.requires_grad = False

    def encode(self, text: str) -> torch.Tensor:
        encoded = self.embedder_tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.inference_mode():
            hidden = self.embedder(**encoded).last_hidden_state
            return _mean_pool(hidden, encoded["attention_mask"])

    def safety_index(self, user_utterances: List[str]) -> float:
        """Walk the user-turn sequence and return B(x, u) last-class minus max other class.

        Matches NBF-LLM `calculate_score_from_dialog`: prior user turns update the SSM;
        the last utterance is the query being scored.
        """
        if not user_utterances:
            raise ValueError("NBF safety_index requires at least one user utterance.")
        x_t = torch.zeros(1, NBF_STATE_DIM, device=self.device)
        nbf_output = None
        with torch.inference_mode():
            for query in user_utterances:
                u_t = self.encode(query)
                nbf_output = self.nbf(x_t, u_t)
                x_t, _y_t = self.ssm(x_t, u_t)
        probs = torch.softmax(nbf_output, dim=-1)
        last_class_prob = probs[:, -1]
        max_other = torch.max(probs[:, :-1], dim=1).values
        return (last_class_prob - max_other)[0].item()

    def is_unsafe(self, user_utterances: List[str]) -> bool:
        # Paper: safe iff safety index < -threshold.
        return self.safety_index(user_utterances) >= -self.threshold

    def generate_or_refuse(self, user_utterances: List[str], generate_fn) -> Tuple[str, int, bool]:
        """Return (text, token_count_placeholder, filtered). Caller fills token count if not filtered."""
        if self.is_unsafe(user_utterances):
            return NBF_REFUSAL, 0, True
        text, token_count = generate_fn()
        return text, token_count, False
