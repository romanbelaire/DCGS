"""Low-level task reward from embedding cosine similarity.

The score is cosine(assistant reply, task text) under a frozen sentence
embedder. Task text is the dataset user goal (benign Goal Completion Rate
definition), not the selected high-level belief.
"""

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

DEFAULT_EMBEDDER = "sentence-transformers/all-mpnet-base-v2"
MAX_LENGTH = 512
BATCH_SIZE = 32


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand_as(last_hidden).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return F.normalize(summed / counts, p=2, dim=1)


def load_task_embedder(name: str = DEFAULT_EMBEDDER, device: str = "cpu"):
    if device != "cpu":
        raise ValueError(f"semantic task embedder must run on cpu, got {device!r}")
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return tokenizer, model


def embed_texts(
    texts: Sequence[str],
    tokenizer,
    model,
    device: str = "cpu",
    batch_size: int = BATCH_SIZE,
) -> torch.Tensor:
    if not texts:
        raise ValueError("embed_texts requires at least one string")
    chunks = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state
            chunks.append(_mean_pool(hidden, encoded["attention_mask"]).cpu())
    return torch.cat(chunks, dim=0)


def unique_text_index(texts: Sequence[str]) -> Tuple[List[str], List[int]]:
    order: List[str] = []
    index: Dict[str, int] = {}
    ids: List[int] = []
    for text in texts:
        if not text.strip():
            raise ValueError("semantic task reward got an empty string")
        if text not in index:
            index[text] = len(order)
            order.append(text)
        ids.append(index[text])
    return order, ids


def pairwise_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape:
        raise ValueError(f"cosine shape mismatch {tuple(left.shape)} vs {tuple(right.shape)}")
    return (left * right).sum(dim=1)


def semantic_task_scores(
    assistants: Sequence[str],
    goals: Sequence[str],
    tokenizer,
    model,
    device: str = "cpu",
    batch_size: int = BATCH_SIZE,
) -> List[float]:
    """Cosine similarity between each assistant reply and its task goal."""
    if len(assistants) != len(goals):
        raise ValueError(
            f"assistants {len(assistants)} and goals {len(goals)} length mismatch"
        )
    if not assistants:
        raise ValueError("semantic_task_scores requires at least one pair")
    unique, ids = unique_text_index(list(assistants) + list(goals))
    embs = embed_texts(unique, tokenizer, model, device=device, batch_size=batch_size)
    n = len(assistants)
    a_ids = ids[:n]
    g_ids = ids[n:]
    a_emb = embs[a_ids]
    g_emb = embs[g_ids]
    return pairwise_cosine(a_emb, g_emb).tolist()
