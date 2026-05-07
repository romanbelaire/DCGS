"""TPO-style inference-time textual optimization wrapper."""

from typing import List, Tuple

from ..utils.llm_utils import batch_generate


_REFUSAL_PREFIXES = [
    "I'm sorry",
    "Sorry",
    "I am sorry",
    "I apologize",
    "As an",
    "As an AI",
    "I'm an",
    "I'm just",
    "As a language model",
    "As an Assistant",
    "I cannot",
    "I do not",
]


def _score_candidate(text: str) -> float:
    lowered = text.strip().lower()
    refusal = any(prefix.lower() in lowered for prefix in _REFUSAL_PREFIXES)
    refusal_bonus = 1.0 if refusal else 0.0
    brevity_bonus = max(0.0, 0.25 - 0.001 * len(lowered))
    return refusal_bonus + brevity_bonus


class TPOWrapper:
    """Generate candidates, rank, and iteratively refine best response."""

    def __init__(
        self,
        model,
        tokenizer,
        sample_size: int,
        max_iters: int,
        temperature: float,
        mode: str,
        max_new_tokens: int,
    ):
        if sample_size <= 0:
            raise ValueError(f"tpo_sample_size must be > 0, got {sample_size}")
        if max_iters < 0:
            raise ValueError(f"tpo_max_iters must be >= 0, got {max_iters}")
        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be > 0, got {max_new_tokens}")
        if mode not in ("tpo", "revision", "bon"):
            raise ValueError(f"Unsupported tpo_mode '{mode}'. Expected one of: tpo, revision, bon")
        self.model = model
        self.tokenizer = tokenizer
        self.sample_size = sample_size
        self.max_iters = max_iters
        self.temperature = temperature
        self.mode = mode
        self.max_new_tokens = max_new_tokens

    def _sample_candidates(self, prompt: str) -> List[str]:
        prompts = [prompt] * self.sample_size
        return batch_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=prompts,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
            prefill_suffix="[RESPONSE]",
            chunk_size=self.sample_size,
        )

    def _rank(self, responses: List[str]) -> Tuple[str, str]:
        if len(responses) == 0:
            raise ValueError("TPO received an empty candidate set.")
        scored = sorted(((resp, _score_candidate(resp)) for resp in responses), key=lambda x: x[1], reverse=True)
        best = scored[0][0]
        worst = scored[-1][0]
        return best, worst

    def _build_refinement_prompt(self, base_prompt: str, best: str, worst: str) -> str:
        if self.mode == "revision":
            return (
                f"{base_prompt}\n\n"
                "You are refining a model response for safety and policy compliance.\n"
                "Current response:\n"
                f"{best}\n\n"
                "Return a safer, concise replacement."
            )
        return (
            f"{base_prompt}\n\n"
            "You are refining a chosen response relative to a rejected one.\n"
            "Chosen response:\n"
            f"{best}\n\n"
            "Rejected response:\n"
            f"{worst}\n\n"
            "Return only an improved chosen response."
        )

    def generate(self, prompt: str) -> Tuple[str, int]:
        total_tokens = 0
        candidates = self._sample_candidates(prompt)
        if len(candidates) == 0:
            raise ValueError("TPO initial candidate set is empty.")
        for c in candidates:
            total_tokens += len(self.tokenizer.encode(c, add_special_tokens=False))
        best, worst = self._rank(candidates)
        if self.mode == "bon":
            if not best.strip():
                raise ValueError("TPO best-of-N selected an empty response.")
            return best, total_tokens

        for _ in range(self.max_iters):
            refinement_prompt = self._build_refinement_prompt(prompt, best, worst)
            refined = batch_generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompts=[refinement_prompt],
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                prefill_suffix="[RESPONSE]",
                chunk_size=1,
            )
            if len(refined) == 0:
                raise ValueError("TPO refinement generated zero outputs.")
            refined_text = refined[0]
            total_tokens += len(self.tokenizer.encode(refined_text, add_special_tokens=False))
            best, worst = self._rank([best, worst, refined_text])

        if not best.strip():
            raise ValueError("TPO final response is empty.")
        return best, total_tokens
