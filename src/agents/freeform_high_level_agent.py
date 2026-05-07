"""Freeform high-level policy used by hierarchical rollout."""

from typing import Dict, List, Optional, Tuple

from .high_level_agent import HighLevelAgent
from ..belief import BeliefCandidate


class FreeformHighLevelAgent(HighLevelAgent):
    """High-level agent that emits freeform instruction candidates."""

    def generate_instructions_batch(
        self,
        histories: List[List[Tuple[str, str]]],
        n_instructions: int,
        temperature: float,
        max_new_tokens: int,
        iterative_candidate_generation: bool = False,
        per_instruction_max_new_tokens: Optional[int] = None,
        chunk_size: Optional[int] = None,
        base_prompts: Optional[List[Optional[str]]] = None,
    ) -> Dict[str, List]:
        """Generate freeform instructions and return both text and belief candidates."""
        if iterative_candidate_generation:
            token_budget = per_instruction_max_new_tokens or max_new_tokens
            all_candidates, raw_outputs = self.generate_candidate_beliefs_batch_iterative(
                histories=histories,
                n_candidates=n_instructions,
                temperature=temperature,
                max_new_tokens_per_candidate=token_budget,
                return_debug_info=True,
                chunk_size=chunk_size,
                base_prompts=base_prompts,
            )
        else:
            all_candidates, raw_outputs = self.generate_candidate_beliefs_batch(
                histories=histories,
                n_candidates=n_instructions,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                return_debug_info=True,
                chunk_size=chunk_size,
                base_prompts=base_prompts,
            )
        instructions = [[candidate.summary for candidate in row] for row in all_candidates]
        return {
            "instructions": instructions,
            "candidates": all_candidates,
            "raw_outputs": raw_outputs,
        }

    def generate_instruction_candidates_as_beliefs(
        self,
        history: List[Tuple[str, str]],
        n_instructions: int,
        temperature: float,
        max_new_tokens: int,
    ) -> List[BeliefCandidate]:
        """Compatibility method for call sites expecting belief candidates."""
        result = self.generate_instructions_batch(
            histories=[history],
            n_instructions=n_instructions,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        return result["candidates"][0]
