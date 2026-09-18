"""High-level agent that maintains candidate beliefs and selects actions."""

import random
import re
from typing import List, Optional, Tuple

from .base_agent import BaseAgent
from ..belief import BeliefCandidate, BeliefState
from ..utils.llm_utils import batch_generate, SuppressWordsLogitsProcessor


class HighLevelAgent(BaseAgent):
    """Maintains candidate belief summaries and generates actions."""
    
    def __init__(
        self,
        model,
        tokenizer,
        prompt_manager=None,
        template_name: str = "belief_generation",
        enable_thinking: Optional[bool] = None,
    ):
        """Initialize high-level agent.
        
        Args:
            model: Language model
            tokenizer: Tokenizer
            prompt_manager: Prompt manager
            template_name: Name of the belief generation template to use
        """
        super().__init__(model, tokenizer, prompt_manager)
        self.template_name = template_name
        self.enable_thinking = enable_thinking
    
    def generate_candidate_beliefs(
        self,
        history: List[Tuple[str, str]],
        n_candidates: int,
        temperature: float = 0.7,
        max_new_tokens: int = 200
    ) -> List[BeliefCandidate]:
        """
        Generate N diverse candidate belief summaries.
        
        Args:
            history: Dialogue history
            n_candidates: Number of candidates to generate
            temperature: Sampling temperature for diversity
            max_new_tokens: Maximum number of tokens to generate
        
        Returns:
            List of BeliefCandidate objects with uniform initial probabilities
        """
        # Use batch method for single history (for consistency)
        results = self.generate_candidate_beliefs_batch(
            histories=[history],
            n_candidates=n_candidates,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )
        return results[0] if results else []
    
    def generate_candidate_beliefs_batch(
        self,
        histories: List[List[Tuple[str, str]]],
        n_candidates: int,
        temperature: float = 0.7,
        max_new_tokens: int = 200,
        return_debug_info: bool = False,
        chunk_size: Optional[int] = None,
        base_prompts: Optional[List[Optional[str]]] = None,
    ):
        """
        Batch generate candidate belief summaries for multiple histories.
        
        Args:
            histories: List of dialogue histories (one per episode/turn)
            n_candidates: Number of candidates to generate per history
            temperature: Sampling temperature for diversity
            max_new_tokens: Maximum number of tokens to generate
            return_debug_info: If True, also return raw LLM outputs for debugging
            chunk_size: Optional chunk size for batch generation to prevent OOM
            base_prompts: Optional list of base_prompt strings. When base_prompts[i] is not None,
                use adversarial belief template (on-topic, intent-varying) for that history.
        
        Returns:
            If return_debug_info=False: List of lists of BeliefCandidate objects (one list per history)
            If return_debug_info=True: Tuple of (List of lists of BeliefCandidate objects, List of raw output strings)
        """
        if not histories:
            return ([], []) if return_debug_info else []
        
        # Generate prompts for all histories (adversarial template when base_prompt provided)
        if base_prompts is not None and len(base_prompts) != len(histories):
            raise ValueError(f"base_prompts length ({len(base_prompts) if base_prompts else 0}) must match histories length ({len(histories)})")
        
        prompts = []
        for i, history in enumerate(histories):
            bp = base_prompts[i] if base_prompts is not None else None
            prompt = self.prompt_manager.get_high_level_prompt(
                template_name=self.template_name,
                dialogue_history=history,
                n_candidates=n_candidates,
                base_prompt=bp,
            )
            prompts.append(prompt)
        
        # Create prefill suffix to force numbered list format
        prefill_suffix = f"Here are {n_candidates} candidate beliefs about the user's desires:\n\n1. "
        
        # Create logits processor to suppress "Example", "example", "Examples", "examples"
        logits_processor = SuppressWordsLogitsProcessor(
            tokenizer=self.tokenizer,
            words_to_suppress=["Example", "example", "Examples", "examples"]
        )
        
        # Batch generate all prompts (with chunking if specified)
        responses = batch_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            prefill_suffix=prefill_suffix,
            chunk_size=chunk_size,
            logits_processor=logits_processor,
            enable_thinking=self.enable_thinking,
        )
        
        # Parse each response into candidates and check for errors
        all_belief_candidates = []
        for idx, (generated_text, prompt, history) in enumerate(zip(responses, prompts, histories)):
            # The prefill_suffix is included in the decoded output, and the regex parser will find the numbered list
            candidates = self._parse_belief_candidates(generated_text, n_candidates)
            
            # Count valid (non-[SKIP]) candidates
            valid_count = sum(1 for c in candidates if c != "[SKIP]")
            
            # Warn if too many candidates are [SKIP] (more than half) - might indicate parsing issues
            if valid_count < n_candidates // 2 and valid_count > 0:
                import sys
                print("\n" + "="*80)
                print(f"[WARNING] History index {idx}: Only {valid_count}/{n_candidates} valid candidates parsed "
                      f"({n_candidates - valid_count} are [SKIP] fallbacks). This may indicate parsing issues.")
                print("="*80)
                print("HIGH-LEVEL AGENT RAW OUTPUT:")
                print("="*80)
                print(generated_text[:2000] + "..." if len(generated_text) > 2000 else generated_text)
                print(f"\n{'='*80}")
                print("Parsed Candidates:")
                print("="*80)
                for i, candidate in enumerate(candidates, 1):
                    status = "VALID" if candidate != "[SKIP]" else "[SKIP]"
                    preview = candidate[:100] + "..." if len(candidate) > 100 else candidate
                    print(f"  {i}. [{status}] {preview}")
                print("="*80 + "\n")
                sys.stdout.flush()
            
            # Initialize with uniform probabilities
            uniform_prob = 1.0 / len(candidates) if candidates else 0.0
            belief_candidates = []
            for summary in candidates:
                context = f"Belief: {summary}"
                candidate = BeliefCandidate(
                    summary=summary,
                    context=context,
                    probability=uniform_prob
                )
                belief_candidates.append(candidate)
            
            all_belief_candidates.append(belief_candidates)
        
        if return_debug_info:
            return all_belief_candidates, responses
        else:
            return all_belief_candidates

    def generate_candidate_beliefs_batch_iterative(
        self,
        histories: List[List[Tuple[str, str]]],
        n_candidates: int,
        temperature: float = 0.7,
        max_new_tokens_per_candidate: int = 96,
        return_debug_info: bool = False,
        chunk_size: Optional[int] = None,
        base_prompts: Optional[List[Optional[str]]] = None,
        max_attempts_per_candidate: int = 3,
    ):
        """Batch generate candidates iteratively, one numbered item per generation step."""
        if max_attempts_per_candidate < 1:
            raise ValueError(
                f"max_attempts_per_candidate must be >= 1, got {max_attempts_per_candidate}"
            )
        if not histories:
            return ([], []) if return_debug_info else []

        if base_prompts is not None and len(base_prompts) != len(histories):
            raise ValueError(f"base_prompts length ({len(base_prompts) if base_prompts else 0}) must match histories length ({len(histories)})")

        candidate_texts_per_history = [[] for _ in histories]
        raw_outputs_per_history = [[] for _ in histories]

        logits_processor = SuppressWordsLogitsProcessor(
            tokenizer=self.tokenizer,
            words_to_suppress=["Example", "example", "Examples", "examples"]
        )

        for candidate_idx in range(1, n_candidates + 1):
            pending_history_indices = list(range(len(histories)))
            attempt_raw_by_history = {history_idx: [] for history_idx in pending_history_indices}

            for attempt in range(max_attempts_per_candidate):
                if not pending_history_indices:
                    break

                step_prompts = []
                for history_idx in pending_history_indices:
                    history = histories[history_idx]
                    bp = base_prompts[history_idx] if base_prompts is not None else None
                    step_prompts.append(
                        self.prompt_manager.get_high_level_iterative_prompt(
                            dialogue_history=history,
                            n_candidates=n_candidates,
                            next_candidate_index=candidate_idx,
                            existing_candidates=candidate_texts_per_history[history_idx],
                            base_prompt=bp,
                        )
                    )

                responses = batch_generate(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompts=step_prompts,
                    max_new_tokens=max_new_tokens_per_candidate,
                    temperature=temperature,
                    do_sample=True,
                    prefill_suffix=f"{candidate_idx}. ",
                    chunk_size=chunk_size,
                    logits_processor=logits_processor,
                    enable_thinking=self.enable_thinking,
                )

                still_pending = []
                for pending_idx, history_idx in enumerate(pending_history_indices):
                    generated_text = responses[pending_idx]
                    attempt_raw_by_history[history_idx].append(generated_text)
                    parsed_candidate = self._parse_single_candidate(generated_text, candidate_idx)
                    if parsed_candidate != "[SKIP]":
                        candidate_texts_per_history[history_idx].append(parsed_candidate)
                    elif attempt == max_attempts_per_candidate - 1:
                        candidate_texts_per_history[history_idx].append("[SKIP]")
                    else:
                        still_pending.append(history_idx)
                pending_history_indices = still_pending

            for history_idx in range(len(histories)):
                raw_outputs_per_history[history_idx].append(
                    "\n".join(attempt_raw_by_history[history_idx])
                )

        all_belief_candidates = []
        joined_raw_outputs = []
        for idx, candidate_texts in enumerate(candidate_texts_per_history):
            valid_count = sum(1 for c in candidate_texts if c != "[SKIP]")
            if valid_count < n_candidates // 2 and valid_count > 0:
                import sys
                print("\n" + "=" * 80)
                print(
                    f"[WARNING] History index {idx}: Only {valid_count}/{n_candidates} valid candidates parsed "
                    f"({n_candidates - valid_count} are [SKIP] fallbacks). This may indicate parsing issues."
                )
                print("=" * 80)
                print("HIGH-LEVEL AGENT RAW OUTPUT:")
                print("=" * 80)
                joined_raw = "\n".join(raw_outputs_per_history[idx])
                print(joined_raw[:2000] + "..." if len(joined_raw) > 2000 else joined_raw)
                print(f"\n{'=' * 80}")
                print("Parsed Candidates:")
                print("=" * 80)
                for candidate_line_idx, candidate in enumerate(candidate_texts, 1):
                    status = "VALID" if candidate != "[SKIP]" else "[SKIP]"
                    preview = candidate[:100] + "..." if len(candidate) > 100 else candidate
                    print(f"  {candidate_line_idx}. [{status}] {preview}")
                print("=" * 80 + "\n")
                sys.stdout.flush()

            uniform_prob = 1.0 / len(candidate_texts) if candidate_texts else 0.0
            belief_candidates = []
            for summary in candidate_texts:
                belief_candidates.append(
                    BeliefCandidate(
                        summary=summary,
                        context=f"Belief: {summary}",
                        probability=uniform_prob,
                    )
                )
            all_belief_candidates.append(belief_candidates)
            joined_raw_outputs.append("\n".join(raw_outputs_per_history[idx]))

        if return_debug_info:
            return all_belief_candidates, joined_raw_outputs
        return all_belief_candidates
    
    def _parse_belief_candidates(self, text: str, n_candidates: int) -> List[str]:
        """Parse numbered list of belief candidates from generated text.

        First removes tags like "[First candidate belief summary]" (keeping text after them),
        then finds the enumerated list by treating only line-start "N." or "N)" as list markers,
        so mid-sentence numbers like "4:25" are not treated as list items.
        Invalid candidates are replaced with "[SKIP]" sentinel value.
        """
        # Remove "[...candidate...belief...summary...]" tags from the whole text; keep everything else
        tag_pattern = re.compile(r'\[.*?candidate.*?belief.*?summary.*?\]', re.IGNORECASE)
        text_cleaned = tag_pattern.sub('', text)

        lines = text_cleaned.split('\n')
        all_items = []  # List of (number, content) tuples
        # Only treat "N." or "N)" at start of a line as list markers (avoids matching e.g. "4:25")
        list_marker = re.compile(r'^\s*(\d+)[.)]\s*(.*)$')

        i = 0
        while i < len(lines):
            line = lines[i]
            match = list_marker.match(line)
            if match:
                number = int(match.group(1))
                rest_of_line = match.group(2).strip()
                # Content is rest of this line plus all following lines until the next list marker
                content_parts = [rest_of_line] if rest_of_line else []
                i += 1
                while i < len(lines) and not list_marker.match(lines[i]):
                    content_parts.append(lines[i].strip())
                    i += 1
                content = ' '.join(p for p in content_parts if p).strip()
                if content:
                    all_items.append((number, content))
                continue
            i += 1
        
        # Find the first sequence of consecutive numbers starting from 1
        # Accept partial sequences if we have at least 2 valid items
        candidates_raw = []
        best_sequence = []  # Track the best (longest) sequence found
        for i in range(len(all_items)):
            # Check if we can form a sequence starting at index i
            sequence = []
            expected_num = 1
            for j in range(i, len(all_items)):
                num, content = all_items[j]
                if num == expected_num:
                    sequence.append(content)
                    expected_num += 1
                    if len(sequence) == n_candidates:
                        # Found complete sequence!
                        candidates_raw = sequence
                        break
                elif num < expected_num:
                    # Number went backwards or repeated, this isn't a valid sequence
                    break
            if len(candidates_raw) == n_candidates:
                break
            # Track the longest sequence found (at least 2 items)
            if len(sequence) >= 2 and len(sequence) > len(best_sequence):
                best_sequence = sequence
        
        # If we didn't find a complete sequence but found at least 2 valid items, use the best partial sequence
        if not candidates_raw and len(best_sequence) >= 2:
            candidates_raw = best_sequence
        
        # Process candidates: clean them and replace invalid ones with [SKIP]
        candidates = []
        for candidate_raw in candidates_raw:
            candidates.append(self._clean_candidate_text(candidate_raw))
        
        # If we have fewer than n_candidates, pad with [SKIP]
        while len(candidates) < n_candidates:
            candidates.append("[SKIP]")
        
        return candidates[:n_candidates]

    def _parse_single_candidate(self, text: str, expected_number: int) -> str:
        """Parse one numbered candidate item and return cleaned text or [SKIP]."""
        lines = text.split('\n')
        list_marker = re.compile(r'^\s*(\d+)[.)]\s*(.*)$')

        for idx, line in enumerate(lines):
            match = list_marker.match(line)
            if not match:
                continue
            number = int(match.group(1))
            if number != expected_number:
                continue
            content_parts = [match.group(2).strip()] if match.group(2).strip() else []
            next_idx = idx + 1
            while next_idx < len(lines) and not list_marker.match(lines[next_idx]):
                if lines[next_idx].strip():
                    content_parts.append(lines[next_idx].strip())
                next_idx += 1
            joined = " ".join(content_parts).strip()
            return self._clean_candidate_text(joined)
        return "[SKIP]"

    def _clean_candidate_text(self, candidate_raw: str) -> str:
        """Normalize and validate one candidate string."""
        candidate = candidate_raw.strip()
        tag_patterns_to_remove = [
            r'\[.*candidate.*belief.*summary.*?\]',
            r'\[.*placeholder.*?\]',
            r'\[.*example.*?\]',
        ]
        for pattern in tag_patterns_to_remove:
            candidate = re.sub(pattern, '', candidate, flags=re.IGNORECASE).strip()

        placeholder_patterns = [
            r'^candidate \d+$',
            r'^belief candidate \d+$',
            r'^\[.*placeholder.*\]$',
            r'^\[.*example.*\]$',
            r'^\[.*candidate.*belief.*summary.*\]$',
        ]
        is_entirely_placeholder = any(
            re.match(pattern, candidate, re.IGNORECASE)
            for pattern in placeholder_patterns
        )
        is_valid = not is_entirely_placeholder and len(candidate) > 10
        if is_valid:
            return candidate
        return "[SKIP]"
    
    def select_action(
        self,
        belief_state: BeliefState,
        value_function,  # ValueFunction type, but avoid circular import
        epsilon: float = 0.1,
        available_actions: List[str] = None,
        current_observation: str = ""
    ) -> str:
        """
        Select action based on expected value over beliefs (V_high).
        
        Uses epsilon-greedy: with probability ε, select random action;
        otherwise select action with max expected value.
        
        Args:
            belief_state: Current belief state
            value_function: Value function for computing expected values
            epsilon: Epsilon-greedy parameter
            available_actions: List of available actions (if None, uses default)
            current_observation: Current observation text (for V_high input)
        
        Returns:
            Selected action string
        """
        if available_actions is None:
            # Default actions for dialogue
            available_actions = [
                "Request information",
                "Provide information",
                "Confirm understanding",
                "Ask clarifying question"
            ]
        
        # Epsilon-greedy: random action with probability epsilon
        if random.random() < epsilon:
            return random.choice(available_actions)
        
        # V_high takes ONLY observation as input
        # For action selection, we use V_high(observation) to get the value of the current state
        # Then select action based on heuristics or random (since V_high doesn't take actions)
        # TODO: This might need a Q-function or different approach for action selection
        
        # For now, use V_high to get state value, but action selection is still heuristic
        state_value = value_function.get_value_for_observation(
            observation=current_observation,
            tokenizer=self.tokenizer
        )
        
        # Since V_high doesn't take actions, we can't compare actions directly
        # Use random selection for now (epsilon-greedy already handled above)
        # In the future, might need Q_high(observation, action) or different approach
        return random.choice(available_actions)
    
    def update_beliefs(
        self,
        belief_state: BeliefState,
        action: str,
        observation: str,
        belief_surrogate,
        belief_updater,
        model,
        tokenizer,
        temperature: float = 1.0
    ) -> BeliefState:
        """
        Update beliefs using DPO-based ranking.
        
        Args:
            belief_state: Current belief state
            action: Agent action
            observation: User observation
            belief_surrogate: Belief surrogate for log-prob computation
            belief_updater: Belief updater for DPO ranking
            model: Language model
            tokenizer: Tokenizer
            temperature: DPO temperature
        
        Returns:
            Updated belief state
        """
        # Compute log-probabilities
        log_probs = belief_surrogate.compute_log_probs_batch(
            beliefs=belief_state.candidates,
            action=action,
            observation=observation,
            model=model,
            tokenizer=tokenizer
        )
        
        # Update belief probabilities via DPO ranking
        updated_state = belief_updater.update_belief_distribution_dpo(
            belief_state=belief_state,
            log_probs=log_probs,
            temperature=temperature
        )
        
        return updated_state
    
    def act(self, belief_state: BeliefState, value_function) -> str:
        """Interface method for acting."""
        return self.select_action(belief_state, value_function)

