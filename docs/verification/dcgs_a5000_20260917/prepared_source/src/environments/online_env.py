"""Online environment for online Q-learning with LLM-based user simulation."""

import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..agents.user_agent import UserAgent
from ..data.multiwoz_loader import MultiWOZDialogue
from ..simulation import (
    ActionConstraintSimulator,
    GoalState,
    extract_desires_from_dialogue,
    extract_slots_from_agent_response,
    extract_service_call_from_agent_response,
)

from .dialogue_env import DialogueEnvironment, StepResult

# Module-level flag to track if we've printed the first completed dialogue
_first_dialogue_printed = False


@dataclass
class OnlineEnvironmentState:
    """Internal state for online environment."""
    turn_idx: int = 0
    dialogue_history: List[Tuple[str, str]] = field(default_factory=list)  # (agent_action, user_response)
    goal_achieved: float = 0.0  # Fraction of goals achieved (0.0 to 1.0)
    episode_ended: bool = False
    belief_generations: List[List[str]] = field(default_factory=list)  # per-turn belief candidate summaries
    hl_agent_inputs: List[str] = field(default_factory=list)  # per-turn high-level agent prompts
    ll_agent_inputs: List[str] = field(default_factory=list)  # per-turn low-level agent prompts
    user_agent_inputs: List[str] = field(default_factory=list)  # per-turn user agent prompts
    hl_agent_outputs: List[str] = field(default_factory=list)  # per-turn high-level agent completions
    ll_agent_outputs: List[str] = field(default_factory=list)  # per-turn low-level agent responses
    user_agent_outputs: List[str] = field(default_factory=list)  # per-turn user agent utterances


class OnlineEnvironment(DialogueEnvironment):
    """
    Online environment for online Q-learning.
    
    Uses UserAgent (LLM) to generate user responses instead of dataset actions.
    User prompts are based only on episode information: persona, goal, dialogue history, and agent action.
    """
    
    def __init__(
        self,
        user_agent: UserAgent,
        persona: Dict,
        goal_json: Dict,
        max_turns: int = 20,
        debug: bool = False,
        source_dialogue: Optional[MultiWOZDialogue] = None
    ):
        """
        Initialize online environment with user LLM.
        
        Args:
            user_agent: UserAgent instance for generating user responses
            persona: Persona dictionary with ground truth beliefs
            goal_json: Ground truth goal for computing rewards
            max_turns: Maximum number of turns per episode
            debug: Whether to print debug statements
            source_dialogue: Optional MultiWOZ dialogue to extract desires and constraints from
        """
        self.user_agent = user_agent
        self.persona = persona
        self.goal_json = goal_json
        self.max_turns = max_turns
        self.debug = debug
        self.env_state = OnlineEnvironmentState()
        
        # Initialize GoalState and ActionConstraintSimulator
        self.goal_state = GoalState()
        self.constraint_simulator = ActionConstraintSimulator(dialogue=source_dialogue)
        self.goal_state.constraint_simulator = self.constraint_simulator
        
        # Extract desires from source dialogue if available
        if source_dialogue:
            desires = extract_desires_from_dialogue(source_dialogue)
            for desire in desires:
                self.goal_state.add_desire(desire)
    
    def reset(self) -> Tuple[str, Dict]:
        """
        Reset environment to initial state.
        
        Generates initial user message using UserAgent with persona and goal.
        
        Returns:
            Initial observation (user's first message), info dict
        """
        self.env_state = OnlineEnvironmentState()
        
        # Reset goal state (reset current desire index, but keep desires)
        self.goal_state.current_desire_idx = 0
        for desire in self.goal_state.desires:
            if desire.status == "satisfied":
                # Keep satisfied desires as satisfied
                pass
            else:
                # Reset other desires to pending
                desire.status = "pending"
                desire.attempt_count = 0
        
        # Generate initial user message
        # Use empty dialogue history for first turn
        initial_agent_action = ""  # No agent action yet for first turn
        progress_summary = self.goal_state.build_progress_summary(self.goal_json)

        raw_observation = self.user_agent.generate_response(
            goal_json=self.goal_json,
            dialogue_history=[],
            agent_action=initial_agent_action,
            persona=self.persona,
            goal_progress_summary=progress_summary,
        )
        # Parse response from [RESPONSE] tags if present
        initial_observation = self._parse_response_from_tags(raw_observation)
        if hasattr(self.env_state, "user_agent_outputs"):
            self.env_state.user_agent_outputs.append(initial_observation)
        
        # Store initial observation
        self.env_state.dialogue_history.append((initial_agent_action, initial_observation))
        self.env_state.turn_idx = 0
        
        info = {
            "goal_json": self.goal_json,
            "persona": self.persona,
            "turn_idx": self.env_state.turn_idx,
            "goal_achieved": 0.0,
            "goal_state": self.goal_state
        }
        
        return initial_observation, info
    
    def step(self, agent_action: str) -> Tuple[str, float, bool, Dict]:
        """
        Execute agent action and get user response.
        
        Args:
            agent_action: Generated action from LL agent
        
        Returns:
            observation: User's response (generated by UserAgent)
            reward: Computed reward based on goal achievement
            done: Whether episode ended
            info: Additional info (goal_achieved, actions, etc.)
        """
        if self.env_state.episode_ended:
            return StepResult("", 0.0, True, {})
        
        prev_fraction = (
            self.goal_state.get_goal_achievement_fraction()
            if self.goal_state.desires else 0.0
        )
        
        # Update dialogue history with agent action.
        # Keep the initial user-only entry intact so we don't lose the first observation.
        if self.env_state.dialogue_history:
            last_agent_action, last_user_response = self.env_state.dialogue_history[-1]
            if last_agent_action and not last_user_response:
                raise RuntimeError(
                    "Attempted to add a new agent action before the previous user "
                    "response was recorded. Dialogue history is inconsistent."
                )
        # Add agent action to dialogue history (user response will be added later)
        self.env_state.dialogue_history.append((agent_action, ""))
        
        # Extract slots and service call from agent action FIRST
        # Determine domain from goal_json or current desire
        domain = None
        if self.goal_state.desires and self.goal_state.current_desire_idx < len(self.goal_state.desires):
            current_desire = self.goal_state.desires[self.goal_state.current_desire_idx]
            domain = current_desire.domain
        elif self.goal_json:
            # Get first domain from goal_json
            domains = list(self.goal_json.keys())
            if domains:
                domain = domains[0]
        
        if domain:
            # Extract slots from agent response
            try:
                extracted_slots = extract_slots_from_agent_response(
                    agent_text=agent_action,
                    domain=domain,
                    goal_json=self.goal_json
                )
            except Exception as e:
                import sys
                print(f"\n[ERROR] Exception in extract_slots_from_agent_response:")
                print(f"  Agent action: {agent_action}")
                print(f"  Domain: {domain}")
                print(f"  Error: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
                raise
            
            # Extract service call if agent is attempting a booking
            try:
                service_call = extract_service_call_from_agent_response(
                    agent_text=agent_action,
                    domain=domain
                )
            except Exception as e:
                import sys
                print(f"\n[ERROR] Exception in extract_service_call_from_agent_response:")
                print(f"  Agent action: {agent_action}")
                print(f"  Domain: {domain}")
                print(f"  Error: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
                raise
            
            # Update goal state with extracted slots
            try:
                self.goal_state.update_from_agent_response(
                    agent_text=agent_action,
                    extracted_slots=extracted_slots,
                    service_call=service_call,
                    goal_json=self.goal_json
                )
            except Exception as e:
                import sys
                print(f"\n[ERROR] Exception in goal_state.update_from_agent_response:")
                print(f"  Agent action: {agent_action}")
                print(f"  Error: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
                raise
        
        # Judge goal satisfaction BEFORE user generates response
        # This ensures goal state is updated before user responds
        goal_satisfied = False
        if self.goal_json:
            try:
                # Get dialogue history up to (but not including) the current agent action
                # The judge should see the full context including the agent's response
                judge_history = self.env_state.dialogue_history.copy()
                goal_satisfied = self.user_agent.judge_goal_satisfaction(
                    dialogue_history=judge_history,
                    goal_json=self.goal_json,
                    agent_response=agent_action,
                    temperature=0.0
                )
            except Exception as e:
                import sys
                print(f"\n[ERROR] Exception in judge_goal_satisfaction:")
                print(f"  Agent action: {agent_action}")
                print(f"  Error: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
                # Fallback to goal state check
                goal_satisfied = False
        
        # If judge says goal is satisfied, update goal state
        if goal_satisfied and self.goal_state.desires and self.goal_state.current_desire_idx < len(self.goal_state.desires):
            current_desire = self.goal_state.desires[self.goal_state.current_desire_idx]
            if current_desire.status == "pending":
                current_desire.status = "satisfied"
                # Move to next desire if available
                if self.goal_state.current_desire_idx + 1 < len(self.goal_state.desires):
                    self.goal_state.current_desire_idx += 1
        
        if self.goal_state.desires:
            goal_achievement_fraction = self.goal_state.get_goal_achievement_fraction()
            progress_made = goal_achievement_fraction > prev_fraction + 1e-9
            all_goals_completed = self.goal_state.is_goal_achieved()
        else:
            goal_achievement_fraction = 1.0 if goal_satisfied else 0.0
            progress_made = goal_satisfied
            all_goals_completed = goal_satisfied
        
        # NOW generate user response (goal state is already updated)
        user_response, raw_response, actions = self._generate_user_response(
            agent_action=agent_action
        )
        
        # Parse episode end tag (if still present in template)
        episode_end = self._parse_episode_end(raw_response)
        
        # Update dialogue history with user response
        if self.env_state.dialogue_history:
            last_agent_action, _ = self.env_state.dialogue_history[-1]
            self.env_state.dialogue_history[-1] = (last_agent_action, user_response)
        if hasattr(self.env_state, "user_agent_outputs"):
            self.env_state.user_agent_outputs.append(user_response)
        
        self.env_state.turn_idx += 1
        
        reward = 1.0 if progress_made else 0.0
        
        # Update goal achievement fraction (keep maximum)
        self.env_state.goal_achieved = max(self.env_state.goal_achieved, goal_achievement_fraction)
        
        # Check if episode should end (use structured tag if available, fallback to heuristic)
        done = episode_end if episode_end is not None else (self._check_episode_ended(user_response) or all_goals_completed)
        done = done or (self.env_state.turn_idx >= self.max_turns)
        self.env_state.episode_ended = done
        
        # Print full dialogue trajectory when first dialogue completes
        if done:
            self._maybe_print_first_dialogue_trajectory()
        
        # Dialogue acts already extracted in _generate_user_response
        info = {
            "actions": actions,
            "goal_achieved": self.env_state.goal_achieved,
            "turn_idx": self.env_state.turn_idx,
            "reward": reward,
            "persona": self.persona,
            "goal_state": self.goal_state
        }

        return StepResult(user_response, reward, done, info)

    def check_goal(self, observation: str) -> bool:
        return self.env_state.goal_achieved >= 1.0 or self.goal_state.is_goal_achieved()

    def _generate_user_response(
        self,
        agent_action: str
    ) -> Tuple[str, str, List[str]]:
        """
        Generate user response based only on episode information (persona, goal, dialogue history, agent action).
        
        Args:
            agent_action: Agent's action
        
        Returns:
            Tuple of (parsed_response, raw_response, dialogue_acts)
        """
        # Get dialogue history (excluding current agent_action)
        dialogue_history = self.env_state.dialogue_history[:-1] if self.env_state.dialogue_history else []
        
        # Generate user response using UserAgent (only uses episode information: persona, goal, history, action)
        progress_summary = self.goal_state.build_progress_summary(self.goal_json)
        raw_response = self.user_agent.generate_response(
            goal_json=self.goal_json,
            dialogue_history=dialogue_history,
            agent_action=agent_action,
            persona=self.persona,
            goal_progress_summary=progress_summary,
        )
        
        # Parse response from [RESPONSE] tags if present
        parsed_response = self._parse_response_from_tags(raw_response)
        
        # Extract dialogue acts (with debug info from raw_response and agent_action)
        dialogue_acts = self._extract_dialogue_acts(parsed_response, raw_response=raw_response, agent_action=agent_action)
        
        return parsed_response, raw_response, dialogue_acts
    
    def _parse_response_from_tags(self, raw_response: str) -> str:
        """
        Parse user response from [RESPONSE] tags.
        
        Extracts content between [RESPONSE]...[/RESPONSE] tags and removes any remaining tags
        to ensure only clean text is returned.
        
        When using prefill_suffix="[RESPONSE]", the generated text starts with [RESPONSE],
        so we want to find the LAST valid occurrence to avoid matching prompt examples.
        
        Args:
            raw_response: Raw response from LLM (should only contain generated text, not prompt)
            
        Returns:
            Parsed response (content between tags with all tags removed, or original if no tags)
        """
        import re
        
        # Look for ALL [RESPONSE]...[/RESPONSE] tag pairs
        pattern = r'\[RESPONSE\](.*?)\[/RESPONSE\]'
        matches = list(re.finditer(pattern, raw_response, re.DOTALL | re.IGNORECASE))
        
        if matches:
            # Filter out matches that only contain whitespace or are very short
            # (these are likely malformed or from metacommentary)
            valid_matches = []
            for match in matches:
                content = match.group(1).strip()
                # Remove tags from content to check actual text length
                text_only = re.sub(r'\[/?[^\]]+\]', '', content).strip()
                if len(text_only) >= 10:  # Require at least 10 characters of actual text
                    valid_matches.append((match, len(text_only)))
            
            if valid_matches:
                # Use the LAST valid match to avoid matching prompt examples
                # (the last one should be the actual generated response)
                best_match = valid_matches[-1][0]
                parsed = best_match.group(1).strip()
            else:
                # If no valid matches, try the last match anyway (might be short but valid)
                # If that's also empty, try earlier matches
                parsed = None
                for match in reversed(matches):
                    content = match.group(1).strip()
                    text_only = re.sub(r'\[/?[^\]]+\]', '', content).strip()
                    if text_only:
                        parsed = content
                        break
                
                if not parsed:
                    # Last resort: use the last match
                    parsed = matches[-1].group(1).strip()
            
            # Remove any remaining tags that might have leaked in (safeguard)
            parsed = re.sub(r'\[/?[^\]]+\]', '', parsed)
            
            result = parsed.strip()
            # If result is still empty, try fallback strategies
            if not result:
                # Try to extract content from an opening tag even without closing tag
                opening_match = re.search(r'\[RESPONSE\](.*)', raw_response, re.DOTALL | re.IGNORECASE)
                if opening_match:
                    content = opening_match.group(1).strip()
                    # Remove all tags and get text
                    result = re.sub(r'\[/?[^\]]+\]', '', content).strip()
                    # Take first reasonable chunk (before any metacommentary)
                    if result:
                        # Stop at common metacommentary patterns
                        for stop_pattern in [r'\[/RESPONSE\]', r'is not needed', r'Here is the revised']:
                            stop_match = re.search(stop_pattern, result, re.IGNORECASE)
                            if stop_match:
                                result = result[:stop_match.start()].strip()
                                break
            
            return result if result else raw_response.strip()
        
        # If no tags found, try to extract from opening tag without closing tag
        opening_match = re.search(r'\[RESPONSE\](.*)', raw_response, re.DOTALL | re.IGNORECASE)
        if opening_match:
            content = opening_match.group(1).strip()
            # Remove all tags
            cleaned = re.sub(r'\[/?[^\]]+\]', '', content).strip()
            if cleaned:
                return cleaned
        
        # Final fallback: remove all tags from raw response
        cleaned = re.sub(r'\[/?[^\]]+\]', '', raw_response)
        return cleaned.strip()
    
    def _parse_goal_status(self, raw_response: str) -> Optional[bool]:
        """
        Parse goal status from second pair of bracketed tags containing 'g'.
        
        Args:
            raw_response: Raw response from LLM
            
        Returns:
            True if goal achieved, False if not achieved, None if tag not found
        """
        import re
        
        # Find all bracketed tags
        tag_matches = list(re.finditer(r'\[([^\]]+)\]', raw_response, re.IGNORECASE))
        
        # Look for a tag pair containing 'g' (skip the first tag which should be [RESPONSE])
        for i, match in enumerate(tag_matches[1:], start=1):  # Skip first tag
            tag_content = match.group(1)
            if 'g' in tag_content.lower():
                # Found opening tag with 'g', extract content until closing tag
                start_pos = match.end()
                remaining = raw_response[start_pos:]
                # Find the next tag (closing tag)
                next_tag_match = re.search(r'\[', remaining)
                if next_tag_match:
                    status_text = remaining[:next_tag_match.start()].strip().lower()
                else:
                    status_text = remaining.strip().lower()
                
                # Check for achieved/not_achieved
                if "achieved" in status_text and "not" not in status_text:
                    return True
                elif "not_achieved" in status_text or ("not" in status_text and "achieved" in status_text):
                    return False
                # Only check first tag with 'g' found
                break
        
        return None
    
    def _parse_episode_end(self, raw_response: str) -> Optional[bool]:
        """
        Parse episode end status from [EPISODE_END] tags.
        
        Args:
            raw_response: Raw response from LLM
            
        Returns:
            True if episode should end, False if not, None if tag not found
        """
        import re
        
        pattern = r'\[EPISODE_END\](.*?)\[/EPISODE_END\]'
        match = re.search(pattern, raw_response, re.DOTALL | re.IGNORECASE)
        
        if match:
            status_text = match.group(1).strip().lower()
            if status_text == "yes":
                return True
            elif status_text == "no":
                return False
        
        return None
    
    def _check_episode_ended(self, user_response: str) -> bool:
        """
        Check if episode should end.
        
        Args:
            user_response: Generated user response
        
        Returns:
            True if episode should end
        """
        actions = self._extract_dialogue_acts(user_response)
        if "GOODBYE" in actions:
            return True
        
        # Check for goodbye keywords
        response_lower = user_response.lower()
        goodbye_keywords = ["goodbye", "bye", "see you", "farewell"]
        if any(keyword in response_lower for keyword in goodbye_keywords):
            return True
        
        return False
    
    def _extract_dialogue_acts(self, text: str, raw_response: Optional[str] = None, agent_action: Optional[str] = None) -> List[str]:
        """
        Extract dialogue acts from generated user response.
        
        Simple keyword-based extraction (can be improved with LLM-based extraction).
        In online mode, if no keywords match but text is non-empty, default to INFORM
        to prevent episodes from ending prematurely.
        
        Args:
            text: User response text (parsed)
            raw_response: Raw user response from LLM (for debugging)
            agent_action: Agent action that prompted the user response (for debugging)
            
        Returns:
            List of dialogue act strings
        """
        text_lower = text.lower().strip()
        acts = []
        
        # If text is empty, return empty list
        if not text_lower:
            return acts
        
        # Simple keyword-based extraction
        if any(word in text_lower for word in ["thank", "thanks", "appreciate"]):
            acts.append("THANK_YOU")
        if any(word in text_lower for word in ["yes", "yeah", "sure", "okay", "ok", "correct"]):
            acts.append("AFFIRM")
        if any(word in text_lower for word in ["no", "not", "don't", "won't", "can't"]):
            acts.append("NEGATE")
        if any(word in text_lower for word in ["goodbye", "bye", "see you", "farewell"]):
            acts.append("GOODBYE")
        if any(word in text_lower for word in ["want", "need", "looking for", "searching"]):
            acts.append("INFORM")
        if any(word in text_lower for word in ["select", "choose", "pick", "book", "reserve"]):
            acts.append("SELECT")
        
        # In online mode, if no acts detected but text is non-empty, default to INFORM
        # This prevents episodes from ending prematurely due to missing dialogue acts
        if not acts and text_lower:
            acts.append("INFORM")
        
        return acts
    
    def get_text_history(self) -> List[Tuple[str, str]]:
        """Get text-only dialogue history (what agent sees)."""
        return self.env_state.dialogue_history.copy()
    
    def get_structured_state(self) -> OnlineEnvironmentState:
        """Get full structured state (for environment use only)."""
        return self.env_state
    
    def record_belief_generation(self, candidates: List[str]) -> None:
        """Store the most recent belief generation (list of candidate summaries)."""
        if candidates:
            self.env_state.belief_generations.append(candidates.copy())
    
    def _maybe_print_first_dialogue_trajectory(self) -> None:
        """Print full dialogue once when the first dialogue finishes."""
        global _first_dialogue_printed
        if _first_dialogue_printed:
            return
        _first_dialogue_printed = True
        self._print_full_dialogue_trajectory()

    def _print_full_dialogue_trajectory(self) -> None:
        """
        Print the full dialogue trajectory with alternating user and agent messages.
        Called when the first dialogue completes.
        
        The dialogue history is stored as (agent_action, user_response) tuples.
        Structure after steps: [(agent_1, user_1), (agent_2, user_2), ...]
        Note: The initial user message may be in the first user_response if the first
        agent_action was empty, but typically it gets overwritten. We print what we have.
        """
        import sys
        
        print("\n" + "="*80)
        print("[DEBUG] FULL DIALOGUE TRAJECTORY (First Completed Dialogue)")
        print("="*80)
        print("User Goal (goal_json):")
        if self.goal_json:
            print(json.dumps(self.goal_json, indent=2))
        else:
            print("  (No goal specified)")
        print("\nUser Goal (natural language):")
        if self.goal_json:
            try:
                from ..data.dialogue_formatter import format_multiwoz_goal
                print(f"  {format_multiwoz_goal(self.goal_json)}")
            except Exception as exc:
                print(f"  [Error formatting goal: {exc}]")
        else:
            print("  (No goal specified)")
        
        if not self.env_state.dialogue_history:
            print("  (No dialogue history available)")
            print("="*80 + "\n")
            sys.stdout.flush()
            return
        
        # Print dialogue in alternating format: User -> Agent -> User -> Agent -> ...
        # Each entry in dialogue_history is (agent_action, user_response)
        # We want to show the conversation flow
        
        turn_num = 0
        for agent_action, user_response in self.env_state.dialogue_history:
            # Handle first entry which may have empty agent_action (initial user message)
            if not agent_action and user_response:
                # Initial user message
                turn_num += 1
                print(f"\nTurn {turn_num} - User:")
                print(f"  {user_response}")
            else:
                # Subsequent entries: Agent -> User
                if agent_action:
                    turn_num += 1
                    print(f"\nTurn {turn_num} - Agent:")
                    print(f"  {agent_action}")
                
                if user_response:
                    print(f"\nTurn {turn_num} - User:")
                    print(f"  {user_response}")
        
        print("\n" + "="*80)
        print(f"[DEBUG] Dialogue completed with {self.env_state.turn_idx} turns")
        print(f"[DEBUG] Goal achieved: {self.env_state.goal_achieved:.2%} ({self.env_state.goal_achieved:.2f})")
        if self.goal_state.desires:
            total_desires = len(self.goal_state.desires)
            satisfied_desires = len(self.goal_state.get_satisfied_desires())
            failed_desires = len(self.goal_state.get_failed_desires())
            pending_desires = total_desires - satisfied_desires - failed_desires
            print(f"[DEBUG] Desires: {satisfied_desires}/{total_desires} satisfied, {failed_desires} failed, {pending_desires} pending")
            if total_desires > 0:
                print(f"[DEBUG] Desire details:")
                for i, desire in enumerate(self.goal_state.desires):
                    print(f"  Desire {i+1}: {desire.intent} ({desire.domain}) - Status: {desire.status}")
        print("="*80 + "\n")
        
        if self.env_state.belief_generations:
            print("[DEBUG] Belief generations per turn:")
            for idx, candidates in enumerate(self.env_state.belief_generations, start=1):
                print(f"\nBelief Gen {idx}:")
                for cand in candidates:
                    print(f"  - {cand}")
        else:
            print("[DEBUG] No belief generations were recorded.")
        
        # Print sample inputs for each model
        print("\n" + "="*80)
        print("[DEBUG] SAMPLE MODEL INPUTS (First Turn Only)")
        print("="*80)
        
        if self.env_state.hl_agent_inputs and len(self.env_state.hl_agent_inputs) > 0:
            print("\n[High-Level Agent Input (Turn 1)]:")
            print("-" * 80)
            print(self.env_state.hl_agent_inputs[0])
            print("-" * 80)
        else:
            print("\n[High-Level Agent Input]: Not available")
        
        if self.env_state.ll_agent_inputs and len(self.env_state.ll_agent_inputs) > 0:
            print("\n[Low-Level Agent Input (Turn 1)]:")
            print("-" * 80)
            print(self.env_state.ll_agent_inputs[0])
            print("-" * 80)
        else:
            print("\n[Low-Level Agent Input]: Not available")
        
        if self.env_state.user_agent_inputs and len(self.env_state.user_agent_inputs) > 0:
            print("\n[User Agent Input (Turn 1)]:")
            print("-" * 80)
            print(self.env_state.user_agent_inputs[0])
            print("-" * 80)
        else:
            print("\n[User Agent Input]: Not available")
        
        print("\n" + "="*80)
        print("[DEBUG] SAMPLE MODEL OUTPUTS (First Turn Only)")
        print("="*80)
        
        if self.env_state.hl_agent_outputs and len(self.env_state.hl_agent_outputs) > 0:
            print("\n[High-Level Agent Output (Turn 1)]:")
            print("-" * 80)
            print(self.env_state.hl_agent_outputs[0])
            print("-" * 80)
        else:
            print("\n[High-Level Agent Output]: Not available")
        
        if self.env_state.ll_agent_outputs and len(self.env_state.ll_agent_outputs) > 0:
            print("\n[Low-Level Agent Output (Turn 1)]:")
            print("-" * 80)
            print(self.env_state.ll_agent_outputs[0])
            print("-" * 80)
        else:
            print("\n[Low-Level Agent Output]: Not available")
        
        if self.env_state.user_agent_outputs and len(self.env_state.user_agent_outputs) > 0:
            print("\n[User Agent Output (Turn 1)]:")
            print("-" * 80)
            print(self.env_state.user_agent_outputs[0])
            print("-" * 80)
        else:
            print("\n[User Agent Output]: Not available")
        
        print("="*80 + "\n")
        sys.stdout.flush()

