"""MultiWOZ environment for offline RL training."""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..data.multiwoz_loader import MultiWOZDialogue

from .dialogue_env import DialogueEnvironment


@dataclass
class EnvironmentState:
    """
    Internal environment state using structured MultiWOZ information.
    
    This is NOT what the agent sees - the agent only sees text observations.
    The environment uses this structured state for:
    - Reward computation (using actions)
    - Episode termination (using GOODBYE)
    - State tracking (using slots, intents, etc.)
    """
    # Current turn index in the dialogue
    turn_idx: int = 0
    
    # Structured dialogue state from MultiWOZ
    # Maps domain -> {active_intent, requested_slots, slot_values}
    dialogue_state: Dict = field(default_factory=dict)
    
    # History of structured turns (for environment tracking)
    structured_history: List[Dict] = field(default_factory=list)
    
    # Text history (what agent sees)
    text_history: List[Tuple[str, str]] = field(default_factory=list)  # (action, observation)
    
    # Episode metadata
    goal_achieved: bool = False
    episode_ended: bool = False
    
    # Belief generation history for debugging
    belief_generations: List[List[str]] = field(default_factory=list)
    
    def get_current_user_turn(self) -> Optional[Dict]:
        """Get current user turn with structured information."""
        if self.structured_history and len(self.structured_history) > 0:
            # Get last turn if it's a user turn
            last_turn = self.structured_history[-1]
            if last_turn.get("speaker", "").lower() in ["user", "usr"]:
                return last_turn
        return None
    
    def get_current_system_turn(self) -> Optional[Dict]:
        """Get current system turn with structured information."""
        if len(self.structured_history) > 1:
            # Get second-to-last turn if it's a system turn
            prev_turn = self.structured_history[-2]
            if prev_turn.get("speaker", "").lower() in ["system", "sys"]:
                return prev_turn
        return None


_offline_first_dialogue_printed = False


class MultiWOZEnvironment(DialogueEnvironment):
    """
    MultiWOZ environment for offline RL.
    
    Uses structured MultiWOZ data internally but provides text-only observations
    to the agent (realistic setting where agent only sees dialogue text).
    """
    
    def __init__(self, dialogue: MultiWOZDialogue, debug: bool = False):
        """
        Initialize environment with a MultiWOZ dialogue.
        
        Args:
            dialogue: MultiWOZ dialogue to replay (offline RL)
            debug: Whether to print debug statements
        """
        self.dialogue = dialogue
        self.env_state = EnvironmentState()
        self._user_turn_idx = 0  # Track which user turn we're on
        self.debug = debug
        
    def reset(self) -> Tuple[str, Dict]:
        self.env_state = EnvironmentState()
        self._user_turn_idx = 0

        first_user_turn = None
        prev_system_turn = None
        for turn in self.dialogue.turns:
            if isinstance(turn, dict):
                speaker = turn.get("speaker", "").lower()
                if speaker in ["system", "sys"]:
                    prev_system_turn = turn
                elif speaker in ["user", "usr"]:
                    first_user_turn = turn
                    break

        if first_user_turn:
            observation = self._extract_text(first_user_turn)
            self.env_state.structured_history.append(first_user_turn)
            self.env_state.text_history.append(("", observation))
            self._user_turn_idx = 1
            info = self._get_info(first_user_turn, prev_system_turn)
            info["raw_turn"] = first_user_turn
            info["prev_system_turn"] = prev_system_turn
            return observation, info

        return "", {}

    def step(self, agent_action: str = None) -> Tuple[str, str, float, bool, Dict]:
        user_turn = None
        prev_system_turn = None
        dataset_system_action = ""
        user_turn_count = 0

        for i, turn in enumerate(self.dialogue.turns):
            if isinstance(turn, dict):
                speaker = turn.get("speaker", "").lower()
                if speaker in ["user", "usr"]:
                    if user_turn_count == self._user_turn_idx:
                        user_turn = turn
                        if i > 0:
                            prev_turn = self.dialogue.turns[i - 1]
                            if isinstance(prev_turn, dict):
                                prev_speaker = prev_turn.get("speaker", "").lower()
                                if prev_speaker in ["system", "sys"]:
                                    dataset_system_action = self._extract_text(prev_turn)
                                    prev_system_turn = prev_turn
                                    self.env_state.structured_history.append(prev_turn)
                                    self.env_state.text_history.append((dataset_system_action, ""))
                        break
                    user_turn_count += 1

        if user_turn is None:
            from .dialogue_env import StepResult
            return StepResult("", 0.0, True, {}, agent_action="", reward_task=0.0, reward_harm=0.0)

        observation = self._extract_text(user_turn)

        if self.env_state.text_history:
            last_action, _ = self.env_state.text_history[-1]
            self.env_state.text_history[-1] = (last_action, observation)

        self.env_state.structured_history.append(user_turn)
        self.env_state.turn_idx += 1
        self._user_turn_idx += 1

        reward = self._compute_reward(user_turn, prev_system_turn)
        done = self._check_episode_ended(user_turn, prev_system_turn)

        self.env_state.goal_achieved = self._check_goal_achieved(user_turn, prev_system_turn)
        self.env_state.episode_ended = done

        info = self._get_info(user_turn, prev_system_turn)
        info["raw_turn"] = user_turn
        info["prev_system_turn"] = prev_system_turn
        info["reward"] = reward
        info["goal_achieved"] = self.env_state.goal_achieved
        # Note: goal_state is not included here because it's created in create_episode_state
        # and forward-filled via last_known_goal_state in EpisodeState

        if done:
            self._maybe_print_first_dialogue_trajectory()

        from .dialogue_env import StepResult
        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
            agent_action=dataset_system_action,
            reward_task=reward,
            reward_harm=reward,
        )

    def check_goal(self, observation: str) -> bool:
        return self.env_state.goal_achieved

    def _extract_text(self, turn: Dict) -> str:
        """Extract text from turn (what agent observes)."""
        if not isinstance(turn, dict):
            return ""
        return turn.get("text") or turn.get("utterance") or ""
    
    def _extract_dialogue_acts(self, turn: Dict) -> List[str]:
        acts: List[str] = []
        if not isinstance(turn, dict):
            return acts

        dialogue_acts = turn.get("dialogue_acts") or turn.get("dialogue_act")
        canonical_map = {
            "GENERAL-THANK": "THANK_YOU",
            "GENERAL-BYE": "GOODBYE",
            "GENERAL-REQMORE": "REQ_MORE",
            "GENERAL-INFORM": "INFORM",
            "GENERAL-AFFIRM": "AFFIRM",
            "GENERAL-NEGATE": "NEGATE",
        }

        def add_act(act_value):
            if not act_value:
                return
            act_upper = str(act_value).upper()
            acts.append(canonical_map.get(act_upper, act_upper))

        if isinstance(dialogue_acts, dict):
            dialog_act = dialogue_acts.get("dialog_act") or dialogue_acts.get("acts")
            if isinstance(dialog_act, dict):
                act_types = dialog_act.get("act_type") or dialog_act.get("type")
                if isinstance(act_types, list):
                    for act in act_types:
                        add_act(act)
                else:
                    add_act(act_types)
            elif isinstance(dialogue_acts, list):
                for item in dialogue_acts:
                    if isinstance(item, dict) and "act_type" in item:
                        add_act(item["act_type"])
                    else:
                        add_act(item)
        elif isinstance(dialogue_acts, list):
            for item in dialogue_acts:
                if isinstance(item, dict) and "act_type" in item:
                    add_act(item["act_type"])
                else:
                    add_act(item)

        return acts

    def _extract_actions(self, turn: Dict, prev_system_turn: Optional[Dict]) -> List[str]:
        actions = self._extract_dialogue_acts(turn)
        if actions:
            return actions

        if prev_system_turn is not None:
            prev_actions = self._extract_dialogue_acts(prev_system_turn)
            if prev_actions:
                return prev_actions

        # If no dialogue acts found, log and return empty list instead of raising
        # This can happen for some turns in the dataset that don't have dialogue acts
        if self.debug:
            self._log_debug_turns(turn, prev_system_turn)
            print(f"[WARNING] No dialogue acts found for turn or previous system turn. Returning empty actions.")
        return []

    def _compute_reward(self, user_turn: Dict, prev_system_turn: Optional[Dict]) -> float:
        actions = self._extract_actions(user_turn, prev_system_turn)
        success_actions = {"THANK_YOU", "AFFIRM", "SELECT"}
        return 1.0 if any(action in success_actions for action in actions) else 0.0

    def _check_goal_achieved(self, user_turn: Dict, prev_system_turn: Optional[Dict]) -> bool:
        actions = self._extract_actions(user_turn, prev_system_turn)
        success_actions = {"THANK_YOU", "AFFIRM", "SELECT"}
        return any(action in success_actions for action in actions)

    def _check_episode_ended(self, turn: Dict, prev_system_turn: Optional[Dict]) -> bool:
        actions = self._extract_actions(turn, prev_system_turn)
        return any(action == "GOODBYE" for action in actions)

    def _log_debug_turns(self, user_turn: Dict, prev_system_turn: Optional[Dict]) -> None:
        if not self.debug:
            return
        print("    [DEBUG] No actions extracted")
        print(f"    [DEBUG]   user_turn keys: {list(user_turn.keys()) if isinstance(user_turn, dict) else user_turn}")
        if isinstance(user_turn, dict):
            print(f"    [DEBUG]   dialogue_acts: {user_turn.get('dialogue_acts') or user_turn.get('dialogue_act')}")
        if prev_system_turn is not None:
            print(f"    [DEBUG]   prev_system_turn keys: {list(prev_system_turn.keys()) if isinstance(prev_system_turn, dict) else prev_system_turn}")

    def _get_info(self, turn: Dict, prev_system_turn: Optional[Dict]) -> Dict:
        actions = self._extract_actions(turn, prev_system_turn)

        dialogue_state = {}
        frames = (turn.get("frames", []) if isinstance(turn, dict) else [])
        for frame in frames:
            state = frame.get("state", {}) if isinstance(frame, dict) else {}
            if state:
                service = frame.get("service", "unknown")
                dialogue_state[service] = {
                    "active_intent": state.get("active_intent", "NONE"),
                    "requested_slots": state.get("requested_slots", []),
                    "slot_values": state.get("slot_values", {})
                }

        return {
            "actions": actions,
            "dialogue_state": dialogue_state,
            "goal_achieved": self.env_state.goal_achieved,
            "turn_idx": self.env_state.turn_idx,
            "prev_system_turn": prev_system_turn
        }
    
    def get_text_history(self) -> List[Tuple[str, str]]:
        """Get text-only dialogue history (what agent sees)."""
        return self.env_state.text_history.copy()
    
    def get_structured_state(self) -> EnvironmentState:
        """Get full structured state (for environment use only)."""
        return self.env_state
    
    def record_belief_generation(self, candidates: List[str]) -> None:
        """Store belief candidates generated for this dialogue (for logging)."""
        if candidates:
            self.env_state.belief_generations.append(candidates.copy())

    def _maybe_print_first_dialogue_trajectory(self) -> None:
        """Print full dialogue once when the first offline dialogue finishes."""
        global _offline_first_dialogue_printed
        if _offline_first_dialogue_printed:
            return
        _offline_first_dialogue_printed = True
        self._print_full_dialogue_trajectory()

    def _print_full_dialogue_trajectory(self) -> None:
        """Print goal context plus the full dataset dialogue transcript."""
        import sys

        print("\n" + "=" * 80)
        print("[DEBUG] FULL DIALOGUE TRAJECTORY (First Completed Dialogue)")
        print("=" * 80)

        goal_json = getattr(self.dialogue, "goal", None)
        print("User Goal (goal_json):")
        # Check if goal is valid (non-empty dict)
        if goal_json and isinstance(goal_json, dict) and bool(goal_json):
            print(json.dumps(goal_json, indent=2))
        else:
            print("  (No goal specified)")
            # Try to infer goal from dialogue if it's missing
            if goal_json is None or (isinstance(goal_json, dict) and not goal_json):
                try:
                    from ..data.dialogue_formatter import infer_goal_from_dialogue
                    inferred_goal = infer_goal_from_dialogue(self.dialogue)
                    if inferred_goal and isinstance(inferred_goal, dict) and bool(inferred_goal):
                        print(f"\n  (Inferred goal from dialogue):")
                        print(json.dumps(inferred_goal, indent=2))
                        goal_json = inferred_goal  # Use inferred goal for natural language formatting
                except Exception as exc:
                    print(f"  [Could not infer goal: {exc}]")

        print("\nUser Goal (natural language):")
        if goal_json and isinstance(goal_json, dict) and bool(goal_json):
            try:
                from ..data.dialogue_formatter import format_multiwoz_goal
                print(f"  {format_multiwoz_goal(goal_json)}")
            except Exception as exc:
                print(f"  [Error formatting goal: {exc}]")
        else:
            print("  (No goal specified)")

        history = self.env_state.text_history
        if not history:
            print("  (No dialogue history available)")
            print("=" * 80 + "\n")
            sys.stdout.flush()
            return

        for turn_idx, (agent_action, user_response) in enumerate(history, start=1):
            agent_text = agent_action.strip() if agent_action and agent_action.strip() else "[NO_AGENT_ACTION]"
            user_text = user_response.strip() if user_response and user_response.strip() else "[NO_USER_RESPONSE]"
            print(f"\nTurn {turn_idx} - Agent:")
            print(f"  {agent_text}")
            print(f"\nTurn {turn_idx} - User:")
            print(f"  {user_text}")

        print("\n" + "=" * 80)
        print(f"[DEBUG] Dialogue completed with {self.env_state.turn_idx} turns")
        print(f"[DEBUG] Goal achieved: {self.env_state.goal_achieved}")
        print("=" * 80 + "\n")
        
        if self.env_state.belief_generations:
            print("[DEBUG] Belief generations per turn:")
            for idx, candidates in enumerate(self.env_state.belief_generations, start=1):
                print(f"\nBelief Gen {idx}:")
                for cand in candidates:
                    print(f"  - {cand}")
        else:
            print("[DEBUG] No belief generations were recorded.")
        print("=" * 80 + "\n")
        sys.stdout.flush()

