"""Convert MultiWOZ format to dialogue format.

Goal formation and evaluation (MultiWOZ vs CARES/WildJailbreak):

- MultiWOZ: Goals are structured (domain/slots). format_multiwoz_goal produces a
  concrete summary (e.g. "User wants Italian restaurant in centre for 2 people.
  They also need phone number."). Offline reward does NOT use this—it uses
  dialogue acts (THANK_YOU, AFFIRM, SELECT). Online mode uses the same LLM judge
  with that structured goal text, so the judge has specific criteria.

- CARES/WildJailbreak: "Goal" is only the base_prompt string (one question or
  request). We set goal_json = {"base_prompt": base_prompt}; format_multiwoz_goal
  just returns that string. The fulfillment judge is asked whether the agent
  "fulfilled the user's request" with that single string as the goal. If
  base_prompt is very general (e.g. "Tell me about X"), almost any on-topic
  response can be judged fulfilled, which can inflate goal completion. For more
  comparable evaluation to MultiWOZ, goals would need to be more specific or
  we would need additional criteria (e.g. slot-like constraints or checklist).
"""

import re
from typing import Dict, List, Tuple, Union

from .multiwoz_loader import MultiWOZDialogue


def extract_canonical_assistant_utterance(raw_output: str) -> str:
    """
    Extract the canonical dialogue content from the assistant's raw output.

    Extracts content from [RESPONSE]...[/RESPONSE] tags. The assistant is prompted to
    use this format; we rely on it rather than heuristic stripping, which could
    incorrectly remove valid instructions to the user (e.g., "You should take this
    with food").

    Args:
        raw_output: Raw output from the assistant model.

    Returns:
        Content between [RESPONSE] tags if present, otherwise the raw output.
    """
    if not raw_output or not raw_output.strip():
        return raw_output or ""

    text = raw_output.strip()

    # 1. Prefer [RESPONSE]...[/RESPONSE] content (the model's intended dialogue)
    pattern = r"\[RESPONSE\](.*?)\[/RESPONSE\]"
    matches = list(re.finditer(pattern, text, re.DOTALL | re.IGNORECASE))
    if matches:
        for m in matches:
            content = m.group(1).strip()
            content = re.sub(r"\[/?[^\]]+\]", "", content).strip()
            if len(content) >= 10:
                return content

    # 2. Try opening tag without closing
    opening = re.search(r"\[RESPONSE\](.*)", text, re.DOTALL | re.IGNORECASE)
    if opening:
        content = opening.group(1).strip()
        content = re.sub(r"\[/?[^\]]+\]", "", content).strip()
        if content:
            return content

    # 3. No tags found: return raw. 
    return text


def ensure_goal_json(goal_or_prompt: Union[Dict, str, None]) -> Dict:
    """
    Normalize goal input to a unified goal_json format for user simulation agents.

    Both MultiWOZ (domain/slots structure) and CARES/WildJailbreak (base_prompt string)
    are aligned to goal_json before being passed to UserAgent or PatientAgent.

    Args:
        goal_or_prompt: Either (a) MultiWOZ goal dict, (b) CARES-style {"base_prompt": "..."},
                        or (c) a raw base_prompt string to wrap.

    Returns:
        goal_json: Dict that format_multiwoz_goal can consume.
    """
    if goal_or_prompt is None:
        return {}
    if isinstance(goal_or_prompt, dict):
        return goal_or_prompt
    if isinstance(goal_or_prompt, str) and goal_or_prompt.strip():
        return {"base_prompt": goal_or_prompt.strip()}
    return {}


def format_multiwoz_goal(goal: Dict) -> str:
    """
    Convert goal structure to natural language summary.

    Supports two formats:
    - CARES/WildJailbreak: {"base_prompt": "What are reliable sources..."} -> returns the base_prompt
    - MultiWOZ: {"restaurant": {"inform_slots": {...}, "request_slots": {...}}} -> domain-based summary

    Example MultiWOZ:
        {
            "restaurant": {
                "inform_slots": {"area": "centre", "food": "italian", "people": "2"},
                "request_slots": {"phone": "?"}
            }
        }
        Returns: "User wants Italian restaurant in centre for 2 people. They also need phone number."
    """
    if not goal:
        return "User goal not specified"
    if "base_prompt" in goal and goal["base_prompt"]:
        return str(goal["base_prompt"]).strip()
    parts = []
    for domain, domain_goal in goal.items():
        domain_parts = []
        
        # Inform slots (requirements)
        if "inform_slots" in domain_goal:
            inform_slots = domain_goal["inform_slots"]
            if inform_slots:
                slot_descriptions = []
                for slot, value in inform_slots.items():
                    if slot == "food":
                        slot_descriptions.append(f"{value} cuisine")
                    elif slot == "area":
                        slot_descriptions.append(f"in {value}")
                    elif slot == "people":
                        slot_descriptions.append(f"for {value} people")
                    elif slot == "price":
                        slot_descriptions.append(f"price range: {value}")
                    else:
                        slot_descriptions.append(f"{slot}: {value}")
                
                if slot_descriptions:
                    domain_parts.append(f"{domain} with " + ", ".join(slot_descriptions))
        
        # Request slots (needs information)
        if "request_slots" in domain_goal:
            request_slots = domain_goal["request_slots"]
            if request_slots:
                requested = [slot for slot in request_slots.keys() if request_slots[slot] == "?"]
                if requested:
                    # Format request slots more naturally
                    requested_natural = []
                    for slot in requested:
                        # Convert slot names to more natural language
                        if "-" in slot:
                            slot_parts = slot.split("-")
                            if len(slot_parts) > 1:
                                slot_name = " ".join(slot_parts[1:])  # Remove domain prefix
                            else:
                                slot_name = slot
                        else:
                            slot_name = slot.replace("_", " ")
                        requested_natural.append(slot_name)
                    # Don't add "needs" prefix here - it will be added in the final formatting
                    domain_parts.append(f"{', '.join(requested_natural)}")
        
        if domain_parts:
            # If only request slots (no inform slots), use "User needs" instead of "User wants needs"
            has_inform = "inform_slots" in domain_goal and domain_goal.get("inform_slots")
            if has_inform:
                parts.append("User wants " + " ".join(domain_parts) + ".")
            else:
                # Only request slots - use "User needs" to avoid double "needs"
                parts.append("User needs " + " ".join(domain_parts) + ".")
    
    if not parts:
        return "User goal not specified"
    
    return " ".join(parts)


def extract_goal_from_turn_state(state: Dict, service: str) -> Dict[str, Dict[str, str]]:
    """
    Extract goal structure from a single turn's state (slot_values, requested_slots, active_intent).
    
    Args:
        state: Dialogue state dict with slot_values, requested_slots, active_intent
        service: Service/domain name
        
    Returns:
        Goal dict for this service: {"inform_slots": {...}, "request_slots": {...}}
    """
    if not isinstance(state, dict):
        return {}
    
    service_goal = {"inform_slots": {}, "request_slots": {}}
    
    slot_values = state.get("slot_values") or {}
    requested_slots = state.get("requested_slots") or []
    active_intent = state.get("active_intent", "NONE")
    
    # Only extract if there's an active intent (not "NONE")
    if active_intent == "NONE" and not slot_values and not requested_slots:
        return {}
    
    # Extract inform_slots from slot_values
    if isinstance(slot_values, dict):
        for slot, values in slot_values.items():
            if isinstance(values, list) and values:
                service_goal["inform_slots"][slot] = values[-1]
            elif values:
                service_goal["inform_slots"][slot] = values
    
    # Extract request_slots from requested_slots
    if isinstance(requested_slots, list):
        for slot in requested_slots:
            if slot:  # Only add non-empty slots
                service_goal["request_slots"][slot] = "?"
    
    # Return empty if no slots found
    if not service_goal["inform_slots"] and not service_goal["request_slots"]:
        return {}
    
    return service_goal


def extract_goal_from_dialogue_state(dialogue: MultiWOZDialogue) -> Dict:
    """
    Extract goal structure from dialogue state fields (slot_values, requested_slots, active_intent).
    
    This uses the actual fields from the raw MultiWOZ data structure rather than a non-existent
    goal field. It processes all turns to capture the full sequence of user desires throughout
    the dialogue, which is needed for the online environment to simulate sequential goal pursuit.
    
    Args:
        dialogue: MultiWOZ dialogue
        
    Returns:
        Goal dict with inform_slots and request_slots per domain, aggregated from all turns
    """
    goal: Dict[str, Dict[str, Dict[str, str]]] = {}
    
    # Process all turns to capture the full sequence of desires
    for turn in dialogue.turns:
        if not isinstance(turn, dict):
            continue
        
        frames = turn.get("frames", [])
        if not isinstance(frames, list):
            continue
        
        # Extract goal from each frame in this turn
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            
            service = frame.get("service")
            if not service:
                continue
            
            state = frame.get("state", {})
            if not isinstance(state, dict):
                continue
            
            # Use slot_values and requested_slots directly from the state
            slot_values = state.get("slot_values") or {}
            requested_slots = state.get("requested_slots") or []
            active_intent = state.get("active_intent", "NONE")
            
            # Skip if no active intent and no slots
            if active_intent == "NONE" and not slot_values and not requested_slots:
                continue
            
            # Initialize service goal if not present
            if service not in goal:
                goal[service] = {"inform_slots": {}, "request_slots": {}}
            
            service_goal = goal[service]
            
            # Aggregate inform_slots from slot_values (use latest value if list)
            if isinstance(slot_values, dict):
                for slot, values in slot_values.items():
                    if isinstance(values, list) and values:
                        service_goal["inform_slots"][slot] = values[-1]
                    elif values:
                        service_goal["inform_slots"][slot] = values
            
            # Aggregate request_slots from requested_slots
            if isinstance(requested_slots, list):
                for slot in requested_slots:
                    if slot:  # Only add non-empty slots
                        service_goal["request_slots"][slot] = "?"
    
    # Clean up: remove empty domains and empty slot dicts
    cleaned_goal: Dict[str, Dict[str, Dict[str, str]]] = {}
    for service, service_goal in goal.items():
        inform_slots = service_goal.get("inform_slots", {})
        request_slots = service_goal.get("request_slots", {})
        
        if inform_slots or request_slots:
            cleaned_goal[service] = {}
            if inform_slots:
                cleaned_goal[service]["inform_slots"] = inform_slots
            if request_slots:
                cleaned_goal[service]["request_slots"] = request_slots
    
    return cleaned_goal


def infer_goal_from_dialogue(dialogue: MultiWOZDialogue) -> Dict:
    """
    Infer a goal structure from all dialogue turns when direct extraction fails.
    
    We aggregate slot values and requested slots from per-turn frame states, prioritising
    the latest values observed in the conversation.
    
    This is a fallback when extract_goal_from_initial_turn() doesn't find a goal.
    """
    inferred_goal: Dict[str, Dict[str, Dict[str, str]]] = {}

    for turn in dialogue.turns:
        if not isinstance(turn, dict):
            continue

        frames = turn.get("frames", [])
        if not isinstance(frames, list):
            continue

        for frame in frames:
            if not isinstance(frame, dict):
                continue

            service = frame.get("service")
            if not service:
                continue

            state = frame.get("state", {})
            if not isinstance(state, dict):
                continue

            slot_values = state.get("slot_values") or {}
            requested_slots = state.get("requested_slots") or []

            service_goal = inferred_goal.setdefault(service, {"inform_slots": {}, "request_slots": {}})

            if isinstance(slot_values, dict):
                for slot, values in slot_values.items():
                    # slot values are typically lists (track latest value)
                    if isinstance(values, list) and values:
                        service_goal["inform_slots"][slot] = values[-1]
                    elif values:
                        service_goal["inform_slots"][slot] = values

            if isinstance(requested_slots, list):
                for slot in requested_slots:
                    service_goal["request_slots"][slot] = "?"

    # Remove domains that ended up empty
    cleaned_goal: Dict[str, Dict[str, Dict[str, str]]] = {}
    for service, goal_dict in inferred_goal.items():
        inform_slots = goal_dict.get("inform_slots") or {}
        request_slots = goal_dict.get("request_slots") or {}

        if inform_slots or request_slots:
            cleaned_goal[service] = {}
            if inform_slots:
                cleaned_goal[service]["inform_slots"] = inform_slots
            if request_slots:
                cleaned_goal[service]["request_slots"] = request_slots

    return cleaned_goal


def format_dialogue_history(
    multiwoz_dialogue: MultiWOZDialogue,
    turn_idx: int
) -> List[Tuple[str, str]]:
    """
    Extract dialogue history up to turn_idx.
    
    Returns:
        List of (action, observation) pairs where:
        - action = system text
        - observation = user text
    """
    history = []
    for i in range(min(turn_idx, len(multiwoz_dialogue.turns))):
        turn = multiwoz_dialogue.turns[i]
        if turn["speaker"] == "system":
            # System turn - this is the action
            if i + 1 < len(multiwoz_dialogue.turns):
                next_turn = multiwoz_dialogue.turns[i + 1]
                if next_turn["speaker"] == "user":
                    # User turn - this is the observation
                    history.append((turn["text"], next_turn["text"]))
    return history


def get_ground_truth_belief(multiwoz_dialogue: MultiWOZDialogue) -> str:
    """
    Extract ground truth goal and format as belief summary.
    
    Returns:
        Natural language summary of user goal
    """
    return format_multiwoz_goal(multiwoz_dialogue.goal)

