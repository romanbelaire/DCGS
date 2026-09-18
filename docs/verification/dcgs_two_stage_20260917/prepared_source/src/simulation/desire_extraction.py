"""Extract sequential user desires from MultiWOZ dialogues."""

from dataclasses import dataclass, field
from typing import Dict, List

from ..data.multiwoz_loader import MultiWOZDialogue


@dataclass
class UserDesire:
    """Represents a single user desire/goal."""
    
    intent: str  # e.g., "ReserveRestaurant"
    domain: str  # e.g., "Restaurants_2"
    required_slots: Dict[str, str] = field(default_factory=dict)  # slot -> value
    optional_slots: Dict[str, str] = field(default_factory=dict)
    status: str = "pending"  # pending, in_progress, satisfied, failed
    attempt_count: int = 0  # How many times user tried this desire
    
    def __post_init__(self):
        """Normalize slot values to handle lists."""
        # MultiWOZ slot_values can be lists, take first value if so
        normalized_required = {}
        for slot, value in self.required_slots.items():
            if isinstance(value, list) and len(value) > 0:
                normalized_required[slot] = value[0]
            else:
                normalized_required[slot] = value
        self.required_slots = normalized_required
        
        normalized_optional = {}
        for slot, value in self.optional_slots.items():
            if isinstance(value, list) and len(value) > 0:
                normalized_optional[slot] = value[0]
            else:
                normalized_optional[slot] = value
        self.optional_slots = normalized_optional


def extract_desires_from_dialogue(dialogue: MultiWOZDialogue) -> List[UserDesire]:
    """
    Extract sequential user desires from a MultiWOZ dialogue.
    
    Returns:
        List of UserDesire objects in chronological order
    """
    desires = []
    current_desire = None
    
    for turn in dialogue.turns:
        if not isinstance(turn, dict):
            continue
            
        speaker = turn.get("speaker", "").upper()
        
        if speaker == "USER":
            # Extract intent and slots from user turn
            for frame in turn.get("frames", []):
                if not isinstance(frame, dict):
                    continue
                    
                state = frame.get("state", {})
                if not isinstance(state, dict):
                    continue
                    
                intent = state.get("active_intent", "NONE")
                slot_values = state.get("slot_values", {})
                
                # Normalize slot_values (handle lists)
                normalized_slots = {}
                for slot, value in slot_values.items():
                    if isinstance(value, list) and len(value) > 0:
                        normalized_slots[slot] = value[0]
                    elif value:
                        normalized_slots[slot] = value
                
                # Check if this is a new desire or continuation
                if intent != "NONE":
                    # Check if we need a new desire
                    if current_desire is None:
                        # First desire
                        current_desire = UserDesire(
                            intent=intent,
                            domain=frame.get("service", ""),
                            required_slots=normalized_slots.copy()
                        )
                    elif (current_desire.intent != intent or 
                          current_desire.domain != frame.get("service", "")):
                        # Different intent or domain - new desire
                        if current_desire.status == "pending":
                            current_desire.status = "in_progress"
                        desires.append(current_desire)
                        current_desire = UserDesire(
                            intent=intent,
                            domain=frame.get("service", ""),
                            required_slots=normalized_slots.copy()
                        )
                    else:
                        # Same intent/domain - update slots
                        # Check if key slots changed (e.g., restaurant_name changed)
                        # This indicates a new attempt at the same desire
                        key_slots = ["restaurant_name", "hotel_name", "train_id", "attraction_name"]
                        slot_changed = False
                        for key_slot in key_slots:
                            if (key_slot in normalized_slots and 
                                key_slot in current_desire.required_slots and
                                normalized_slots[key_slot] != current_desire.required_slots[key_slot]):
                                slot_changed = True
                                break
                        
                        if slot_changed and current_desire.status == "failed":
                            # User is trying a different option after failure - new desire
                            desires.append(current_desire)
                            current_desire = UserDesire(
                                intent=intent,
                                domain=frame.get("service", ""),
                                required_slots=normalized_slots.copy()
                            )
                        else:
                            # Update existing desire with new slots
                            current_desire.required_slots.update(normalized_slots)
        
        elif speaker == "SYSTEM":
            # Check for NOTIFY_FAILURE or NOTIFY_SUCCESS
            for frame in turn.get("frames", []):
                if not isinstance(frame, dict):
                    continue
                    
                actions = frame.get("actions", [])
                if not isinstance(actions, list):
                    actions = [actions] if actions else []
                
                for action in actions:
                    if not isinstance(action, dict):
                        continue
                        
                    act = action.get("act", "")
                    if act == "NOTIFY_FAILURE":
                        if current_desire:
                            current_desire.status = "failed"
                            current_desire.attempt_count += 1
                    elif act == "NOTIFY_SUCCESS":
                        if current_desire:
                            current_desire.status = "satisfied"
    
    # Add final desire if exists
    if current_desire:
        if current_desire.status == "pending":
            # Check if any desire was satisfied - if so, mark this as satisfied too
            # Otherwise, mark as in_progress
            if any(d.status == "satisfied" for d in desires):
                current_desire.status = "satisfied"
            else:
                current_desire.status = "in_progress"
        desires.append(current_desire)
    
    return desires

