"""Track goal satisfaction state with sequential desires."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .constraint_simulator import ActionConstraintSimulator
from .desire_extraction import UserDesire


@dataclass
class GoalState:
    """Tracks goal satisfaction state with sequential desires."""
    
    # Sequential list of user desires
    desires: List[UserDesire] = field(default_factory=list)
    
    # Current desire being pursued
    current_desire_idx: int = 0
    
    # Track slot values across all desires
    slot_values: Dict[str, Dict[str, str]] = field(default_factory=dict)  # domain -> slot -> value
    
    # Action constraint simulator
    constraint_simulator: Optional[ActionConstraintSimulator] = None
    
    def add_desire(self, desire: UserDesire):
        """Add a new desire to the sequence."""
        self.desires.append(desire)
    
    def update_from_agent_response(
        self,
        agent_text: str,
        extracted_slots: Dict[str, Dict[str, str]],  # domain -> slot -> value
        service_call: Optional[Dict] = None,
        goal_json: Optional[Dict] = None
    ):
        """
        Update goal state from agent response.
        
        Args:
            agent_text: Agent's natural language response
            extracted_slots: Slots extracted from agent response (domain -> slot -> value)
            service_call: Optional service call dict with method and parameters
            goal_json: Optional goal JSON to check request_slots satisfaction
        """
        # Update slot values
        for domain, slots in extracted_slots.items():
            if domain not in self.slot_values:
                self.slot_values[domain] = {}
            self.slot_values[domain].update(slots)
        
        # Check for service call (booking attempt)
        if service_call and self.constraint_simulator:
            if isinstance(service_call, dict):
                service = service_call.get("service", "")
                method = service_call.get("method", "")
                parameters = service_call.get("parameters", {})
                
                if service and method and parameters:
                    success, results = self.constraint_simulator.check_booking_constraint(
                        service=service,
                        method=method,
                        parameters=parameters
                    )
                    
                    if self.current_desire_idx < len(self.desires):
                        current_desire = self.desires[self.current_desire_idx]
                        if success:
                            current_desire.status = "satisfied"
                            # Move to next desire if available
                            if self.current_desire_idx + 1 < len(self.desires):
                                self.current_desire_idx += 1
                        else:
                            current_desire.status = "failed"
                            current_desire.attempt_count += 1
                            
                            # If failed, move to next desire if available
                            if self.current_desire_idx + 1 < len(self.desires):
                                self.current_desire_idx += 1
        
        # Check if request_slots from goal_json have been satisfied
        # This handles information requests (not just bookings)
        if goal_json and extracted_slots and self.current_desire_idx < len(self.desires):
            current_desire = self.desires[self.current_desire_idx]
            if current_desire.status == "pending":
                # Check if any domain in goal_json has request_slots that match the current desire's domain
                for domain, domain_goal in goal_json.items():
                    if not isinstance(domain_goal, dict):
                        continue
                    
                    # Normalize domain names (e.g., "restaurant" vs "Restaurants_2")
                    domain_normalized = domain.lower().replace("_", "").replace("-", "")
                    desire_domain_normalized = current_desire.domain.lower().replace("_", "").replace("-", "")
                    
                    # Check if this domain matches the current desire's domain
                    if domain_normalized in desire_domain_normalized or desire_domain_normalized in domain_normalized:
                        request_slots = domain_goal.get("request_slots", {})
                        if request_slots:
                            # Check if extracted slots satisfy the request_slots
                            extracted_for_domain = extracted_slots.get(domain, {})
                            if not extracted_for_domain:
                                # Try to find extracted slots for the desire's domain
                                for ext_domain, ext_slots in extracted_slots.items():
                                    ext_domain_normalized = ext_domain.lower().replace("_", "").replace("-", "")
                                    if domain_normalized in ext_domain_normalized or ext_domain_normalized in domain_normalized:
                                        extracted_for_domain = ext_slots
                                        break
                            
                            # Check if we have values for the requested slots
                            # Normalize slot names (e.g., "restaurant-food" vs "food")
                            satisfied_count = 0
                            total_requested = len(request_slots)
                            
                            for requested_slot, requested_value in request_slots.items():
                                # Normalize slot name (remove domain prefix if present)
                                slot_normalized = requested_slot.lower().replace("_", "-")
                                if "-" in slot_normalized:
                                    # Remove domain prefix (e.g., "restaurant-food" -> "food")
                                    slot_name = slot_normalized.split("-", 1)[-1]
                                else:
                                    slot_name = slot_normalized
                                
                                # Check if any extracted slot matches (by name or value)
                                slot_satisfied = False
                                for ext_slot, ext_value in extracted_for_domain.items():
                                    ext_slot_normalized = ext_slot.lower().replace("_", "-")
                                    ext_slot_name = ext_slot_normalized.split("-", 1)[-1] if "-" in ext_slot_normalized else ext_slot_normalized
                                    
                                    # Check if slot names match
                                    if slot_name == ext_slot_name or slot_normalized == ext_slot_normalized:
                                        # Check if we have a non-empty value
                                        if ext_value and str(ext_value).strip() and str(ext_value).strip().lower() not in ["none", "n/a", "unknown", ""]:
                                            slot_satisfied = True
                                            break
                                
                                # Also check slot_values for this slot
                                if not slot_satisfied and domain in self.slot_values:
                                    domain_slots = self.slot_values[domain]
                                    for slot_key, slot_val in domain_slots.items():
                                        slot_key_normalized = slot_key.lower().replace("_", "-")
                                        slot_key_name = slot_key_normalized.split("-", 1)[-1] if "-" in slot_key_normalized else slot_key_normalized
                                        if slot_name == slot_key_name and slot_val and str(slot_val).strip():
                                            slot_satisfied = True
                                            break
                                
                                if slot_satisfied:
                                    satisfied_count += 1
                            
                            # If all request_slots are satisfied, mark desire as satisfied
                            if total_requested > 0 and satisfied_count == total_requested:
                                current_desire.status = "satisfied"
                                # Move to next desire if available
                                if self.current_desire_idx + 1 < len(self.desires):
                                    self.current_desire_idx += 1
                                break
    
    def update_desire_status(self, status: str):
        """Manually update the current desire's status."""
        if self.current_desire_idx < len(self.desires):
            self.desires[self.current_desire_idx].status = status
    
    def has_any_satisfied_desire(self) -> bool:
        """Return True if at least one desire has been satisfied."""
        return any(d.status == "satisfied" for d in self.desires)
    
    def is_goal_achieved(self) -> bool:
        """
        Check if overall goal is achieved.
        
        Goal is considered achieved only when all desires have been satisfied.
        """
        if not self.desires:
            return False
        
        return all(d.status == "satisfied" for d in self.desires)
    
    def get_goal_achievement_fraction(self) -> float:
        """
        Get the fraction of goals that are satisfied (0.0 to 1.0).
        
        Returns:
            Fraction of satisfied desires (0.0 = none satisfied, 1.0 = all satisfied)
        """
        if not self.desires:
            return 0.0
        
        satisfied_count = len(self.get_satisfied_desires())
        total_count = len(self.desires)
        
        return satisfied_count / total_count if total_count > 0 else 0.0
    
    def get_current_desire(self) -> Optional[UserDesire]:
        """Get the current desire being pursued."""
        if self.current_desire_idx < len(self.desires):
            return self.desires[self.current_desire_idx]
        return None
    
    def get_satisfied_desires(self) -> List[UserDesire]:
        """Get all satisfied desires."""
        return [d for d in self.desires if d.status == "satisfied"]
    
    def get_failed_desires(self) -> List[UserDesire]:
        """Get all failed desires."""
        return [d for d in self.desires if d.status == "failed"]

    @staticmethod
    def _normalize_name(name: str) -> str:
        return name.lower().replace("_", "-") if name else ""

    def _match_domain_slots(self, domain: str) -> Dict[str, str]:
        """Return slot dictionary for domain, handling name variants."""
        if not self.slot_values:
            return {}

        target = self._normalize_name(domain)
        for stored_domain, slots in self.slot_values.items():
            normalized = self._normalize_name(stored_domain)
            if target and (target in normalized or normalized in target):
                return slots

        return self.slot_values.get(domain, {})

    def _slot_is_satisfied(self, domain: str, slot_name: str) -> bool:
        """Heuristic check if a requested slot has a provided value."""
        slots = self._match_domain_slots(domain)
        if not slots:
            return False

        target = self._normalize_name(slot_name)
        for provided_slot, value in slots.items():
            provided_norm = self._normalize_name(provided_slot)
            if provided_norm == target or provided_norm.endswith(f"-{target}") or target.endswith(f"-{provided_norm}"):
                if value is not None and str(value).strip() and str(value).strip().lower() not in ["none", "n/a", "unknown", ""]:
                    return True

        return False

    def build_progress_summary(self, goal_json: Optional[Dict] = None) -> str:
        """
        Build a natural language summary focused on the current desire only.

        Args:
            goal_json: Original goal specification (used to determine pending info requests)

        Returns:
            String describing the current desire's status plus remaining info requests.
        """
        if not self.desires:
            return "No structured desires defined yet. Wait for a new goal."

        current_desire = self.get_current_desire()
        if current_desire is None:
            return "All desires are satisfied. Awaiting confirmation or new goals."

        lines: List[str] = []
        lines.append(
            f"Current desire: {current_desire.intent} ({current_desire.domain}). "
            f"Status: {current_desire.status}."
        )

        # Provide quick progress summary for context
        total_desires = len(self.desires)
        satisfied_desires = len(self.get_satisfied_desires())
        lines.append(f"Overall progress: {satisfied_desires}/{total_desires} desires satisfied.")

        # Summarize request slot progress for the current desire only
        pending_slots: List[str] = []
        satisfied_slots: List[str] = []

        if goal_json and current_desire.domain:
            # Find matching domain entry (allow fuzzy match)
            matched_domain = None
            desired_domain_norm = self._normalize_name(current_desire.domain)
            for domain_name in goal_json.keys():
                domain_norm = self._normalize_name(domain_name)
                if desired_domain_norm in domain_norm or domain_norm in desired_domain_norm:
                    matched_domain = domain_name
                    break

            if matched_domain:
                domain_goal = goal_json.get(matched_domain, {})
                request_slots = domain_goal.get("request_slots") or {}
                for slot_name in request_slots.keys():
                    if self._slot_is_satisfied(matched_domain, slot_name):
                        satisfied_slots.append(slot_name)
                    else:
                        pending_slots.append(slot_name)

        if satisfied_slots:
            lines.append(f"Already satisfied: {', '.join(satisfied_slots)}.")
        if pending_slots:
            lines.append(f"Still need: {', '.join(pending_slots)}.")

        if not pending_slots and current_desire.status != "satisfied":
            lines.append("All required information appears provided. Confirm completion.")

        if len(lines) == 2 and not pending_slots and not satisfied_slots:
            lines.append("No specific info requests for this desire; focus on next required step.")

        return "\n".join(lines)

