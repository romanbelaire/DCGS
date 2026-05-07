"""Simulate constraints on agent actions (e.g., booking failures)."""

from typing import Dict, List, Optional, Tuple

from ..data.multiwoz_loader import MultiWOZDialogue


class ActionConstraintSimulator:
    """Simulates constraints on agent actions (e.g., booking failures)."""
    
    def __init__(self, dialogue: Optional[MultiWOZDialogue] = None):
        """
        Initialize with optional dialogue to extract constraints.
        
        If dialogue provided, extract failure patterns:
        - Which restaurants fail
        - Which time slots are unavailable
        - etc.
        """
        self.failure_patterns: Dict[str, str] = {}  # key -> "failure" or "success"
        if dialogue:
            self._extract_failure_patterns(dialogue)
    
    def _extract_failure_patterns(self, dialogue: MultiWOZDialogue):
        """Extract failure patterns from dialogue."""
        for turn in dialogue.turns:
            if not isinstance(turn, dict):
                continue
                
            speaker = turn.get("speaker", "").upper()
            if speaker != "SYSTEM":
                continue
            
            for frame in turn.get("frames", []):
                if not isinstance(frame, dict):
                    continue
                    
                actions = frame.get("actions", [])
                if not isinstance(actions, list):
                    actions = [actions] if actions else []
                
                service_call = frame.get("service_call")
                
                # Check for NOTIFY_FAILURE or NOTIFY_SUCCESS
                has_failure = False
                has_success = False
                for action in actions:
                    if not isinstance(action, dict):
                        continue
                    act = action.get("act", "")
                    if act == "NOTIFY_FAILURE":
                        has_failure = True
                    elif act == "NOTIFY_SUCCESS":
                        has_success = True
                
                # Extract what failed or succeeded
                if service_call and isinstance(service_call, dict):
                    method = service_call.get("method", "")
                    parameters = service_call.get("parameters", {})
                    
                    if isinstance(parameters, dict) and parameters:
                        key = self._create_failure_key(parameters)
                        if has_failure:
                            self.failure_patterns[key] = "failure"
                        elif has_success:
                            self.failure_patterns[key] = "success"
    
    def check_booking_constraint(
        self,
        service: str,
        method: str,
        parameters: Dict[str, str]
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Check if a booking attempt would succeed or fail.
        
        Args:
            service: Service name (e.g., "Restaurants_2")
            method: Method name (e.g., "ReserveRestaurant")
            parameters: Booking parameters (e.g., {"restaurant_name": "P.f. Chang's", "date": "2019-03-08"})
        
        Returns:
            (success: bool, service_results: Optional[Dict])
        """
        # Check against failure patterns
        key = self._create_failure_key(parameters)
        if key in self.failure_patterns:
            if self.failure_patterns[key] == "failure":
                return False, None
            elif self.failure_patterns[key] == "success":
                # Return success with mock results
                return True, self._generate_mock_results(service, parameters)
        
        # Default: success (or use more sophisticated logic)
        # For now, return success with mock results
        return True, self._generate_mock_results(service, parameters)
    
    def _create_failure_key(self, parameters: Dict) -> str:
        """Create a key to identify failure patterns."""
        # Use key parameters to create a unique identifier
        # For restaurants: restaurant_name + date + time
        # For hotels: hotel_name + date
        # For trains: train_id or departure + destination + date
        key_parts = []
        
        # Common keys across domains
        if "restaurant_name" in parameters:
            key_parts.append(f"restaurant:{parameters['restaurant_name']}")
        if "hotel_name" in parameters:
            key_parts.append(f"hotel:{parameters['hotel_name']}")
        if "train_id" in parameters:
            key_parts.append(f"train:{parameters['train_id']}")
        if "attraction_name" in parameters:
            key_parts.append(f"attraction:{parameters['attraction_name']}")
        
        # Date and time for bookings
        if "date" in parameters:
            key_parts.append(f"date:{parameters['date']}")
        if "time" in parameters:
            key_parts.append(f"time:{parameters['time']}")
        
        # If no specific entity, use all parameters
        if not key_parts:
            key_parts = [f"{k}:{v}" for k, v in sorted(parameters.items())]
        
        return "|".join(key_parts)
    
    def _generate_mock_results(self, service: str, parameters: Dict) -> Dict:
        """Generate mock service results for successful booking."""
        # Base results from parameters
        results = parameters.copy()
        
        # Add common fields that might be in service_results
        if "restaurant_name" in parameters:
            results["restaurant_name"] = parameters["restaurant_name"]
            # Add mock additional info
            results["rating"] = "4.0"
            results["price_range"] = "moderate"
        
        if "hotel_name" in parameters:
            results["hotel_name"] = parameters["hotel_name"]
            results["rating"] = "4.0"
            results["price_range"] = "moderate"
        
        return results
    
    def add_failure_pattern(self, parameters: Dict[str, str]):
        """Manually add a failure pattern."""
        key = self._create_failure_key(parameters)
        self.failure_patterns[key] = "failure"
    
    def add_success_pattern(self, parameters: Dict[str, str]):
        """Manually add a success pattern."""
        key = self._create_failure_key(parameters)
        self.failure_patterns[key] = "success"

