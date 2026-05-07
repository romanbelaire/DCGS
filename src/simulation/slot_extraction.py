"""Extract slot values from agent natural language responses."""

import re
from typing import Dict, List, Optional


def extract_slots_from_agent_response(
    agent_text: str,
    domain: str,
    goal_json: Optional[Dict] = None,
    use_llm: bool = False
) -> Dict[str, Dict[str, str]]:
    """
    Extract slot values from agent's natural language response.
    
    Args:
        agent_text: Agent's natural language response
        domain: Domain name (e.g., "Restaurants_2", "Hotels_1")
        goal_json: Optional goal JSON to know which slots to look for
        use_llm: Whether to use LLM-based extraction (not implemented yet)
    
    Returns:
        Dict mapping domain -> slot -> value
    """
    if use_llm:
        # TODO: Implement LLM-based extraction
        return _extract_slots_with_llm(agent_text, domain, goal_json)
    else:
        return _extract_slots_rule_based(agent_text, domain, goal_json)


def _extract_slots_rule_based(
    agent_text: str,
    domain: str,
    goal_json: Optional[Dict] = None
) -> Dict[str, Dict[str, str]]:
    """
    Rule-based slot extraction using patterns.
    
    This is a basic implementation that looks for common patterns.
    Can be extended with more sophisticated rules.
    """
    extracted = {domain: {}}
    text_lower = agent_text.lower()
    
    # Domain-specific patterns
    if "restaurant" in domain.lower():
        # Restaurant slots
        restaurant_patterns = {
            "restaurant_name": [
                r"(?:restaurant|place|venue)\s+(?:called|named|is|at)\s+([A-Z][a-zA-Z\s&'\.]+)",
                r"([A-Z][a-zA-Z\s&'\.]+)\s+(?:restaurant|bar|grill|cafe)",
            ],
            "location": [
                r"(?:in|at|located in)\s+([A-Z][a-zA-Z\s]+)",
                r"([A-Z][a-zA-Z\s]+)\s+(?:area|neighborhood|district)",
            ],
            "time": [
                r"(?:at|for)\s+(\d{1,2}:\d{2}|\d{1,2}\s*(?:am|pm|AM|PM))",
                r"(\d{1,2}:\d{2})\s+(?:o'clock|hours)",
            ],
            "date": [
                r"(?:on|for)\s+([A-Z][a-z]+\s+\d{1,2}(?:st|nd|rd|th)?)",
                r"(\d{1,2}(?:st|nd|rd|th)?\s+[A-Z][a-z]+)",
            ],
            "number_of_seats": [
                r"(?:for|party of|table for)\s+(\d+)",
                r"(\d+)\s+(?:people|persons|guests)",
            ],
        }
        
        for slot, patterns in restaurant_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, agent_text, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    extracted[domain][slot] = value
                    break
    
    elif "hotel" in domain.lower():
        # Hotel slots
        hotel_patterns = {
            "hotel_name": [
                r"(?:hotel|accommodation)\s+(?:called|named|is|at)\s+([A-Z][a-zA-Z\s&'\.]+)",
                r"([A-Z][a-zA-Z\s&'\.]+)\s+(?:hotel|inn|lodge)",
            ],
            "location": [
                r"(?:in|at|located in)\s+([A-Z][a-zA-Z\s]+)",
            ],
            "date": [
                r"(?:on|for|check-in|check-out)\s+([A-Z][a-z]+\s+\d{1,2}(?:st|nd|rd|th)?)",
            ],
            "number_of_days": [
                r"(?:for|staying)\s+(\d+)\s+(?:days|nights)",
            ],
        }
        
        for slot, patterns in hotel_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, agent_text, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    extracted[domain][slot] = value
                    break
    
    # Common slots across domains
    # Price range
    price_match = re.search(r"(?:price|cost|expensive|cheap|budget).*?(low|moderate|high|expensive|cheap|budget)", text_lower)
    if price_match:
        price_value = price_match.group(1)
        if price_value in ["low", "cheap", "budget"]:
            extracted[domain]["price_range"] = "cheap"
        elif price_value in ["moderate", "medium"]:
            extracted[domain]["price_range"] = "moderate"
        elif price_value in ["high", "expensive"]:
            extracted[domain]["price_range"] = "expensive"
    
    # Phone number patterns (common across domains)
    # Look for phone numbers in various formats: (555) 123-4567, 555-123-4567, 555.123.4567
    phone_patterns = [
        r"\((\d{3})\)\s*(\d{3}[-.\s]?\d{4})",  # (555) 123-4567
        r"(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})",  # 555-123-4567 or 555.123.4567
        r"phone\s*(?:number|#)?\s*:?\s*(?:is\s*)?\(?(\d{3})\)?\s*[-.\s]?(\d{3})[-.\s]?(\d{4})",
    ]
    for pattern in phone_patterns:
        matches = re.finditer(pattern, agent_text, re.IGNORECASE)
        for match in matches:
            # Reconstruct phone number from groups
            if match.lastindex and match.lastindex >= 2:
                # Multiple groups - combine them
                phone_parts = [match.group(i) for i in range(1, match.lastindex + 1)]
                phone = "".join(phone_parts)
            else:
                phone = match.group(0)
            # Clean up and format
            # Fix regex: escape '-' to avoid bad character range
            phone = re.sub(r'[^\d\-\(\)]', '', phone).strip()
            if phone and len(re.sub(r'[^\d]', '', phone)) >= 10:  # At least 10 digits
                extracted[domain]["phone"] = phone
                # Also try to extract restaurant-phone if that's what was requested
                if "restaurant" in domain.lower():
                    extracted[domain]["restaurant-phone"] = phone
                break
    
    # Food/cuisine patterns (for restaurants)
    if "restaurant" in domain.lower():
        food_patterns = [
            r"(?:food|cuisine|type)\s*(?:is|:)?\s*(?:an?\s*)?([A-Z][a-zA-Z\s]+?)(?:\s+restaurant|\s+cuisine|,|\.|$)",
            r"(?:serves?|offers?|specializes?\s+in)\s+([A-Z][a-zA-Z\s]+?)(?:\s+cuisine|,|\.|$)",
            r"(?:italian|mexican|asian|chinese|japanese|indian|french|american|thai|mediterranean|vegetarian|vegan|gluten-free)",
        ]
        for pattern in food_patterns:
            match = re.search(pattern, agent_text, re.IGNORECASE)
            if match:
                food = match.group(1).strip() if match.lastindex is not None and match.lastindex >= 1 else match.group(0).strip()
                if food and len(food) > 2:  # Avoid single letters
                    extracted[domain]["food"] = food
                    break
    
    # Use goal_json to extract specific requested slots if available
    if goal_json and domain in goal_json:
        domain_goal = goal_json[domain]
        if isinstance(domain_goal, dict):
            request_slots = domain_goal.get("request_slots", {})
            # Look for patterns that match requested slot names
            for slot_name, slot_value in request_slots.items():
                if slot_name not in extracted[domain]:  # Don't overwrite if already extracted
                    # Normalize slot name (remove domain prefix)
                    slot_normalized = slot_name.lower().replace("_", "-")
                    if "-" in slot_normalized:
                        slot_key = slot_normalized.split("-", 1)[-1]  # e.g., "restaurant-food" -> "food"
                    else:
                        slot_key = slot_normalized
                    
                    # Try to find this slot in the text
                    # Look for patterns like "food: Italian" or "phone: 555-1234"
                    slot_patterns = [
                        rf"{slot_key}\s*:?\s*([^\n,\.]+?)(?:\s*[,\.]|\s*$)",
                        rf"{slot_key}\s+(?:is|:)?\s*([^\n,\.]+?)(?:\s*[,\.]|\s*$)",
                    ]
                    for pattern in slot_patterns:
                        match = re.search(pattern, agent_text, re.IGNORECASE)
                        if match:
                            value = match.group(1).strip()
                            if value and len(value) > 0:
                                extracted[domain][slot_name] = value
                                break
    
    return extracted


def _extract_slots_with_llm(
    agent_text: str,
    domain: str,
    goal_json: Optional[Dict] = None
) -> Dict[str, Dict[str, str]]:
    """
    LLM-based slot extraction (not implemented yet).
    
    This would use an LLM to extract structured slot information from agent text.
    """
    # TODO: Implement LLM-based extraction
    # Would use a prompt like:
    # "Extract slot values from this agent response: {agent_text}
    #  Domain: {domain}
    #  Expected slots: {slots_from_goal_json}
    #  Return JSON: {domain: {slot: value}}"
    return {domain: {}}


def extract_service_call_from_agent_response(
    agent_text: str,
    domain: str
) -> Optional[Dict]:
    """
    Extract service call information from agent response.
    
    Looks for indicators that agent is attempting a booking/reservation.
    
    Args:
        agent_text: Agent's natural language response
        domain: Domain name
    
    Returns:
        Optional dict with service, method, and parameters, or None if no service call detected
    """
    text_lower = agent_text.lower()
    
    # Check for booking/reservation language
    booking_indicators = [
        "reservation has been made",
        "booking confirmed",
        "reserved",
        "booked",
        "confirmed your",
        "your reservation",
        "your booking",
    ]
    
    has_booking = any(indicator in text_lower for indicator in booking_indicators)
    
    if not has_booking:
        return None
    
    # Extract parameters from the text
    # This is a simplified version - could be enhanced
    parameters = {}
    
    # Try to extract slots from the text
    # Note: goal_json is not available here, but it's optional for extract_slots_from_agent_response
    slots = extract_slots_from_agent_response(agent_text, domain, goal_json=None)
    if domain in slots:
        parameters = slots[domain]
    
    # Determine method based on domain
    method_map = {
        "restaurants": "ReserveRestaurant",
        "hotels": "ReserveHotel",
        "trains": "BookTrain",
        "attractions": "BookAttraction",
    }
    
    method = None
    for key, value in method_map.items():
        if key in domain.lower():
            method = value
            break
    
    if method and parameters:
        return {
            "service": domain,
            "method": method,
            "parameters": parameters
        }
    
    return None

