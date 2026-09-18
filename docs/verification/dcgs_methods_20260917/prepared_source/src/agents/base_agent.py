"""Base agent interface."""

from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseAgent(ABC):
    """Abstract base class for agents."""
    
    def __init__(self, model, tokenizer, prompt_manager=None):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_manager = prompt_manager
    
    @abstractmethod
    def act(self, *args, **kwargs):
        """Perform an action based on current state."""
        pass

