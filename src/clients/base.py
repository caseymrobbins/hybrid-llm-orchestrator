# src/clients/base.py

from abc import ABC, abstractmethod

class LlmClient(ABC):
    """Abstract base class for all LLM clients."""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generates a response from the LLM based on the provided prompt.
        
        Args:
            prompt: The input text to the model.
            **kwargs: Additional provider-specific parameters.
            
        Returns:
            The text response from the LLM.
        """
        pass