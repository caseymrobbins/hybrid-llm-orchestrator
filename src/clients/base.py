# src/clients/base.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class LlmClient(ABC):
    """
    Abstract base class for all LLM clients.
    
    Defines the common interface that all LLM clients must implement,
    ensuring consistency across different providers and model types.
    """

    @abstractmethod
    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """
        Generate a response from the LLM based on the provided prompt.
        
        Args:
            prompt: The input text/prompt for the model
            system_prompt: Optional system message or context
            **kwargs: Additional provider-specific parameters such as:
                - model: Override the default model
                - max_tokens: Maximum tokens to generate
                - temperature: Sampling temperature (0.0 to 2.0)
                - top_p: Top-p sampling parameter
                - frequency_penalty: Frequency penalty
                - presence_penalty: Presence penalty
                - timeout: Request timeout in seconds
                
        Returns:
            The text response from the LLM
            
        Raises:
            ConnectionError: When API communication fails
            ValueError: When input parameters are invalid
            TimeoutError: When request times out
        """
        pass

    async def validate_connection(self) -> bool:
        """
        Validate that the client can successfully connect to the LLM service.
        
        Returns:
            True if connection is valid and working, False otherwise
        """
        try:
            # Default implementation - try a simple generation
            await self.generate("Hello", max_tokens=1)
            return True
        except Exception:
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model configuration.
        
        Returns:
            Dictionary containing model information such as:
            - model_name: Name of the current model
            - provider: Name of the provider
            - max_retries: Maximum retry attempts
            - base_url: API base URL (if applicable)
        """
        return {
            "model_name": getattr(self, 'model_name', 'unknown'),
            "provider": self.__class__.__name__.replace('Client', '').lower()
        }

    def get_generation_params(self) -> Dict[str, Any]:
        """
        Get the default generation parameters for this client.
        
        Returns:
            Dictionary of default parameters used for text generation
        """
        return {
            "temperature": getattr(self, 'temperature', 0.7),
            "max_tokens": getattr(self, 'max_tokens', 1024),
            "top_p": getattr(self, 'top_p', 0.9)
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a comprehensive health check of the client.
        
        Returns:
            Dictionary containing health check results
        """
        try:
            connection_valid = await self.validate_connection()
            model_info = self.get_model_info()
            
            return {
                "status": "healthy" if connection_valid else "unhealthy",
                "connection_valid": connection_valid,
                "model_info": model_info,
                "client_type": self.__class__.__name__
            }
        except Exception as e:
            return {
                "status": "error",
                "connection_valid": False,
                "error": str(e),
                "client_type": self.__class__.__name__
            }

    def __repr__(self) -> str:
        """String representation of the client."""
        model_name = getattr(self, 'model_name', 'unknown')
        return f"{self.__class__.__name__}(model='{model_name}')"