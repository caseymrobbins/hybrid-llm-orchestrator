# src/clients/openai_compatible.py

import asyncio
import logging
from typing import Optional, Dict, Any
from openai import AsyncOpenAI, OpenAIError, RateLimitError
from .base import LlmClient

logger = logging.getLogger(__name__)

class OpenAICompatibleClient(LlmClient):
    """Client for OpenAI and any OpenAI-compatible APIs (e.g., Grok, Anthropic via proxy)."""

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        if not api_key:
            raise ValueError("API key is required for OpenAICompatibleClient.")
        
        self.client = AsyncOpenAI(
            api_key=api_key, 
            base_url=base_url or "https://api.openai.com/v1"
        )
        self.model_name = self._determine_model_name(base_url or "")
        self.max_retries = 3
        self.base_delay = 1  # Base delay for exponential backoff

    def _determine_model_name(self, base_url: str) -> str:
        """Determine appropriate model based on the provider URL."""
        base_url = base_url.lower()
        
        # Provider-specific model mapping
        if "api.x.ai" in base_url:
            return "grok-beta"  # Correct Grok model name
        elif "api.groq.com" in base_url:
            return "mixtral-8x7b-32768"
        elif "api.together.xyz" in base_url:
            return "mistralai/Mixtral-8x7B-Instruct-v0.1"
        elif "api.anthropic.com" in base_url:
            return "claude-3-sonnet-20240229"
        
        # Default OpenAI model
        return "gpt-4o-mini"

    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generates text using an OpenAI-compatible chat completion endpoint.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system message
            **kwargs: Additional parameters (model, max_tokens, temperature, etc.)
            
        Returns:
            Generated text response
            
        Raises:
            ConnectionError: When API communication fails
            ValueError: When response is invalid
        """
        model = kwargs.get("model", self.model_name)
        max_tokens = kwargs.get("max_tokens", 1024)
        temperature = kwargs.get("temperature", 0.7)

        # Build messages array
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Retry with exponential backoff
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **{k: v for k, v in kwargs.items() 
                       if k not in ['model', 'max_tokens', 'temperature']}
                )
                
                # Validate response structure
                if not response.choices or len(response.choices) == 0:
                    raise ValueError("No response choices received from API")
                
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("Response content is None")
                
                return content.strip()
                
            except RateLimitError as e:
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Rate limit exceeded. Retrying in {delay} seconds... (attempt {attempt + 1}/{self.max_retries})")
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise ConnectionError(f"Rate limit exceeded after {self.max_retries} retries: {e}")
                    
            except OpenAIError as e:
                # Handle other OpenAI errors
                error_msg = f"OpenAI API error: {e}"
                logger.error(error_msg)
                raise ConnectionError(error_msg)
                
            except Exception as e:
                # Handle unexpected errors
                error_msg = f"Unexpected error during generation: {e}"
                logger.error(error_msg)
                raise ConnectionError(error_msg)
        
        # This should never be reached due to the loop structure, but just in case
        raise ConnectionError("Maximum retries exceeded")

    async def validate_connection(self) -> bool:
        """
        Validate that the client can connect to the API.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Try a minimal request to validate connection
            await self.generate("Hello", max_tokens=1)
            return True
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model configuration.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "base_url": self.client.base_url,
            "max_retries": self.max_retries
        }