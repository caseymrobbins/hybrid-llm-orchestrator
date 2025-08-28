# src/clients/anthropic.py

import asyncio
import logging
from typing import Optional, Dict, Any
from anthropic import AsyncAnthropic, AnthropicError, RateLimitError
from .base import LlmClient

logger = logging.getLogger(__name__)

class AnthropicClient(LlmClient):
    """Client for Anthropic's Claude models with comprehensive error handling and retry logic."""

    def __init__(self, api_key: str, model_name: Optional[str] = None):
        if not api_key:
            raise ValueError("API key is required for AnthropicClient.")
        
        self.client = AsyncAnthropic(api_key=api_key)
        self.model_name = model_name or "claude-3-sonnet-20240229"
        self.max_retries = 3
        self.base_delay = 1

    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate text using Anthropic's Messages API.
        
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
        top_p = kwargs.get("top_p", 0.9)

        # Build messages array
        messages = []
        messages.append({"role": "user", "content": prompt})

        # Prepare API call parameters
        api_params = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages
        }

        # Add system prompt if provided
        if system_prompt:
            api_params["system"] = system_prompt

        # Add optional parameters
        if "temperature" in kwargs:
            api_params["temperature"] = temperature
        if "top_p" in kwargs:
            api_params["top_p"] = top_p

        # Retry with exponential backoff
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.client.messages.create(**api_params)
                
                # Validate response structure
                if not response.content:
                    raise ValueError("No response content received from Anthropic API")
                
                if not isinstance(response.content, list) or len(response.content) == 0:
                    raise ValueError("Invalid response content structure from Anthropic API")
                
                # Extract text from the first content block
                content_block = response.content[0]
                if not hasattr(content_block, 'text'):
                    raise ValueError("Response content block does not contain text")
                
                return content_block.text.strip()
                
            except RateLimitError as e:
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(f"Anthropic rate limit exceeded. Retrying in {delay} seconds... (attempt {attempt + 1}/{self.max_retries})")
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise ConnectionError(f"Anthropic rate limit exceeded after {self.max_retries} retries: {e}")
                    
            except AnthropicError as e:
                error_msg = f"Anthropic API error: {e}"
                logger.error(error_msg)
                raise ConnectionError(error_msg)
                
            except Exception as e:
                error_msg = f"Unexpected error during Anthropic generation: {e}"
                logger.error(error_msg)
                raise ConnectionError(error_msg)
        
        raise ConnectionError("Maximum retries exceeded")

    async def validate_connection(self) -> bool:
        """
        Validate that the client can connect to Anthropic API.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Try a minimal request to validate connection
            await self.generate("Hello", max_tokens=1)
            return True
        except Exception as e:
            logger.error(f"Anthropic connection validation failed: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model configuration.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "provider": "anthropic",
            "max_retries": self.max_retries
        }