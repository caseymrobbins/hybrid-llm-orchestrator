# src/clients/openai_compatible.py

import asyncio
from openai import AsyncOpenAI, OpenAIError
from.base import LlmClient

class OpenAICompatibleClient(LlmClient):
    """Client for OpenAI and any OpenAI-compatible APIs (e.g., Grok)."""

    def __init__(self, api_key: str, base_url: str):
        if not api_key:
            raise ValueError("API key is required for OpenAICompatibleClient.")
        
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model_name = self._determine_model_name(base_url)

    def _determine_model_name(self, base_url: str) -> str:
        """A simple heuristic to select a default model based on the provider."""
        if "api.x.ai" in base_url:
            return "grok-1.5-claude-3.5-sonnet" # Example Grok model
        # Default to a standard OpenAI model
        return "gpt-4o"

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generates text using an OpenAI-compatible chat completion endpoint."""
        model = kwargs.get("model", self.model_name)
        max_tokens = kwargs.get("max_tokens", 1024)
        temperature = kwargs.get("temperature", 0.7)

        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=,
                max_tokens=max_tokens,
                temperature=temperature
            )
            if response.choices:
                return response.choices.message.content.strip()
            return "Error: No response choices received."
        except OpenAIError as e:
            # Implement exponential backoff for rate limiting
            if "rate limit" in str(e).lower():
                print("Rate limit exceeded. Retrying in 5 seconds...")
                await asyncio.sleep(5)
                # In a production system, this would be a more robust retry loop
                return await self.generate(prompt, **kwargs)
            raise ConnectionError(f"OpenAI API error: {e}")