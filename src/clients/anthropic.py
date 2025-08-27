# src/clients/anthropic.py

from anthropic import AsyncAnthropic, AnthropicError
from.base import LlmClient

class AnthropicClient(LlmClient):
    """Client for Anthropic's Claude models."""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key is required for AnthropicClient.")
        
        self.client = AsyncAnthropic(api_key=api_key)
        self.model_name = "claude-3-sonnet-20240229" # A strong, cost-effective default

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generates text using the Anthropic messages API."""
        model = kwargs.get("model", self.model_name)
        max_tokens = kwargs.get("max_tokens", 1024)

        try:
            response = await self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            if response.content and isinstance(response.content, list):
                return response.content.text.strip()
            return "Error: No response content received."
        except AnthropicError as e:
            raise ConnectionError(f"Anthropic API error: {e}")
# src/utils/caching.py

import diskcache as dc
import hashlib
from pathlib import Path

class Cache:
    """A simple persistent on-disk cache for LLM responses."""

    def __init__(self, cache_dir: str = ".cache"):
        Path(cache_dir).mkdir(exist_ok=True)
        self.cache = dc.Cache(cache_dir)

    def _generate_key(self, text: str) -> str:
        """Creates a SHA256 hash of the input text to use as a cache key."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def get(self, key_text: str) -> str | None:
        """Retrieves an item from the cache."""
        key = self._generate_key(key_text)
        return self.cache.get(key)

    def set(self, key_text: str, value: str):
        """Saves an item to the cache."""
        key = self._generate_key(key_text)
        self.cache.set(key, value)