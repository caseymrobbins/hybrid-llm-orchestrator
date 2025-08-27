# src/clients/local.py

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from.base import LlmClient

class LocalClient(LlmClient):
    """Client for running local models via Hugging Face transformers."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._device = self._get_device()
        print(f"Initializing local model '{model_path}' on device '{self._device}'...")
        
        try:
            # The pipeline is a high-level, easy-to-use abstraction
            self.pipe = pipeline(
                "text-generation",
                model=self.model_path,
                device=self._device,
                torch_dtype=torch.bfloat16 # Use bfloat16 for better performance on modern GPUs
            )
            print(f"Local model '{model_path}' loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load local model from '{model_path}': {e}")

    def _get_device(self) -> str:
        """Determines the best available device for inference."""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generates text using the local pipeline."""
        # Note: Transformers pipeline is synchronous. For a truly async local setup,
        # one would need to run it in a separate thread or process pool.
        # For this laptop-based system, we assume one local model runs at a time.
        max_new_tokens = kwargs.get("max_new_tokens", 512)
        
        outputs = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )
        
        if outputs and isinstance(outputs, list) and 'generated_text' in outputs:
            return outputs['generated_text']
        
        return "Error: Could not generate response from local model."