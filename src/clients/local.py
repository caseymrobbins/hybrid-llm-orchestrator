# src/clients/local.py

import asyncio
import logging
import torch
from typing import Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Pipeline,
    PreTrainedTokenizer,
    PreTrainedModel
)

from .base import LlmClient

logger = logging.getLogger(__name__)

class LocalClientError(Exception):
    """Custom exception for local client errors."""
    pass

class LocalClient(LlmClient):
    """
    Client for running local models via Hugging Face transformers.
    
    Features:
    - Automatic device detection (CUDA, MPS, CPU)
    - Proper async handling with thread pool execution
    - Memory-efficient loading with optional quantization
    - Resource cleanup and management
    - Comprehensive error handling
    """

    def __init__(
        self, 
        model_path: str, 
        use_pipeline: bool = True,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        max_workers: int = 1
    ):
        """
        Initialize the local client.
        
        Args:
            model_path: Path to the model (local path or HuggingFace model ID)
            use_pipeline: Whether to use HuggingFace pipeline (recommended)
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto-detection)
            torch_dtype: Torch data type (None for automatic)
            max_workers: Number of worker threads for async execution
        """
        self.model_path = model_path
        self.use_pipeline = use_pipeline
        self._device = device or self._get_optimal_device()
        self._torch_dtype = torch_dtype or self._get_optimal_dtype()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Model components
        self.pipe: Optional[Pipeline] = None
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        
        # Configuration
        self.model_name = model_path
        self.max_tokens = 512
        self.temperature = 0.7
        self.top_p = 0.95
        
        logger.info(f"Initializing LocalClient for '{model_path}' on device '{self._device}'...")
        
        # Initialize the model
        self._initialize_model()

    def _get_optimal_device(self) -> str:
        """Determine the best available device for inference."""
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"CUDA available: {device_count} devices, {memory:.1f}GB memory")
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("MPS (Apple Silicon) available")
            return "mps"
        else:
            logger.info("Using CPU for inference")
            return "cpu"

    def _get_optimal_dtype(self) -> torch.dtype:
        """Determine optimal torch dtype based on device capabilities."""
        if self._device == "cuda":
            # Check if bfloat16 is supported (Ampere or newer)
            if torch.cuda.is_available():
                capability = torch.cuda.get_device_capability()
                if capability[0] >= 8:  # Ampere or newer
                    logger.info("Using bfloat16 for optimal performance")
                    return torch.bfloat16
                else:
                    logger.info("Using float16 for memory efficiency")
                    return torch.float16
        elif self._device == "mps":
            # MPS works well with float16
            return torch.float16
        
        # CPU fallback
        return torch.float32

    def _initialize_model(self) -> None:
        """Initialize the model and tokenizer."""
        try:
            if self.use_pipeline:
                self._initialize_pipeline()
            else:
                self._initialize_manual()
                
            logger.info(f"Local model '{self.model_path}' loaded successfully on {self._device}")
            
        except Exception as e:
            error_msg = f"Failed to load local model '{self.model_path}': {e}"
            logger.error(error_msg)
            raise LocalClientError(error_msg)

    def _initialize_pipeline(self) -> None:
        """Initialize using HuggingFace pipeline (recommended approach)."""
        self.pipe = pipeline(
            "text-generation",
            model=self.model_path,
            device=self._device if self._device != "mps" else -1,  # MPS uses device=-1
            torch_dtype=self._torch_dtype,
            model_kwargs={
                "low_cpu_mem_usage": True,
                "torch_dtype": self._torch_dtype
            }
        )

    def _initialize_manual(self) -> None:
        """Initialize model and tokenizer manually (for more control)."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self._torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto" if self._device == "cuda" else None
        )
        
        # Move model to device if not using device_map
        if self._device != "cuda":
            self.model = self.model.to(self._device)

    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """
        Generate text using the local model.
        
        Args:
            prompt: The input prompt
            system_prompt: Optional system message (will be prepended)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
            
        Raises:
            LocalClientError: When generation fails
            ValueError: When input is invalid
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        # Prepare the full prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"

        # Extract generation parameters
        max_new_tokens = kwargs.get("max_new_tokens", kwargs.get("max_tokens", self.max_tokens))
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        do_sample = kwargs.get("do_sample", temperature > 0)

        try:
            # Run generation in thread pool to avoid blocking the async loop
            if self.use_pipeline and self.pipe:
                result = await self._generate_with_pipeline(
                    full_prompt, max_new_tokens, temperature, top_p, do_sample
                )
            elif self.model and self.tokenizer:
                result = await self._generate_manual(
                    full_prompt, max_new_tokens, temperature, top_p, do_sample
                )
            else:
                raise LocalClientError("Model not properly initialized")

            return result.strip()

        except Exception as e:
            error_msg = f"Text generation failed: {e}"
            logger.error(error_msg)
            raise LocalClientError(error_msg)

    async def _generate_with_pipeline(
        self, 
        prompt: str, 
        max_new_tokens: int, 
        temperature: float, 
        top_p: float,
        do_sample: bool
    ) -> str:
        """Generate text using the pipeline."""
        def _sync_generate():
            outputs = self.pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                return_full_text=False,  # Only return generated text
                clean_up_tokenization_spaces=True
            )
            
            if outputs and isinstance(outputs, list) and len(outputs) > 0:
                if 'generated_text' in outputs[0]:
                    return outputs[0]['generated_text']
                else:
                    raise LocalClientError("Pipeline output missing 'generated_text' field")
            else:
                raise LocalClientError("Pipeline returned empty or invalid output")

        return await asyncio.get_event_loop().run_in_executor(
            self._executor, _sync_generate
        )

    async def _generate_manual(
        self, 
        prompt: str, 
        max_new_tokens: int, 
        temperature: float, 
        top_p: float,
        do_sample: bool
    ) -> str:
        """Generate text using model and tokenizer manually."""
        def _sync_generate():
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if do_sample else 1.0,
                    top_p=top_p if do_sample else 1.0,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode only the generated tokens
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            
            return self.tokenizer.decode(
                generated_tokens, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

        return await asyncio.get_event_loop().run_in_executor(
            self._executor, _sync_generate
        )

    async def validate_connection(self) -> bool:
        """
        Validate that the local model is loaded and working.
        
        Returns:
            True if model is working, False otherwise
        """
        try:
            # Try a very simple generation
            result = await self.generate("Hello", max_tokens=5)
            return len(result.strip()) > 0
        except Exception as e:
            logger.error(f"Local model validation failed: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the local model."""
        try:
            # Get memory usage if CUDA
            memory_info = {}
            if self._device == "cuda" and torch.cuda.is_available():
                memory_info = {
                    "allocated_memory_gb": torch.cuda.memory_allocated() / 1e9,
                    "cached_memory_gb": torch.cuda.memory_reserved() / 1e9,
                    "max_memory_gb": torch.cuda.max_memory_allocated() / 1e9
                }

            # Get model parameters count
            param_count = 0
            if self.pipe and hasattr(self.pipe.model, 'num_parameters'):
                param_count = self.pipe.model.num_parameters()
            elif self.model and hasattr(self.model, 'num_parameters'):
                param_count = self.model.num_parameters()

            return {
                "model_name": self.model_name,
                "model_path": self.model_path,
                "provider": "local_huggingface",
                "device": self._device,
                "torch_dtype": str(self._torch_dtype),
                "use_pipeline": self.use_pipeline,
                "parameters": param_count,
                "memory_info": memory_info
            }
            
        except Exception as e:
            logger.warning(f"Could not get complete model info: {e}")
            return {
                "model_name": self.model_name,
                "provider": "local_huggingface",
                "device": self._device,
                "error": str(e)
            }

    def cleanup(self) -> None:
        """Clean up model resources."""
        try:
            logger.info("Cleaning up local model resources...")
            
            # Clear CUDA cache if using GPU
            if self._device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Shutdown thread pool
            if self._executor:
                self._executor.shutdown(wait=True)
            
            # Clear model references
            self.pipe = None
            self.model = None
            self.tokenizer = None
            
            logger.info("Local model resources cleaned up")
            
        except Exception as e:
            logger.warning(f"Error during local model cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore errors in destructor