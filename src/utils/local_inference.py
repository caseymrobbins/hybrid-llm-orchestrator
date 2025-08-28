# src/utils/local_inference.py

import torch
import asyncio
import logging
from asyncio import Queue
from typing import List, Dict, Any, Optional, Tuple
from contextlib import asynccontextmanager
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    GenerationConfig
)

logger = logging.getLogger(__name__)

class OptimizedLocalModelLoader:
    """Loads Hugging Face models with optimizations for memory efficiency."""
    
    @staticmethod
    def _get_optimal_dtype() -> torch.dtype:
        """Determine the optimal dtype based on hardware capabilities."""
        if torch.cuda.is_available():
            # Check if bfloat16 is supported
            if torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
                return torch.bfloat16
            else:
                return torch.float16
        return torch.float32

    @staticmethod
    def _create_quantization_config() -> Optional[BitsAndBytesConfig]:
        """Create quantization config if CUDA is available."""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping quantization")
            return None
            
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=OptimizedLocalModelLoader._get_optimal_dtype(),
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.uint8
        )

    @classmethod
    def load_model_and_tokenizer(
        self, 
        model_name: str,
        use_quantization: bool = True,
        trust_remote_code: bool = False
    ) -> Tuple[Any, Any]:
        """
        Load model and tokenizer with optimizations.
        
        Args:
            model_name: HuggingFace model identifier
            use_quantization: Whether to use 4-bit quantization
            trust_remote_code: Whether to trust remote code execution
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if not model_name:
            raise ValueError("Model name cannot be empty")

        try:
            # Load tokenizer first
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code
            )
            
            # Set padding token if not present
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            # Configure padding for batched inference
            tokenizer.padding_side = "left"  # Important for generation

            # Prepare model loading arguments
            model_kwargs = {
                "torch_dtype": OptimizedLocalModelLoader._get_optimal_dtype(),
                "trust_remote_code": trust_remote_code,
                "low_cpu_mem_usage": True
            }

            # Add quantization if requested and available
            if use_quantization and torch.cuda.is_available():
                model_kwargs["quantization_config"] = OptimizedLocalModelLoader._create_quantization_config()
                model_kwargs["device_map"] = "auto"
            elif torch.cuda.is_available():
                model_kwargs["device_map"] = "auto"

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # Resize token embeddings if we added new tokens
            if len(tokenizer) > model.config.vocab_size:
                model.resize_token_embeddings(len(tokenizer))

            logger.info(f"Successfully loaded model '{model_name}' with device map: {getattr(model, 'hf_device_map', 'single device')}")
            return model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            raise

class BatchedLocalInference:
    """A server that batches incoming requests for higher throughput."""
    
    def __init__(
        self, 
        model: Any, 
        tokenizer: Any, 
        max_batch_size: int = 8,
        timeout_ms: int = 50,
        max_new_tokens: int = 512,
        generation_config: Optional[Dict[str, Any]] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.queue: Queue = Queue()
        self.max_batch_size = max_batch_size
        self.timeout = timeout_ms / 1000.0
        self.max_new_tokens = max_new_tokens
        self._running = False
        self._runner_task: Optional[asyncio.Task] = None
        
        # Setup generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **(generation_config or {})
        )

    async def start(self) -> None:
        """Start the batch processor."""
        if self._running:
            logger.warning("Batch processor is already running")
            return
            
        self._running = True
        self._runner_task = asyncio.create_task(self._batch_processor())
        logger.info("Batch processor started")

    async def stop(self) -> None:
        """Stop the batch processor and cleanup resources."""
        self._running = False
        
        if self._runner_task and not self._runner_task.done():
            try:
                await asyncio.wait_for(self._runner_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Batch processor did not stop gracefully, cancelling...")
                self._runner_task.cancel()
                try:
                    await self._runner_task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Batch processor stopped")

    @asynccontextmanager
    async def managed_inference(self):
        """Context manager for automatic start/stop of the inference server."""
        await self.start()
        try:
            yield self
        finally:
            await self.stop()

    async def _batch_processor(self) -> None:
        """Continuously processes batches of requests from the queue."""
        logger.info("Batch processor loop started")
        
        while self._running:
            try:
                batch: List[Dict[str, Any]] = []
                start_time = asyncio.get_event_loop().time()
                
                # Collect batch within timeout
                while len(batch) < self.max_batch_size and self._running:
                    try:
                        remaining_time = max(0, self.timeout - (asyncio.get_event_loop().time() - start_time))
                        if remaining_time <= 0:
                            break
                            
                        request = await asyncio.wait_for(
                            self.queue.get(), 
                            timeout=remaining_time
                        )
                        batch.append(request)
                        
                    except asyncio.TimeoutError:
                        break
                
                # Process batch if we have requests
                if batch:
                    await self._process_batch(batch)
                    
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
                # Continue processing other batches
                await asyncio.sleep(0.1)

    async def _process_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Process a batch of generation requests."""
        try:
            prompts = [req['prompt'] for req in batch]
            
            # Tokenize batch
            inputs = self.tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=2048  # Reasonable max length
            )
            
            # Move to model device
            inputs = inputs.to(self.model.device)
            
            # Generate responses
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode results (remove input tokens from output)
            input_lengths = inputs['attention_mask'].sum(dim=1)
            results = []
            
            for i, output in enumerate(outputs):
                # Extract only the newly generated tokens
                generated_tokens = output[input_lengths[i]:]
                decoded = self.tokenizer.decode(
                    generated_tokens, 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                results.append(decoded.strip())
            
            # Set results for all requests in the batch
            for req, result in zip(batch, results):
                if not req['future'].cancelled():
                    req['future'].set_result(result)
                    
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Set exception for all requests in the batch
            for req in batch:
                if not req['future'].cancelled():
                    req['future'].set_exception(e)

    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Add a generation request to the queue and wait for the result.
        
        Args:
            prompt: Input prompt for generation
            **kwargs: Additional generation parameters (currently unused)
            
        Returns:
            Generated text
            
        Raises:
            RuntimeError: If the batch processor is not running
            ValueError: If prompt is empty
        """
        if not self._running:
            raise RuntimeError("Batch processor is not running. Call start() first.")
            
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        future = asyncio.Future()
        await self.queue.put({
            'prompt': prompt.strip(), 
            'future': future,
            **kwargs
        })
        
        try:
            return await future
        except Exception as e:
            logger.error(f"Generation failed for prompt: {prompt[:50]}... Error: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics about the inference server."""
        return {
            "queue_size": self.queue.qsize(),
            "max_batch_size": self.max_batch_size,
            "timeout_ms": self.timeout * 1000,
            "running": self._running,
            "model_device": str(self.model.device) if hasattr(self.model, 'device') else 'unknown'
        }