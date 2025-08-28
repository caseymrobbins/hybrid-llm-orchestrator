# src/orchestrator.py

import asyncio
import sqlite3
import aiohttp
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from .config import ConfigLoader, ModuleConfig
from .utils.security import SecurityManager
from .clients.base import LlmClient
from .clients.openai_compatible import OpenAICompatibleClient
from .utils.local_inference import BatchedLocalInference, OptimizedLocalModelLoader

# External dependencies
from routellm.controller import Controller
from upstash_semantic_cache import SemanticCache
from pybreaker import CircuitBreaker, CircuitBreakerError

logger = logging.getLogger(__name__)

class OrchestrationError(Exception):
    """Custom exception for orchestration errors."""
    pass

class Orchestrator:
    """Manages the execution of the AI workflow with production-grade features."""

    def __init__(self, config_path: str, db_path: str = "users.db"):
        self.db_path = db_path
        self.config_loader = ConfigLoader(Path(config_path))
        self.security_manager = SecurityManager()
        
        # Client registry for different LLM providers
        self.clients: Dict[str, LlmClient] = {}
        self.local_inference: Optional[BatchedLocalInference] = None
        
        # Production features
        self.router: Optional[Controller] = None
        self.semantic_cache: Optional[SemanticCache] = None
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # State management
        self._initialized = False
        self._session: Optional[aiohttp.ClientSession] = None

    async def initialize(self) -> None:
        """Initialize all components asynchronously."""
        if self._initialized:
            return
            
        try:
            logger.info("Initializing Orchestrator...")
            
            # Initialize HTTP session
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
            
            # Initialize components
            await self._initialize_router()
            await self._initialize_cache()
            await self._initialize_circuit_breakers()
            await self._initialize_clients()
            
            self._initialized = True
            logger.info("Orchestrator initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize Orchestrator: {e}")
            await self.cleanup()
            raise OrchestrationError(f"Initialization failed: {e}")

    async def _initialize_router(self) -> None:
        """Initialize the RouteLLM controller."""
        try:
            # RouteLLM initialization - this may be sync, so we run in executor
            self.router = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: Controller(
                    routers=["mf"],  # Matrix Factorization router
                    strong_model="gpt-4o",
                    weak_model="huggingface:google/gemma-2b-it"
                )
            )
            logger.info("RouteLLM controller initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize router: {e}. Routing will be disabled.")
            self.router = None

    async def _initialize_cache(self) -> None:
        """Initialize semantic cache."""
        try:
            self.semantic_cache = SemanticCache(min_proximity=0.90)
            logger.info("Semantic cache initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize cache: {e}. Caching will be disabled.")
            self.semantic_cache = None

    async def _initialize_circuit_breakers(self) -> None:
        """Initialize circuit breakers for external services."""
        # Get unique external endpoints from config
        external_endpoints = self._get_external_endpoints()
        
        for endpoint in external_endpoints:
            self.circuit_breakers[endpoint] = CircuitBreaker(
                fail_max=3,
                reset_timeout=60,
                recovery_timeout=30,
                expected_exception=Exception
            )
        
        logger.info(f"Initialized circuit breakers for {len(self.circuit_breakers)} endpoints")

    def _get_external_endpoints(self) -> List[str]:
        """Extract external endpoints from configuration."""
        endpoints = set()
        
        for module_config in self.config_loader.module_configs.values():
            if hasattr(module_config, 'llm_address'):
                # Extract provider from address (e.g., "openai:gpt-4o" -> "openai")
                address = module_config.llm_address
                if ":" in address and not address.startswith("huggingface"):
                    provider = address.split(":")[0]
                    endpoints.add(provider)
                    
        return list(endpoints)

    async def _initialize_clients(self) -> None:
        """Initialize LLM clients for different providers."""
        for module_name, config in self.config_loader.module_configs.items():
            if hasattr(config, 'llm_address'):
                client = await self._create_client(config)
                if client:
                    self.clients[module_name] = client

    async def _create_client(self, config: ModuleConfig) -> Optional[LlmClient]:
        """Create appropriate client based on configuration."""
        try:
            address = config.llm_address
            
            if address.startswith("openai:"):
                return OpenAICompatibleClient(
                    api_key=config.get("api_key", ""),
                    base_url="https://api.openai.com/v1"
                )
            elif address.startswith("anthropic:"):
                return OpenAICompatibleClient(
                    api_key=config.get("api_key", ""),
                    base_url="https://api.anthropic.com/v1"
                )
            elif address.startswith("huggingface:"):
                # For local models, initialize if not already done
                if not self.local_inference:
                    model_name = address.split(":", 1)[1]
                    model, tokenizer = OptimizedLocalModelLoader.load_model_and_tokenizer(model_name)
                    self.local_inference = BatchedLocalInference(model, tokenizer)
                    await self.local_inference.start()
                return self.local_inference
            else:
                logger.warning(f"Unknown provider in address: {address}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create client for {config.llm_address}: {e}")
            return None

    async def _execute_module(self, module_name: str, context: Dict[str, Any]) -> str:
        """Execute a single module with security, caching, routing, and resilience."""
        config = self.config_loader.module_configs.get(module_name)
        if not config:
            raise OrchestrationError(f"Module '{module_name}' not found in configuration")

        # Format prompt with context
        try:
            prompt = config.instructions.format(**context)
        except KeyError as e:
            raise OrchestrationError(f"Missing context variable for module '{module_name}': {e}")

        # 1. Security: Prompt Injection Check
        try:
            is_injection, confidence = self.security_manager.detect_injection(prompt)
            if is_injection and confidence > 0.8:  # High confidence threshold
                logger.warning(f"Potential prompt injection detected in module '{module_name}' (confidence: {confidence})")
                return "Error: Potentially unsafe input detected."
        except Exception as e:
            logger.warning(f"Security check failed for module '{module_name}': {e}")

        # 2. Caching: Check Semantic Cache
        if self.semantic_cache:
            try:
                cached_result = self.semantic_cache.get(prompt)
                if cached_result:
                    logger.info(f"Cache hit for module '{module_name}'")
                    return cached_result
            except Exception as e:
                logger.warning(f"Cache lookup failed: {e}")

        logger.info(f"Executing module '{module_name}'...")

        # 3. PII Scrubbing
        try:
            scrubbed_prompt = self.security_manager.scrub_pii(prompt)
        except Exception as e:
            logger.warning(f"PII scrubbing failed: {e}")
            scrubbed_prompt = prompt

        # 4. Execution with resilience
        result = await self._execute_with_resilience(module_name, scrubbed_prompt, config)

        # 5. Store in cache
        if self.semantic_cache and result:
            try:
                self.semantic_cache.set(prompt, result)
            except Exception as e:
                logger.warning(f"Cache storage failed: {e}")

        return result

    async def _execute_with_resilience(self, module_name: str, prompt: str, config: ModuleConfig) -> str:
        """Execute module with circuit breaker protection."""
        # Get appropriate client
        client = self.clients.get(module_name)
        if not client:
            return f"Error: No client available for module '{module_name}'"

        # Determine if we need circuit breaker protection
        provider = self._get_provider_from_config(config)
        circuit_breaker = self.circuit_breakers.get(provider) if provider else None

        try:
            if circuit_breaker:
                # Wrap execution in circuit breaker
                result = await self._execute_with_circuit_breaker(
                    circuit_breaker, client, prompt, **config.get_generation_params()
                )
            else:
                # Direct execution for local models
                result = await client.generate(prompt, **config.get_generation_params())

            return result

        except CircuitBreakerError:
            logger.warning(f"Circuit breaker open for module '{module_name}', using fallback")
            return await self._fallback_execution(prompt)
        except Exception as e:
            logger.error(f"Module '{module_name}' execution failed: {e}")
            return f"Error: Module {module_name} failed to produce a result."

    async def _execute_with_circuit_breaker(self, breaker: CircuitBreaker, client: LlmClient, prompt: str, **kwargs) -> str:
        """Execute client call wrapped in circuit breaker."""
        def sync_call():
            # This is a workaround for circuit breakers that expect sync functions
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(client.generate(prompt, **kwargs))
            finally:
                loop.close()

        # Run the circuit-breaker-protected call in executor
        return await asyncio.get_event_loop().run_in_executor(None, breaker(sync_call))

    async def _fallback_execution(self, prompt: str) -> str:
        """Fallback execution when primary service is unavailable."""
        if self.local_inference:
            try:
                return await self.local_inference.generate(prompt)
            except Exception as e:
                logger.error(f"Fallback execution failed: {e}")
        
        return "Error: Service temporarily unavailable. Please try again later."

    def _get_provider_from_config(self, config: ModuleConfig) -> Optional[str]:
        """Extract provider name from module configuration."""
        if hasattr(config, 'llm_address'):
            address = config.llm_address
            if ":" in address and not address.startswith("huggingface"):
                return address.split(":")[0]
        return None

    async def execute_workflow(self, query: str) -> str:
        """Execute the full workflow defined in configuration."""
        if not self._initialized:
            await self.initialize()

        execution_plan = self.config_loader.workflow_config.execution_plan
        context = {"query": query}

        try:
            # Execute modules according to plan
            for step in execution_plan:
                module_name = step["module"]
                dependencies = step.get("dependencies", [])

                # Check dependencies
                missing_deps = [dep for dep in dependencies if dep not in context]
                if missing_deps:
                    raise OrchestrationError(f"Missing dependencies for module {module_name}: {missing_deps}")

                logger.info(f"Running module: {module_name}")
                result = await self._execute_module(module_name, context)
                
                # Store results with multiple keys for flexibility
                context[module_name] = result
                context[f"{module_name}_output"] = result

            # Final synthesis
            logger.info("Synthesizing final response")
            final_response = await self._synthesize_response(query, context)
            
            return final_response

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise OrchestrationError(f"Workflow execution failed: {e}")

    async def _synthesize_response(self, query: str, context: Dict[str, Any]) -> str:
        """Synthesize final response from module outputs."""
        synthesis_prompt = self._build_synthesis_prompt(query, context)
        
        # Use local model for synthesis to reduce costs and latency
        if self.local_inference:
            try:
                return await self.local_inference.generate(synthesis_prompt)
            except Exception as e:
                logger.warning(f"Local synthesis failed: {e}")

        # Fallback to any available client
        for client in self.clients.values():
            try:
                return await client.generate(synthesis_prompt, max_tokens=512)
            except Exception as e:
                logger.warning(f"Synthesis fallback failed: {e}")
                continue

        return "Error: Unable to synthesize final response."

    def _build_synthesis_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """Build the synthesis prompt from context."""
        prompt_parts = [
            f"Synthesize the following information into a coherent response for: \"{query}\"",
            ""
        ]

        # Add module outputs
        for key, value in context.items():
            if key.endswith("_output") and value:
                module_name = key.replace("_output", "")
                prompt_parts.append(f"{module_name} Output:")
                prompt_parts.append(str(value))
                prompt_parts.append("")

        prompt_parts.append("Final Response:")
        return "\n".join(prompt_parts)

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check of all system components."""
        if not self._initialized:
            return {"overall_status": "not_initialized"}

        # Check database
        db_status = await self._check_db_connection()
        
        # Check external APIs
        api_status = await self._check_api_connectivity()
        
        # Check circuit breakers
        breaker_status = {
            name: {
                "state": str(cb.current_state),
                "failure_count": cb.fail_counter,
                "last_failure_time": getattr(cb, 'last_failure_time', None)
            }
            for name, cb in self.circuit_breakers.items()
        }

        # Check clients
        client_status = {}
        for name, client in self.clients.items():
            try:
                if hasattr(client, 'validate_connection'):
                    client_status[name] = "ok" if await client.validate_connection() else "error"
                else:
                    client_status[name] = "unknown"
            except Exception:
                client_status[name] = "error"

        overall_healthy = (
            db_status["status"] == "ok" and
            all(status == "ok" for status in api_status.values()) and
            all(status in ["ok", "unknown"] for status in client_status.values())
        )

        return {
            "overall_status": "healthy" if overall_healthy else "degraded",
            "timestamp": asyncio.get_event_loop().time(),
            "components": {
                "database": db_status,
                "external_apis": api_status,
                "circuit_breakers": breaker_status,
                "clients": client_status,
                "cache": "enabled" if self.semantic_cache else "disabled",
                "router": "enabled" if self.router else "disabled"
            }
        }

    async def _check_db_connection(self) -> Dict[str, Any]:
        """Check database connectivity."""
        try:
            def db_check():
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("SELECT 1").fetchone()
                    
            await asyncio.get_event_loop().run_in_executor(None, db_check)
            return {"status": "ok"}
        except Exception as e:
            return {"status": "error", "details": str(e)}

    async def _check_api_connectivity(self) -> Dict[str, Any]:
        """Check connectivity to external APIs."""
        endpoints = {
            "OpenAI": "https://api.openai.com/v1/models",
            "Anthropic": "https://api.anthropic.com/v1/messages",
            "xAI": "https://api.x.ai/v1/models"
        }
        
        statuses = {}
        
        if not self._session:
            return {name: "session_not_available" for name in endpoints}

        for name, url in endpoints.items():
            try:
                async with self._session.head(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    statuses[name] = "ok" if response.status < 500 else "error"
            except asyncio.TimeoutError:
                statuses[name] = "timeout"
            except Exception:
                statuses[name] = "unreachable"
                
        return statuses

    @asynccontextmanager
    async def managed_execution(self):
        """Context manager for automatic initialization and cleanup."""
        try:
            await self.initialize()
            yield self
        finally:
            await self.cleanup()

    async def cleanup(self) -> None:
        """Clean up all resources."""
        logger.info("Cleaning up Orchestrator resources...")
        
        # Close HTTP session
        if self._session and not self._session.closed:
            await self._session.close()
            
        # Cleanup local inference
        if self.local_inference:
            try:
                await self.local_inference.stop()
            except Exception as e:
                logger.warning(f"Error stopping local inference: {e}")

        # Reset state
        self._initialized = False
        logger.info("Orchestrator cleanup complete")

    def get_stats(self) -> Dict[str, Any]:
        """Get current orchestrator statistics."""
        return {
            "initialized": self._initialized,
            "modules_configured": len(self.config_loader.module_configs) if hasattr(self.config_loader, 'module_configs') else 0,
            "clients_active": len(self.clients),
            "circuit_breakers": len(self.circuit_breakers),
            "cache_enabled": self.semantic_cache is not None,
            "router_enabled": self.router is not None,
            "local_inference": self.local_inference is not None
        }