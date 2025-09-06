# src/orchestrator.py

import asyncio
import sqlite3
import aiohttp
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
import time

from .config import ConfigLoader, ModuleConfig
from .utils.security import SecurityManager
from .clients.base import LlmClient
from .clients.openai_compatible import OpenAICompatibleClient
from .clients.anthropic import AnthropicClient
from .clients.local import LocalClient
from .utils.caching import Cache, SemanticCache

# External dependencies (with fallback if not available)
try:
    from routellm.controller import Controller
    HAS_ROUTELLM = True
except ImportError:
    HAS_ROUTELLM = False
    Controller = None

try:
    from pybreaker import CircuitBreaker, CircuitBreakerError
    HAS_CIRCUIT_BREAKER = True
except ImportError:
    HAS_CIRCUIT_BREAKER = False
    CircuitBreaker = None
    CircuitBreakerError = Exception

logger = logging.getLogger(__name__)

class OrchestrationError(Exception):
    """Custom exception for orchestration errors."""
    pass

class SimpleRouter:
    """Simple router fallback when RouteLLM is not available."""
    def __init__(self, strong_model: str, weak_model: str):
        self.strong_model = strong_model
        self.weak_model = weak_model
    
    def route(self, query: str) -> str:
        """Simple routing based on query length."""
        if len(query) > 100:
            return self.strong_model
        return self.weak_model

class SimpleCircuitBreaker:
    """Simple circuit breaker fallback."""
    def __init__(self, fail_max=3, reset_timeout=60):
        self.fail_max = fail_max
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            current_time = time.time()
            
            # Check if we should reset
            if (self.state == "open" and 
                current_time - self.last_failure_time > self.reset_timeout):
                self.state = "half-open"
                self.failure_count = 0
            
            # If circuit is open, fail fast
            if self.state == "open":
                raise Exception("Circuit breaker is open")
            
            try:
                result = func(*args, **kwargs)
                # Success - close circuit if it was half-open
                if self.state == "half-open":
                    self.state = "closed"
                    self.failure_count = 0
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = current_time
                
                if self.failure_count >= self.fail_max:
                    self.state = "open"
                
                raise e
        
        return wrapper

class Orchestrator:
    """Manages the execution of the AI workflow with production-grade features."""

    def __init__(self, config_path: str, db_path: str = "users.db"):
        self.db_path = db_path
        self.config_loader = ConfigLoader(Path(config_path))
        self.security_manager = SecurityManager()
        
        # Client registry for different LLM providers
        self.clients: Dict[str, LlmClient] = {}
        
        # Production features
        self.router: Optional[Any] = None
        self.semantic_cache: Optional[Cache] = None
        self.circuit_breakers: Dict[str, Any] = {}
        
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
        """Initialize the RouteLLM controller or fallback."""
        try:
            if HAS_ROUTELLM and Controller:
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
            else:
                # Fallback simple router
                self.router = SimpleRouter("gpt-4o", "huggingface:google/gemma-2b-it")
                logger.info("Simple router fallback initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize router: {e}. Routing will be disabled.")
            self.router = None

    async def _initialize_cache(self) -> None:
        """Initialize semantic cache."""
        try:
            self.semantic_cache = SemanticCache(
                cache_dir=".cache/semantic",
                similarity_threshold=0.90
            )
            logger.info("Semantic cache initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize cache: {e}. Caching will be disabled.")
            self.semantic_cache = None

    async def _initialize_circuit_breakers(self) -> None:
        """Initialize circuit breakers for external services."""
        # Get unique external endpoints from config
        external_endpoints = self._get_external_endpoints()
        
        for endpoint in external_endpoints:
            if HAS_CIRCUIT_BREAKER and CircuitBreaker:
                self.circuit_breakers[endpoint] = CircuitBreaker(
                    fail_max=3,
                    reset_timeout=60,
                    recovery_timeout=30,
                    expected_exception=Exception
                )
            else:
                self.circuit_breakers[endpoint] = SimpleCircuitBreaker()
        
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
                    if provider not in ["local", "huggingface"]:
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
            api_key = getattr(config, 'api_key', 'local')
            
            if address.startswith("https://api.openai.com"):
                return OpenAICompatibleClient(
                    api_key=api_key,
                    base_url="https://api.openai.com/v1"
                )
            elif address.startswith("https://api.anthropic.com"):
                return AnthropicClient(api_key=api_key)
            elif address.startswith("https://api.x.ai"):
                return OpenAICompatibleClient(
                    api_key=api_key,
                    base_url="https://api.x.ai/v1"
                )
            elif "openai:" in address:
                return OpenAICompatibleClient(
                    api_key=api_key,
                    base_url="https://api.openai.com/v1"
                )
            elif "anthropic:" in address:
                return AnthropicClient(api_key=api_key)
            elif address.startswith("huggingface:") or "huggingface" in address:
                # For local models
                model_name = address.split(":", 1)[1] if ":" in address else "google/gemma-2b-it"
                return LocalClient(model_name)
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
            missing_key = str(e).strip("'\"")
            logger.warning(f"Missing context variable '{missing_key}' for module '{module_name}', using empty string")
            # Create a copy of context with missing keys as empty strings
            safe_context = context.copy()
            safe_context[missing_key] = ""
            prompt = config.instructions.format(**safe_context)
        except Exception as e:
            raise OrchestrationError(f"Failed to format prompt for module '{module_name}': {e}")

        # 1. Security: Prompt Injection Check
        try:
            is_injection, confidence = self.security_manager.detect_injection(prompt)
            if is_injection and confidence > 0.8:  # High confidence threshold
                logger.warning(f"Potential prompt injection detected in module '{module_name}' (confidence: {confidence})")
                return "Error: Potentially unsafe input detected."
        except Exception as e:
            logger.warning(f"Security check failed for module '{module_name}': {e}")

        # 2. Caching: Check Semantic Cache
        cache_key = f"{module_name}:{prompt[:100]}"  # Truncate for cache key
        if self.semantic_cache:
            try:
                cached_result = await self.semantic_cache.get(cache_key, prefix=module_name)
                if cached_result:
                    logger.info(f"Cache hit for module '{module_name}'")
                    return f"[CACHED] {cached_result}"
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
        if self.semantic_cache and result and not result.startswith("Error:"):
            try:
                await self.semantic_cache.set(cache_key, result, prefix=module_name)
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
            if circuit_breaker and provider not in ["local", "huggingface"]:
                # Wrap execution in circuit breaker for external services
                result = await self._execute_with_circuit_breaker(
                    circuit_breaker, client, prompt, **config.get_generation_params()
                )
            else:
                # Direct execution for local models or when no circuit breaker
                result = await client.generate(prompt, **config.get_generation_params())

            return result

        except (CircuitBreakerError, Exception) as e:
            if "circuit breaker" in str(e).lower():
                logger.warning(f"Circuit breaker open for module '{module_name}', using fallback")
                return await self._fallback_execution(prompt, config)
            else:
                logger.error(f"Module '{module_name}' execution failed: {e}")
                return await self._fallback_execution(prompt, config)

    async def _execute_with_circuit_breaker(self, breaker: Any, client: LlmClient, prompt: str, **kwargs) -> str:
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

    async def _fallback_execution(self, prompt: str, config: ModuleConfig) -> str:
        """Fallback execution when primary service is unavailable."""
        # Try to find a local client as fallback
        for client in self.clients.values():
            if isinstance(client, LocalClient):
                try:
                    logger.info("Using local client as fallback")
                    return await client.generate(prompt, max_tokens=config.get("max_tokens", 512))
                except Exception as e:
                    logger.error(f"Fallback execution failed: {e}")
                    continue
        
        return "Error: Service temporarily unavailable. Please try again later."

    def _get_provider_from_config(self, config: ModuleConfig) -> Optional[str]:
        """Extract provider name from module configuration."""
        if hasattr(config, 'llm_address'):
            address = config.llm_address
            if ":" in address:
                provider = address.split(":")[0]
                if "http" in provider:
                    # Extract from URL
                    if "api.openai.com" in address:
                        return "openai"
                    elif "api.anthropic.com" in address:
                        return "anthropic"
                    elif "api.x.ai" in address:
                        return "xai"
                return provider
        return None

    async def execute_workflow(self, query: str, user_id: int = 1) -> str:
        """Execute the full workflow defined in configuration."""
        if not self._initialized:
            await self.initialize()

        execution_plan = self.config_loader.workflow_config.execution_plan
        context = {"query": query, "user_id": user_id}

        try:
            # Execute modules according to plan
            for step in execution_plan:
                module_name = step.module
                dependencies = step.dependencies

                # Check dependencies
                missing_deps = [dep for dep in dependencies if dep not in context]
                if missing_deps:
                    logger.warning(f"Missing dependencies for module {module_name}: {missing_deps}")
                    # Continue with available context

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
        local_client = None
        for client in self.clients.values():
            if isinstance(client, LocalClient):
                local_client = client
                break
        
        if local_client:
            try:
                return await local_client.generate(synthesis_prompt, max_tokens=512)
            except Exception as e:
                logger.warning(f"Local synthesis failed: {e}")

        # Fallback to any available client
        for client in self.clients.values():
            try:
                return await client.generate(synthesis_prompt, max_tokens=512, temperature=0.3)
            except Exception as e:
                logger.warning(f"Synthesis fallback failed: {e}")
                continue

        # Ultimate fallback - simple concatenation
        return self._simple_synthesis(query, context)

    def _simple_synthesis(self, query: str, context: Dict[str, Any]) -> str:
        """Simple synthesis fallback when all else fails."""
        outputs = []
        for key, value in context.items():
            if key.endswith("_output") and isinstance(value, str) and value.strip():
                module_name = key.replace("_output", "")
                clean_value = value.replace("[CACHED]", "").strip()
                if clean_value and not clean_value.startswith("Error:"):
                    outputs.append(f"**{module_name}**: {clean_value}")
        
        if not outputs:
            return f"I apologize, but I was unable to process your query: {query}"
        
        return f"Based on your query: '{query}'\n\n" + "\n\n".join(outputs)

    def _build_synthesis_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """Build the synthesis prompt from context."""
        prompt_parts = [
            f"Please provide a comprehensive, well-structured response to the following query: \"{query}\"",
            "",
            "Use the following analysis to inform your response:",
            ""
        ]

        # Add module outputs
        for key, value in context.items():
            if key.endswith("_output") and isinstance(value, str) and value.strip():
                module_name = key.replace("_output", "")
                clean_value = value.replace("[CACHED]", "").strip()
                if clean_value and not clean_value.startswith("Error:"):
                    prompt_parts.append(f"## {module_name}:")
                    prompt_parts.append(clean_value)
                    prompt_parts.append("")

        prompt_parts.extend([
            "## Instructions:",
            "- Synthesize the above information into a coherent, helpful response",
            "- Address the user's query directly and comprehensively", 
            "- Include relevant details from each analysis where appropriate",
            "- Use a natural, conversational tone",
            "- Ensure the response is well-structured and easy to follow",
            "",
            "## Response:"
        ])
        
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
        breaker_status = {}
        for name, cb in self.circuit_breakers.items():
            if hasattr(cb, 'current_state'):
                breaker_status[name] = {
                    "state": str(cb.current_state),
                    "failure_count": getattr(cb, 'fail_counter', 0),
                    "last_failure_time": getattr(cb, 'last_failure_time', None)
                }
            else:
                # Simple circuit breaker
                breaker_status[name] = {
                    "state": cb.state,
                    "failure_count": cb.failure_count,
                    "last_failure_time": cb.last_failure_time
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
            "timestamp": time.time(),
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
        for client in self.clients.values():
            if hasattr(client, 'cleanup'):
                try:
                    await client.cleanup()
                except Exception as e:
                    logger.warning(f"Error cleaning up client: {e}")

        # Close cache
        if self.semantic_cache and hasattr(self.semantic_cache, 'close'):
            try:
                self.semantic_cache.close()
            except Exception as e:
                logger.warning(f"Error closing cache: {e}")

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
            "has_routellm": HAS_ROUTELLM,
            "has_circuit_breaker": HAS_CIRCUIT_BREAKER
        }