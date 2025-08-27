# src/orchestrator.py

import asyncio
import sqlite3
import requests
from pathlib import Path
from typing import Dict, Any

from.config import ConfigLoader, ModuleConfig
from.utils.security import SecurityManager

from routellm.controller import Controller
from upstash_semantic_cache import SemanticCache
from pybreaker import CircuitBreaker, CircuitBreakerError

class Orchestrator:
    """Manages the execution of the AI workflow with production-grade features."""

    def __init__(self, config_path: str):
        self.db_path = "users.db" # Example DB path
        self.config_loader = ConfigLoader(Path(config_path))
        self.security_manager = SecurityManager()

        # 1. Initialize Intelligent Router (RouteLLM)
        # Defines a powerful external model and a fast local model for routing decisions.
        self.router = Controller(
            routers=["mf"],  # Matrix Factorization is a great general-purpose router [4]
            strong_model="gpt-4o",
            weak_model="huggingface:google/gemma-2b-it"
        )
        
        # 2. Initialize Semantic Cache
        # Connects to a vector DB (configured via env vars) to store results based on meaning.
        self.semantic_cache = SemanticCache(min_proximity=0.90) # [3]

        # 3. Initialize Circuit Breakers for external services
        # Creates a breaker for each unique external LLM address in the configs.
        self.circuit_breakers = self._initialize_circuit_breakers()

    def _initialize_circuit_breakers(self) -> Dict:
        """Creates a CircuitBreaker for each unique external service."""
        breakers = {}
        # This is a simplified example; in a real system, you'd scan configs
        # for all potential external endpoints.
        external_endpoints = ["gpt-4o", "claude-3-sonnet-20240229"] # Example endpoints
        for endpoint in external_endpoints:
            breakers[endpoint] = CircuitBreaker(
                fail_max=3, 
                reset_timeout=60
            ) # [6]
        return breakers

    async def _execute_module(self, module_name: str, context: Dict[str, Any]) -> str:
        """Executes a single module with security, caching, routing, and resilience."""
        config = self.config_loader.module_configs[module_name]
        prompt = config.instructions.format(**context)

        # 1. Security: Prompt Injection Check
        is_injection, _ = self.security_manager.detect_injection(prompt)
        if is_injection:
            print(f"SECURITY ALERT: Potential prompt injection detected in module '{module_name}'. Aborting.")
            return "Error: Malicious input detected."

        # 2. Caching: Check Semantic Cache
        cached_result = self.semantic_cache.get(prompt)
        if cached_result:
            print(f"SEMANTIC CACHE HIT: Module '{module_name}'")
            return cached_result
        print(f"CACHE MISS: Executing module '{module_name}'...")

        # 3. Routing and Execution with Resilience
        try:
            # The router returns the name of the model it chose (e.g., "gpt-4o")
            # We use this to select the correct circuit breaker.
            # Note: RouteLLM's create() is a synchronous call. For a fully async system,
            # you would run this in an executor.
            
            # The model name format "router-{type}-{threshold}" tells RouteLLM how to route.[5]
            calibrated_model_name = "router-mf-0.11593"

            # PII Scrubbing before sending to router (which may call external API)
            scrubbed_prompt = self.security_manager.scrub_pii(prompt)

            # Get the appropriate circuit breaker for the potential strong model
            # This is a simplification; a more robust implementation would get the
            # model name from the router *before* wrapping in the breaker.
            breaker = self.circuit_breakers.get("gpt-4o")

            if breaker:
                @breaker
                def run_routed_completion():
                    completion = self.router.chat.completions.create(
                        model=calibrated_model_name,
                        messages=[{"role": "user", "content": scrubbed_prompt}]
                    )
                    return completion.choices.message.content
                
                result = run_routed_completion()
            else: # Fallback for local-only or non-breakered models
                completion = self.router.chat.completions.create(
                    model=calibrated_model_name,
                    messages=[{"role": "user", "content": scrubbed_prompt}]
                )
                result = completion.choices.message.content

        except CircuitBreakerError:
            print(f"RESILIENCE: Circuit open for module '{module_name}'. Using fallback.")
            # Implement fallback logic here (e.g., call a local model directly)
            return "Error: Service is temporarily unavailable. Please try again later."
        except Exception as e:
            print(f"ERROR: Module '{module_name}' failed: {e}")
            return f"Error: Module {module_name} failed to produce a result."

        # 4. Store in cache and return
        self.semantic_cache.set(prompt, result)
        return result
    
    async def execute_workflow(self, query: str) -> str:
        """Executes the full workflow defined in workflow.yaml."""
        execution_plan = self.config_loader.workflow_config.execution_plan
        context = {"query": query}

        for step in execution_plan:
            module_name = step["module"]
            dependencies = step.get("dependencies",)

            # This is a simple sequential execution. A more complex system
            # could use asyncio.gather for parallel execution of independent steps.
            if not all(dep in context for dep in dependencies):
                raise ValueError(f"Missing dependencies for module {module_name}: {dependencies}")

            print(f"\n--- Running Module: {module_name} ---")
            result = await self._execute_module(module_name, context)
            context[module_name] = result
            context[f"{module_name}_output"] = result # For clearer reference

        # Final synthesis step (always local)
        print("\n--- Synthesizing Final Response ---")
        synthesis_prompt = f"""
        Synthesize the following information into a coherent final response for the user query: "{query}"
        
        Curiosity Module Output:
        {context.get('Curiosity_output', 'N/A')}

        Opposing Opinion Module Output:
        {context.get('OpposingOpinion_output', 'N/A')}

        Ethics Module Decision:
        {context.get('Ethics_output', 'N/A')}

        Final Answer:
        """
        # Using a small, fast local model for synthesis is a good practice
        synthesis_config = ModuleConfig(
            aspect="Synthesis",
            llm_address="huggingface:google/gemma-2b", # Example small local model
            instructions=synthesis_prompt
        )
        synthesis_client = self._get_client(synthesis_config)
        final_response = await synthesis_client.generate(synthesis_prompt)

        return final_response
     async def health_check(self) -> dict:
        """Performs a comprehensive health check of all critical system components."""
        db_status = self._check_db_connection()
        api_status = await self._check_api_connectivity()
        
        return {
            "overall_status": "ok" if db_status["status"] == "ok" and all(s == "ok" for s in api_status.values()) else "degraded",
            "database": db_status,
            "external_apis": api_status,
            "circuit_breakers": {name: cb.current_state for name, cb in self.circuit_breakers.items()}
        }

    def _check_db_connection(self) -> dict:
        """Verifies the connection to the SQLite database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("SELECT 1")
            return {"status": "ok"}
        except Exception as e:
            return {"status": "error", "details": str(e)}

    async def _check_api_connectivity(self) -> dict:
        """Pings the base URLs of critical external APIs."""
        # In a real system, these URLs would be dynamically loaded from configs
        endpoints = {
            "OpenAI": "https://api.openai.com/v1",
            "Anthropic": "https://api.anthropic.com/v1",
            "xAI": "https://api.x.ai/v1"
        }
        statuses = {}
        for name, url in endpoints.items():
            try:
                # A simple HEAD request is a lightweight way to check connectivity [11]
                response = await asyncio.to_thread(requests.head, url, timeout=5)
                statuses[name] = "ok" if response.status_code < 500 else "error"
            except requests.RequestException:
                statuses[name] = "unreachable"
        return statuses
