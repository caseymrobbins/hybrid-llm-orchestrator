# example_usage.py
"""
Complete example showing how to use all the improved components together.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

# Import the improved components
from src import Orchestrator, ConfigLoader
from src.utils import setup_logging, get_logger, Cache
from src.database import DatabaseManager, InteractionLog, ModuleOutput
from src.clients import create_client, list_providers

async def main():
    """Demonstrate the complete system working together."""
    
    # 1. Set up logging
    setup_logging(
        level="INFO",
        log_dir="./logs",
        json_logs=False,  # Use colored console for development
        console_colors=True
    )
    
    logger = get_logger(__name__)
    logger.info("Starting Hybrid LLM Orchestrator Demo")
    
    # 2. Initialize database
    db = DatabaseManager(
        db_path="./data/orchestrator.db",
        max_connections=5
    )
    await db.initialize()
    
    # 3. Set up caching
    cache = Cache(
        cache_dir="./cache",
        max_size=500 * 1024 * 1024,  # 500MB
        default_expire=3600  # 1 hour
    )
    
    # 4. Create a test user
    user_id = await db.create_user(
        profile_data={
            "name": "Test User",
            "role": "developer",
            "interests": ["AI", "Python", "Machine Learning"]
        },
        preferences={
            "preferred_models": ["gpt-4", "claude-3"],
            "max_cost_per_query": 100,  # cents
            "language": "en"
        }
    )
    
    logger.info(f"Created test user: {user_id}")
    
    # 5. Test individual clients
    logger.info("Testing individual LLM clients...")
    
    # List available providers
    providers = list_providers()
    logger.info(f"Available providers: {providers}")
    
    # Test cache
    test_prompt = "What is artificial intelligence?"
    cached_response = await cache.get(test_prompt)
    
    if not cached_response:
        logger.info("Cache miss - this is expected for first run")
        
        # Simulate a response (normally from LLM)
        response = "Artificial Intelligence (AI) refers to computer systems that can perform tasks typically requiring human intelligence..."
        
        # Cache the response
        await cache.set(test_prompt, response)
        logger.info("Response cached successfully")
    else:
        logger.info("Cache hit! Retrieved cached response")
        response = cached_response
    
    # 6. Log interaction to database
    query_hash = db.generate_query_hash(test_prompt)
    
    interaction = InteractionLog(
        user_id=user_id,
        query_hash=query_hash,
        modules_used=["TestModule"],
        total_cost_cents=5,
        total_latency_ms=1200,
        cache_hit_rate=0.0 if not cached_response else 1.0
    )
    
    interaction_id = await db.log_interaction(interaction)
    logger.info(f"Logged interaction: {interaction_id}")
    
    # Log module output
    module_output = ModuleOutput(
        interaction_id=interaction_id,
        module_name="TestModule",
        output_hash=db.generate_query_hash(response),
        latency_ms=1200,
        model_used="gpt-4-simulated"
    )
    
    await db.log_module_output(module_output)
    
    # 7. Record some analytics
    await db.record_metric("query_count", 1, {"user_type": "developer"})
    await db.record_metric("response_time_ms", 1200, {"module": "TestModule"})
    await db.record_metric("cache_hit_rate", 1.0 if cached_response else 0.0)
    
    # 8. Get user interaction history
    interactions = await db.get_user_interactions(user_id, limit=10)
    logger.info(f"User has {len(interactions)} interactions")
    
    # 9. Get system statistics
    cache_stats = cache.get_stats()
    db_stats = await db.get_database_stats()
    
    logger.info("System Statistics:")
    logger.info(f"Cache hit rate: {cache_stats['statistics']['hit_rate_percent']}%")
    logger.info(f"Database size: {db_stats['database_size_mb']:.2f} MB")
    logger.info(f"Total queries executed: {db_stats['query_stats']['queries_executed']}")
    
    # 10. Demonstrate full orchestrator (if config exists)
    config_path = Path("./configs")
    if config_path.exists():
        logger.info("Found configuration directory, testing full orchestrator...")
        
        try:
            async with Orchestrator(str(config_path)).managed_execution() as orchestrator:
                # Perform health check
                health = await orchestrator.health_check()
                logger.info(f"System health: {health['overall_status']}")
                
                # Example query (would normally come from user input)
                if health['overall_status'] == 'healthy':
                    result = await orchestrator.execute_workflow(
                        "What are the key considerations for ethical AI development?"
                    )
                    print(f"\nOrchestrator Response:\n{result}")
                
        except Exception as e:
            logger.warning(f"Full orchestrator test failed: {e}")
            logger.info("This is expected if configuration files are not set up")
    else:
        logger.info("No configuration directory found, skipping full orchestrator test")
    
    # 11. Cleanup
    await cache.close() if hasattr(cache, 'close') else None
    await db.close()
    
    logger.info("Demo completed successfully!")

async def configuration_example():
    """Show how to set up configuration files."""
    
    # Create example configuration structure
    config_dir = Path("./configs")
    config_dir.mkdir(exist_ok=True)
    
    modules_dir = config_dir / "modules"
    modules_dir.mkdir(exist_ok=True)
    
    # Example workflow configuration
    workflow_config = {
        "name": "AI Ethics Workflow",
        "description": "Comprehensive AI ethics evaluation pipeline",
        "version": "1.0.0",
        "execution_plan": [
            {
                "module": "Analysis",
                "dependencies": [],
                "parallel": False
            },
            {
                "module": "Ethics", 
                "dependencies": ["Analysis"],
                "parallel": False
            }
        ]
    }
    
    with open(config_dir / "workflow.yaml", "w") as f:
        import yaml
        yaml.dump(workflow_config, f, default_flow_style=False)
    
    # Example module configurations
    analysis_config = {
        "aspect": "Analysis",
        "instructions": "Analyze the following query for key themes and implications: {query}",
        "llm_address": "openai:gpt-3.5-turbo",
        "api_key": "${OPENAI_API_KEY}",
        "temperature": 0.3,
        "max_tokens": 512
    }
    
    ethics_config = {
        "aspect": "Ethics", 
        "instructions": "Evaluate the ethical implications of: {query}. Consider the analysis: {Analysis_output}",
        "llm_address": "anthropic:claude-3-sonnet-20240229",
        "api_key": "${ANTHROPIC_API_KEY}",
        "temperature": 0.2,
        "max_tokens": 1024
    }
    
    with open(modules_dir / "analysis.yaml", "w") as f:
        yaml.dump(analysis_config, f, default_flow_style=False)
    
    with open(modules_dir / "ethics.yaml", "w") as f:
        yaml.dump(ethics_config, f, default_flow_style=False)
    
    print("Example configuration files created in ./configs/")
    print("Set OPENAI_API_KEY and ANTHROPIC_API_KEY environment variables to test")

def environment_setup():
    """Show how to set up the environment."""
    
    env_example = """
# .env file example
OPENAI_API_KEY=sk-your-openai-api-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# Optional: Local model settings
HF_TOKEN=your-huggingface-token
CUDA_VISIBLE_DEVICES=0

# Database settings
DATABASE_PATH=./data/orchestrator.db
CACHE_DIR=./cache
LOG_DIR=./logs

# Performance settings  
MAX_DB_CONNECTIONS=10
CACHE_MAX_SIZE_MB=500
LOG_LEVEL=INFO
    """
    
    with open(".env.example", "w") as f:
        f.write(env_example)
    
    print("Created .env.example file with configuration template")

if __name__ == "__main__":
    print("Hybrid LLM Orchestrator - Complete System Demo")
    print("=" * 50)
    
    # Set up example configurations
    configuration_example()
    environment_setup()
    
    # Run the main demo
    asyncio.run(main())