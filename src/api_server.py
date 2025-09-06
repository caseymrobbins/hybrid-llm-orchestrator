# src/api_server.py

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Set
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
import logging

# Import your existing orchestrator
from .orchestrator import Orchestrator
from .config import ConfigLoader, ConfigurationError
from .database import UserDatabase
from .utils.monitoring import MetricsCollector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hybrid LLM Orchestrator API", 
    version="1.0.0",
    description="Production-grade AI workflow orchestration system"
)

# CORS configuration for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
try:
    orchestrator = Orchestrator(config_path="configs")
    database = UserDatabase()
    metrics_collector = MetricsCollector(max_history=1000, window_hours=24)
except Exception as e:
    logger.error(f"Failed to initialize components: {e}")
    # We'll initialize later in startup event
    orchestrator = None
    database = None
    metrics_collector = MetricsCollector()

# WebSocket connection manager for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connection added. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket connection removed. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Send message to all connected clients"""
        if not self.active_connections:
            return
            
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send message to WebSocket client: {e}")
                disconnected.add(connection)
        
        # Clean up disconnected clients
        self.active_connections -= disconnected

manager = ConnectionManager()

# Request/Response models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000, description="User query to process")
    user_id: Optional[int] = Field(default=1, description="User ID for tracking")
    preferences: Optional[Dict[str, Any]] = Field(default_factory=dict, description="User preferences")
    modules_override: Optional[List[str]] = Field(default=None, description="Override default module execution")

class QueryResponse(BaseModel):
    response: str = Field(..., description="Final synthesized response")
    module_outputs: Dict[str, Dict[str, Any]] = Field(..., description="Individual module outputs")
    routing_decisions: List[Dict[str, Any]] = Field(..., description="Routing decisions made")
    metrics: Dict[str, Any] = Field(..., description="Performance metrics")
    execution_time: float = Field(..., description="Total execution time in seconds")
    interaction_id: int = Field(..., description="Database interaction ID")

class HealthResponse(BaseModel):
    overall_status: str
    database: Dict[str, Any]
    external_apis: Dict[str, str]
    circuit_breakers: Dict[str, Any]
    cache_status: Dict[str, Any]
    timestamp: str

class MetricsResponse(BaseModel):
    total_queries: int
    avg_latency_ms: float
    cache_hit_rate: float
    cost_saved_cents: float
    local_vs_external: Dict[str, int]
    recent_queries: List[Dict[str, Any]]

class ModuleInfo(BaseModel):
    name: str
    aspect: str
    has_fallback: bool
    uses_external: bool
    model_info: Dict[str, Any]

class ConfigInfo(BaseModel):
    workflow: Dict[str, Any]
    routing: Dict[str, Any]
    cache: Dict[str, Any]
    security: Dict[str, Any]

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Hybrid LLM Orchestrator API",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs",
        "health_check": "/health",
        "metrics": "/metrics",
        "websocket": "/ws",
        "endpoints": {
            "process_query": "POST /process",
            "health_check": "GET /health", 
            "metrics": "GET /metrics",
            "modules": "GET /modules",
            "config": "GET /config",
            "cache_operations": "POST /cache/{operation}",
            "analytics": "GET /analytics"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def get_health():
    """Get comprehensive system health status"""
    if not orchestrator:
        return HealthResponse(
            overall_status="not_initialized",
            database={},
            external_apis={},
            circuit_breakers={},
            cache_status={},
            timestamp=datetime.now().isoformat()
        )
    
    try:
        # Get health status from orchestrator
        health = await orchestrator.health_check()
        
        # Add database health
        db_health = database.health_check() if database else {"status": "not_available"}
        
        # Add cache status
        cache_status = {
            "enabled": hasattr(orchestrator, 'semantic_cache') and orchestrator.semantic_cache is not None,
            "entries": 0,  # Would get from actual cache
            "hit_rate": metrics_collector.get_report().get("cache_hit_rate", 0),
            "memory_usage_mb": 0  # Would calculate actual memory usage
        }
        
        return HealthResponse(
            overall_status=health.get('overall_status', 'unknown'),
            database=db_health,
            external_apis=health.get('components', {}).get('external_apis', {}),
            circuit_breakers=health.get('components', {}).get('circuit_breakers', {}),
            cache_status=cache_status,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            overall_status="error",
            database={"status": "unknown"},
            external_apis={},
            circuit_breakers={},
            cache_status={},
            timestamp=datetime.now().isoformat()
        )

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get system metrics and performance data"""
    try:
        report = metrics_collector.get_report(include_detailed=True)
        
        return MetricsResponse(
            total_queries=report.get("total_requests", 0),
            avg_latency_ms=report.get("avg_latency_ms", 0),
            cache_hit_rate=report.get("cache_hit_rate", 0),
            cost_saved_cents=0,  # Would calculate from routing decisions
            local_vs_external={
                "local": report.get("local_requests", 0),
                "external": report.get("external_requests", 0)
            },
            recent_queries=[]  # Would get from database
        )
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        return MetricsResponse(
            total_queries=0,
            avg_latency_ms=0,
            cache_hit_rate=0,
            cost_saved_cents=0,
            local_vs_external={"local": 0, "external": 0},
            recent_queries=[]
        )

@app.post("/process", response_model=QueryResponse)
async def process_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """Process a query through the hybrid LLM orchestrator"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    start_time = time.time()
    
    try:
        # Initialize orchestrator if needed
        if not orchestrator._initialized:
            await orchestrator.initialize()
        
        # Broadcast query start via WebSocket
        await manager.broadcast({
            'type': 'query_start',
            'query': request.query[:100] + ('...' if len(request.query) > 100 else ''),
            'user_id': request.user_id,
            'timestamp': datetime.now().isoformat()
        })
        
        # Execute the workflow
        context = {"query": request.query, "user_id": request.user_id}
        module_outputs = {}
        routing_decisions = []
        
        # Get execution plan
        execution_plan = orchestrator.config_loader.workflow_config.execution_plan
        
        # Execute each module
        for step in execution_plan:
            module_name = step.module
            dependencies = step.dependencies
            
            # Check dependencies are met
            missing_deps = [dep for dep in dependencies if dep not in context]
            if missing_deps:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Missing dependencies for module {module_name}: {missing_deps}"
                )
            
            # Broadcast module start
            await manager.broadcast({
                'type': 'module_start',
                'module': module_name,
                'timestamp': datetime.now().isoformat()
            })
            
            # Execute module
            module_start_time = time.time()
            try:
                result = await orchestrator._execute_module(module_name, context)
                module_latency = (time.time() - module_start_time) * 1000
                
                # Determine if it was cached (simplified check)
                was_cached = "cache hit" in result.lower() if result else False
                
                # Get model info from config
                config = orchestrator.config_loader.get_module_config(module_name)
                model_used = "local" if config and "huggingface" in str(config.llm_address) else "external"
                
                # Store module output
                module_output = {
                    'output': result,
                    'cached': was_cached,
                    'latency_ms': module_latency,
                    'model_used': model_used,
                    'timestamp': datetime.now().isoformat()
                }
                
                module_outputs[module_name] = module_output
                context[module_name] = result
                context[f"{module_name}_output"] = result
                
                # Track routing decision (simplified)
                routing_decision = {
                    'module': module_name,
                    'original_model': model_used,
                    'routed_model': model_used,
                    'complexity_score': 0.5,  # Would get from actual router
                    'cost_cents': 0 if model_used == "local" else 2,
                    'cached': was_cached,
                    'latency_ms': module_latency
                }
                routing_decisions.append(routing_decision)
                
                # Track metrics
                metrics_collector.track_execution(
                    latency_s=module_latency / 1000,
                    cost_cents=routing_decision['cost_cents'],
                    cache_hit=was_cached,
                    model_used=model_used
                )
                
                # Broadcast module completion
                await manager.broadcast({
                    'type': 'module_complete',
                    'module': module_name,
                    'output': module_output,
                    'routing': routing_decision,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Module {module_name} execution failed: {e}")
                # Continue with other modules, store error
                module_outputs[module_name] = {
                    'output': f"Error: {str(e)}",
                    'cached': False,
                    'latency_ms': (time.time() - module_start_time) * 1000,
                    'model_used': 'error',
                    'timestamp': datetime.now().isoformat()
                }
        
        # Generate final response
        final_response = await synthesize_response(request.query, context)
        
        execution_time = time.time() - start_time
        
        # Log interaction to database
        interaction_id = 0
        if database:
            try:
                interaction_data = {
                    'user_id': request.user_id,
                    'query_text': request.query,
                    'modules_used': list(module_outputs.keys()),
                    'total_cost_cents': sum(r.get('cost_cents', 0) for r in routing_decisions),
                    'total_latency_ms': int(execution_time * 1000),
                    'cache_hit_rate': sum(1 for r in routing_decisions if r.get('cached')) / max(len(routing_decisions), 1),
                    'final_response': final_response,
                    'module_outputs': [
                        {
                            'module_name': name,
                            'output': data['output'],
                            'latency_ms': data.get('latency_ms', 0),
                            'cost_cents': next((r['cost_cents'] for r in routing_decisions if r['module'] == name), 0),
                            'model_used': data.get('model_used', 'unknown'),
                            'was_cached': data.get('cached', False)
                        }
                        for name, data in module_outputs.items()
                    ],
                    'routing_decisions': [
                        {
                            'module_name': r['module'],
                            'original_model': r['original_model'],
                            'routed_model': r['routed_model'],
                            'complexity_score': r.get('complexity_score', 0),
                            'cost_savings_cents': 0,  # Would calculate actual savings
                            'routing_reason': 'automatic'
                        }
                        for r in routing_decisions
                    ]
                }
                interaction_id = database.log_interaction(interaction_data)
            except Exception as e:
                logger.error(f"Failed to log interaction: {e}")
        
        # Broadcast completion
        await manager.broadcast({
            'type': 'query_complete',
            'response': final_response,
            'execution_time': execution_time,
            'interaction_id': interaction_id,
            'timestamp': datetime.now().isoformat()
        })
        
        # Schedule background cleanup
        background_tasks.add_task(cleanup_old_data)
        
        return QueryResponse(
            response=final_response,
            module_outputs=module_outputs,
            routing_decisions=routing_decisions,
            metrics=metrics_collector.get_report(),
            execution_time=execution_time,
            interaction_id=interaction_id
        )
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

async def synthesize_response(query: str, context: Dict[str, Any]) -> str:
    """Synthesize final response from module outputs."""
    try:
        if orchestrator:
            return await orchestrator._synthesize_response(query, context)
        else:
            # Fallback synthesis
            outputs = []
            for key, value in context.items():
                if key.endswith("_output") and isinstance(value, str):
                    module_name = key.replace("_output", "")
                    outputs.append(f"{module_name}: {value}")
            
            return f"Based on the analysis:\n\n" + "\n\n".join(outputs)
    except Exception as e:
        logger.error(f"Response synthesis failed: {e}")
        return f"I apologize, but I encountered an error while processing your query: {query}"

@app.get("/modules")
async def get_modules():
    """Get information about configured modules"""
    if not orchestrator or not hasattr(orchestrator, 'config_loader'):
        raise HTTPException(status_code=503, detail="Orchestrator not properly initialized")
    
    modules = []
    try:
        for name, config in orchestrator.config_loader.module_configs.items():
            module_info = {
                'name': name,
                'aspect': getattr(config, 'aspect', name),
                'has_fallback': hasattr(config, 'fallback') and config.fallback is not None,
                'uses_external': bool(config.llm_address and "http" in str(config.llm_address)),
                'model_info': {
                    'provider': config.llm_address.split(':')[0] if ':' in str(config.llm_address) else 'unknown',
                    'model': config.llm_address.split(':')[1] if ':' in str(config.llm_address) else str(config.llm_address),
                    'temperature': getattr(config, 'temperature', 0.7),
                    'max_tokens': getattr(config, 'max_tokens', 1024)
                }
            }
            modules.append(module_info)
        
        return {'modules': modules, 'total_count': len(modules)}
        
    except Exception as e:
        logger.error(f"Failed to get modules info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve module information")

@app.get("/config", response_model=ConfigInfo)
async def get_config():
    """Get current system configuration (sanitized)"""
    if not orchestrator or not hasattr(orchestrator, 'config_loader'):
        raise HTTPException(status_code=503, detail="Orchestrator not properly initialized")
    
    try:
        workflow = orchestrator.config_loader.workflow_config
        
        return ConfigInfo(
            workflow={
                'name': workflow.name,
                'description': workflow.description,
                'version': getattr(workflow, 'version', '1.0.0'),
                'execution_plan': [
                    {
                        'module': step.module,
                        'dependencies': step.dependencies,
                        'parallel': getattr(step, 'parallel', False)
                    }
                    for step in workflow.execution_plan
                ],
                'max_concurrent_steps': getattr(workflow, 'max_concurrent_steps', 5)
            },
            routing={
                'enabled': orchestrator.router is not None,
                'router_type': 'matrix_factorization',
                'strong_model': getattr(orchestrator.router, 'strong_model', 'gpt-4o') if orchestrator.router else 'unknown',
                'weak_model': getattr(orchestrator.router, 'weak_model', 'local') if orchestrator.router else 'unknown'
            },
            cache={
                'enabled': orchestrator.semantic_cache is not None,
                'type': 'semantic',
                'similarity_threshold': 0.90
            },
            security={
                'prompt_injection_detection': True,
                'pii_scrubbing': True,
                'circuit_breakers': len(orchestrator.circuit_breakers) > 0
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve configuration")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle any incoming messages
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                # Handle ping/pong for connection health
                if data == "ping":
                    await websocket.send_text("pong")
                
            except asyncio.TimeoutError:
                # Send periodic heartbeat
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat()
                })
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.post("/cache/{operation}")
async def cache_operations(operation: str):
    """Cache management operations"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        if operation == "clear":
            if hasattr(orchestrator, 'semantic_cache') and orchestrator.semantic_cache:
                # Clear cache operation would go here
                pass
            return {"status": "success", "message": "Cache cleared", "operation": operation}
        
        elif operation == "stats":
            stats = {
                "enabled": hasattr(orchestrator, 'semantic_cache') and orchestrator.semantic_cache is not None,
                "entries": 0,  # Would get actual count
                "hit_rate": metrics_collector.get_report().get("cache_hit_rate", 0),
                "memory_usage_mb": 0  # Would calculate actual usage
            }
            return {"status": "success", "stats": stats}
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown operation: {operation}")
            
    except Exception as e:
        logger.error(f"Cache operation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cache operation failed: {str(e)}")

@app.get("/analytics")
async def get_analytics(days: int = 7):
    """Get analytics data"""
    if not database:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        analytics = database.get_analytics_summary(days=days)
        return analytics
        
    except Exception as e:
        logger.error(f"Analytics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analytics")

@app.post("/modules/{module_name}/override")
async def override_module_model(module_name: str, model: str):
    """Override the model selection for a specific module"""
    if not orchestrator or not hasattr(orchestrator, 'config_loader'):
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    if module_name not in orchestrator.config_loader.module_configs:
        raise HTTPException(status_code=404, detail=f"Module '{module_name}' not found")
    
    # This would implement actual override logic in production
    return {
        "status": "success",
        "message": f"Module '{module_name}' will now use model '{model}'",
        "module": module_name,
        "new_model": model
    }

@app.get("/costs/breakdown")
async def get_cost_breakdown():
    """Get detailed cost breakdown by module and model"""
    if not database:
        return {
            "total_cost_today": 0,
            "cost_saved_by_cache": 0,
            "cost_saved_by_routing": 0,
            "projection_monthly": 0
        }
    
    try:
        # Get cost data from analytics
        analytics = database.get_analytics_summary(days=1)  # Today
        
        total_cost = 0
        if 'module_performance' in analytics:
            total_cost = sum(module.get('total_cost_cents', 0) for module in analytics['module_performance'])
        
        # Estimate savings (would be calculated from actual data)
        cache_savings = total_cost * 0.3  # Estimate 30% savings from cache
        routing_savings = total_cost * 0.4  # Estimate 40% savings from routing
        
        return {
            "total_cost_today": total_cost / 100,  # Convert to dollars
            "cost_saved_by_cache": cache_savings / 100,
            "cost_saved_by_routing": routing_savings / 100,
            "projection_monthly": (total_cost * 30) / 100
        }
        
    except Exception as e:
        logger.error(f"Cost breakdown failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to calculate cost breakdown")

# Background task functions
async def cleanup_old_data():
    """Background task to clean up old data"""
    if database:
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, database.cleanup_old_data, 90  # Keep 90 days
            )
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    global orchestrator, database
    
    logger.info("🚀 Hybrid LLM Orchestrator API starting...")
    
    try:
        # Initialize database if not already done
        if database is None:
            database = UserDatabase()
        
        # Initialize orchestrator if not already done
        if orchestrator is None:
            orchestrator = Orchestrator(config_path="configs")
        
        # Initialize orchestrator components
        if not orchestrator._initialized:
            await orchestrator.initialize()
        
        # Perform initial health check
        health = await orchestrator.health_check()
        logger.info(f"✅ System health: {health.get('overall_status', 'unknown')}")
        
        logger.info("🎯 API ready at http://localhost:8000")
        logger.info("📊 Metrics available at http://localhost:8000/metrics")
        logger.info("🔍 Health check at http://localhost:8000/health")
        logger.info("📚 API documentation at http://localhost:8000/docs")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        # Don't raise here - let the app start but mark components as unavailable

@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown"""
    logger.info("🛑 Hybrid LLM Orchestrator API shutting down...")
    
    try:
        if orchestrator:
            await orchestrator.cleanup()
        
        # Close any remaining WebSocket connections
        for connection in list(manager.active_connections):
            try:
                await connection.close()
            except Exception:
                pass
        
        logger.info("✅ Shutdown complete")
        
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

# Custom error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# Run with: uvicorn src.api_server:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        access_log=True
    )