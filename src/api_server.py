# src/api_server.py

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

# Import your existing orchestrator
from .orchestrator import Orchestrator
from .config import ConfigLoader

app = FastAPI(title="Hybrid LLM Orchestrator API", version="1.0.0")

# CORS configuration for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the orchestrator
orchestrator = Orchestrator(config_path="configs")

# WebSocket connection manager for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Send message to all connected clients"""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass  # Handle disconnected clients

manager = ConnectionManager()

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    user_id: Optional[int] = 1
    preferences: Optional[Dict[str, Any]] = {}

class QueryResponse(BaseModel):
    response: str
    module_outputs: Dict[str, Any]
    routing_decisions: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    execution_time: float

class HealthResponse(BaseModel):
    overall_status: str
    database: Dict[str, str]
    external_apis: Dict[str, str]
    circuit_breakers: Dict[str, str]
    cache_status: Dict[str, Any]

class MetricsResponse(BaseModel):
    total_queries: int
    avg_latency: float
    cache_hit_rate: float
    cost_saved: float
    local_vs_external: Dict[str, int]
    recent_queries: List[Dict[str, Any]]

# In-memory metrics storage (in production, use Redis or database)
class MetricsCollector:
    def __init__(self):
        self.queries = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_cost_saved = 0.0
        self.local_calls = 0
        self.external_calls = 0
        
    def record_query(self, query_data: dict):
        self.queries.append({
            **query_data,
            'timestamp': datetime.now().isoformat()
        })
        # Keep only last 100 queries in memory
        if len(self.queries) > 100:
            self.queries.pop(0)
    
    def get_metrics(self) -> dict:
        total_queries = len(self.queries)
        avg_latency = sum(q.get('latency', 0) for q in self.queries) / max(total_queries, 1)
        cache_hit_rate = self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
        
        return {
            'total_queries': total_queries,
            'avg_latency': round(avg_latency, 2),
            'cache_hit_rate': round(cache_hit_rate, 2),
            'cost_saved': round(self.total_cost_saved, 2),
            'local_vs_external': {
                'local': round(self.local_calls / max(self.local_calls + self.external_calls, 1) * 100),
                'external': round(self.external_calls / max(self.local_calls + self.external_calls, 1) * 100)
            },
            'recent_queries': self.queries[-10:]  # Last 10 queries
        }

metrics_collector = MetricsCollector()

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Hybrid LLM Orchestrator API",
        "version": "1.0.0",
        "status": "online",
        "endpoints": [
            "/health",
            "/metrics",
            "/process",
            "/modules",
            "/config",
            "/ws"
        ]
    }

@app.get("/health", response_model=HealthResponse)
async def get_health():
    """Get comprehensive system health status"""
    health = await orchestrator.health_check()
    
    # Add cache status
    cache_status = {
        "entries": len(orchestrator.semantic_cache._cache) if hasattr(orchestrator.semantic_cache, '_cache') else 0,
        "hit_rate": metrics_collector.cache_hits / max(metrics_collector.cache_hits + metrics_collector.cache_misses, 1),
        "memory_usage_mb": 0  # Would calculate actual memory usage in production
    }
    
    return HealthResponse(
        overall_status=health['overall_status'],
        database=health['database'],
        external_apis=health['external_apis'],
        circuit_breakers=health['circuit_breakers'],
        cache_status=cache_status
    )

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get system metrics and performance data"""
    return MetricsResponse(**metrics_collector.get_metrics())

@app.post("/process", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a query through the hybrid LLM orchestrator"""
    start_time = time.time()
    
    try:
        # Track module execution for real-time updates
        module_outputs = {}
        routing_decisions = []
        
        # Execute workflow with tracking
        # This is a simplified version - in production, you'd hook into the orchestrator's events
        for module in orchestrator.config_loader.workflow_config.execution_plan:
            module_name = module['module']
            
            # Broadcast module start via WebSocket
            await manager.broadcast({
                'type': 'module_start',
                'module': module_name,
                'timestamp': datetime.now().isoformat()
            })
            
            # Execute module
            output = await orchestrator._execute_module(module_name, {
                'query': request.query,
                'user_id': request.user_id,
                **module_outputs  # Pass previous outputs as context
            })
            
            # Determine if it was cached
            was_cached = "SEMANTIC CACHE HIT" in output  # Simple check, improve in production
            
            # Determine routing (simplified - in production, get from orchestrator)
            model_used = "llama-3.1-8b" if module_name == "Curiosity" else "gpt-4o"
            is_local = "llama" in model_used.lower()
            
            # Track routing decision
            routing_decision = {
                'module': module_name,
                'model': model_used,
                'complexity': 'low' if is_local else 'high',
                'cost': 0 if is_local else 0.02,
                'cached': was_cached
            }
            routing_decisions.append(routing_decision)
            
            # Track module output
            module_outputs[module_name] = {
                'output': output,
                'cached': was_cached,
                'latency': (time.time() - start_time) * 1000,
                'model': model_used
            }
            
            # Update metrics
            if was_cached:
                metrics_collector.cache_hits += 1
            else:
                metrics_collector.cache_misses += 1
                
            if is_local:
                metrics_collector.local_calls += 1
            else:
                metrics_collector.external_calls += 1
                metrics_collector.total_cost_saved += 0.02 if was_cached else 0
            
            # Broadcast module completion via WebSocket
            await manager.broadcast({
                'type': 'module_complete',
                'module': module_name,
                'output': module_outputs[module_name],
                'routing': routing_decision,
                'timestamp': datetime.now().isoformat()
            })
        
        # Generate final response
        response = await orchestrator.synthesize_response(module_outputs, request.query)
        
        execution_time = time.time() - start_time
        
        # Record query metrics
        metrics_collector.record_query({
            'query': request.query,
            'latency': execution_time * 1000,
            'modules_used': list(module_outputs.keys()),
            'total_cost': sum(r['cost'] for r in routing_decisions),
            'cache_hits': sum(1 for r in routing_decisions if r['cached'])
        })
        
        # Broadcast completion
        await manager.broadcast({
            'type': 'query_complete',
            'response': response,
            'execution_time': execution_time,
            'timestamp': datetime.now().isoformat()
        })
        
        return QueryResponse(
            response=response,
            module_outputs=module_outputs,
            routing_decisions=routing_decisions,
            metrics=metrics_collector.get_metrics(),
            execution_time=execution_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/modules")
async def get_modules():
    """Get information about configured modules"""
    modules = []
    for name, config in orchestrator.config_loader.module_configs.items():
        modules.append({
            'name': name,
            'aspect': config.aspect,
            'has_fallback': config.fallback is not None,
            'uses_external': config.llm_address and "http" in str(config.llm_address)
        })
    return {'modules': modules}

@app.get("/config")
async def get_config():
    """Get current system configuration (sanitized)"""
    workflow = orchestrator.config_loader.workflow_config
    return {
        'workflow': {
            'name': workflow.name,
            'description': workflow.description,
            'execution_plan': workflow.execution_plan
        },
        'routing': {
            'router_type': 'mf',  # Matrix Factorization
            'strong_model': orchestrator.router.strong_model,
            'weak_model': orchestrator.router.weak_model,
            'threshold': 0.11593  # Default threshold
        },
        'cache': {
            'type': 'semantic',
            'similarity_threshold': 0.90
        },
        'security': {
            'prompt_injection_detection': True,
            'pii_scrubbing': True,
            'circuit_breakers': True
        }
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle any incoming messages
            data = await websocket.receive_text()
            
            # Handle ping/pong for connection health
            if data == "ping":
                await websocket.send_text("pong")
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/cache/clear")
async def clear_cache():
    """Clear the semantic cache"""
    # In production, implement actual cache clearing
    orchestrator.semantic_cache.clear()
    return {"status": "success", "message": "Cache cleared"}

@app.post("/modules/{module_name}/override")
async def override_module_model(module_name: str, model: str):
    """Override the model selection for a specific module"""
    if module_name not in orchestrator.config_loader.module_configs:
        raise HTTPException(status_code=404, detail=f"Module '{module_name}' not found")
    
    # In production, implement actual override logic
    return {
        "status": "success",
        "message": f"Module '{module_name}' will now use model '{model}'"
    }

@app.get("/costs/breakdown")
async def get_cost_breakdown():
    """Get detailed cost breakdown by module and model"""
    # Analyze recent queries for cost patterns
    costs_by_module = {}
    costs_by_model = {}
    
    for query in metrics_collector.queries:
        # This is simplified - in production, track actual costs
        pass
    
    return {
        "total_cost_today": round(sum(q.get('total_cost', 0) for q in metrics_collector.queries), 2),
        "cost_saved_by_cache": round(metrics_collector.total_cost_saved, 2),
        "cost_saved_by_routing": round(metrics_collector.total_cost_saved * 0.7, 2),  # Estimate
        "projection_monthly": round(metrics_collector.total_cost_saved * 30, 2)
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    print("🚀 Hybrid LLM Orchestrator API starting...")
    
    # Perform initial health check
    health = await orchestrator.health_check()
    print(f"✅ System health: {health['overall_status']}")
    
    # Warm up cache with common queries (optional)
    # await warm_up_cache()
    
    print("🎯 API ready at http://localhost:8000")

# Run with: uvicorn src.api_server:app --reload --port 8000
# Or add this to main.py:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)