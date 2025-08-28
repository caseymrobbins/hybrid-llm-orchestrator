# src/utils/monitoring.py

import time
import logging
import threading
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
from enum import Enum

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logging.warning("NumPy not available, using basic statistics")

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Enumeration for model types."""
    LOCAL = "local"
    EXTERNAL = "external"
    UNKNOWN = "unknown"

@dataclass
class ExecutionMetrics:
    """Data class for a single execution metric."""
    timestamp: float
    latency_ms: float
    cost_cents: float
    cache_hit: bool
    model_used: str
    model_type: ModelType
    
    def __post_init__(self):
        """Validate metrics after initialization."""
        if self.latency_ms < 0:
            raise ValueError("Latency cannot be negative")
        if self.cost_cents < 0:
            raise ValueError("Cost cannot be negative")

class MetricsCollector:
    """Thread-safe collector for performance and quality metrics with sliding window support."""
    
    def __init__(
        self, 
        max_history: int = 10000,
        window_hours: float = 24.0,
        enable_detailed_tracking: bool = True
    ):
        """
        Initialize metrics collector.
        
        Args:
            max_history: Maximum number of metrics to keep in memory
            window_hours: Time window for metrics (older metrics are discarded)
            enable_detailed_tracking: Whether to track detailed per-request metrics
        """
        self._lock = threading.RLock()
        self.max_history = max_history
        self.window_hours = window_hours
        self.enable_detailed_tracking = enable_detailed_tracking
        
        # Use deque for efficient FIFO operations
        self._metrics_history: deque = deque(maxlen=max_history)
        
        # Aggregate counters (thread-safe)
        self._cache_hits = 0
        self._total_requests = 0
        self._local_count = 0
        self._external_count = 0
        self._total_cost_cents = 0.0
        self._total_latency_ms = 0.0
        
        # Router accuracy tracking
        self._router_correct_decisions = 0
        self._router_total_decisions = 0
        
        # Start time for rate calculations
        self._start_time = time.time()
        
        logger.info(f"MetricsCollector initialized with max_history={max_history}, window_hours={window_hours}")

    def track_execution(
        self, 
        latency_s: float, 
        cost_cents: float, 
        cache_hit: bool, 
        model_used: str,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Track metrics for a single execution.
        
        Args:
            latency_s: Execution latency in seconds
            cost_cents: Cost in cents
            cache_hit: Whether this was a cache hit
            model_used: Name/identifier of the model used
            additional_data: Optional additional metadata
        """
        try:
            # Validate inputs
            if latency_s < 0:
                logger.warning(f"Negative latency received: {latency_s}s, setting to 0")
                latency_s = 0
            
            if cost_cents < 0:
                logger.warning(f"Negative cost received: {cost_cents} cents, setting to 0")
                cost_cents = 0
                
            latency_ms = latency_s * 1000
            timestamp = time.time()
            model_type = self._determine_model_type(model_used)
            
            with self._lock:
                # Update aggregate counters
                self._total_requests += 1
                self._total_latency_ms += latency_ms
                self._total_cost_cents += cost_cents
                
                if cache_hit:
                    self._cache_hits += 1
                
                if model_type == ModelType.LOCAL:
                    self._local_count += 1
                elif model_type == ModelType.EXTERNAL:
                    self._external_count += 1
                
                # Store detailed metrics if enabled
                if self.enable_detailed_tracking:
                    try:
                        metric = ExecutionMetrics(
                            timestamp=timestamp,
                            latency_ms=latency_ms,
                            cost_cents=cost_cents,
                            cache_hit=cache_hit,
                            model_used=model_used,
                            model_type=model_type
                        )
                        self._metrics_history.append(metric)
                    except ValueError as e:
                        logger.error(f"Invalid metrics data: {e}")
                        return
                
                # Clean up old data
                self._cleanup_old_metrics()
                
        except Exception as e:
            logger.error(f"Error tracking execution metrics: {e}")

    def track_router_decision(self, correct: bool) -> None:
        """
        Track router accuracy.
        
        Args:
            correct: Whether the router decision was correct
        """
        with self._lock:
            self._router_total_decisions += 1
            if correct:
                self._router_correct_decisions += 1

    def _determine_model_type(self, model_used: str) -> ModelType:
        """Determine if a model is local or external based on its name."""
        model_lower = model_used.lower()
        if any(keyword in model_lower for keyword in ['local', 'huggingface', 'hf', 'ollama', 'llamacpp']):
            return ModelType.LOCAL
        elif any(keyword in model_lower for keyword in ['openai', 'anthropic', 'claude', 'gpt', 'gemini', 'grok']):
            return ModelType.EXTERNAL
        else:
            return ModelType.UNKNOWN

    def _cleanup_old_metrics(self) -> None:
        """Remove metrics older than the window."""
        if not self.enable_detailed_tracking or not self._metrics_history:
            return
            
        cutoff_time = time.time() - (self.window_hours * 3600)
        
        # Remove old metrics from the front of deque
        while self._metrics_history and self._metrics_history[0].timestamp < cutoff_time:
            self._metrics_history.popleft()

    def _calculate_percentiles(self, values: List[float], percentiles: List[int]) -> Dict[str, float]:
        """Calculate percentiles, handling the case where numpy might not be available."""
        if not values:
            return {f"p{p}": 0.0 for p in percentiles}
        
        if HAS_NUMPY:
            return {f"p{p}": float(np.percentile(values, p)) for p in percentiles}
        else:
            # Fallback implementation without numpy
            sorted_values = sorted(values)
            n = len(sorted_values)
            result = {}
            
            for p in percentiles:
                if p == 0:
                    result[f"p{p}"] = sorted_values[0]
                elif p == 100:
                    result[f"p{p}"] = sorted_values[-1]
                else:
                    index = int((p / 100.0) * (n - 1))
                    result[f"p{p}"] = sorted_values[index]
            
            return result

    def get_report(self, include_detailed: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive metrics report.
        
        Args:
            include_detailed: Whether to include detailed percentile calculations
            
        Returns:
            Dictionary containing metrics report
        """
        with self._lock:
            if self._total_requests == 0:
                return {
                    "message": "No data collected yet.",
                    "timestamp": datetime.now().isoformat()
                }
            
            current_time = time.time()
            uptime_hours = (current_time - self._start_time) / 3600
            
            # Basic metrics
            report = {
                "timestamp": datetime.now().isoformat(),
                "uptime_hours": round(uptime_hours, 2),
                "total_requests": self._total_requests,
                "requests_per_hour": round(self._total_requests / max(uptime_hours, 0.001), 2),
                
                # Cost metrics
                "avg_cost_per_query_cents": round(self._total_cost_cents / self._total_requests, 4),
                "total_cost_dollars": round(self._total_cost_cents / 100, 2),
                
                # Cache metrics
                "cache_hit_rate": round(self._cache_hits / self._total_requests, 4),
                "cache_hits": self._cache_hits,
                
                # Model distribution
                "local_requests": self._local_count,
                "external_requests": self._external_count,
                "local_vs_external_ratio": f"{self._local_count}:{self._external_count}",
                "local_percentage": round((self._local_count / self._total_requests) * 100, 1),
                
                # Average latency
                "avg_latency_ms": round(self._total_latency_ms / self._total_requests, 2)
            }
            
            # Router accuracy
            if self._router_total_decisions > 0:
                report["router_accuracy"] = round(
                    self._router_correct_decisions / self._router_total_decisions, 4
                )
                report["router_decisions"] = self._router_total_decisions
            
            # Detailed metrics (if enabled and requested)
            if include_detailed and self.enable_detailed_tracking and self._metrics_history:
                recent_metrics = [m for m in self._metrics_history 
                                if current_time - m.timestamp <= self.window_hours * 3600]
                
                if recent_metrics:
                    latencies = [m.latency_ms for m in recent_metrics]
                    costs = [m.cost_cents for m in recent_metrics]
                    
                    # Latency percentiles
                    latency_percentiles = self._calculate_percentiles(latencies, [50, 95, 99])
                    report.update({
                        f"latency_{k}_ms": round(v, 2) for k, v in latency_percentiles.items()
                    })
                    
                    # Cost percentiles
                    cost_percentiles = self._calculate_percentiles(costs, [50, 95, 99])
                    report.update({
                        f"cost_{k}_cents": round(v, 4) for k, v in cost_percentiles.items()
                    })
                    
                    # Recent window stats
                    report["recent_window_hours"] = self.window_hours
                    report["recent_requests"] = len(recent_metrics)
            
            return report

    def reset_metrics(self) -> None:
        """Reset all collected metrics."""
        with self._lock:
            self._metrics_history.clear()
            self._cache_hits = 0
            self._total_requests = 0
            self._local_count = 0
            self._external_count = 0
            self._total_cost_cents = 0.0
            self._total_latency_ms = 0.0
            self._router_correct_decisions = 0
            self._router_total_decisions = 0
            self._start_time = time.time()
            
        logger.info("Metrics reset successfully")

    def export_raw_data(self) -> List[Dict[str, Any]]:
        """
        Export raw metrics data for external analysis.
        
        Returns:
            List of dictionaries containing raw metric data
        """
        with self._lock:
            if not self.enable_detailed_tracking:
                logger.warning("Detailed tracking is disabled, no raw data available")
                return []
            
            return [
                {
                    "timestamp": m.timestamp,
                    "datetime": datetime.fromtimestamp(m.timestamp).isoformat(),
                    "latency_ms": m.latency_ms,
                    "cost_cents": m.cost_cents,
                    "cache_hit": m.cache_hit,
                    "model_used": m.model_used,
                    "model_type": m.model_type.value
                }
                for m in self._metrics_history
            ]

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get current health status of the metrics collector.
        
        Returns:
            Dictionary containing health information
        """
        with self._lock:
            current_time = time.time()
            
            # Check if we're receiving recent data
            last_request_time = None
            if self._metrics_history:
                last_request_time = self._metrics_history[-1].timestamp
                time_since_last = current_time - last_request_time
            else:
                time_since_last = current_time - self._start_time
            
            return {
                "status": "healthy" if time_since_last < 300 else "idle",  # 5 minutes threshold
                "total_requests": self._total_requests,
                "memory_usage_metrics": len(self._metrics_history),
                "max_memory_usage": self.max_history,
                "memory_utilization": round(len(self._metrics_history) / self.max_history, 2),
                "time_since_last_request_seconds": round(time_since_last, 1),
                "last_request_time": datetime.fromtimestamp(last_request_time).isoformat() if last_request_time else None
            }