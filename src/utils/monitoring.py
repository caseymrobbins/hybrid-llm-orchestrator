# src/utils/monitoring.py

import time
import numpy as np

class MetricsCollector:
    """Collects and calculates key performance and quality metrics."""
    def __init__(self):
        self.latencies =
        self.costs_cents =
        self.cache_hits = 0
        self.total_requests = 0
        self.local_count = 0
        self.external_count = 0
        # In a real system, you'd also track router accuracy against a baseline
        self.router_correct_decisions = 0
        self.router_total_decisions = 0

    def track_execution(self, latency_s: float, cost_cents: float, cache_hit: bool, model_used: str):
        """Tracks the metrics for a single module execution."""
        self.total_requests += 1
        self.latencies.append(latency_s * 1000) # Store in ms
        self.costs_cents.append(cost_cents)
        
        if cache_hit:
            self.cache_hits += 1
        
        if 'local' in model_used.lower() or 'huggingface' in model_used.lower():
            self.local_count += 1
        else:
            self.external_count += 1

    def get_report(self) -> dict:
        """Generates a report of the current metrics."""
        if not self.latencies:
            return {"message": "No data collected yet."}
            
        total_cost = sum(self.costs_cents)
        
        return {
            'latency_p50_ms': np.percentile(self.latencies, 50),
            'latency_p99_ms': np.percentile(self.latencies, 99),
            'avg_cost_per_query_cents': total_cost / self.total_requests if self.total_requests > 0 else 0,
            'total_cost_dollars': total_cost / 100,
            'cache_hit_rate': self.cache_hits / self.total_requests if self.total_requests > 0 else 0,
            'local_vs_external_ratio': f"{self.local_count}:{self.external_count}",
            # 'router_accuracy': self.router_correct_decisions / self.router_total_decisions
        }