"""
Data models and configurations for performance optimization.

This module contains all the data classes and exception types used throughout
the performance package for metrics tracking and error handling.
"""

from typing import Dict, Any
from datetime import datetime
from dataclasses import dataclass, field


class OptimizationError(Exception):
    """Raised when optimization operations fail."""
    pass


@dataclass
class PerformanceMetrics:
    """
    Performance metrics for system monitoring and optimization.

    Tracks CPU, memory, latency, throughput, and error metrics.
    """
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    latency_ms: float = 0.0
    throughput_ops_per_sec: float = 0.0
    error_rate_percent: float = 0.0
    concurrent_operations: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'cpu_usage_percent': self.cpu_usage_percent,
            'memory_usage_mb': self.memory_usage_mb,
            'latency_ms': self.latency_ms,
            'throughput_ops_per_sec': self.throughput_ops_per_sec,
            'error_rate_percent': self.error_rate_percent,
            'concurrent_operations': self.concurrent_operations,
            'timestamp': self.timestamp.isoformat()
        }

    def calculate_efficiency_score(self) -> float:
        """
        Calculate efficiency score based on metrics.

        Returns:
            Efficiency score between 0 and 1 (higher is better)
        """
        # Normalize metrics to 0-1 scale
        cpu_score = max(0, (100 - self.cpu_usage_percent) / 100)
        memory_score = max(0, (1000 - self.memory_usage_mb) / 1000)
        latency_score = max(0, (100 - self.latency_ms) / 100)
        throughput_score = min(1, self.throughput_ops_per_sec / 2000)  # Max 2000 ops/sec
        error_score = max(0, (100 - self.error_rate_percent) / 100)

        # Weighted average
        weights = {
            'cpu': 0.2,
            'memory': 0.2,
            'latency': 0.3,
            'throughput': 0.2,
            'error': 0.1
        }

        efficiency = (
            weights['cpu'] * cpu_score +
            weights['memory'] * memory_score +
            weights['latency'] * latency_score +
            weights['throughput'] * throughput_score +
            weights['error'] * error_score
        )

        return max(0.0, min(1.0, efficiency))