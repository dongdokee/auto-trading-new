"""
Database optimization models and data structures.

This module contains the core data models for database optimization including:
- Query execution plans and performance metrics
- Query statistics and monitoring data
- Index recommendations and optimization suggestions
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class OptimizationError(Exception):
    """Raised when database optimization operations fail."""
    pass


@dataclass
class QueryPlan:
    """
    Query execution plan with performance metrics.

    Represents a database query execution plan with cost estimates,
    timing information, and optimization recommendations.
    """
    query: str
    estimated_cost: float = 0.0
    execution_time_ms: float = 0.0
    rows_examined: int = 0
    rows_returned: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    plan_details: Optional[str] = None

    def get_efficiency_ratio(self) -> float:
        """
        Calculate query efficiency ratio.

        Returns:
            Efficiency ratio between 0 and 1 (higher is better)
        """
        if self.rows_examined == 0:
            return 1.0

        # Calculate selectivity (rows returned / rows examined)
        selectivity = self.rows_returned / self.rows_examined

        # Factor in execution time (lower is better)
        time_factor = max(0, 1 - (self.execution_time_ms / 1000))  # Normalize to seconds

        # Factor in cost (lower is better)
        cost_factor = max(0, 1 - (self.estimated_cost / 1000))  # Normalize cost

        # Combined efficiency score
        efficiency = (selectivity + time_factor + cost_factor) / 3
        return max(0.0, min(1.0, efficiency))

    def identify_issues(self) -> List[str]:
        """Identify potential performance issues with the query."""
        issues = []

        # High execution time
        if self.execution_time_ms > 100:
            issues.append(f"High execution time: {self.execution_time_ms:.1f}ms")

        # Low selectivity
        if self.rows_examined > 0:
            selectivity = self.rows_returned / self.rows_examined
            if selectivity < 0.1:
                issues.append(f"Low selectivity: {selectivity:.2%} (examining too many rows)")

        # High cost
        if self.estimated_cost > 100:
            issues.append(f"High estimated cost: {self.estimated_cost:.1f}")

        # Full table scan detection
        if self.plan_details and "Seq Scan" in self.plan_details:
            issues.append("Sequential scan detected (consider adding index)")

        return issues

    def to_dict(self) -> Dict[str, Any]:
        """Convert query plan to dictionary."""
        return {
            'query': self.query,
            'estimated_cost': self.estimated_cost,
            'execution_time_ms': self.execution_time_ms,
            'rows_examined': self.rows_examined,
            'rows_returned': self.rows_returned,
            'created_at': self.created_at.isoformat(),
            'efficiency_ratio': self.get_efficiency_ratio(),
            'issues': self.identify_issues(),
            'plan_details': self.plan_details
        }


@dataclass
class QueryStats:
    """
    Query execution statistics and performance metrics.

    Tracks query performance, caching efficiency, and optimization opportunities.
    """
    total_queries: int = 0
    slow_queries: int = 0
    avg_execution_time_ms: float = 0.0
    total_execution_time_ms: float = 0.0
    query_cache_hits: int = 0
    query_cache_misses: int = 0
    slow_query_threshold_ms: float = 100.0

    def record_query(self, execution_time_ms: float, cache_hit: bool = False) -> None:
        """Record a query execution."""
        self.total_queries += 1
        self.total_execution_time_ms += execution_time_ms

        if execution_time_ms > self.slow_query_threshold_ms:
            self.slow_queries += 1

        if cache_hit:
            self.query_cache_hits += 1
        else:
            self.query_cache_misses += 1

    def get_avg_execution_time(self) -> float:
        """Calculate average execution time."""
        if self.total_queries == 0:
            return 0.0
        return self.total_execution_time_ms / self.total_queries

    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_cache_requests = self.query_cache_hits + self.query_cache_misses
        if total_cache_requests == 0:
            return 0.0
        return self.query_cache_hits / total_cache_requests

    def get_slow_query_rate(self) -> float:
        """Calculate slow query rate."""
        if self.total_queries == 0:
            return 0.0
        return self.slow_queries / self.total_queries

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            'total_queries': self.total_queries,
            'slow_queries': self.slow_queries,
            'avg_execution_time_ms': self.get_avg_execution_time(),
            'total_execution_time_ms': self.total_execution_time_ms,
            'cache_hit_rate': self.get_cache_hit_rate(),
            'slow_query_rate': self.get_slow_query_rate(),
            'query_cache_hits': self.query_cache_hits,
            'query_cache_misses': self.query_cache_misses
        }


@dataclass
class IndexRecommendation:
    """
    Index recommendation for query optimization.

    Suggests database indexes to improve query performance.
    """
    table_name: str
    column_names: List[str]
    index_type: str = "btree"
    estimated_benefit: float = 0.0
    reason: str = ""
    query_frequency: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)

    def generate_create_sql(self) -> str:
        """Generate CREATE INDEX SQL statement."""
        index_name = f"idx_{self.table_name}_{'_'.join(self.column_names)}"
        columns = ', '.join(self.column_names)

        if self.index_type.lower() == 'partial':
            return f"CREATE INDEX {index_name} ON {self.table_name} ({columns}) WHERE condition;"
        else:
            return f"CREATE INDEX {index_name} ON {self.table_name} USING {self.index_type} ({columns});"

    def get_priority_score(self) -> float:
        """Calculate priority score for the recommendation."""
        # Combine benefit estimate with query frequency
        frequency_factor = min(1.0, self.query_frequency / 1000)  # Normalize to max 1000 queries
        priority = (self.estimated_benefit + frequency_factor) / 2
        return max(0.0, min(1.0, priority))

    def to_dict(self) -> Dict[str, Any]:
        """Convert recommendation to dictionary."""
        return {
            'table_name': self.table_name,
            'column_names': self.column_names,
            'index_type': self.index_type,
            'estimated_benefit': self.estimated_benefit,
            'reason': self.reason,
            'query_frequency': self.query_frequency,
            'priority_score': self.get_priority_score(),
            'create_sql': self.generate_create_sql(),
            'created_at': self.created_at.isoformat()
        }