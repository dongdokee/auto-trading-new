"""
Query optimizer with caching and performance analysis.

This module provides intelligent query optimization capabilities including:
- Query execution plan analysis
- Query optimization suggestions
- Index recommendations
- Query result caching
- Performance monitoring
"""

import re
import time
import logging
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Pattern, Tuple

from .models import QueryPlan, QueryStats, IndexRecommendation, OptimizationError

logger = logging.getLogger(__name__)


class QueryOptimizer:
    """
    Query optimizer with caching and performance analysis.

    Provides intelligent query optimization, caching, and performance monitoring.
    """

    def __init__(self, slow_query_threshold_ms: float = 100.0):
        """
        Initialize query optimizer.

        Args:
            slow_query_threshold_ms: Threshold for slow query detection
        """
        self.stats = QueryStats(slow_query_threshold_ms=slow_query_threshold_ms)
        self.query_cache: Dict[str, Tuple[Any, datetime]] = {}
        self.slow_query_threshold_ms = slow_query_threshold_ms
        self.cache_ttl_seconds = 300  # 5 minutes

        # Common query patterns for optimization
        self.optimization_patterns = self._compile_optimization_patterns()

    def _compile_optimization_patterns(self) -> Dict[str, Pattern]:
        """Compile regex patterns for query optimization."""
        return {
            'select_star': re.compile(r'SELECT\s+\*\s+FROM', re.IGNORECASE),
            'no_limit': re.compile(r'SELECT\s+.*\s+FROM\s+\w+(?!\s+LIMIT)', re.IGNORECASE),
            'subquery_in': re.compile(r'IN\s*\(\s*SELECT', re.IGNORECASE),
            'cartesian_join': re.compile(r'FROM\s+\w+\s*,\s*\w+', re.IGNORECASE),
            'missing_where': re.compile(r'SELECT\s+.*\s+FROM\s+\w+(?!\s+WHERE)', re.IGNORECASE)
        }

    async def analyze_query_plan(self, query: str, connection) -> QueryPlan:
        """
        Analyze query execution plan.

        Args:
            query: SQL query to analyze
            connection: Database connection

        Returns:
            QueryPlan with analysis results
        """
        try:
            # Get execution plan
            explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"
            start_time = time.time()

            plan_result = await connection.fetch(explain_query)
            execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            if plan_result:
                plan_data = plan_result[0]['QUERY PLAN'][0]

                return QueryPlan(
                    query=query,
                    estimated_cost=plan_data.get('Total Cost', 0.0),
                    execution_time_ms=plan_data.get('Actual Total Time', execution_time),
                    rows_examined=plan_data.get('Actual Rows', 0),
                    rows_returned=plan_data.get('Plan Rows', 0),
                    plan_details=str(plan_data)
                )
            else:
                return QueryPlan(query=query, execution_time_ms=execution_time)

        except Exception as e:
            logger.error(f"Failed to analyze query plan: {e}")
            return QueryPlan(query=query)

    async def optimize_query(self, query: str) -> Dict[str, Any]:
        """
        Optimize a SQL query and provide suggestions.

        Args:
            query: SQL query to optimize

        Returns:
            Dictionary with optimization suggestions
        """
        suggestions = {
            'optimized_query': query,
            'recommendations': [],
            'estimated_improvement': 0.0
        }

        # Detect and suggest fixes for common issues
        issues = self.detect_query_issues(query)

        for issue in issues:
            if 'SELECT *' in issue:
                suggestions['recommendations'].append(
                    "Replace SELECT * with specific column names to reduce data transfer"
                )
                suggestions['estimated_improvement'] += 0.2

            elif 'missing WHERE' in issue:
                suggestions['recommendations'].append(
                    "Add WHERE clause to filter results and improve performance"
                )
                suggestions['estimated_improvement'] += 0.3

            elif 'no LIMIT' in issue:
                suggestions['recommendations'].append(
                    "Add LIMIT clause to prevent returning excessive rows"
                )
                suggestions['estimated_improvement'] += 0.15

            elif 'subquery IN' in issue:
                suggestions['recommendations'].append(
                    "Consider rewriting IN subquery as JOIN for better performance"
                )
                optimized = self._rewrite_in_subquery(query)
                if optimized != query:
                    suggestions['optimized_query'] = optimized
                    suggestions['estimated_improvement'] += 0.25

        return suggestions

    def suggest_indexes(self, slow_queries: List[str]) -> List[IndexRecommendation]:
        """
        Suggest indexes based on slow query analysis.

        Args:
            slow_queries: List of slow SQL queries

        Returns:
            List of index recommendations
        """
        recommendations = []
        table_columns = {}

        # Analyze queries to extract table and column usage
        for query in slow_queries:
            table_info = self._extract_table_columns(query)

            for table, columns in table_info.items():
                if table not in table_columns:
                    table_columns[table] = {}

                for column in columns:
                    if column not in table_columns[table]:
                        table_columns[table][column] = 0
                    table_columns[table][column] += 1

        # Generate recommendations
        for table, columns in table_columns.items():
            # Sort columns by frequency
            sorted_columns = sorted(columns.items(), key=lambda x: x[1], reverse=True)

            # Recommend single-column indexes for frequently used columns
            for column, frequency in sorted_columns[:3]:  # Top 3 columns
                if frequency >= 2:  # Used in at least 2 queries
                    recommendations.append(IndexRecommendation(
                        table_name=table,
                        column_names=[column],
                        index_type="btree",
                        estimated_benefit=min(0.9, frequency / 10),
                        reason=f"Frequently used in WHERE clauses ({frequency} times)",
                        query_frequency=frequency
                    ))

            # Recommend composite indexes for column combinations
            if len(sorted_columns) >= 2:
                top_columns = [col for col, _ in sorted_columns[:2]]
                recommendations.append(IndexRecommendation(
                    table_name=table,
                    column_names=top_columns,
                    index_type="btree",
                    estimated_benefit=0.7,
                    reason="Composite index for commonly used column combination",
                    query_frequency=sum(freq for _, freq in sorted_columns[:2])
                ))

        return recommendations

    async def execute_cached_query(
        self,
        query: str,
        connection,
        *args,
        cache_key: Optional[str] = None
    ) -> Any:
        """
        Execute query with result caching.

        Args:
            query: SQL query
            connection: Database connection
            *args: Query parameters
            cache_key: Custom cache key (optional)

        Returns:
            Query results
        """
        # Generate cache key
        if cache_key is None:
            key_content = f"{query}:{':'.join(str(arg) for arg in args)}"
            cache_key = hashlib.md5(key_content.encode()).hexdigest()

        # Check cache
        if cache_key in self.query_cache:
            result, cached_at = self.query_cache[cache_key]
            age = (datetime.utcnow() - cached_at).total_seconds()

            if age < self.cache_ttl_seconds:
                self.stats.record_query(0, cache_hit=True)
                return result

            # Remove expired entry
            del self.query_cache[cache_key]

        # Execute query
        start_time = time.time()
        try:
            result = await connection.fetch(query, *args)
            execution_time = (time.time() - start_time) * 1000

            # Cache the result
            self.query_cache[cache_key] = (result, datetime.utcnow())

            # Record statistics
            self.stats.record_query(execution_time, cache_hit=False)

            return result

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.stats.record_query(execution_time, cache_hit=False)
            raise OptimizationError(f"Query execution failed: {e}")

    def detect_query_issues(self, query: str) -> List[str]:
        """Detect common query performance issues."""
        issues = []

        for pattern_name, pattern in self.optimization_patterns.items():
            if pattern.search(query):
                if pattern_name == 'select_star':
                    issues.append("SELECT * detected - consider specifying columns")
                elif pattern_name == 'no_limit':
                    issues.append("no LIMIT clause - may return excessive rows")
                elif pattern_name == 'subquery_in':
                    issues.append("subquery IN detected - consider JOIN")
                elif pattern_name == 'cartesian_join':
                    issues.append("Cartesian join detected - add JOIN conditions")
                elif pattern_name == 'missing_where':
                    issues.append("missing WHERE clause - may scan entire table")

        return issues

    def _rewrite_in_subquery(self, query: str) -> str:
        """Attempt to rewrite IN subquery as JOIN."""
        # Simple pattern matching for basic IN subquery rewriting
        in_pattern = re.compile(
            r'WHERE\s+(\w+)\s+IN\s*\(\s*SELECT\s+(\w+)\s+FROM\s+(\w+)(?:\s+WHERE\s+(.+?))?\s*\)',
            re.IGNORECASE
        )

        match = in_pattern.search(query)
        if match:
            column, subquery_column, subquery_table, subquery_where = match.groups()

            # Build JOIN replacement
            join_clause = f"INNER JOIN {subquery_table} ON {column} = {subquery_table}.{subquery_column}"
            if subquery_where:
                join_clause += f" AND {subquery_where}"

            # Replace the IN clause with JOIN
            rewritten = in_pattern.sub('', query)
            rewritten = rewritten.replace('FROM', f'FROM ... {join_clause} ')

            return rewritten

        return query

    def _extract_table_columns(self, query: str) -> Dict[str, List[str]]:
        """Extract table and column information from query."""
        tables = {}

        # Simple pattern matching for tables and columns
        # This is a basic implementation - a full parser would be more robust
        from_pattern = re.compile(r'FROM\s+(\w+)', re.IGNORECASE)
        where_pattern = re.compile(r'WHERE\s+.*?(\w+)\s*[=<>]', re.IGNORECASE)

        # Extract table names
        table_matches = from_pattern.findall(query)
        for table in table_matches:
            if table not in tables:
                tables[table] = []

        # Extract column names from WHERE clauses
        where_matches = where_pattern.findall(query)
        for column in where_matches:
            # Assign to first table (simplified logic)
            if table_matches:
                table = table_matches[0]
                if column not in tables[table]:
                    tables[table].append(column)

        return tables

    def clear_cache(self) -> None:
        """Clear the query cache."""
        self.query_cache.clear()
        logger.info("Query cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_size': len(self.query_cache),
            'cache_hit_rate': self.stats.get_cache_hit_rate(),
            'total_queries': self.stats.total_queries,
            'avg_execution_time_ms': self.stats.get_avg_execution_time()
        }