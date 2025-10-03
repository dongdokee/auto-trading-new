"""
Tests for Database Query Optimizer.

Following TDD methodology: Red -> Green -> Refactor
Tests for query optimization, connection pooling, indexing strategies, and performance monitoring.
"""

import pytest
import asyncio
import time
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from src.optimization.db_optimizer import (
    DatabaseOptimizer,
    QueryOptimizer,
    QueryPlan,
    QueryStats,
    IndexRecommendation,
    ConnectionPoolManager,
    OptimizationError
)


class TestQueryPlan:
    """Test suite for QueryPlan class."""

    def test_should_initialize_with_query_details(self):
        """Test that QueryPlan initializes with query details."""
        plan = QueryPlan(
            query="SELECT * FROM orders WHERE status = 'active'",
            estimated_cost=100.5,
            execution_time_ms=25.3,
            rows_examined=1000,
            rows_returned=50
        )

        assert plan.query == "SELECT * FROM orders WHERE status = 'active'"
        assert plan.estimated_cost == 100.5
        assert plan.execution_time_ms == 25.3
        assert plan.rows_examined == 1000
        assert plan.rows_returned == 50
        assert plan.created_at is not None

    def test_should_calculate_efficiency_ratio(self):
        """Test that QueryPlan can calculate efficiency ratio."""
        # Efficient query
        efficient_plan = QueryPlan(
            query="SELECT id FROM orders WHERE status = 'active'",
            estimated_cost=10.0,
            execution_time_ms=5.0,
            rows_examined=100,
            rows_returned=50
        )

        # Inefficient query
        inefficient_plan = QueryPlan(
            query="SELECT * FROM orders",
            estimated_cost=1000.0,
            execution_time_ms=200.0,
            rows_examined=10000,
            rows_returned=50
        )

        efficient_ratio = efficient_plan.get_efficiency_ratio()
        inefficient_ratio = inefficient_plan.get_efficiency_ratio()

        assert efficient_ratio > inefficient_ratio
        assert 0 <= efficient_ratio <= 1
        assert 0 <= inefficient_ratio <= 1

    def test_should_identify_performance_issues(self):
        """Test that QueryPlan can identify performance issues."""
        # Slow query
        slow_plan = QueryPlan(
            query="SELECT * FROM trades",
            estimated_cost=500.0,
            execution_time_ms=150.0,
            rows_examined=50000,
            rows_returned=10
        )

        issues = slow_plan.identify_issues()

        assert isinstance(issues, list)
        assert len(issues) > 0
        assert any("high execution time" in issue.lower() for issue in issues)
        assert any("low selectivity" in issue.lower() for issue in issues)

    def test_should_convert_to_dictionary(self):
        """Test that QueryPlan can be converted to dictionary."""
        plan = QueryPlan(
            query="SELECT * FROM positions",
            estimated_cost=50.0,
            execution_time_ms=12.5
        )

        plan_dict = plan.to_dict()

        assert isinstance(plan_dict, dict)
        assert plan_dict['query'] == "SELECT * FROM positions"
        assert plan_dict['estimated_cost'] == 50.0
        assert plan_dict['execution_time_ms'] == 12.5
        assert 'created_at' in plan_dict


class TestQueryStats:
    """Test suite for QueryStats class."""

    def test_should_initialize_with_default_values(self):
        """Test that QueryStats initializes with default values."""
        stats = QueryStats()

        assert stats.total_queries == 0
        assert stats.slow_queries == 0
        assert stats.avg_execution_time_ms == 0.0
        assert stats.total_execution_time_ms == 0.0
        assert stats.query_cache_hits == 0
        assert stats.query_cache_misses == 0

    def test_should_record_query_execution(self):
        """Test that QueryStats can record query execution."""
        stats = QueryStats()

        # Record some queries
        stats.record_query(execution_time_ms=25.5, cache_hit=False)
        stats.record_query(execution_time_ms=150.0, cache_hit=True)
        stats.record_query(execution_time_ms=30.2, cache_hit=False)

        assert stats.total_queries == 3
        assert stats.slow_queries == 1  # 150ms query
        assert stats.query_cache_hits == 1
        assert stats.query_cache_misses == 2
        assert stats.total_execution_time_ms == 205.7

    def test_should_calculate_average_execution_time(self):
        """Test that QueryStats calculates average execution time."""
        stats = QueryStats()

        stats.record_query(execution_time_ms=10.0)
        stats.record_query(execution_time_ms=20.0)
        stats.record_query(execution_time_ms=30.0)

        assert stats.get_avg_execution_time() == 20.0

    def test_should_calculate_cache_hit_rate(self):
        """Test that QueryStats calculates cache hit rate."""
        stats = QueryStats()

        # 8 hits out of 10 total
        for _ in range(8):
            stats.record_query(execution_time_ms=50.0, cache_hit=True)
        for _ in range(2):
            stats.record_query(execution_time_ms=100.0, cache_hit=False)

        assert stats.get_cache_hit_rate() == 0.8

    def test_should_convert_to_dictionary(self):
        """Test that QueryStats can be converted to dictionary."""
        stats = QueryStats()
        stats.record_query(execution_time_ms=25.0, cache_hit=True)

        stats_dict = stats.to_dict()

        assert isinstance(stats_dict, dict)
        assert stats_dict['total_queries'] == 1
        assert stats_dict['avg_execution_time_ms'] == 25.0
        assert stats_dict['cache_hit_rate'] == 1.0


class TestIndexRecommendation:
    """Test suite for IndexRecommendation class."""

    def test_should_initialize_with_recommendation_details(self):
        """Test that IndexRecommendation initializes with details."""
        recommendation = IndexRecommendation(
            table_name="orders",
            column_names=["user_id", "created_at"],
            index_type="btree",
            estimated_benefit=0.75,
            reason="Improve WHERE clause performance"
        )

        assert recommendation.table_name == "orders"
        assert recommendation.column_names == ["user_id", "created_at"]
        assert recommendation.index_type == "btree"
        assert recommendation.estimated_benefit == 0.75
        assert recommendation.reason == "Improve WHERE clause performance"

    def test_should_generate_create_index_sql(self):
        """Test that IndexRecommendation can generate CREATE INDEX SQL."""
        recommendation = IndexRecommendation(
            table_name="trades",
            column_names=["symbol", "timestamp"],
            index_type="btree"
        )

        sql = recommendation.generate_create_sql()

        assert "CREATE INDEX" in sql
        assert "trades" in sql
        assert "symbol" in sql
        assert "timestamp" in sql

    def test_should_calculate_priority_score(self):
        """Test that IndexRecommendation calculates priority score."""
        high_benefit = IndexRecommendation(
            table_name="orders",
            column_names=["status"],
            estimated_benefit=0.9,
            query_frequency=1000
        )

        low_benefit = IndexRecommendation(
            table_name="logs",
            column_names=["level"],
            estimated_benefit=0.3,
            query_frequency=10
        )

        high_score = high_benefit.get_priority_score()
        low_score = low_benefit.get_priority_score()

        assert high_score > low_score
        assert 0 <= high_score <= 1
        assert 0 <= low_score <= 1

    def test_should_convert_to_dictionary(self):
        """Test that IndexRecommendation can be converted to dictionary."""
        recommendation = IndexRecommendation(
            table_name="positions",
            column_names=["symbol", "side"],
            index_type="btree"
        )

        rec_dict = recommendation.to_dict()

        assert isinstance(rec_dict, dict)
        assert rec_dict['table_name'] == "positions"
        assert rec_dict['column_names'] == ["symbol", "side"]
        assert rec_dict['index_type'] == "btree"


class TestConnectionPoolManager:
    """Test suite for ConnectionPoolManager class."""

    @pytest.fixture
    def pool_manager(self):
        """Create ConnectionPoolManager instance for testing."""
        return ConnectionPoolManager(
            database_url="postgresql://user:pass@localhost/test",
            min_connections=5,
            max_connections=20
        )

    def test_should_initialize_with_configuration(self, pool_manager):
        """Test that ConnectionPoolManager initializes with configuration."""
        assert pool_manager.database_url == "postgresql://user:pass@localhost/test"
        assert pool_manager.min_connections == 5
        assert pool_manager.max_connections == 20
        assert pool_manager.pool is None
        assert pool_manager.is_initialized is False

    @pytest.mark.asyncio
    async def test_should_initialize_connection_pool(self, pool_manager):
        """Test that ConnectionPoolManager can initialize connection pool."""
        # Mock asyncpg pool creation
        with patch('src.optimization.db_optimizer.asyncpg.create_pool', new_callable=AsyncMock) as mock_create_pool:
            mock_pool = AsyncMock()
            mock_create_pool.return_value = mock_pool

            await pool_manager.initialize()

            assert pool_manager.pool is not None
            assert pool_manager.is_initialized is True
            mock_create_pool.assert_called_once()

    @pytest.mark.asyncio
    async def test_should_handle_connection_pool_errors(self, pool_manager):
        """Test that ConnectionPoolManager handles pool creation errors."""
        with patch('src.optimization.db_optimizer.asyncpg.create_pool', new_callable=AsyncMock) as mock_create_pool:
            mock_create_pool.side_effect = Exception("Connection failed")

            with pytest.raises(OptimizationError, match="Failed to initialize connection pool"):
                await pool_manager.initialize()

            assert pool_manager.is_initialized is False

    @pytest.mark.asyncio
    async def test_should_acquire_and_release_connections(self, pool_manager):
        """Test that ConnectionPoolManager can acquire and release connections."""
        # Mock pool
        mock_pool = MagicMock()
        mock_connection = AsyncMock()

        # Create a proper context manager mock
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_connection
        mock_context.__aexit__.return_value = None
        mock_pool.acquire.return_value = mock_context

        pool_manager.pool = mock_pool
        pool_manager.is_initialized = True

        async with pool_manager.acquire_connection() as conn:
            assert conn == mock_connection

        mock_pool.acquire.assert_called_once()

    @pytest.mark.asyncio
    async def test_should_execute_queries_with_connection(self, pool_manager):
        """Test that ConnectionPoolManager can execute queries."""
        # Mock pool and connection
        mock_pool = MagicMock()
        mock_connection = AsyncMock()
        mock_connection.fetch.return_value = [{'id': 1, 'name': 'test'}]

        # Create a proper context manager mock
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_connection
        mock_context.__aexit__.return_value = None
        mock_pool.acquire.return_value = mock_context

        pool_manager.pool = mock_pool
        pool_manager.is_initialized = True

        result = await pool_manager.execute_query("SELECT * FROM test")

        assert result == [{'id': 1, 'name': 'test'}]
        mock_connection.fetch.assert_called_once_with("SELECT * FROM test")

    @pytest.mark.asyncio
    async def test_should_get_pool_statistics(self, pool_manager):
        """Test that ConnectionPoolManager provides pool statistics."""
        # Mock pool with statistics
        mock_pool = MagicMock()
        mock_pool.get_size.return_value = 10
        mock_pool.get_min_size.return_value = 5
        mock_pool.get_max_size.return_value = 20
        mock_pool.get_idle_size.return_value = 3
        pool_manager.pool = mock_pool
        pool_manager.is_initialized = True

        stats = pool_manager.get_pool_stats()

        assert isinstance(stats, dict)
        assert stats['current_size'] == 10
        assert stats['min_size'] == 5
        assert stats['max_size'] == 20
        assert stats['idle_connections'] == 3

    @pytest.mark.asyncio
    async def test_should_close_connection_pool(self, pool_manager):
        """Test that ConnectionPoolManager can close connection pool."""
        mock_pool = AsyncMock()
        pool_manager.pool = mock_pool
        pool_manager.is_initialized = True

        await pool_manager.close()

        assert pool_manager.pool is None
        assert pool_manager.is_initialized is False
        mock_pool.close.assert_called_once()


class TestQueryOptimizer:
    """Test suite for QueryOptimizer class."""

    @pytest.fixture
    def query_optimizer(self):
        """Create QueryOptimizer instance for testing."""
        return QueryOptimizer()

    def test_should_initialize_with_default_settings(self, query_optimizer):
        """Test that QueryOptimizer initializes with default settings."""
        assert query_optimizer.stats is not None
        assert isinstance(query_optimizer.stats, QueryStats)
        assert query_optimizer.query_cache == {}
        assert query_optimizer.slow_query_threshold_ms == 100

    @pytest.mark.asyncio
    async def test_should_analyze_query_plan(self, query_optimizer):
        """Test that QueryOptimizer can analyze query execution plans."""
        # Mock database connection that returns explain plan
        mock_connection = AsyncMock()
        mock_connection.fetch.return_value = [
            {
                'QUERY PLAN': [{
                    'Total Cost': 100.0,
                    'Actual Total Time': 50.0,
                    'Actual Rows': 1000,
                    'Plan Rows': 1000
                }]
            }
        ]

        query = "SELECT * FROM orders WHERE status = 'active'"
        plan = await query_optimizer.analyze_query_plan(query, mock_connection)

        assert isinstance(plan, QueryPlan)
        assert plan.query == query
        assert plan.estimated_cost > 0

    @pytest.mark.asyncio
    async def test_should_optimize_query_with_suggestions(self, query_optimizer):
        """Test that QueryOptimizer provides optimization suggestions."""
        query = "SELECT * FROM orders WHERE status = 'active' AND created_at > '2023-01-01'"

        suggestions = await query_optimizer.optimize_query(query)

        assert isinstance(suggestions, dict)
        assert 'optimized_query' in suggestions
        assert 'recommendations' in suggestions
        assert 'estimated_improvement' in suggestions

    def test_should_identify_missing_indexes(self, query_optimizer):
        """Test that QueryOptimizer identifies missing indexes."""
        slow_queries = [
            "SELECT * FROM orders WHERE user_id = 123",
            "SELECT * FROM orders WHERE status = 'pending'",
            "SELECT * FROM trades WHERE symbol = 'BTCUSDT' AND timestamp > '2023-01-01'"
        ]

        recommendations = query_optimizer.suggest_indexes(slow_queries)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(rec, IndexRecommendation) for rec in recommendations)

    @pytest.mark.asyncio
    async def test_should_cache_query_results(self, query_optimizer):
        """Test that QueryOptimizer caches query results."""
        mock_connection = AsyncMock()
        mock_connection.fetch.return_value = [{'id': 1, 'status': 'active'}]

        query = "SELECT * FROM orders WHERE id = 1"

        # First execution - should hit database
        result1 = await query_optimizer.execute_cached_query(query, mock_connection)
        assert result1 == [{'id': 1, 'status': 'active'}]
        assert query_optimizer.stats.query_cache_misses == 1

        # Second execution - should hit cache
        result2 = await query_optimizer.execute_cached_query(query, mock_connection)
        assert result2 == [{'id': 1, 'status': 'active'}]
        assert query_optimizer.stats.query_cache_hits == 1

        # Database should only be called once
        mock_connection.fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_should_handle_query_timeouts(self, query_optimizer):
        """Test that QueryOptimizer handles query timeouts."""
        mock_connection = AsyncMock()
        mock_connection.fetch.side_effect = asyncio.TimeoutError("Query timeout")

        query = "SELECT * FROM large_table"

        with pytest.raises(OptimizationError, match="Query execution timeout"):
            await query_optimizer.execute_with_timeout(query, mock_connection, timeout=5)

    def test_should_detect_expensive_operations(self, query_optimizer):
        """Test that QueryOptimizer detects expensive operations."""
        expensive_queries = [
            "SELECT * FROM orders",  # Full table scan
            "SELECT * FROM orders o1 JOIN orders o2 ON o1.user_id = o2.user_id",  # Cartesian join
            "SELECT COUNT(*) FROM orders GROUP BY user_id HAVING COUNT(*) > 100"  # Expensive aggregation
        ]

        for query in expensive_queries:
            issues = query_optimizer.detect_query_issues(query)
            assert isinstance(issues, list)
            assert len(issues) > 0

    @pytest.mark.asyncio
    async def test_should_provide_query_rewrite_suggestions(self, query_optimizer):
        """Test that QueryOptimizer provides query rewrite suggestions."""
        original_query = "SELECT * FROM orders WHERE status IN (SELECT status FROM order_statuses WHERE active = true)"

        suggestions = await query_optimizer.suggest_query_rewrites(original_query)

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert all('rewritten_query' in suggestion for suggestion in suggestions)
        assert all('improvement_reason' in suggestion for suggestion in suggestions)

    def test_should_get_optimization_statistics(self, query_optimizer):
        """Test that QueryOptimizer provides optimization statistics."""
        # Simulate some query executions
        query_optimizer.stats.record_query(execution_time_ms=25.0, cache_hit=False)
        query_optimizer.stats.record_query(execution_time_ms=150.0, cache_hit=True)
        query_optimizer.stats.record_query(execution_time_ms=30.0, cache_hit=False)

        stats = query_optimizer.get_statistics()

        assert isinstance(stats, dict)
        assert stats['total_queries'] == 3
        assert stats['slow_queries'] == 1
        assert stats['cache_hit_rate'] == pytest.approx(0.333, rel=1e-2)


class TestDatabaseOptimizer:
    """Test suite for DatabaseOptimizer class."""

    @pytest.fixture
    def db_optimizer(self):
        """Create DatabaseOptimizer instance for testing."""
        return DatabaseOptimizer(
            database_url="postgresql://user:pass@localhost/test",
            enable_query_optimization=True,
            enable_connection_pooling=True
        )

    def test_should_initialize_with_configuration(self, db_optimizer):
        """Test that DatabaseOptimizer initializes with configuration."""
        assert db_optimizer.database_url == "postgresql://user:pass@localhost/test"
        assert db_optimizer.enable_query_optimization is True
        assert db_optimizer.enable_connection_pooling is True
        assert db_optimizer.pool_manager is not None
        assert db_optimizer.query_optimizer is not None

    @pytest.mark.asyncio
    async def test_should_initialize_all_components(self, db_optimizer):
        """Test that DatabaseOptimizer initializes all components."""
        with patch.object(db_optimizer.pool_manager, 'initialize') as mock_init:
            await db_optimizer.initialize()

            assert db_optimizer.is_initialized is True
            mock_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_should_execute_optimized_queries(self, db_optimizer):
        """Test that DatabaseOptimizer executes optimized queries."""
        # Mock components
        db_optimizer.is_initialized = True
        mock_connection = AsyncMock()
        mock_connection.fetch.return_value = [{'id': 1}]

        with patch.object(db_optimizer.pool_manager, 'acquire_connection') as mock_acquire:
            mock_acquire.return_value.__aenter__.return_value = mock_connection

            query = "SELECT * FROM orders WHERE status = 'active'"
            result = await db_optimizer.execute_optimized_query(query)

            assert result == [{'id': 1}]

    @pytest.mark.asyncio
    async def test_should_provide_optimization_recommendations(self, db_optimizer):
        """Test that DatabaseOptimizer provides comprehensive optimization recommendations."""
        db_optimizer.is_initialized = True

        # Mock slow queries data
        slow_queries = [
            "SELECT * FROM orders WHERE user_id = 123",
            "SELECT * FROM trades WHERE symbol = 'BTCUSDT'"
        ]

        with patch.object(db_optimizer.query_optimizer, 'suggest_indexes') as mock_suggest:
            mock_suggest.return_value = [
                IndexRecommendation("orders", ["user_id"], "btree", 0.8),
                IndexRecommendation("trades", ["symbol"], "btree", 0.7)
            ]

            recommendations = await db_optimizer.get_optimization_recommendations(slow_queries)

            assert isinstance(recommendations, dict)
            assert 'index_recommendations' in recommendations
            assert 'query_optimizations' in recommendations
            assert 'performance_metrics' in recommendations

    @pytest.mark.asyncio
    async def test_should_monitor_database_performance(self, db_optimizer):
        """Test that DatabaseOptimizer monitors database performance."""
        db_optimizer.is_initialized = True

        with patch.object(db_optimizer.pool_manager, 'get_pool_stats') as mock_stats:
            mock_stats.return_value = {
                'current_size': 10,
                'idle_connections': 3,
                'max_size': 20
            }

            performance_data = await db_optimizer.monitor_performance()

            assert isinstance(performance_data, dict)
            assert 'connection_pool' in performance_data
            assert 'query_statistics' in performance_data
            assert 'optimization_opportunities' in performance_data

    @pytest.mark.asyncio
    async def test_should_auto_optimize_database(self, db_optimizer):
        """Test that DatabaseOptimizer can perform automatic optimization."""
        db_optimizer.is_initialized = True

        # Mock database analysis
        with patch.object(db_optimizer, 'analyze_database_performance') as mock_analyze:
            mock_analyze.return_value = {
                'slow_queries': ["SELECT * FROM orders"],
                'missing_indexes': ["orders(user_id)"],
                'connection_issues': []
            }

            optimization_result = await db_optimizer.auto_optimize()

            assert isinstance(optimization_result, dict)
            assert 'optimizations_applied' in optimization_result
            assert 'performance_improvement' in optimization_result
            assert 'recommendations' in optimization_result

    @pytest.mark.asyncio
    async def test_should_handle_database_errors_gracefully(self, db_optimizer):
        """Test that DatabaseOptimizer handles database errors gracefully."""
        db_optimizer.is_initialized = True

        with patch.object(db_optimizer.pool_manager, 'execute_query') as mock_execute:
            mock_execute.side_effect = Exception("Database connection error")

            # Should not raise exception but return error info
            result = await db_optimizer.execute_optimized_query("SELECT 1")

            assert result is None or 'error' in str(result).lower()

    def test_should_export_optimization_report(self, db_optimizer):
        """Test that DatabaseOptimizer can export optimization report."""
        # Add some mock statistics
        db_optimizer.query_optimizer.stats.record_query(execution_time_ms=50.0)
        db_optimizer.query_optimizer.stats.record_query(execution_time_ms=120.0)

        report = db_optimizer.export_optimization_report()

        assert isinstance(report, dict)
        assert 'database_configuration' in report
        assert 'performance_statistics' in report
        assert 'optimization_opportunities' in report
        assert 'recommendations' in report

    @pytest.mark.asyncio
    async def test_should_close_all_connections(self, db_optimizer):
        """Test that DatabaseOptimizer properly closes all connections."""
        db_optimizer.is_initialized = True

        with patch.object(db_optimizer.pool_manager, 'close') as mock_close:
            await db_optimizer.close()

            assert db_optimizer.is_initialized is False
            mock_close.assert_called_once()