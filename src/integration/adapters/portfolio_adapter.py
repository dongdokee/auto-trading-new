# src/integration/adapters/portfolio_adapter.py
"""
Portfolio Adapter

Bridges the portfolio management system with the event-driven integration system.
Handles portfolio optimization, rebalancing, and performance attribution.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from decimal import Decimal
import pandas as pd
import numpy as np

from src.integration.events.event_bus import EventBus
from src.integration.events.models import (
    PortfolioEvent, StrategySignalEvent, ExecutionEvent, OrderEvent, SystemEvent,
    EventType, EventPriority
)
from src.integration.events.handlers import BaseEventHandler, HandlerResult
from src.integration.state.manager import StateManager

# Import portfolio management components
from src.portfolio.portfolio_optimizer import PortfolioOptimizer
from src.portfolio.performance_attributor import PerformanceAttributor
from src.portfolio.correlation_analyzer import CorrelationAnalyzer
from src.portfolio.adaptive_allocator import AdaptiveAllocator


class PortfolioAdapter:
    """
    Adapter for the portfolio management system

    Responsibilities:
    - Process strategy signals for portfolio optimization
    - Generate rebalancing orders
    - Track portfolio performance
    - Analyze strategy correlations
    - Manage dynamic allocation adjustments
    """

    def __init__(self,
                 event_bus: EventBus,
                 state_manager: StateManager,
                 rebalance_threshold: float = 0.05):

        self.event_bus = event_bus
        self.state_manager = state_manager
        self.rebalance_threshold = rebalance_threshold

        # Initialize portfolio management components
        self.portfolio_optimizer = PortfolioOptimizer()
        self.performance_attributor = PerformanceAttributor()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.adaptive_allocator = AdaptiveAllocator()

        # Adapter state
        self.is_active = False
        self.current_allocation = {}
        self.target_allocation = {}
        self.strategy_signals = {}
        self.strategy_performance = {}
        self.rebalancing_in_progress = False

        # Performance tracking
        self.optimizations_performed = 0
        self.rebalances_executed = 0
        self.last_optimization = None
        self.last_rebalance = None

        # Logger
        self.logger = logging.getLogger("portfolio_adapter")

        # Register event handlers
        self._register_handlers()

    def _register_handlers(self):
        """Register event handlers with the event bus"""
        signal_handler = PortfolioSignalHandler(self)
        self.event_bus.register_handler(EventType.STRATEGY_SIGNAL, signal_handler)

        execution_handler = PortfolioExecutionHandler(self)
        self.event_bus.register_handler(EventType.EXECUTION, execution_handler)

        portfolio_handler = PortfolioPortfolioHandler(self)
        self.event_bus.register_handler(EventType.PORTFOLIO, portfolio_handler)

        system_handler = PortfolioSystemHandler(self)
        self.event_bus.register_handler(EventType.SYSTEM, system_handler)

    async def start(self):
        """Start the portfolio adapter"""
        self.is_active = True

        # Initialize portfolio state
        await self._initialize_portfolio_state()

        # Send startup event
        startup_event = SystemEvent(
            source_component="portfolio_adapter",
            system_action="START",
            status="RUNNING",
            message="Portfolio adapter started"
        )
        await self.event_bus.publish(startup_event)

        self.logger.info("Portfolio adapter started")

    async def stop(self):
        """Stop the portfolio adapter"""
        self.is_active = False

        # Send shutdown event
        shutdown_event = SystemEvent(
            source_component="portfolio_adapter",
            system_action="STOP",
            status="STOPPED",
            message="Portfolio adapter stopped"
        )
        await self.event_bus.publish(shutdown_event)

        self.logger.info("Portfolio adapter stopped")

    async def process_strategy_signal(self, signal_event: StrategySignalEvent):
        """Process strategy signal for portfolio optimization"""
        if not self.is_active:
            return

        try:
            # Store strategy signal
            strategy_key = f"{signal_event.strategy_name}_{signal_event.symbol}"
            self.strategy_signals[strategy_key] = {
                'strategy': signal_event.strategy_name,
                'symbol': signal_event.symbol,
                'action': signal_event.action,
                'strength': signal_event.strength,
                'confidence': signal_event.confidence,
                'timestamp': signal_event.timestamp
            }

            # Update correlation analysis
            await self._update_correlation_analysis(signal_event)

            # Check if portfolio optimization is needed
            await self._check_portfolio_optimization()

        except Exception as e:
            self.logger.error(f"Error processing strategy signal: {e}")

    async def perform_portfolio_optimization(self):
        """Perform portfolio optimization based on current signals"""
        if not self.is_active or self.rebalancing_in_progress:
            return

        try:
            self.optimizations_performed += 1
            self.last_optimization = datetime.now()

            # Prepare strategy returns data
            returns_data = await self._prepare_returns_data()

            if returns_data.empty:
                self.logger.warning("No returns data available for optimization")
                return

            # Perform Markowitz optimization
            optimization_result = self.portfolio_optimizer.optimize_weights(
                returns=returns_data,
                constraints={
                    'min_weight': 0.05,  # Minimum 5% allocation
                    'max_weight': 0.60,  # Maximum 60% allocation
                    'max_strategies': 4  # Maximum 4 strategies
                },
                objective='max_sharpe'
            )

            # Get adaptive allocation adjustments
            allocation_update = await self._get_adaptive_allocation(optimization_result.weights)

            # Update target allocation
            self.target_allocation = allocation_update.new_weights

            # Check if rebalancing is needed
            rebalance_needed = await self._check_rebalancing_needed()

            if rebalance_needed:
                await self._generate_rebalancing_orders()

            # Generate portfolio optimization event
            portfolio_event = PortfolioEvent(
                source_component="portfolio_adapter",
                action="OPTIMIZE",
                current_weights=self.current_allocation.copy(),
                target_weights=self.target_allocation.copy(),
                optimization_result={
                    'expected_return': float(optimization_result.expected_return),
                    'volatility': float(optimization_result.volatility),
                    'sharpe_ratio': float(optimization_result.sharpe_ratio),
                    'optimization_success': optimization_result.success
                }
            )

            await self.event_bus.publish(portfolio_event)

            self.logger.info(f"Portfolio optimization completed. Sharpe ratio: {optimization_result.sharpe_ratio:.3f}")

        except Exception as e:
            self.logger.error(f"Error performing portfolio optimization: {e}")

    async def process_execution_result(self, execution_event: ExecutionEvent):
        """Process execution result for portfolio tracking"""
        try:
            if execution_event.status == "FILLED":
                # Update current allocation based on execution
                await self._update_current_allocation(execution_event)

                # Update performance attribution
                await self._update_performance_attribution(execution_event)

                # Check if rebalancing is complete
                if self.rebalancing_in_progress:
                    await self._check_rebalancing_completion()

        except Exception as e:
            self.logger.error(f"Error processing execution result: {e}")

    async def _initialize_portfolio_state(self):
        """Initialize portfolio state from current positions"""
        try:
            positions = await self.state_manager.get_positions()

            # Calculate current allocation from positions
            total_value = 0.0
            position_values = {}

            for symbol, position in positions.items():
                size = float(position.get('size', 0))
                price = float(position.get('current_price', 0))
                value = size * price
                position_values[symbol] = value
                total_value += value

            # Calculate allocation percentages
            if total_value > 0:
                for symbol, value in position_values.items():
                    self.current_allocation[symbol] = value / total_value
            else:
                self.current_allocation = {}

            self.logger.info(f"Initialized portfolio with {len(self.current_allocation)} positions")

        except Exception as e:
            self.logger.error(f"Error initializing portfolio state: {e}")

    async def _prepare_returns_data(self) -> pd.DataFrame:
        """Prepare historical returns data for optimization"""
        try:
            # In a real implementation, this would fetch historical strategy returns
            # For now, we'll create synthetic data based on recent signals

            returns_dict = {}
            strategies = set(signal['strategy'] for signal in self.strategy_signals.values())

            for strategy in strategies:
                # Generate synthetic returns based on signal strength and confidence
                strategy_signals = [s for s in self.strategy_signals.values() if s['strategy'] == strategy]

                if strategy_signals:
                    avg_strength = np.mean([s['strength'] for s in strategy_signals])
                    avg_confidence = np.mean([s['confidence'] for s in strategy_signals])

                    # Create synthetic returns (simplified)
                    base_return = avg_strength * avg_confidence * 0.1  # Base monthly return
                    volatility = 0.15 * (1 - avg_confidence)  # Lower confidence = higher volatility

                    # Generate 60 daily returns
                    daily_returns = np.random.normal(base_return/30, volatility/np.sqrt(252), 60)
                    returns_dict[strategy] = daily_returns

            if returns_dict:
                return pd.DataFrame(returns_dict)
            else:
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Error preparing returns data: {e}")
            return pd.DataFrame()

    async def _get_adaptive_allocation(self, base_weights: Dict[str, float]) -> Any:
        """Get adaptive allocation adjustments"""
        try:
            # Add performance data to adaptive allocator
            for strategy, weight in base_weights.items():
                if strategy in self.strategy_performance:
                    performance_data = self.strategy_performance[strategy]
                    self.adaptive_allocator.add_strategy_performance(strategy, performance_data)

            # Calculate allocation update
            allocation_update = self.adaptive_allocator.calculate_allocation_update(
                current_allocation=base_weights,
                performance_threshold=0.05,  # 5% performance threshold
                max_allocation_change=0.10   # Maximum 10% allocation change
            )

            return allocation_update

        except Exception as e:
            self.logger.error(f"Error getting adaptive allocation: {e}")
            # Return simple allocation update as fallback
            class SimpleAllocationUpdate:
                def __init__(self, weights):
                    self.new_weights = weights
                    self.rebalance_needed = True
                    self.transaction_costs = {}

            return SimpleAllocationUpdate(base_weights)

    async def _check_rebalancing_needed(self) -> bool:
        """Check if portfolio rebalancing is needed"""
        try:
            if not self.current_allocation or not self.target_allocation:
                return False

            # Calculate allocation differences
            max_diff = 0.0
            for strategy, target_weight in self.target_allocation.items():
                current_weight = self.current_allocation.get(strategy, 0.0)
                diff = abs(target_weight - current_weight)
                max_diff = max(max_diff, diff)

            return max_diff > self.rebalance_threshold

        except Exception as e:
            self.logger.error(f"Error checking rebalancing needed: {e}")
            return False

    async def _generate_rebalancing_orders(self):
        """Generate orders for portfolio rebalancing"""
        try:
            self.rebalancing_in_progress = True
            self.rebalances_executed += 1
            self.last_rebalance = datetime.now()

            rebalance_orders = []
            portfolio_state = await self.state_manager.get_portfolio_state()
            total_equity = float(portfolio_state.get('equity', 100000))  # Default for testing

            for strategy, target_weight in self.target_allocation.items():
                current_weight = self.current_allocation.get(strategy, 0.0)
                weight_diff = target_weight - current_weight

                if abs(weight_diff) > 0.01:  # Only rebalance if difference > 1%
                    # Calculate order size
                    target_value = total_equity * target_weight
                    current_value = total_equity * current_weight
                    order_value = target_value - current_value

                    # Determine order side and size
                    if order_value > 0:
                        side = "BUY"
                        size = abs(order_value) / 50000  # Assuming $50,000 per unit (simplified)
                    else:
                        side = "SELL"
                        size = abs(order_value) / 50000

                    # Create order event
                    order_event = OrderEvent(
                        source_component="portfolio_adapter",
                        action="CREATE",
                        symbol=f"{strategy}USDT",  # Simplified symbol mapping
                        side=side,
                        size=Decimal(str(size)),
                        order_type="MARKET",
                        urgency="MEDIUM",
                        source_signal="portfolio_rebalance"
                    )

                    rebalance_orders.append(order_event)

            # Publish rebalancing orders
            for order in rebalance_orders:
                await self.event_bus.publish(order)

            # Generate portfolio rebalancing event
            rebalance_event = PortfolioEvent(
                source_component="portfolio_adapter",
                action="REBALANCE",
                current_weights=self.current_allocation.copy(),
                target_weights=self.target_allocation.copy(),
                rebalance_orders=[{
                    'symbol': order.symbol,
                    'side': order.side,
                    'size': float(order.size)
                } for order in rebalance_orders]
            )

            await self.event_bus.publish(rebalance_event)

            self.logger.info(f"Generated {len(rebalance_orders)} rebalancing orders")

        except Exception as e:
            self.logger.error(f"Error generating rebalancing orders: {e}")
            self.rebalancing_in_progress = False

    async def _update_correlation_analysis(self, signal_event: StrategySignalEvent):
        """Update correlation analysis with new signal"""
        try:
            # Add strategy signal to correlation analyzer
            strategy_returns = self._generate_strategy_returns(signal_event)
            self.correlation_analyzer.add_strategy_returns(
                signal_event.strategy_name,
                strategy_returns
            )

        except Exception as e:
            self.logger.error(f"Error updating correlation analysis: {e}")

    def _generate_strategy_returns(self, signal_event: StrategySignalEvent) -> pd.Series:
        """Generate synthetic strategy returns for correlation analysis"""
        # This is a simplified implementation
        # In practice, you would use actual historical strategy returns
        base_return = signal_event.strength * signal_event.confidence * 0.001
        returns = pd.Series(np.random.normal(base_return, 0.02, 30))
        return returns

    async def _update_current_allocation(self, execution_event: ExecutionEvent):
        """Update current allocation based on execution"""
        try:
            # This is a simplified update
            # In practice, you would recalculate allocation based on all positions
            symbol = execution_event.symbol
            strategy = symbol.replace('USDT', '')  # Simplified mapping

            if execution_event.side == "BUY":
                self.current_allocation[strategy] = self.current_allocation.get(strategy, 0) + 0.05
            else:
                self.current_allocation[strategy] = max(0, self.current_allocation.get(strategy, 0) - 0.05)

            # Normalize allocations
            total_weight = sum(self.current_allocation.values())
            if total_weight > 0:
                for strategy in self.current_allocation:
                    self.current_allocation[strategy] /= total_weight

        except Exception as e:
            self.logger.error(f"Error updating current allocation: {e}")

    async def _update_performance_attribution(self, execution_event: ExecutionEvent):
        """Update performance attribution data"""
        try:
            symbol = execution_event.symbol
            strategy = symbol.replace('USDT', '')

            # Calculate trade return (simplified)
            if execution_event.executed_qty > 0:
                trade_return = 0.001 if execution_event.side == "BUY" else -0.001  # Simplified

                if strategy not in self.strategy_performance:
                    self.strategy_performance[strategy] = {
                        'total_return': 0.0,
                        'trade_count': 0,
                        'win_rate': 0.0
                    }

                perf = self.strategy_performance[strategy]
                perf['total_return'] += trade_return
                perf['trade_count'] += 1

                # Add to performance attributor
                strategy_data = {
                    'returns': pd.Series([trade_return]),
                    'weights': pd.Series([self.current_allocation.get(strategy, 0)])
                }
                self.performance_attributor.add_strategy_data(strategy, strategy_data)

        except Exception as e:
            self.logger.error(f"Error updating performance attribution: {e}")

    async def _check_portfolio_optimization(self):
        """Check if portfolio optimization should be triggered"""
        try:
            # Check if enough new signals have been received
            if len(self.strategy_signals) >= 4:  # At least 4 signals
                last_optimization_time = self.last_optimization or datetime.min

                # Optimize if last optimization was more than 5 minutes ago
                if datetime.now() - last_optimization_time > timedelta(minutes=5):
                    await self.perform_portfolio_optimization()

        except Exception as e:
            self.logger.error(f"Error checking portfolio optimization: {e}")

    async def _check_rebalancing_completion(self):
        """Check if rebalancing is complete"""
        try:
            # Check if allocation differences are within tolerance
            if await self._check_rebalancing_needed():
                return  # Still need rebalancing

            # Rebalancing complete
            self.rebalancing_in_progress = False

            completion_event = PortfolioEvent(
                source_component="portfolio_adapter",
                action="REBALANCE_COMPLETE",
                current_weights=self.current_allocation.copy(),
                target_weights=self.target_allocation.copy()
            )

            await self.event_bus.publish(completion_event)

            self.logger.info("Portfolio rebalancing completed")

        except Exception as e:
            self.logger.error(f"Error checking rebalancing completion: {e}")

    def get_adapter_metrics(self) -> Dict[str, Any]:
        """Get portfolio adapter metrics"""
        return {
            'is_active': self.is_active,
            'optimizations_performed': self.optimizations_performed,
            'rebalances_executed': self.rebalances_executed,
            'rebalancing_in_progress': self.rebalancing_in_progress,
            'last_optimization': self.last_optimization.isoformat() if self.last_optimization else None,
            'last_rebalance': self.last_rebalance.isoformat() if self.last_rebalance else None,
            'current_allocation': self.current_allocation.copy(),
            'target_allocation': self.target_allocation.copy(),
            'strategy_signals_count': len(self.strategy_signals),
            'strategies_tracked': len(set(s['strategy'] for s in self.strategy_signals.values()))
        }


# Event handlers for portfolio adapter

class PortfolioSignalHandler(BaseEventHandler):
    """Handler for strategy signal events in portfolio adapter"""

    def __init__(self, adapter: PortfolioAdapter):
        super().__init__("portfolio_signal_handler")
        self.adapter = adapter

    async def handle_event(self, event) -> HandlerResult:
        """Handle strategy signal event"""
        try:
            await self.adapter.process_strategy_signal(event)
            return HandlerResult(success=True)

        except Exception as e:
            self.logger.error(f"Error handling strategy signal event: {e}")
            return HandlerResult(
                success=False,
                message=f"Strategy signal processing failed: {str(e)}"
            )


class PortfolioExecutionHandler(BaseEventHandler):
    """Handler for execution events in portfolio adapter"""

    def __init__(self, adapter: PortfolioAdapter):
        super().__init__("portfolio_execution_handler")
        self.adapter = adapter

    async def handle_event(self, event) -> HandlerResult:
        """Handle execution event"""
        try:
            await self.adapter.process_execution_result(event)
            return HandlerResult(success=True)

        except Exception as e:
            self.logger.error(f"Error handling execution event: {e}")
            return HandlerResult(
                success=False,
                message=f"Execution processing failed: {str(e)}"
            )


class PortfolioPortfolioHandler(BaseEventHandler):
    """Handler for portfolio events in portfolio adapter"""

    def __init__(self, adapter: PortfolioAdapter):
        super().__init__("portfolio_portfolio_handler")
        self.adapter = adapter

    async def handle_event(self, event) -> HandlerResult:
        """Handle portfolio event"""
        try:
            if event.action == "OPTIMIZE":
                await self.adapter.perform_portfolio_optimization()

            return HandlerResult(success=True)

        except Exception as e:
            self.logger.error(f"Error handling portfolio event: {e}")
            return HandlerResult(
                success=False,
                message=f"Portfolio event processing failed: {str(e)}"
            )


class PortfolioSystemHandler(BaseEventHandler):
    """Handler for system events in portfolio adapter"""

    def __init__(self, adapter: PortfolioAdapter):
        super().__init__("portfolio_system_handler")
        self.adapter = adapter

    async def handle_event(self, event) -> HandlerResult:
        """Handle system event"""
        try:
            if event.system_action == "PAUSE":
                self.adapter.is_active = False
                self.logger.info("Portfolio adapter paused")

            elif event.system_action == "RESUME":
                self.adapter.is_active = True
                self.logger.info("Portfolio adapter resumed")

            elif event.system_action == "STOP":
                await self.adapter.stop()

            return HandlerResult(success=True)

        except Exception as e:
            self.logger.error(f"Error handling system event: {e}")
            return HandlerResult(
                success=False,
                message=f"System event processing failed: {str(e)}"
            )