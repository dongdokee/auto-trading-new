# src/integration/adapters/risk_adapter.py
"""
Risk Adapter

Bridges the risk management system with the event-driven integration system.
Monitors risk metrics, validates orders, and generates risk alerts.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from decimal import Decimal

from src.integration.events.event_bus import EventBus
from src.integration.events.models import (
    RiskEvent, OrderEvent, ExecutionEvent, PortfolioEvent, SystemEvent,
    EventType, EventPriority
)
from src.integration.events.handlers import BaseEventHandler, HandlerResult
from src.integration.state.manager import StateManager

# Import risk management components
from src.risk_management.risk_management import RiskController
from src.risk_management.position_sizing import PositionSizer
from src.risk_management.position_management import PositionManager


class RiskAdapter:
    """
    Adapter for the risk management system

    Responsibilities:
    - Monitor portfolio risk metrics
    - Validate trading orders against risk limits
    - Generate risk alerts and events
    - Manage position risk and leverage
    """

    def __init__(self,
                 event_bus: EventBus,
                 state_manager: StateManager,
                 initial_capital: float = 10000.0):

        self.event_bus = event_bus
        self.state_manager = state_manager

        # Initialize risk management components
        self.risk_controller = RiskController(initial_capital_usdt=initial_capital)
        self.position_sizer = PositionSizer(self.risk_controller)
        self.position_manager = PositionManager(self.risk_controller)

        # Adapter state
        self.is_active = False
        self.last_risk_check = None
        self.risk_violations = []
        self.order_validations = 0
        self.orders_rejected = 0

        # Risk monitoring
        self.risk_check_interval = 30  # seconds
        self.monitoring_task: Optional[asyncio.Task] = None

        # Logger
        self.logger = logging.getLogger("risk_adapter")

        # Register event handlers
        self._register_handlers()

    def _register_handlers(self):
        """Register event handlers with the event bus"""
        order_handler = RiskOrderHandler(self)
        self.event_bus.register_handler(EventType.ORDER, order_handler)

        execution_handler = RiskExecutionHandler(self)
        self.event_bus.register_handler(EventType.EXECUTION, execution_handler)

        portfolio_handler = RiskPortfolioHandler(self)
        self.event_bus.register_handler(EventType.PORTFOLIO, portfolio_handler)

        system_handler = RiskSystemHandler(self)
        self.event_bus.register_handler(EventType.SYSTEM, system_handler)

    async def start(self):
        """Start the risk adapter"""
        self.is_active = True

        # Start risk monitoring task
        self.monitoring_task = asyncio.create_task(self._risk_monitoring_loop())

        # Send startup event
        startup_event = SystemEvent(
            source_component="risk_adapter",
            system_action="START",
            status="RUNNING",
            message="Risk adapter started"
        )
        await self.event_bus.publish(startup_event)

        self.logger.info("Risk adapter started")

    async def stop(self):
        """Stop the risk adapter"""
        self.is_active = False

        # Stop monitoring task
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        # Send shutdown event
        shutdown_event = SystemEvent(
            source_component="risk_adapter",
            system_action="STOP",
            status="STOPPED",
            message="Risk adapter stopped"
        )
        await self.event_bus.publish(shutdown_event)

        self.logger.info("Risk adapter stopped")

    async def validate_order(self, order_event: OrderEvent) -> bool:
        """Validate order against risk limits"""
        if not self.is_active:
            return False

        try:
            self.order_validations += 1

            # Get current portfolio state
            portfolio_state = await self.state_manager.get_portfolio_state()
            if not portfolio_state:
                self.logger.warning("No portfolio state available for risk validation")
                return False

            # Prepare signal for position sizing
            signal = {
                'symbol': order_event.symbol,
                'side': order_event.side,
                'strength': 0.8,  # Default strength for validation
                'confidence': 0.7  # Default confidence for validation
            }

            # Prepare market state (simplified)
            market_state = {
                'price': float(order_event.price) if order_event.price else 50000.0,
                'atr': 1000.0,  # Default ATR
                'daily_volatility': 0.05,
                'regime': 'NEUTRAL',
                'min_notional': 10.0,
                'lot_size': 0.001,
                'symbol_leverage': 10
            }

            # Calculate recommended position size
            recommended_size = self.position_sizer.calculate_position_size(
                signal=signal,
                market_state=market_state,
                portfolio_state=portfolio_state
            )

            # Check if requested size exceeds recommendation
            if float(order_event.size) > recommended_size * 1.1:  # 10% tolerance
                await self._generate_risk_event(
                    "POSITION_SIZE_EXCEEDED",
                    "WARNING",
                    order_event.symbol,
                    {
                        'requested_size': float(order_event.size),
                        'recommended_size': recommended_size,
                        'order_id': order_event.order_id
                    }
                )
                self.orders_rejected += 1
                return False

            # Check VaR limits
            var_violations = self.risk_controller.check_var_limit(portfolio_state)
            if var_violations:
                await self._generate_risk_event(
                    "VAR_LIMIT_BREACH",
                    "CRITICAL",
                    order_event.symbol,
                    {
                        'violations': var_violations,
                        'order_id': order_event.order_id
                    }
                )
                self.orders_rejected += 1
                return False

            # Check leverage limits
            leverage_violations = self.risk_controller.check_leverage_limit(portfolio_state)
            if leverage_violations:
                await self._generate_risk_event(
                    "LEVERAGE_LIMIT_BREACH",
                    "CRITICAL",
                    order_event.symbol,
                    {
                        'violations': leverage_violations,
                        'order_id': order_event.order_id
                    }
                )
                self.orders_rejected += 1
                return False

            self.logger.debug(f"Order validation passed for {order_event.symbol}")
            return True

        except Exception as e:
            self.logger.error(f"Error validating order: {e}")
            self.orders_rejected += 1
            return False

    async def process_execution(self, execution_event: ExecutionEvent):
        """Process execution and update risk metrics"""
        try:
            # Update position in position manager
            if execution_event.status == "FILLED":
                # Check if position exists or create new one
                positions = await self.state_manager.get_positions()

                position_data = {
                    'side': execution_event.side,
                    'size': float(execution_event.executed_qty),
                    'entry_price': float(execution_event.avg_price),
                    'current_price': float(execution_event.avg_price),
                    'leverage': 5.0,  # Default leverage
                    'margin': float(execution_event.executed_qty) * float(execution_event.avg_price) / 5.0
                }

                # Update position in state manager
                await self.state_manager.update_position(
                    execution_event.symbol,
                    position_data
                )

                # Update portfolio risk metrics
                await self._update_risk_metrics()

                self.logger.info(f"Processed execution for {execution_event.symbol}: "
                               f"{execution_event.executed_qty} @ {execution_event.avg_price}")

        except Exception as e:
            self.logger.error(f"Error processing execution: {e}")

    async def _risk_monitoring_loop(self):
        """Background risk monitoring loop"""
        while self.is_active:
            try:
                await self._perform_risk_check()
                await asyncio.sleep(self.risk_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in risk monitoring loop: {e}")
                await asyncio.sleep(10)

    async def _perform_risk_check(self):
        """Perform comprehensive risk check"""
        try:
            # Get current portfolio state
            portfolio_state = await self.state_manager.get_portfolio_state()
            if not portfolio_state:
                return

            # Check VaR limits
            var_violations = self.risk_controller.check_var_limit(portfolio_state)
            if var_violations:
                await self._generate_risk_event(
                    "VAR_LIMIT_BREACH",
                    "WARNING",
                    None,
                    {'violations': var_violations}
                )

            # Check leverage limits
            leverage_violations = self.risk_controller.check_leverage_limit(portfolio_state)
            if leverage_violations:
                await self._generate_risk_event(
                    "LEVERAGE_LIMIT_BREACH",
                    "WARNING",
                    None,
                    {'violations': leverage_violations}
                )

            # Check drawdown
            current_equity = float(portfolio_state.get('equity', 0))
            if current_equity > 0:
                drawdown = self.risk_controller.update_drawdown(current_equity)

                if drawdown > 0.05:  # 5% drawdown warning
                    severity = "CRITICAL" if drawdown > 0.10 else "WARNING"
                    await self._generate_risk_event(
                        "DRAWDOWN_WARNING",
                        severity,
                        None,
                        {'current_drawdown_pct': drawdown * 100}
                    )

            self.last_risk_check = datetime.now()

            # Update state manager with risk metrics
            await self._update_risk_metrics()

        except Exception as e:
            self.logger.error(f"Error performing risk check: {e}")

    async def _update_risk_metrics(self):
        """Update risk metrics in state manager"""
        try:
            portfolio_state = await self.state_manager.get_portfolio_state()
            if not portfolio_state:
                return

            # Calculate risk metrics
            risk_metrics = {
                'var_daily_usdt': portfolio_state.get('current_var_usdt', 0),
                'leverage_ratio': self._calculate_portfolio_leverage(portfolio_state),
                'last_risk_check': datetime.now().isoformat()
            }

            # Update state manager
            await self.state_manager.update_risk_metrics(risk_metrics)

        except Exception as e:
            self.logger.error(f"Error updating risk metrics: {e}")

    def _calculate_portfolio_leverage(self, portfolio_state: Dict) -> float:
        """Calculate portfolio-wide leverage"""
        try:
            total_notional = 0.0
            total_margin = 0.0

            positions = portfolio_state.get('positions', {})
            for symbol, position in positions.items():
                size = float(position.get('size', 0))
                price = float(position.get('current_price', 0))
                leverage = float(position.get('leverage', 1))

                position_notional = size * price
                position_margin = position_notional / leverage

                total_notional += position_notional
                total_margin += position_margin

            return total_notional / total_margin if total_margin > 0 else 1.0

        except Exception as e:
            self.logger.error(f"Error calculating portfolio leverage: {e}")
            return 1.0

    async def _generate_risk_event(self,
                                 risk_type: str,
                                 severity: str,
                                 symbol: Optional[str],
                                 risk_data: Dict[str, Any]):
        """Generate and publish risk event"""
        try:
            risk_event = RiskEvent(
                source_component="risk_adapter",
                risk_type=risk_type,
                severity=severity,
                symbol=symbol,
                risk_metrics=risk_data,
                priority=EventPriority.CRITICAL if severity == "CRITICAL" else EventPriority.HIGH
            )

            await self.event_bus.publish(risk_event)

            # Store violation in history
            self.risk_violations.append({
                'timestamp': datetime.now(),
                'risk_type': risk_type,
                'severity': severity,
                'symbol': symbol,
                'data': risk_data
            })

            # Limit history size
            if len(self.risk_violations) > 1000:
                self.risk_violations = self.risk_violations[-1000:]

            self.logger.warning(f"Risk event generated: {risk_type} ({severity}) for {symbol}")

        except Exception as e:
            self.logger.error(f"Error generating risk event: {e}")

    def get_adapter_metrics(self) -> Dict[str, Any]:
        """Get risk adapter metrics"""
        return {
            'is_active': self.is_active,
            'order_validations': self.order_validations,
            'orders_rejected': self.orders_rejected,
            'rejection_rate_pct': (self.orders_rejected / self.order_validations * 100)
                                  if self.order_validations > 0 else 0.0,
            'last_risk_check': self.last_risk_check.isoformat() if self.last_risk_check else None,
            'risk_violations_count': len(self.risk_violations),
            'risk_controller_limits': self.risk_controller.risk_limits
        }


# Event handlers for risk adapter

class RiskOrderHandler(BaseEventHandler):
    """Handler for order events in risk adapter"""

    def __init__(self, adapter: RiskAdapter):
        super().__init__("risk_order_handler")
        self.adapter = adapter

    async def handle_event(self, event) -> HandlerResult:
        """Handle order event for risk validation"""
        try:
            if event.action == "CREATE":
                is_valid = await self.adapter.validate_order(event)

                if not is_valid:
                    return HandlerResult(
                        success=False,
                        message="Order rejected due to risk limits"
                    )

            return HandlerResult(success=True)

        except Exception as e:
            self.logger.error(f"Error handling order event: {e}")
            return HandlerResult(
                success=False,
                message=f"Order risk validation failed: {str(e)}"
            )


class RiskExecutionHandler(BaseEventHandler):
    """Handler for execution events in risk adapter"""

    def __init__(self, adapter: RiskAdapter):
        super().__init__("risk_execution_handler")
        self.adapter = adapter

    async def handle_event(self, event) -> HandlerResult:
        """Handle execution event for risk tracking"""
        try:
            await self.adapter.process_execution(event)
            return HandlerResult(success=True)

        except Exception as e:
            self.logger.error(f"Error handling execution event: {e}")
            return HandlerResult(
                success=False,
                message=f"Execution risk processing failed: {str(e)}"
            )


class RiskPortfolioHandler(BaseEventHandler):
    """Handler for portfolio events in risk adapter"""

    def __init__(self, adapter: RiskAdapter):
        super().__init__("risk_portfolio_handler")
        self.adapter = adapter

    async def handle_event(self, event) -> HandlerResult:
        """Handle portfolio event for risk analysis"""
        try:
            # Portfolio events trigger risk metric updates
            await self.adapter._update_risk_metrics()
            return HandlerResult(success=True)

        except Exception as e:
            self.logger.error(f"Error handling portfolio event: {e}")
            return HandlerResult(
                success=False,
                message=f"Portfolio risk analysis failed: {str(e)}"
            )


class RiskSystemHandler(BaseEventHandler):
    """Handler for system events in risk adapter"""

    def __init__(self, adapter: RiskAdapter):
        super().__init__("risk_system_handler")
        self.adapter = adapter

    async def handle_event(self, event) -> HandlerResult:
        """Handle system event"""
        try:
            if event.system_action == "PAUSE":
                self.adapter.is_active = False
                self.logger.info("Risk adapter paused")

            elif event.system_action == "RESUME":
                self.adapter.is_active = True
                self.logger.info("Risk adapter resumed")

            elif event.system_action == "STOP":
                await self.adapter.stop()

            return HandlerResult(success=True)

        except Exception as e:
            self.logger.error(f"Error handling system event: {e}")
            return HandlerResult(
                success=False,
                message=f"System event processing failed: {str(e)}"
            )