# src/integration/state/manager.py
"""
State Manager

Centralized state management for the entire trading system.
Provides persistence, recovery, and synchronization capabilities.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from decimal import Decimal
import uuid

from .positions import PositionTracker


@dataclass
class SystemState:
    """Complete system state snapshot"""
    timestamp: datetime
    state_id: str
    trading_active: bool
    component_status: Dict[str, str]
    portfolio_state: Dict[str, Any]
    position_state: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    active_orders: List[Dict[str, Any]]
    active_signals: List[Dict[str, Any]]


class StateManager:
    """
    Centralized state management for the trading system

    Features:
    - Real-time state tracking
    - State persistence
    - State recovery
    - Component synchronization
    - Historical state snapshots
    """

    def __init__(self,
                 snapshot_interval_seconds: int = 60,
                 max_snapshots: int = 1440,  # 24 hours of minute snapshots
                 enable_persistence: bool = True):

        self.snapshot_interval = snapshot_interval_seconds
        self.max_snapshots = max_snapshots
        self.enable_persistence = enable_persistence

        # Current state
        self.current_state: Optional[SystemState] = None
        self.state_version = 0

        # State history
        self.state_snapshots: List[SystemState] = []

        # Component states
        self.component_states: Dict[str, Dict[str, Any]] = {}
        self.component_last_update: Dict[str, datetime] = {}

        # Position tracking
        self.position_tracker = PositionTracker()

        # Portfolio state
        self.portfolio_state = {
            'equity': Decimal('0'),
            'margin_used': Decimal('0'),
            'margin_available': Decimal('0'),
            'unrealized_pnl': Decimal('0'),
            'realized_pnl': Decimal('0'),
            'positions': {},
            'balances': {},
            'risk_metrics': {}
        }

        # Risk metrics
        self.risk_metrics = {
            'var_daily_usdt': Decimal('0'),
            'cvar_daily_usdt': Decimal('0'),
            'current_drawdown_pct': 0.0,
            'max_drawdown_pct': 0.0,
            'leverage_ratio': 1.0,
            'concentration_risk': 0.0,
            'correlation_risk': 0.0
        }

        # Performance metrics
        self.performance_metrics = {
            'total_return_pct': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown_pct': 0.0,
            'win_rate_pct': 0.0,
            'profit_factor': 0.0,
            'trades_count': 0,
            'avg_trade_return_pct': 0.0
        }

        # Active orders and signals
        self.active_orders: Dict[str, Dict[str, Any]] = {}
        self.active_signals: Dict[str, Dict[str, Any]] = {}

        # Background tasks
        self.snapshot_task: Optional[asyncio.Task] = None
        self.is_running = False

        # State change callbacks
        self.state_change_callbacks: List[callable] = []

        # Logger
        self.logger = logging.getLogger("state_manager")

    async def start(self):
        """Start the state manager"""
        if self.is_running:
            self.logger.warning("State manager is already running")
            return

        self.is_running = True

        # Initialize state
        await self._initialize_state()

        # Start background tasks
        if self.enable_persistence:
            self.snapshot_task = asyncio.create_task(self._snapshot_loop())

        self.logger.info("State manager started")

    async def stop(self):
        """Stop the state manager"""
        if not self.is_running:
            self.logger.warning("State manager is not running")
            return

        self.is_running = False

        # Stop background tasks
        if self.snapshot_task:
            self.snapshot_task.cancel()
            try:
                await self.snapshot_task
            except asyncio.CancelledError:
                pass

        # Take final snapshot
        if self.enable_persistence:
            await self._take_snapshot()

        self.logger.info("State manager stopped")

    async def update_component_state(self, component_name: str, state_data: Dict[str, Any]):
        """Update state for a specific component"""
        self.component_states[component_name] = state_data
        self.component_last_update[component_name] = datetime.now()
        self.state_version += 1

        self.logger.debug(f"Updated state for component: {component_name}")

        # Notify callbacks
        await self._notify_state_change("component_update", component_name, state_data)

    async def update_portfolio_state(self, portfolio_data: Dict[str, Any]):
        """Update portfolio state"""
        # Convert Decimal strings to Decimal objects
        for key, value in portfolio_data.items():
            if isinstance(value, str) and key in ['equity', 'margin_used', 'margin_available',
                                                 'unrealized_pnl', 'realized_pnl']:
                try:
                    portfolio_data[key] = Decimal(value)
                except:
                    pass

        self.portfolio_state.update(portfolio_data)
        self.state_version += 1

        self.logger.debug("Updated portfolio state")

        # Notify callbacks
        await self._notify_state_change("portfolio_update", "portfolio", portfolio_data)

    async def update_position(self, symbol: str, position_data: Dict[str, Any]):
        """Update position data"""
        await self.position_tracker.update_position(symbol, position_data)

        # Update portfolio state with new position data
        self.portfolio_state['positions'] = await self.position_tracker.get_all_positions()
        self.state_version += 1

        self.logger.debug(f"Updated position for {symbol}")

        # Notify callbacks
        await self._notify_state_change("position_update", symbol, position_data)

    async def update_risk_metrics(self, risk_data: Dict[str, Any]):
        """Update risk metrics"""
        # Convert Decimal strings
        for key, value in risk_data.items():
            if isinstance(value, str) and 'usdt' in key:
                try:
                    risk_data[key] = Decimal(value)
                except:
                    pass

        self.risk_metrics.update(risk_data)
        self.state_version += 1

        self.logger.debug("Updated risk metrics")

        # Notify callbacks
        await self._notify_state_change("risk_update", "risk", risk_data)

    async def update_performance_metrics(self, performance_data: Dict[str, Any]):
        """Update performance metrics"""
        self.performance_metrics.update(performance_data)
        self.state_version += 1

        self.logger.debug("Updated performance metrics")

        # Notify callbacks
        await self._notify_state_change("performance_update", "performance", performance_data)

    async def add_active_order(self, order_id: str, order_data: Dict[str, Any]):
        """Add an active order"""
        self.active_orders[order_id] = {
            **order_data,
            'created_at': datetime.now(),
            'last_updated': datetime.now()
        }
        self.state_version += 1

        self.logger.debug(f"Added active order: {order_id}")

    async def update_active_order(self, order_id: str, order_data: Dict[str, Any]):
        """Update an active order"""
        if order_id in self.active_orders:
            self.active_orders[order_id].update(order_data)
            self.active_orders[order_id]['last_updated'] = datetime.now()
            self.state_version += 1

            self.logger.debug(f"Updated active order: {order_id}")

    async def remove_active_order(self, order_id: str):
        """Remove an active order"""
        if order_id in self.active_orders:
            del self.active_orders[order_id]
            self.state_version += 1

            self.logger.debug(f"Removed active order: {order_id}")

    async def add_active_signal(self, signal_id: str, signal_data: Dict[str, Any]):
        """Add an active signal"""
        self.active_signals[signal_id] = {
            **signal_data,
            'created_at': datetime.now(),
            'last_updated': datetime.now()
        }
        self.state_version += 1

        self.logger.debug(f"Added active signal: {signal_id}")

    async def remove_active_signal(self, signal_id: str):
        """Remove an active signal"""
        if signal_id in self.active_signals:
            del self.active_signals[signal_id]
            self.state_version += 1

            self.logger.debug(f"Removed active signal: {signal_id}")

    async def get_system_state(self) -> SystemState:
        """Get current complete system state"""
        component_status = {}
        for component, last_update in self.component_last_update.items():
            time_since_update = datetime.now() - last_update
            if time_since_update < timedelta(minutes=2):
                component_status[component] = "ACTIVE"
            elif time_since_update < timedelta(minutes=5):
                component_status[component] = "STALE"
            else:
                component_status[component] = "INACTIVE"

        return SystemState(
            timestamp=datetime.now(),
            state_id=str(uuid.uuid4()),
            trading_active=self.is_running,
            component_status=component_status,
            portfolio_state=self._serialize_portfolio_state(),
            position_state=await self.position_tracker.get_all_positions(),
            risk_metrics=self._serialize_risk_metrics(),
            performance_metrics=self.performance_metrics.copy(),
            active_orders=list(self.active_orders.values()),
            active_signals=list(self.active_signals.values())
        )

    async def get_portfolio_state(self) -> Dict[str, Any]:
        """Get current portfolio state"""
        return self._serialize_portfolio_state()

    async def get_positions(self) -> Dict[str, Any]:
        """Get all positions"""
        return await self.position_tracker.get_all_positions()

    async def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get specific position"""
        return await self.position_tracker.get_position(symbol)

    async def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics"""
        return self._serialize_risk_metrics()

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.performance_metrics.copy()

    async def get_active_orders(self) -> Dict[str, Dict[str, Any]]:
        """Get all active orders"""
        return self.active_orders.copy()

    async def get_active_signals(self) -> Dict[str, Dict[str, Any]]:
        """Get all active signals"""
        return self.active_signals.copy()

    def add_state_change_callback(self, callback: callable):
        """Add callback for state changes"""
        self.state_change_callbacks.append(callback)

    def remove_state_change_callback(self, callback: callable):
        """Remove state change callback"""
        if callback in self.state_change_callbacks:
            self.state_change_callbacks.remove(callback)

    async def _initialize_state(self):
        """Initialize state on startup"""
        # Initialize with default values
        self.current_state = await self.get_system_state()
        self.logger.info("State manager initialized")

    async def _snapshot_loop(self):
        """Background loop for taking state snapshots"""
        while self.is_running:
            try:
                await self._take_snapshot()
                await asyncio.sleep(self.snapshot_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in snapshot loop: {e}")
                await asyncio.sleep(10)

    async def _take_snapshot(self):
        """Take a state snapshot"""
        try:
            snapshot = await self.get_system_state()
            self.state_snapshots.append(snapshot)

            # Maintain snapshot limit
            if len(self.state_snapshots) > self.max_snapshots:
                self.state_snapshots.pop(0)

            self.current_state = snapshot

            self.logger.debug(f"State snapshot taken, total snapshots: {len(self.state_snapshots)}")

        except Exception as e:
            self.logger.error(f"Error taking state snapshot: {e}")

    async def _notify_state_change(self, change_type: str, component: str, data: Dict[str, Any]):
        """Notify callbacks of state changes"""
        for callback in self.state_change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(change_type, component, data)
                else:
                    callback(change_type, component, data)
            except Exception as e:
                self.logger.error(f"Error in state change callback: {e}")

    def _serialize_portfolio_state(self) -> Dict[str, Any]:
        """Serialize portfolio state for JSON compatibility"""
        serialized = {}
        for key, value in self.portfolio_state.items():
            if isinstance(value, Decimal):
                serialized[key] = str(value)
            else:
                serialized[key] = value
        return serialized

    def _serialize_risk_metrics(self) -> Dict[str, Any]:
        """Serialize risk metrics for JSON compatibility"""
        serialized = {}
        for key, value in self.risk_metrics.items():
            if isinstance(value, Decimal):
                serialized[key] = str(value)
            else:
                serialized[key] = value
        return serialized

    def get_state_history(self,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> List[SystemState]:
        """Get state history within time range"""
        if not start_time and not end_time:
            return self.state_snapshots.copy()

        filtered_snapshots = []
        for snapshot in self.state_snapshots:
            if start_time and snapshot.timestamp < start_time:
                continue
            if end_time and snapshot.timestamp > end_time:
                continue
            filtered_snapshots.append(snapshot)

        return filtered_snapshots

    def get_state_metrics(self) -> Dict[str, Any]:
        """Get state manager metrics"""
        return {
            'is_running': self.is_running,
            'state_version': self.state_version,
            'snapshots_count': len(self.state_snapshots),
            'components_tracked': len(self.component_states),
            'active_orders_count': len(self.active_orders),
            'active_signals_count': len(self.active_signals),
            'callbacks_registered': len(self.state_change_callbacks),
            'position_tracker_metrics': self.position_tracker.get_metrics()
        }