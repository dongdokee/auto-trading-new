# src/integration/adapters/strategy_adapter.py
"""
Strategy Adapter

Bridges the strategy engine with the event-driven integration system.
Converts strategy signals to events and processes market data events.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from decimal import Decimal

from src.integration.events.event_bus import EventBus
from src.integration.events.models import (
    MarketDataEvent, StrategySignalEvent, SystemEvent,
    EventType, EventPriority
)
from src.integration.events.handlers import BaseEventHandler, HandlerResult

# Import strategy engine components
from src.strategy_engine.strategy_manager import StrategyManager
from src.strategy_engine.base_strategy import StrategySignal


class StrategyAdapter:
    """
    Adapter for the strategy engine system

    Responsibilities:
    - Process market data events and feed to strategy engine
    - Convert strategy signals to events
    - Manage strategy lifecycle
    - Monitor strategy performance
    """

    def __init__(self, event_bus: EventBus, strategy_manager: Optional[StrategyManager] = None):
        self.event_bus = event_bus
        self.strategy_manager = strategy_manager or StrategyManager()

        # Adapter state
        self.is_active = False
        self.last_market_data = {}
        self.signal_history = []

        # Performance tracking
        self.signals_generated = 0
        self.signals_processed = 0
        self.last_signal_time = None

        # Logger
        self.logger = logging.getLogger("strategy_adapter")

        # Register event handlers
        self._register_handlers()

    def _register_handlers(self):
        """Register event handlers with the event bus"""
        market_data_handler = StrategyMarketDataHandler(self)
        self.event_bus.register_handler(EventType.MARKET_DATA, market_data_handler)

        system_handler = StrategySystemHandler(self)
        self.event_bus.register_handler(EventType.SYSTEM, system_handler)

    async def start(self):
        """Start the strategy adapter"""
        self.is_active = True

        # Send startup event
        startup_event = SystemEvent(
            source_component="strategy_adapter",
            system_action="START",
            status="RUNNING",
            message="Strategy adapter started"
        )
        await self.event_bus.publish(startup_event)

        self.logger.info("Strategy adapter started")

    async def stop(self):
        """Stop the strategy adapter"""
        self.is_active = False

        # Send shutdown event
        shutdown_event = SystemEvent(
            source_component="strategy_adapter",
            system_action="STOP",
            status="STOPPED",
            message="Strategy adapter stopped"
        )
        await self.event_bus.publish(shutdown_event)

        self.logger.info("Strategy adapter stopped")

    async def process_market_data(self, market_data_event: MarketDataEvent):
        """Process market data and generate strategy signals"""
        if not self.is_active:
            return

        try:
            # Store market data
            symbol = market_data_event.symbol
            self.last_market_data[symbol] = {
                'symbol': symbol,
                'price': float(market_data_event.price),
                'volume': float(market_data_event.volume),
                'timestamp': market_data_event.timestamp,
                'bid': float(market_data_event.bid) if market_data_event.bid else None,
                'ask': float(market_data_event.ask) if market_data_event.ask else None
            }

            # Generate trading signals
            await self._generate_trading_signals(symbol)

        except Exception as e:
            self.logger.error(f"Error processing market data for {market_data_event.symbol}: {e}")

    async def _generate_trading_signals(self, symbol: str):
        """Generate trading signals for a symbol"""
        try:
            # Prepare market data for strategy manager
            market_data = self._prepare_market_data(symbol)

            if not market_data:
                return

            # Generate signals using strategy manager
            signal_result = self.strategy_manager.generate_trading_signals(
                market_data=market_data,
                current_index=-1  # Use latest data
            )

            # Process primary signal
            primary_signal = signal_result.get('primary_signal')
            if primary_signal and primary_signal.action in ['BUY', 'SELL']:
                await self._emit_strategy_signal(primary_signal, signal_result)

            # Update performance metrics
            self.signals_generated += 1
            self.last_signal_time = datetime.now()

        except Exception as e:
            self.logger.error(f"Error generating trading signals for {symbol}: {e}")

    async def _emit_strategy_signal(self, signal: StrategySignal, signal_result: Dict):
        """Convert strategy signal to event and emit"""
        try:
            # Create strategy signal event
            signal_event = StrategySignalEvent(
                source_component="strategy_adapter",
                strategy_name="aggregated",  # Combined signal from multiple strategies
                symbol=signal.symbol,
                action=signal.action,
                strength=signal.strength,
                confidence=signal.confidence,
                target_price=signal.metadata.get('target_price'),
                stop_loss=signal.stop_loss,
                take_profit=signal.metadata.get('take_profit'),
                regime_info=signal_result.get('regime_info'),
                strategy_metadata={
                    'individual_signals': self._extract_individual_signals(signal_result),
                    'allocation': signal_result.get('allocation'),
                    'signal_timestamp': datetime.now().isoformat()
                }
            )

            # Publish the signal event
            await self.event_bus.publish(signal_event)

            # Store in history
            self.signal_history.append({
                'timestamp': datetime.now(),
                'symbol': signal.symbol,
                'action': signal.action,
                'strength': signal.strength,
                'confidence': signal.confidence
            })

            # Limit history size
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-1000:]

            self.logger.info(f"Generated signal: {signal.action} {signal.symbol} "
                           f"(strength: {signal.strength:.2f}, confidence: {signal.confidence:.2f})")

        except Exception as e:
            self.logger.error(f"Error emitting strategy signal: {e}")

    def _prepare_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Prepare market data for strategy manager"""
        if symbol not in self.last_market_data:
            return None

        market_data = self.last_market_data[symbol].copy()

        # Add any additional required fields
        market_data.update({
            'close': market_data['price'],  # Use current price as close
            'timestamp': market_data['timestamp'],
            'symbol': symbol
        })

        # In a real implementation, you would include OHLCV data
        # For now, we'll use simplified data
        return market_data

    def _extract_individual_signals(self, signal_result: Dict) -> List[Dict]:
        """Extract individual strategy signals from result"""
        individual_signals = []

        strategy_signals = signal_result.get('strategy_signals', {})
        for strategy_name, signal in strategy_signals.items():
            if hasattr(signal, '__dict__'):
                individual_signals.append({
                    'strategy': strategy_name,
                    'action': signal.action,
                    'strength': signal.strength,
                    'confidence': signal.confidence
                })

        return individual_signals

    def get_adapter_metrics(self) -> Dict[str, Any]:
        """Get adapter performance metrics"""
        return {
            'is_active': self.is_active,
            'signals_generated': self.signals_generated,
            'signals_processed': self.signals_processed,
            'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,
            'symbols_tracked': len(self.last_market_data),
            'signal_history_size': len(self.signal_history)
        }

    def get_strategy_status(self) -> Dict[str, Any]:
        """Get strategy engine status"""
        return self.strategy_manager.get_system_status()


class StrategyMarketDataHandler(BaseEventHandler):
    """Handler for market data events in strategy adapter"""

    def __init__(self, adapter: StrategyAdapter):
        super().__init__("strategy_market_data_handler")
        self.adapter = adapter

    async def handle_event(self, event) -> HandlerResult:
        """Handle market data event"""
        try:
            await self.adapter.process_market_data(event)
            return HandlerResult(success=True)

        except Exception as e:
            self.logger.error(f"Error handling market data event: {e}")
            return HandlerResult(
                success=False,
                message=f"Market data processing failed: {str(e)}"
            )


class StrategySystemHandler(BaseEventHandler):
    """Handler for system events in strategy adapter"""

    def __init__(self, adapter: StrategyAdapter):
        super().__init__("strategy_system_handler")
        self.adapter = adapter

    async def handle_event(self, event) -> HandlerResult:
        """Handle system event"""
        try:
            system_event = event

            if system_event.system_action == "PAUSE":
                self.adapter.is_active = False
                self.logger.info("Strategy adapter paused")

            elif system_event.system_action == "RESUME":
                self.adapter.is_active = True
                self.logger.info("Strategy adapter resumed")

            elif system_event.system_action == "STOP":
                await self.adapter.stop()

            return HandlerResult(success=True)

        except Exception as e:
            self.logger.error(f"Error handling system event: {e}")
            return HandlerResult(
                success=False,
                message=f"System event processing failed: {str(e)}"
            )