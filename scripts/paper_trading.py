#!/usr/bin/env python3
"""
Paper Trading System Entry Point
Binance Testnet 기반 모의 트레이딩 시스템

이 스크립트는 실제 돈을 사용하지 않고 Binance Testnet을 통해
트레이딩 전략을 테스트할 수 있는 완전한 paper trading 환경을 제공합니다.
"""

import asyncio
import sys
import signal
import json
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.config.config_manager import ConfigManager
from src.api.binance.executor import BinanceExecutor
from src.api.base import ExchangeConfig
from src.strategy_engine.strategy_manager import StrategyManager
from src.risk_management.risk_management import RiskController
from src.portfolio.portfolio_manager import PortfolioManager
from src.execution.order_manager import OrderManager
from src.execution.models import Order, OrderSide, OrderUrgency
from src.utils.trading_logger import UnifiedTradingLogger, TradingMode
from src.core.patterns import LoggerFactory


class PaperTradingSystem:
    """
    Complete paper trading system using Binance Testnet

    Features:
    - Binance Testnet integration
    - Real market data with simulated execution
    - Strategy execution without real money
    - Risk management validation
    - Performance tracking and reporting
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/trading.yaml"
        self.config: Dict[str, Any] = {}
        self.running = False
        self.session_id = f"paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Core components
        self.executor: Optional[BinanceExecutor] = None
        self.strategy_manager: Optional[StrategyManager] = None
        self.risk_controller: Optional[RiskController] = None
        self.portfolio_manager: Optional[PortfolioManager] = None
        self.order_manager: Optional[OrderManager] = None

        # Logging
        self.logger: Optional[UnifiedTradingLogger] = None
        self.main_logger = logging.getLogger(__name__)

        # Virtual portfolio state
        self.virtual_balance = Decimal('100000.0')  # Starting with $100,000
        self.virtual_positions: Dict[str, Decimal] = {}
        self.total_pnl = Decimal('0.0')
        self.trades_executed = 0
        self.winning_trades = 0

        # Performance tracking
        self.start_time = datetime.now()
        self.last_report_time = datetime.now()
        self.performance_history = []

    async def initialize(self) -> None:
        """Initialize the paper trading system"""
        try:
            print("🚀 Initializing Paper Trading System...")

            # Load configuration
            await self._load_configuration()

            # Setup logging
            self._setup_logging()

            # Initialize components
            await self._initialize_components()

            # Setup signal handlers
            self._setup_signal_handlers()

            print("✅ Paper Trading System initialized successfully!")
            self.logger.log_system_event(
                message="Paper trading system initialized",
                session_id=self.session_id,
                config=self.config.get('paper_trading', {})
            )

        except Exception as e:
            print(f"❌ Failed to initialize paper trading system: {e}")
            self.main_logger.error(f"Initialization failed: {e}")
            raise

    async def _load_configuration(self) -> None:
        """Load trading configuration"""
        try:
            # Load base configuration
            config_manager = ConfigManager()
            self.config = await config_manager.load_config(self.config_path)

            # Ensure paper trading mode
            self.config['trading']['mode'] = 'paper'
            self.config['exchanges']['binance']['testnet'] = True
            self.config['exchanges']['binance']['paper_trading'] = True

            # Set paper trading defaults if not configured
            if 'paper_trading' not in self.config:
                self.config['paper_trading'] = {
                    'initial_balance': 100000.0,
                    'commission_rate': 0.001,  # 0.1%
                    'slippage_simulation': True,
                    'max_slippage': 0.002,  # 0.2%
                    'latency_simulation': True,
                    'min_latency_ms': 10,
                    'max_latency_ms': 50,
                    'report_interval_minutes': 15
                }

            # Override virtual balance if configured
            if 'initial_balance' in self.config['paper_trading']:
                self.virtual_balance = Decimal(str(self.config['paper_trading']['initial_balance']))

            print(f"📋 Configuration loaded - Starting balance: ${self.virtual_balance:,.2f}")

        except Exception as e:
            print(f"❌ Failed to load configuration: {e}")
            raise

    def _setup_logging(self) -> None:
        """Setup comprehensive logging for paper trading"""
        try:
            # Create unified trading logger
            self.logger = UnifiedTradingLogger(
                name="paper_trading_system",
                mode=TradingMode.PAPER,
                config=self.config
            )

            # Setup component loggers
            LoggerFactory.setup_trading_session(
                session_id=self.session_id,
                strategy="paper_trading_multi_strategy",
                mode=TradingMode.PAPER
            )

            print("📝 Logging system configured")

        except Exception as e:
            print(f"❌ Failed to setup logging: {e}")
            raise

    async def _initialize_components(self) -> None:
        """Initialize all trading components"""
        try:
            # Create exchange configuration
            binance_config = self.config.get('exchanges', {}).get('binance', {})
            exchange_config = ExchangeConfig(
                name="BINANCE",
                api_key=binance_config.get('api_key', ''),
                api_secret=binance_config.get('api_secret', ''),
                testnet=True,
                paper_trading=True,
                rate_limit_requests=binance_config.get('rate_limit', 1200),
                timeout=30
            )

            # Initialize executor
            self.executor = BinanceExecutor(exchange_config)
            await self.executor.connect()
            print("🔗 Connected to Binance Testnet")

            # Initialize order manager
            self.order_manager = OrderManager(config=self.config)

            # Initialize risk controller
            self.risk_controller = RiskController(config=self.config)

            # Initialize portfolio manager
            self.portfolio_manager = PortfolioManager(config=self.config)

            # Initialize strategy manager
            self.strategy_manager = StrategyManager(config=self.config)

            # Setup market data callbacks
            await self._setup_market_data()

            print("🧩 All components initialized")

        except Exception as e:
            print(f"❌ Failed to initialize components: {e}")
            raise

    async def _setup_market_data(self) -> None:
        """Setup market data subscriptions"""
        try:
            trading_pairs = self.config.get('trading', {}).get('trading_pairs', ['BTC/USDT'])

            for pair in trading_pairs:
                # Convert pair format (BTC/USDT -> BTCUSDT)
                symbol = pair.replace('/', '')

                # Subscribe to market data
                await self.executor.subscribe_market_data(symbol)

                # Add callbacks for strategy updates
                self.executor.add_trade_callback(symbol, self._handle_trade_update)
                self.executor.add_orderbook_callback(symbol, self._handle_orderbook_update)

                print(f"📊 Subscribed to market data for {symbol}")

        except Exception as e:
            print(f"❌ Failed to setup market data: {e}")
            raise

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n⏹️  Received signal {signum}, shutting down gracefully...")
        self.running = False

    async def run(self) -> None:
        """Main trading loop"""
        self.running = True
        print("🎯 Starting paper trading session...")
        print(f"💰 Initial Balance: ${self.virtual_balance:,.2f}")
        print("📈 Monitoring market for trading opportunities...")

        try:
            while self.running:
                # Generate trading signals
                await self._process_trading_signals()

                # Generate periodic reports
                await self._generate_periodic_report()

                # Sleep before next iteration
                await asyncio.sleep(5)  # Check every 5 seconds

        except KeyboardInterrupt:
            print("\n⏹️  Trading session stopped by user")
        except Exception as e:
            print(f"❌ Trading session error: {e}")
            self.main_logger.error(f"Trading session error: {e}")
        finally:
            await self._shutdown()

    async def _process_trading_signals(self) -> None:
        """Process trading signals and execute orders"""
        try:
            trading_pairs = self.config.get('trading', {}).get('trading_pairs', ['BTC/USDT'])

            for pair in trading_pairs:
                symbol = pair.replace('/', '')

                # Generate strategy signals
                signals = await self.strategy_manager.generate_signals(symbol)

                if signals and any(signal.get('strength', 0) > 0.6 for signal in signals):
                    # Aggregate signals
                    aggregated_signal = self._aggregate_signals(signals)

                    if aggregated_signal['should_trade']:
                        await self._execute_paper_trade(symbol, aggregated_signal)

        except Exception as e:
            self.main_logger.error(f"Error processing trading signals: {e}")

    def _aggregate_signals(self, signals: list) -> Dict[str, Any]:
        """Aggregate multiple strategy signals"""
        if not signals:
            return {'should_trade': False}

        # Simple aggregation - average strength and confidence
        total_strength = sum(signal.get('strength', 0) for signal in signals)
        total_confidence = sum(signal.get('confidence', 0) for signal in signals)
        avg_strength = total_strength / len(signals)
        avg_confidence = total_confidence / len(signals)

        # Determine trade direction
        buy_signals = sum(1 for signal in signals if signal.get('signal_type') == 'BUY')
        sell_signals = sum(1 for signal in signals if signal.get('signal_type') == 'SELL')

        side = OrderSide.BUY if buy_signals > sell_signals else OrderSide.SELL
        should_trade = avg_strength > 0.6 and avg_confidence > 0.7

        return {
            'should_trade': should_trade,
            'side': side,
            'strength': avg_strength,
            'confidence': avg_confidence,
            'signal_count': len(signals)
        }

    async def _execute_paper_trade(self, symbol: str, signal: Dict[str, Any]) -> None:
        """Execute a paper trade based on signal"""
        try:
            # Calculate position size based on Kelly Criterion
            position_size = await self._calculate_position_size(symbol, signal)

            if position_size <= 0:
                return

            # Get current market price
            market_data = await self.executor.get_market_conditions(symbol)
            current_price = Decimal(str(market_data['market_data']['lastPrice']))

            # Create order
            order = Order(
                symbol=symbol,
                side=signal['side'],
                size=position_size,
                price=current_price,
                urgency=OrderUrgency.NORMAL
            )

            # Risk validation
            risk_passed = await self.risk_controller.validate_order(order)
            if not risk_passed:
                print(f"🚫 Risk check failed for {symbol} order")
                return

            # Simulate order execution
            await self._simulate_order_execution(order, signal)

        except Exception as e:
            self.main_logger.error(f"Error executing paper trade for {symbol}: {e}")

    async def _calculate_position_size(self, symbol: str, signal: Dict[str, Any]) -> Decimal:
        """Calculate optimal position size using Kelly Criterion"""
        try:
            # Base position size as percentage of balance
            max_position_pct = Decimal('0.05')  # 5% max per trade

            # Adjust based on signal strength and confidence
            strength_factor = Decimal(str(signal['strength']))
            confidence_factor = Decimal(str(signal['confidence']))

            position_pct = max_position_pct * strength_factor * confidence_factor
            position_value = self.virtual_balance * position_pct

            # Get current price to calculate size
            market_data = await self.executor.get_market_conditions(symbol)
            current_price = Decimal(str(market_data['market_data']['lastPrice']))

            position_size = position_value / current_price

            return position_size

        except Exception as e:
            self.main_logger.error(f"Error calculating position size: {e}")
            return Decimal('0')

    async def _simulate_order_execution(self, order: Order, signal: Dict[str, Any]) -> None:
        """Simulate order execution with realistic slippage and latency"""
        try:
            # Simulate latency
            if self.config['paper_trading'].get('latency_simulation', True):
                import random
                latency_ms = random.randint(
                    self.config['paper_trading']['min_latency_ms'],
                    self.config['paper_trading']['max_latency_ms']
                )
                await asyncio.sleep(latency_ms / 1000)

            # Simulate slippage
            execution_price = order.price
            if self.config['paper_trading'].get('slippage_simulation', True):
                import random
                max_slippage = self.config['paper_trading'].get('max_slippage', 0.002)
                slippage_factor = Decimal(str(random.uniform(-max_slippage, max_slippage)))

                if order.side == OrderSide.BUY:
                    execution_price = order.price * (1 + abs(slippage_factor))
                else:
                    execution_price = order.price * (1 - abs(slippage_factor))

            # Calculate commission
            commission_rate = Decimal(str(self.config['paper_trading']['commission_rate']))
            trade_value = order.size * execution_price
            commission = trade_value * commission_rate

            # Update virtual portfolio
            if order.side == OrderSide.BUY:
                # Buy order - decrease balance, increase position
                total_cost = trade_value + commission
                if total_cost <= self.virtual_balance:
                    self.virtual_balance -= total_cost
                    self.virtual_positions[order.symbol] = self.virtual_positions.get(order.symbol, Decimal('0')) + order.size

                    self.trades_executed += 1

                    # Log the trade
                    order_id = f"paper_{self.session_id}_{self.trades_executed}"
                    await self._log_trade_execution(order_id, order, execution_price, commission, signal)

                    print(f"✅ BUY executed: {order.size:.6f} {order.symbol} @ ${execution_price:.2f}")
                    print(f"💰 Balance: ${self.virtual_balance:,.2f} | Position: {self.virtual_positions[order.symbol]:.6f}")

            else:
                # Sell order - increase balance, decrease position
                current_position = self.virtual_positions.get(order.symbol, Decimal('0'))
                if current_position >= order.size:
                    proceeds = trade_value - commission
                    self.virtual_balance += proceeds
                    self.virtual_positions[order.symbol] -= order.size

                    self.trades_executed += 1

                    # Log the trade
                    order_id = f"paper_{self.session_id}_{self.trades_executed}"
                    await self._log_trade_execution(order_id, order, execution_price, commission, signal)

                    print(f"✅ SELL executed: {order.size:.6f} {order.symbol} @ ${execution_price:.2f}")
                    print(f"💰 Balance: ${self.virtual_balance:,.2f} | Position: {self.virtual_positions[order.symbol]:.6f}")

        except Exception as e:
            self.main_logger.error(f"Error simulating order execution: {e}")

    async def _log_trade_execution(self, order_id: str, order: Order, execution_price: Decimal, commission: Decimal, signal: Dict[str, Any]) -> None:
        """Log trade execution details"""
        try:
            self.logger.log_order(
                message=f"Paper trade executed: {order.side.value} {order.size} {order.symbol}",
                order_id=order_id,
                symbol=order.symbol,
                side=order.side.value,
                size=float(order.size),
                price=float(order.price),
                order_type="MARKET",
                status="FILLED",
                execution_price=float(execution_price),
                commission=float(commission),
                session_id=self.session_id,
                paper_trading=True,
                signal_strength=signal.get('strength'),
                signal_confidence=signal.get('confidence')
            )

        except Exception as e:
            self.main_logger.error(f"Error logging trade execution: {e}")

    async def _handle_trade_update(self, data: Dict[str, Any]) -> None:
        """Handle real-time trade updates"""
        try:
            # Update strategy manager with trade data
            await self.strategy_manager.process_trade_data(data)

        except Exception as e:
            self.main_logger.error(f"Error handling trade update: {e}")

    async def _handle_orderbook_update(self, data: Dict[str, Any]) -> None:
        """Handle real-time orderbook updates"""
        try:
            # Update strategy manager with orderbook data
            await self.strategy_manager.process_orderbook_data(data)

        except Exception as e:
            self.main_logger.error(f"Error handling orderbook update: {e}")

    async def _generate_periodic_report(self) -> None:
        """Generate periodic performance reports"""
        try:
            now = datetime.now()
            report_interval = timedelta(minutes=self.config['paper_trading'].get('report_interval_minutes', 15))

            if now - self.last_report_time >= report_interval:
                await self._generate_performance_report()
                self.last_report_time = now

        except Exception as e:
            self.main_logger.error(f"Error generating periodic report: {e}")

    async def _generate_performance_report(self) -> None:
        """Generate comprehensive performance report"""
        try:
            # Calculate current portfolio value
            total_value = self.virtual_balance

            for symbol, position in self.virtual_positions.items():
                if position > 0:
                    try:
                        market_data = await self.executor.get_market_conditions(symbol)
                        current_price = Decimal(str(market_data['market_data']['lastPrice']))
                        position_value = position * current_price
                        total_value += position_value
                    except:
                        pass  # Skip if can't get price

            # Calculate PnL
            initial_balance = Decimal(str(self.config['paper_trading']['initial_balance']))
            total_pnl = total_value - initial_balance
            pnl_pct = (total_pnl / initial_balance) * 100 if initial_balance > 0 else Decimal('0')

            # Calculate session duration
            session_duration = datetime.now() - self.start_time

            # Generate report
            print("\n" + "="*60)
            print("📊 PAPER TRADING PERFORMANCE REPORT")
            print("="*60)
            print(f"🕒 Session Duration: {session_duration}")
            print(f"💰 Initial Balance: ${initial_balance:,.2f}")
            print(f"💰 Current Balance: ${self.virtual_balance:,.2f}")
            print(f"📈 Total Portfolio Value: ${total_value:,.2f}")
            print(f"💵 Total PnL: ${total_pnl:,.2f} ({pnl_pct:.2f}%)")
            print(f"📊 Trades Executed: {self.trades_executed}")

            if self.virtual_positions:
                print(f"\n🎯 Current Positions:")
                for symbol, position in self.virtual_positions.items():
                    if position > 0:
                        print(f"   {symbol}: {position:.6f}")

            print("="*60 + "\n")

            # Log performance metrics
            self.logger.log_performance(
                message="Paper trading performance report",
                session_id=self.session_id,
                total_value=float(total_value),
                total_pnl=float(total_pnl),
                pnl_percentage=float(pnl_pct),
                trades_executed=self.trades_executed,
                session_duration_hours=session_duration.total_seconds() / 3600
            )

        except Exception as e:
            self.main_logger.error(f"Error generating performance report: {e}")

    async def _shutdown(self) -> None:
        """Cleanup and shutdown"""
        try:
            print("🔄 Shutting down paper trading system...")

            # Generate final report
            await self._generate_performance_report()

            # Disconnect from exchange
            if self.executor:
                await self.executor.disconnect()

            # Log session end
            if self.logger:
                self.logger.log_system_event(
                    message="Paper trading session ended",
                    session_id=self.session_id,
                    total_trades=self.trades_executed
                )

            print("✅ Paper trading system shutdown complete")

        except Exception as e:
            print(f"❌ Error during shutdown: {e}")


async def main():
    """Main entry point for paper trading"""
    print("📝 AutoTrading Paper Trading System")
    print("🏦 Using Binance Testnet - No real money at risk!")
    print("=" * 50)

    # Check for configuration file
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/trading.yaml"

    try:
        # Initialize paper trading system
        paper_trading = PaperTradingSystem(config_path)
        await paper_trading.initialize()

        # Run paper trading
        await paper_trading.run()

        return 0

    except KeyboardInterrupt:
        print("\n⏹️  Paper trading stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Paper trading error: {e}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Paper trading stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Critical error: {e}")
        sys.exit(1)