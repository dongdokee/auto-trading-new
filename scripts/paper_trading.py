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
import yaml
import os
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import available modules with graceful fallback
try:
    from src.api.binance.executor import BinanceExecutor
    from src.api.base import ExchangeConfig
    BINANCE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Binance API not available - {e}")
    BinanceExecutor = None
    ExchangeConfig = None
    BINANCE_AVAILABLE = False

try:
    from src.strategy_engine.strategy_manager import StrategyManager
    STRATEGY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Strategy engine not available - {e}")
    StrategyManager = None
    STRATEGY_AVAILABLE = False

try:
    from src.risk_management.risk_management import RiskController
    RISK_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Risk management not available - {e}")
    RiskController = None
    RISK_AVAILABLE = False

try:
    from src.execution.order_manager import OrderManager
    from src.execution.models import Order, OrderSide, OrderUrgency
    EXECUTION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Execution engine not available - {e}")
    OrderManager = None
    Order = None
    OrderSide = None
    OrderUrgency = None
    EXECUTION_AVAILABLE = False

try:
    from src.utils.trading_logger import UnifiedTradingLogger, TradingMode
    from src.core.patterns import LoggerFactory
    LOGGING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Advanced logging not available - {e}")
    UnifiedTradingLogger = None
    TradingMode = None
    LoggerFactory = None
    LOGGING_AVAILABLE = False

try:
    from src.utils.testnet_loader import TestnetParameterLoader
    TESTNET_LOADER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Testnet loader not available - {e}")
    TestnetParameterLoader = None
    TESTNET_LOADER_AVAILABLE = False


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
        self.executor = None
        self.strategy_manager = None
        self.risk_controller = None
        self.order_manager = None

        # Logging
        self.logger = None
        self.main_logger = logging.getLogger(__name__)

        # Virtual portfolio state
        self.virtual_balance = Decimal('1000.0')  # Starting with $1,000
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
            print("Initializing Paper Trading System...")

            # Load configuration
            await self._load_configuration()

            # Setup logging
            self._setup_logging()

            # Initialize components
            await self._initialize_components()

            # Validate critical components (Fail-Fast)
            self._validate_critical_components()

            # Setup signal handlers
            self._setup_signal_handlers()

            print("Paper Trading System initialized successfully!")
            print("All critical components ready: StrategyManager, BinanceExecutor, RiskController")
            if self.logger and hasattr(self.logger, 'info'):
                self.logger.info("Paper trading system initialized with all critical components")

        except SystemExit:
            # Re-raise SystemExit to allow graceful exit
            raise
        except Exception as e:
            print(f"Failed to initialize paper trading system: {e}")
            self.main_logger.error(f"Initialization failed: {e}")
            raise

    async def _load_configuration(self) -> None:
        """Load trading configuration"""
        try:
            # Load environment variables from .env file
            self._load_env_file()

            # Load base configuration from YAML file
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)

            # Resolve environment variables in config
            self._resolve_env_variables(self.config)

            # Ensure paper trading mode
            self.config['trading']['mode'] = 'paper'
            if 'exchanges' in self.config and 'binance' in self.config['exchanges']:
                self.config['exchanges']['binance']['testnet'] = True

            # Load actual testnet parameters if available
            testnet_params = await self._load_testnet_parameters()

            # Set paper trading defaults if not configured
            if 'paper_trading' not in self.config:
                self.config['paper_trading'] = {
                    'initial_balance': testnet_params['initial_balance'],
                    'commission_rate': testnet_params['commission_rate'],
                    'slippage_simulation': True,
                    'max_slippage': 0.002,  # 0.2%
                    'latency_simulation': True,
                    'min_latency_ms': 10,
                    'max_latency_ms': 50,
                    'report_interval_minutes': 15
                }
            else:
                # Update with testnet values if not explicitly set
                if 'initial_balance' not in self.config['paper_trading']:
                    self.config['paper_trading']['initial_balance'] = testnet_params['initial_balance']
                if 'commission_rate' not in self.config['paper_trading']:
                    self.config['paper_trading']['commission_rate'] = testnet_params['commission_rate']

            # Override virtual balance if configured
            if 'initial_balance' in self.config['paper_trading']:
                self.virtual_balance = Decimal(str(self.config['paper_trading']['initial_balance']))

            print(f"Configuration loaded - Starting balance: ${self.virtual_balance:,.2f}")
            print(f"Commission rate: {self.config['paper_trading']['commission_rate']*100:.3f}%")

        except Exception as e:
            print(f"Failed to load configuration: {e}")
            raise

    def _resolve_env_variables(self, config: Dict[str, Any]) -> None:
        """Resolve environment variables in configuration"""
        def resolve_value(value):
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                return os.getenv(env_var, value)
            elif isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(item) for item in value]
            return value

        for key, value in config.items():
            config[key] = resolve_value(value)

    def _load_env_file(self) -> None:
        """Load environment variables from .env file"""
        env_file = Path(".env")
        if env_file.exists():
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()

    async def _load_testnet_parameters(self) -> Dict[str, Any]:
        """Load actual parameters from Binance Testnet (REQUIRED)"""
        if not TESTNET_LOADER_AVAILABLE:
            raise RuntimeError(
                "Testnet loader module not available. "
                "Please ensure src/utils/testnet_loader.py exists and dependencies are installed."
            )

        # Get API credentials
        api_key = os.getenv("BINANCE_TESTNET_API_KEY")
        api_secret = os.getenv("BINANCE_TESTNET_API_SECRET")

        if not api_key or not api_secret:
            raise RuntimeError(
                "Binance Testnet API credentials not found.\n\n"
                "Paper trading requires valid testnet credentials to operate.\n"
                "Please set the following environment variables in your .env file:\n"
                "  - BINANCE_TESTNET_API_KEY\n"
                "  - BINANCE_TESTNET_API_SECRET\n\n"
                "Get your testnet API keys from: https://testnet.binancefuture.com/\n"
            )

        print("Loading parameters from Binance Testnet...")

        try:
            params = await TestnetParameterLoader.load_testnet_parameters(
                api_key=api_key,
                api_secret=api_secret
            )

            print(f"✓ Testnet balance loaded: ${params['initial_balance']:,.2f} USDT")
            print(f"✓ Testnet commission rate: {params['commission_rate']*100:.3f}%")

            return params

        except Exception as e:
            raise RuntimeError(
                f"Failed to load parameters from Binance Testnet: {e}\n\n"
                "Paper trading requires successful testnet connection.\n"
                "Please check:\n"
                "  1. Your testnet API credentials are valid\n"
                "  2. Network connectivity to testnet.binancefuture.com\n"
                "  3. Your testnet account is active\n"
            )

    def _setup_logging(self) -> None:
        """Setup comprehensive logging for paper trading"""
        try:
            # Setup basic logging
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)

            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler(f'logs/paper_trading_{self.session_id}.log')
                ]
            )
            self.logger = self.main_logger

            print("Logging system configured")

        except Exception as e:
            print(f"Failed to setup logging: {e}")
            # Continue with basic logging
            logging.basicConfig(level=logging.INFO)
            self.logger = self.main_logger

    def _fail_with_error(self, component: str, reason: str, module_error: str, suggestions: list) -> None:
        """
        Print detailed error message and exit immediately (Fail-Fast principle)

        Args:
            component: Name of the component that failed
            reason: High-level reason for failure
            module_error: Technical error details
            suggestions: List of actionable suggestions to fix the issue
        """
        error_message = f"""
{'='*70}
CRITICAL ERROR: {component} Initialization Failed
{'='*70}

Reason: {reason}
Error: {module_error}

Paper trading cannot proceed without {component}.

Why this is critical:
  - Cannot generate meaningful trading signals without strategy manager
  - Cannot execute orders without exchange connection
  - Cannot validate risk limits without risk controller
  - Results would not reflect real trading behavior

Please ensure:
"""

        for suggestion in suggestions:
            error_message += f"  - {suggestion}\n"

        error_message += f"\n{'='*70}\n"

        print(error_message, file=sys.stderr)
        self.main_logger.error(f"{component} initialization failed: {module_error}")

        raise SystemExit(1)

    def _validate_critical_components(self) -> None:
        """
        Validate that all critical components are initialized
        Called at the end of initialize() to ensure system is ready
        """
        missing = []

        if not self.strategy_manager:
            missing.append("StrategyManager")

        if not self.executor:
            missing.append("BinanceExecutor")

        if not self.risk_controller:
            missing.append("RiskController")

        if missing:
            error_message = f"""
{'='*70}
CRITICAL ERROR: Missing Required Components
{'='*70}

The following critical components failed to initialize:
  {', '.join(missing)}

Paper trading requires ALL critical components to function properly.

Component roles:
  - StrategyManager: Generates trading signals based on market data
  - BinanceExecutor: Executes orders and provides market data feed
  - RiskController: Validates position sizing and enforces risk limits

Please review initialization logs above for specific error messages.

{'='*70}
"""
            print(error_message, file=sys.stderr)
            self.main_logger.error(f"Critical components missing: {', '.join(missing)}")
            raise SystemExit(1)

    async def _initialize_components(self) -> None:
        """Initialize all trading components"""
        try:
            print("Initializing core components...")

            # Initialize components that are available
            components_initialized = []

            # Try to initialize each component safely
            if self._try_initialize_executor():
                components_initialized.append("Executor")

            if self._try_initialize_order_manager():
                components_initialized.append("OrderManager")

            if self._try_initialize_risk_controller():
                components_initialized.append("RiskController")

            if self._try_initialize_strategy_manager():
                components_initialized.append("StrategyManager")

            # Setup market data if executor is available
            if self.executor:
                try:
                    await self._setup_market_data()
                    components_initialized.append("MarketData")
                except Exception as e:
                    print(f"Market data setup failed: {e}")

            if components_initialized:
                print(f"Initialized components: {', '.join(components_initialized)}")

        except Exception as e:
            print(f"Component initialization error: {e}")
            raise

    def _try_initialize_executor(self) -> bool:
        """Initialize Binance executor (REQUIRED for paper trading)"""
        try:
            if not BINANCE_AVAILABLE:
                self._fail_with_error(
                    component="BinanceExecutor",
                    reason="Binance API module not available",
                    module_error="No module named 'src.api.binance.executor'",
                    suggestions=[
                        "Install required dependencies: pip install -r requirements.txt",
                        "Verify Binance API module exists: src/api/binance/executor.py",
                        "Check ccxt library is installed: pip install ccxt"
                    ]
                )

            binance_config = self.config.get('exchanges', {}).get('binance', {})

            # Check if API keys are available
            api_key = binance_config.get('api_key', '')
            api_secret = binance_config.get('api_secret', '')

            if not api_key or not api_secret:
                self._fail_with_error(
                    component="BinanceExecutor",
                    reason="Binance API credentials not configured",
                    module_error="Missing BINANCE_TESTNET_API_KEY or BINANCE_TESTNET_API_SECRET",
                    suggestions=[
                        "Create .env file with Binance Testnet credentials",
                        "Get testnet API keys from: https://testnet.binancefuture.com",
                        "Set environment variables: BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET",
                        "Example .env file:\n    BINANCE_TESTNET_API_KEY=your_key_here\n    BINANCE_TESTNET_API_SECRET=your_secret_here"
                    ]
                )

            exchange_config = ExchangeConfig(
                api_key=api_key,
                api_secret=api_secret,
                testnet=True,
                timeout=30,
                rate_limit_per_minute=binance_config.get('rate_limit', 1200)
            )

            self.executor = BinanceExecutor(exchange_config)
            print("Binance executor initialized")
            return True

        except SystemExit:
            raise
        except Exception as e:
            self._fail_with_error(
                component="BinanceExecutor",
                reason="Failed to instantiate BinanceExecutor",
                module_error=str(e),
                suggestions=[
                    "Verify Binance Testnet API credentials are valid",
                    "Check network connectivity to testnet.binancefuture.com",
                    "Review configuration: config/trading.yaml"
                ]
            )

    def _try_initialize_order_manager(self) -> bool:
        """Try to initialize order manager"""
        try:
            if not EXECUTION_AVAILABLE:
                return False

            self.order_manager = OrderManager(config=self.config)
            print("Order manager initialized")
            return True

        except Exception as e:
            print(f"Order manager initialization failed: {e}")
            self.order_manager = None
            return False

    def _try_initialize_risk_controller(self) -> bool:
        """Initialize risk controller (REQUIRED for paper trading)"""
        try:
            if not RISK_AVAILABLE:
                self._fail_with_error(
                    component="RiskController",
                    reason="Risk management module not available",
                    module_error="No module named 'src.risk_management.risk_management'",
                    suggestions=[
                        "Ensure all dependencies are installed: pip install -r requirements.txt",
                        "Verify risk management module exists: src/risk_management/risk_management.py"
                    ]
                )

            initial_capital = float(self.config.get('paper_trading', {}).get('initial_balance', 1000))
            self.risk_controller = RiskController(
                initial_capital_usdt=initial_capital,
                var_daily_pct=0.02,
                max_drawdown_pct=0.12,
                max_leverage=10.0,
                allow_short=False
            )
            print("Risk controller initialized")
            return True

        except SystemExit:
            raise
        except Exception as e:
            self._fail_with_error(
                component="RiskController",
                reason="Failed to instantiate RiskController",
                module_error=str(e),
                suggestions=[
                    "Check configuration file: config/trading.yaml",
                    "Verify risk management parameters are valid",
                    "Review error logs for detailed traceback"
                ]
            )

    def _try_initialize_strategy_manager(self) -> bool:
        """Initialize strategy manager (REQUIRED for paper trading)"""
        try:
            if not STRATEGY_AVAILABLE:
                self._fail_with_error(
                    component="StrategyManager",
                    reason="Strategy engine module not available",
                    module_error="No module named 'src.strategy_engine.strategy_manager'",
                    suggestions=[
                        "Ensure all dependencies are installed: pip install -r requirements.txt",
                        "Verify strategy engine module exists: src/strategy_engine/strategy_manager.py",
                        "Check Python environment is properly configured"
                    ]
                )

            self.strategy_manager = StrategyManager(config=self.config)
            print("Strategy manager initialized")
            return True

        except SystemExit:
            raise
        except Exception as e:
            self._fail_with_error(
                component="StrategyManager",
                reason="Failed to instantiate StrategyManager",
                module_error=str(e),
                suggestions=[
                    "Check configuration file: config/trading.yaml",
                    "Verify strategy parameters are correctly configured",
                    "Review error logs for detailed traceback"
                ]
            )

    async def _setup_market_data(self) -> None:
        """Setup market data subscriptions"""
        try:
            trading_pairs = self.config.get('trading', {}).get('trading_pairs', ['BTC/USDT'])

            for pair in trading_pairs:
                # Convert pair format (BTC/USDT -> BTCUSDT)
                symbol = pair.replace('/', '')

                # Subscribe to market data
                if hasattr(self.executor, 'subscribe_market_data'):
                    await self.executor.subscribe_market_data(symbol)

                # Add callbacks for strategy updates
                if hasattr(self.executor, 'add_trade_callback'):
                    self.executor.add_trade_callback(symbol, self._handle_trade_update)
                if hasattr(self.executor, 'add_orderbook_callback'):
                    self.executor.add_orderbook_callback(symbol, self._handle_orderbook_update)

                print(f"Subscribed to market data for {symbol}")

        except Exception as e:
            print(f"Failed to setup market data: {e}")
            raise

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        self.running = False

    async def run(self) -> None:
        """Main trading loop"""
        # Runtime validation: ensure strategy manager is available
        if not self.strategy_manager:
            raise RuntimeError(
                "StrategyManager is required for paper trading. "
                "This should have been caught during initialization."
            )

        self.running = True
        print("Starting paper trading session...")
        print(f"Initial Balance: ${self.virtual_balance:,.2f}")
        print("Mode: Strategy-based trading (real strategies)")
        print("Monitoring market for trading opportunities...")

        try:
            while self.running:
                # Process trading signals from strategy manager
                await self._process_trading_signals()

                # Generate periodic reports
                await self._generate_periodic_report()

                # Sleep before next iteration
                await asyncio.sleep(5)  # Check every 5 seconds

        except KeyboardInterrupt:
            print("\nTrading session stopped by user")
        except Exception as e:
            print(f"Trading session error: {e}")
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
                if hasattr(self.strategy_manager, 'generate_signals'):
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

        side = 'BUY' if buy_signals > sell_signals else 'SELL'
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
            # Calculate position size based on simple percentage
            position_size = await self._calculate_position_size(symbol, signal)

            if position_size <= 0:
                return

            # Get current market price (mock price for simulation)
            current_price = await self._get_current_price(symbol)

            # Simulate order execution
            await self._simulate_order_execution(symbol, signal['side'], position_size, current_price, signal)

        except Exception as e:
            self.main_logger.error(f"Error executing paper trade for {symbol}: {e}")

    async def _calculate_position_size(self, symbol: str, signal: Dict[str, Any]) -> Decimal:
        """Calculate optimal position size"""
        try:
            # Base position size as percentage of balance
            max_position_pct = Decimal('0.05')  # 5% max per trade

            # Adjust based on signal strength and confidence
            strength_factor = Decimal(str(signal['strength']))
            confidence_factor = Decimal(str(signal['confidence']))

            position_pct = max_position_pct * strength_factor * confidence_factor
            position_value = self.virtual_balance * position_pct

            # Get current price to calculate size
            current_price = await self._get_current_price(symbol)
            position_size = position_value / current_price

            return position_size

        except Exception as e:
            self.main_logger.error(f"Error calculating position size: {e}")
            return Decimal('0')

    async def _get_current_price(self, symbol: str) -> Decimal:
        """Get current market price"""
        try:
            # Try to get real price from executor
            if self.executor and hasattr(self.executor, 'get_market_conditions'):
                market_data = await self.executor.get_market_conditions(symbol)
                return Decimal(str(market_data['market_data']['lastPrice']))
            else:
                # Mock prices for simulation
                mock_prices = {
                    'BTCUSDT': 50000.0,
                    'ETHUSDT': 3000.0,
                    'BNBUSDT': 400.0,
                    'ADAUSDT': 0.5,
                    'SOLUSDT': 100.0
                }
                return Decimal(str(mock_prices.get(symbol, 100.0)))

        except Exception as e:
            self.main_logger.error(f"Error getting price for {symbol}: {e}")
            return Decimal('100.0')  # Default fallback price

    async def _simulate_order_execution(self, symbol: str, side: str, size: Decimal, price: Decimal, signal: Dict[str, Any]) -> None:
        """Simulate order execution with realistic slippage and latency"""
        try:
            # Simulate latency
            if self.config.get('paper_trading', {}).get('latency_simulation', True):
                import random
                latency_ms = random.randint(
                    self.config['paper_trading'].get('min_latency_ms', 10),
                    self.config['paper_trading'].get('max_latency_ms', 50)
                )
                await asyncio.sleep(latency_ms / 1000)

            # Simulate slippage
            execution_price = price
            if self.config.get('paper_trading', {}).get('slippage_simulation', True):
                import random
                max_slippage = self.config['paper_trading'].get('max_slippage', 0.002)
                slippage_factor = Decimal(str(random.uniform(-max_slippage, max_slippage)))

                if side == 'BUY':
                    execution_price = price * (1 + abs(slippage_factor))
                else:
                    execution_price = price * (1 - abs(slippage_factor))

            # Calculate commission
            commission_rate = Decimal(str(self.config['paper_trading'].get('commission_rate', 0.001)))
            trade_value = size * execution_price
            commission = trade_value * commission_rate

            # Update virtual portfolio
            if side == 'BUY':
                # Buy order - decrease balance, increase position
                total_cost = trade_value + commission
                if total_cost <= self.virtual_balance:
                    self.virtual_balance -= total_cost
                    self.virtual_positions[symbol] = self.virtual_positions.get(symbol, Decimal('0')) + size

                    self.trades_executed += 1

                    # Log the trade
                    order_id = f"paper_{self.session_id}_{self.trades_executed}"
                    await self._log_trade_execution(order_id, symbol, side, size, execution_price, commission, signal)

                    print(f"BUY executed: {size:.6f} {symbol} @ ${execution_price:.2f}")
                    print(f"Balance: ${self.virtual_balance:,.2f} | Position: {self.virtual_positions[symbol]:.6f}")

            else:
                # Sell order - increase balance, decrease position
                current_position = self.virtual_positions.get(symbol, Decimal('0'))
                if current_position >= size:
                    proceeds = trade_value - commission
                    self.virtual_balance += proceeds
                    self.virtual_positions[symbol] -= size

                    self.trades_executed += 1

                    # Log the trade
                    order_id = f"paper_{self.session_id}_{self.trades_executed}"
                    await self._log_trade_execution(order_id, symbol, side, size, execution_price, commission, signal)

                    print(f"SELL executed: {size:.6f} {symbol} @ ${execution_price:.2f}")
                    print(f"Balance: ${self.virtual_balance:,.2f} | Position: {self.virtual_positions[symbol]:.6f}")

        except Exception as e:
            self.main_logger.error(f"Error simulating order execution: {e}")

    async def _log_trade_execution(self, order_id: str, symbol: str, side: str, size: Decimal, execution_price: Decimal, commission: Decimal, signal: Dict[str, Any]) -> None:
        """Log trade execution details"""
        try:
            log_entry = {
                'order_id': order_id,
                'symbol': symbol,
                'side': side,
                'size': float(size),
                'price': float(execution_price),
                'commission': float(commission),
                'signal_strength': signal.get('strength'),
                'signal_confidence': signal.get('confidence'),
                'timestamp': datetime.now().isoformat()
            }

            if self.logger:
                self.logger.info(f"Paper trade executed: {log_entry}")

        except Exception as e:
            self.main_logger.error(f"Error logging trade execution: {e}")

    async def _handle_trade_update(self, data: Dict[str, Any]) -> None:
        """Handle real-time trade updates"""
        try:
            # Update strategy manager with trade data
            if self.strategy_manager and hasattr(self.strategy_manager, 'process_trade_data'):
                await self.strategy_manager.process_trade_data(data)

        except Exception as e:
            self.main_logger.error(f"Error handling trade update: {e}")

    async def _handle_orderbook_update(self, data: Dict[str, Any]) -> None:
        """Handle real-time orderbook updates"""
        try:
            # Update strategy manager with orderbook data
            if self.strategy_manager and hasattr(self.strategy_manager, 'process_orderbook_data'):
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
                        current_price = await self._get_current_price(symbol)
                        position_value = position * current_price
                        total_value += position_value
                    except:
                        pass  # Skip if can't get price

            # Calculate PnL
            initial_balance = Decimal(str(self.config['paper_trading'].get('initial_balance', 1000)))
            total_pnl = total_value - initial_balance
            pnl_pct = (total_pnl / initial_balance) * 100 if initial_balance > 0 else Decimal('0')

            # Calculate session duration
            session_duration = datetime.now() - self.start_time

            # Generate report
            print("\n" + "="*60)
            print("PAPER TRADING PERFORMANCE REPORT")
            print("="*60)
            print(f"Session Duration: {session_duration}")
            print(f"Initial Balance: ${initial_balance:,.2f}")
            print(f"Current Balance: ${self.virtual_balance:,.2f}")
            print(f"Total Portfolio Value: ${total_value:,.2f}")
            print(f"Total PnL: ${total_pnl:,.2f} ({pnl_pct:.2f}%)")
            print(f"Trades Executed: {self.trades_executed}")

            if self.virtual_positions:
                print(f"\nCurrent Positions:")
                for symbol, position in self.virtual_positions.items():
                    if position > 0:
                        print(f"   {symbol}: {position:.6f}")

            print("="*60 + "\n")

            # Log performance metrics
            if self.logger:
                self.logger.info(f"Performance report: PnL={total_pnl:.2f} ({pnl_pct:.2f}%), Trades={self.trades_executed}")

        except Exception as e:
            self.main_logger.error(f"Error generating performance report: {e}")

    async def _shutdown(self) -> None:
        """Cleanup and shutdown"""
        try:
            print("Shutting down paper trading system...")

            # Generate final report
            await self._generate_performance_report()

            # Disconnect from exchange
            if self.executor and hasattr(self.executor, 'disconnect'):
                await self.executor.disconnect()

            # Log session end
            if self.logger:
                self.logger.info(f"Paper trading session ended - {self.trades_executed} trades executed")

            print("Paper trading system shutdown complete")

        except Exception as e:
            print(f"Error during shutdown: {e}")


async def main():
    """Main entry point for paper trading"""
    print("AutoTrading Paper Trading System")
    print("Using Binance Testnet - No real money at risk!")
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
        print("\nPaper trading stopped by user")
        return 0
    except Exception as e:
        print(f"Paper trading error: {e}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nPaper trading stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"Critical error: {e}")
        sys.exit(1)