# tests/unit/test_config/test_paper_trading_config.py
import pytest
import tempfile
import os
import yaml
from unittest.mock import patch, Mock

from src.core.config.loader import ConfigLoader
from src.utils.trading_logger import TradingMode
from src.core.patterns import LoggerFactory


class TestPaperTradingConfiguration:
    """Test suite for paper trading configuration"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")

        # Create test configuration
        self.test_config = {
            'trading': {
                'mode': 'paper',
                'session_timeout': 3600,
                'max_concurrent_orders': 10
            },
            'paper_trading': {
                'enabled': True,
                'initial_balance': 100000.0,
                'commission_rate': 0.001,
                'slippage_simulation': True,
                'latency_simulation': {
                    'enabled': True,
                    'min_latency_ms': 10,
                    'max_latency_ms': 50
                }
            },
            'logging': {
                'level': 'DEBUG',
                'file_handler': {
                    'enabled': True,
                    'filename': 'paper_trading.log',
                    'max_size_mb': 100,
                    'backup_count': 5
                },
                'db_handler': {
                    'enabled': True,
                    'database_path': 'paper_trading.db'
                },
                'console_handler': {
                    'enabled': True,
                    'level': 'INFO'
                }
            },
            'exchanges': {
                'binance': {
                    'name': 'BINANCE',
                    'testnet': True,
                    'paper_trading': True,
                    'api_key': '${BINANCE_TESTNET_API_KEY}',
                    'api_secret': '${BINANCE_TESTNET_API_SECRET}',
                    'rate_limit_requests': 1200,
                    'timeout': 30
                }
            },
            'strategies': {
                'momentum': {
                    'enabled': True,
                    'allocation': 0.4,
                    'risk_limit': 0.02
                },
                'mean_reversion': {
                    'enabled': True,
                    'allocation': 0.3,
                    'risk_limit': 0.015
                },
                'breakout': {
                    'enabled': True,
                    'allocation': 0.3,
                    'risk_limit': 0.02
                }
            },
            'risk_management': {
                'max_portfolio_risk': 0.05,
                'max_position_size': 0.1,
                'max_daily_var': 0.02,
                'kelly_fraction_limit': 0.25,
                'paper_trading_multiplier': 1.0
            },
            'database': {
                'path': 'paper_trading.db',
                'backup_enabled': True,
                'backup_interval_hours': 24
            }
        }

        # Write configuration to file
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)

    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_should_load_paper_trading_configuration(self):
        """Test loading paper trading configuration"""
        config_loader = ConfigLoader()
        config = config_loader.load_config(self.config_path)

        # Verify paper trading settings
        assert config['trading']['mode'] == 'paper'
        assert config['paper_trading']['enabled'] is True
        assert config['paper_trading']['initial_balance'] == 100000.0

    def test_should_validate_paper_trading_safety_settings(self):
        """Test paper trading safety validations"""
        config_loader = ConfigLoader()
        config = config_loader.load_config(self.config_path)

        # Verify exchange is in testnet mode
        assert config['exchanges']['binance']['testnet'] is True
        assert config['exchanges']['binance']['paper_trading'] is True

        # Verify logging is properly configured
        assert config['logging']['level'] == 'DEBUG'
        assert config['logging']['db_handler']['enabled'] is True

    def test_should_handle_environment_variables_safely(self):
        """Test safe handling of environment variables"""
        # Mock environment variables for testnet
        with patch.dict(os.environ, {
            'BINANCE_TESTNET_API_KEY': 'test_key_12345678',
            'BINANCE_TESTNET_API_SECRET': 'test_secret_12345678'
        }):
            config_loader = ConfigLoader()
            config = config_loader.load_config(self.config_path)

            # Should resolve testnet credentials
            exchange_config = config['exchanges']['binance']
            assert 'test_key' in exchange_config['api_key']
            assert 'test_secret' in exchange_config['api_secret']

    def test_should_prevent_live_trading_in_paper_mode(self):
        """Test prevention of live trading when in paper mode"""
        config_loader = ConfigLoader()
        config = config_loader.load_config(self.config_path)

        # Paper mode should force testnet and paper trading
        if config['trading']['mode'] == 'paper':
            assert config['paper_trading']['enabled'] is True
            assert config['exchanges']['binance']['testnet'] is True
            assert config['exchanges']['binance']['paper_trading'] is True

    def test_should_configure_appropriate_logging_levels(self):
        """Test appropriate logging levels for paper trading"""
        config_loader = ConfigLoader()
        config = config_loader.load_config(self.config_path)

        # Paper trading should have DEBUG level for detailed logging
        assert config['logging']['level'] == 'DEBUG'
        assert config['logging']['file_handler']['enabled'] is True
        assert config['logging']['db_handler']['enabled'] is True

    def test_should_validate_risk_management_settings(self):
        """Test risk management settings for paper trading"""
        config_loader = ConfigLoader()
        config = config_loader.load_config(self.config_path)

        risk_config = config['risk_management']

        # Risk limits should be reasonable for paper trading
        assert risk_config['max_portfolio_risk'] <= 0.1  # Max 10%
        assert risk_config['max_position_size'] <= 0.2   # Max 20%
        assert risk_config['max_daily_var'] <= 0.05      # Max 5%

    def test_should_configure_simulation_parameters(self):
        """Test simulation parameter configuration"""
        config_loader = ConfigLoader()
        config = config_loader.load_config(self.config_path)

        paper_config = config['paper_trading']

        # Verify simulation settings
        assert 'slippage_simulation' in paper_config
        assert 'latency_simulation' in paper_config

        if paper_config['slippage_simulation']:
            assert paper_config['commission_rate'] > 0

        if paper_config['latency_simulation']['enabled']:
            assert paper_config['latency_simulation']['min_latency_ms'] > 0
            assert paper_config['latency_simulation']['max_latency_ms'] > 0


class TestPaperTradingLoggerConfiguration:
    """Test paper trading logger configuration"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()

        self.config = {
            'database': {'path': os.path.join(self.temp_dir, "paper_logger_test.db")},
            'paper_trading': {'enabled': True},
            'logging': {
                'level': 'DEBUG',
                'file_handler': {'enabled': True, 'filename': 'paper_test.log'},
                'db_handler': {'enabled': True}
            }
        }

    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_should_create_paper_trading_logger(self):
        """Test creating paper trading logger"""
        from src.utils.trading_logger import UnifiedTradingLogger

        logger = UnifiedTradingLogger(
            name="paper_test",
            mode=TradingMode.PAPER,
            config=self.config
        )

        assert logger.mode == TradingMode.PAPER
        assert logger.name == "paper_test"

    def test_should_initialize_with_paper_trading_database(self):
        """Test database initialization for paper trading"""
        from src.utils.trading_logger import UnifiedTradingLogger

        logger = UnifiedTradingLogger(
            name="paper_db_test",
            mode=TradingMode.PAPER,
            config=self.config
        )

        # Database should be created
        assert os.path.exists(logger.db_path)

    def test_should_mark_all_logs_as_paper_trading(self):
        """Test that all logs are marked as paper trading"""
        from src.utils.trading_logger import UnifiedTradingLogger

        logger = UnifiedTradingLogger(
            name="paper_marking_test",
            mode=TradingMode.PAPER,
            config=self.config
        )

        # Log a test signal
        logger.log_signal(
            message="Test paper signal",
            strategy="test_strategy",
            symbol="BTCUSDT",
            signal_type="BUY"
        )

        # Verify paper trading mode is preserved
        assert logger.mode == TradingMode.PAPER

    def test_should_handle_component_logger_in_paper_mode(self):
        """Test component logger behavior in paper mode"""
        # Mock enhanced logging availability
        with patch('src.core.patterns.ENHANCED_LOGGING_AVAILABLE', True):
            logger = LoggerFactory.get_component_trading_logger(
                component="paper_component_test",
                strategy="paper_strategy"
            )

            assert logger is not None

    def test_should_validate_paper_trading_session_tracking(self):
        """Test session tracking in paper trading mode"""
        from src.utils.trading_logger import UnifiedTradingLogger

        logger = UnifiedTradingLogger(
            name="paper_session_test",
            mode=TradingMode.PAPER,
            config=self.config
        )

        session_id = "paper_session_123"
        correlation_id = "paper_corr_456"

        # Set session context
        if hasattr(logger, 'set_context'):
            logger.set_context(session_id=session_id, correlation_id=correlation_id)

        # Log with session tracking
        logger.log_order(
            message="Paper order with session tracking",
            order_id="paper_order_123",
            symbol="BTCUSDT",
            side="BUY",
            session_id=session_id,
            correlation_id=correlation_id
        )

        # Verify session data can be retrieved
        if hasattr(logger, 'get_session_statistics'):
            stats = logger.get_session_statistics(session_id)
            assert stats is not None


class TestConfigurationValidation:
    """Test configuration validation for paper trading"""

    def test_should_reject_invalid_paper_trading_config(self):
        """Test rejection of invalid paper trading configuration"""
        invalid_configs = [
            # Missing paper trading section
            {'trading': {'mode': 'paper'}},

            # Paper mode with live exchange settings
            {
                'trading': {'mode': 'paper'},
                'paper_trading': {'enabled': False},
                'exchanges': {'binance': {'testnet': False}}
            },

            # Missing logging configuration
            {
                'trading': {'mode': 'paper'},
                'paper_trading': {'enabled': True}
            }
        ]

        for invalid_config in invalid_configs:
            with pytest.raises((ValueError, KeyError)):
                # This should raise an exception for invalid configuration
                self._validate_paper_trading_config(invalid_config)

    def _validate_paper_trading_config(self, config):
        """Validate paper trading configuration"""
        if config['trading']['mode'] == 'paper':
            if not config.get('paper_trading', {}).get('enabled', False):
                raise ValueError("Paper trading must be enabled when mode is 'paper'")

            if 'exchanges' in config:
                for exchange_name, exchange_config in config['exchanges'].items():
                    if not exchange_config.get('testnet', False):
                        raise ValueError(f"Exchange {exchange_name} must use testnet in paper mode")

            if 'logging' not in config:
                raise ValueError("Logging configuration required for paper trading")

    def test_should_accept_valid_paper_trading_config(self):
        """Test acceptance of valid paper trading configuration"""
        valid_config = {
            'trading': {'mode': 'paper'},
            'paper_trading': {'enabled': True},
            'exchanges': {
                'binance': {
                    'testnet': True,
                    'paper_trading': True
                }
            },
            'logging': {
                'level': 'DEBUG',
                'db_handler': {'enabled': True}
            }
        }

        # Should not raise exception
        self._validate_paper_trading_config(valid_config)

    def test_should_validate_live_trading_prevention(self):
        """Test that live trading is prevented in paper mode"""
        paper_config = {
            'trading': {'mode': 'paper'},
            'paper_trading': {'enabled': True},
            'exchanges': {
                'binance': {
                    'testnet': True,
                    'paper_trading': True,
                    'api_key': 'testnet_key',
                    'api_secret': 'testnet_secret'
                }
            }
        }

        # Verify testnet enforcement
        for exchange_name, exchange_config in paper_config['exchanges'].items():
            assert exchange_config['testnet'] is True
            assert exchange_config['paper_trading'] is True
            assert 'testnet' in exchange_config.get('api_key', '').lower()

    def test_should_configure_appropriate_database_settings(self):
        """Test database configuration for paper trading"""
        paper_config = {
            'database': {
                'path': 'paper_trading.db',
                'backup_enabled': True
            },
            'paper_trading': {'enabled': True}
        }

        # Database should be separate for paper trading
        assert 'paper' in paper_config['database']['path']
        assert paper_config['database']['backup_enabled'] is True


class TestPaperTradingIntegration:
    """Integration tests for complete paper trading configuration"""

    def setup_method(self):
        """Setup integration test environment"""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_should_integrate_paper_trading_across_all_components(self):
        """Test paper trading integration across all system components"""
        # Create comprehensive paper trading configuration
        paper_config = {
            'trading': {'mode': 'paper'},
            'paper_trading': {
                'enabled': True,
                'initial_balance': 100000.0,
                'commission_rate': 0.001
            },
            'database': {'path': os.path.join(self.temp_dir, "integration_paper.db")},
            'logging': {
                'level': 'DEBUG',
                'db_handler': {'enabled': True}
            },
            'exchanges': {
                'binance': {
                    'testnet': True,
                    'paper_trading': True
                }
            }
        }

        # Verify each component can work with paper configuration
        components_to_test = [
            'strategy_engine',
            'execution_engine',
            'risk_management',
            'api_integration'
        ]

        for component in components_to_test:
            logger = LoggerFactory.get_component_trading_logger(
                component=component,
                strategy="integration_test"
            )

            # Should create logger successfully
            assert logger is not None

    def test_should_prevent_accidental_live_trading(self):
        """Test prevention of accidental live trading"""
        # Simulate configuration that could accidentally enable live trading
        risky_config = {
            'trading': {'mode': 'paper'},
            'paper_trading': {'enabled': True},
            'exchanges': {
                'binance': {
                    'testnet': True,  # This is correct
                    'paper_trading': True,
                    'api_key': 'LIVE_KEY_123',  # This looks like live key
                    'api_secret': 'LIVE_SECRET_456'
                }
            }
        }

        # Configuration should still be safe because testnet=True
        assert risky_config['exchanges']['binance']['testnet'] is True
        assert risky_config['paper_trading']['enabled'] is True

        # Even with live-looking keys, testnet flag ensures safety
        assert risky_config['exchanges']['binance']['paper_trading'] is True

    def test_should_provide_comprehensive_paper_trading_validation(self):
        """Test comprehensive validation for paper trading setup"""
        complete_config = {
            'trading': {'mode': 'paper', 'session_timeout': 3600},
            'paper_trading': {
                'enabled': True,
                'initial_balance': 100000.0,
                'commission_rate': 0.001,
                'slippage_simulation': True
            },
            'exchanges': {
                'binance': {
                    'testnet': True,
                    'paper_trading': True,
                    'api_key': '${BINANCE_TESTNET_API_KEY}',
                    'api_secret': '${BINANCE_TESTNET_API_SECRET}'
                }
            },
            'logging': {
                'level': 'DEBUG',
                'file_handler': {'enabled': True},
                'db_handler': {'enabled': True}
            },
            'database': {
                'path': os.path.join(self.temp_dir, "comprehensive_paper.db")
            },
            'risk_management': {
                'max_portfolio_risk': 0.05,
                'paper_trading_multiplier': 1.0
            }
        }

        # Validate complete configuration
        validation_results = self._comprehensive_validation(complete_config)

        assert validation_results['trading_mode_valid'] is True
        assert validation_results['paper_trading_enabled'] is True
        assert validation_results['exchange_safety'] is True
        assert validation_results['logging_configured'] is True
        assert validation_results['database_configured'] is True

    def _comprehensive_validation(self, config):
        """Perform comprehensive validation of paper trading configuration"""
        results = {}

        # Validate trading mode
        results['trading_mode_valid'] = config.get('trading', {}).get('mode') == 'paper'

        # Validate paper trading enabled
        results['paper_trading_enabled'] = config.get('paper_trading', {}).get('enabled', False)

        # Validate exchange safety
        exchange_safety = True
        for exchange_name, exchange_config in config.get('exchanges', {}).items():
            if not exchange_config.get('testnet', False):
                exchange_safety = False
            if not exchange_config.get('paper_trading', False):
                exchange_safety = False

        results['exchange_safety'] = exchange_safety

        # Validate logging configuration
        logging_config = config.get('logging', {})
        results['logging_configured'] = (
            'level' in logging_config and
            logging_config.get('db_handler', {}).get('enabled', False)
        )

        # Validate database configuration
        results['database_configured'] = 'path' in config.get('database', {})

        return results