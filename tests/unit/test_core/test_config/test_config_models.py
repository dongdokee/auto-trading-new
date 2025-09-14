"""
Tests for Pydantic-based configuration management system.
Following TDD methodology: Red -> Green -> Refactor
"""

import pytest
import os
from decimal import Decimal
from pathlib import Path

# These imports will fail initially (Red phase)
# We'll implement them to make tests pass (Green phase)
try:
    from src.core.config.models import (
        DatabaseConfig, ExchangeConfig, RiskConfig, TradingConfig,
        MonitoringConfig, SystemConfig, AppConfig
    )
    from src.core.config.loader import ConfigLoader
    from src.core.config.validator import ConfigValidator
except ImportError:
    pytest.skip("Configuration system not yet implemented", allow_module_level=True)


class TestConfigModels:
    """Test Pydantic configuration models"""

    def test_should_create_database_config_with_defaults(self):
        """DatabaseConfig should have reasonable defaults"""
        db_config = DatabaseConfig()

        assert db_config.host == 'localhost'
        assert db_config.port == 5432
        assert db_config.database == 'autotrading'
        assert db_config.max_connections == 20
        assert db_config.connection_timeout == 30

    def test_should_validate_database_config_fields(self):
        """DatabaseConfig should validate field types and constraints"""
        # Valid configuration
        db_config = DatabaseConfig(
            host='db.example.com',
            port=5432,
            database='trading_prod',
            username='trader',
            password='secure_password',
            max_connections=50
        )

        assert db_config.host == 'db.example.com'
        assert db_config.port == 5432
        assert db_config.max_connections == 50

        # Invalid port should raise validation error
        with pytest.raises(ValueError):
            DatabaseConfig(port=70000)  # Port too high

        # Invalid max_connections should raise validation error
        with pytest.raises(ValueError):
            DatabaseConfig(max_connections=0)  # Must be positive

    def test_should_create_exchange_config_with_validation(self):
        """ExchangeConfig should validate API keys and settings"""
        exchange_config = ExchangeConfig(
            name='BINANCE',
            api_key='test_api_key_12345',
            api_secret='test_secret_67890',
            testnet=True,
            rate_limit_requests=1200,
            rate_limit_window=60
        )

        assert exchange_config.name == 'BINANCE'
        assert exchange_config.testnet is True
        assert exchange_config.rate_limit_requests == 1200

        # Invalid rate limit should raise error
        with pytest.raises(ValueError):
            ExchangeConfig(
                name='BINANCE',
                api_key='key',
                api_secret='secret',
                rate_limit_requests=0  # Must be positive
            )

    def test_should_create_risk_config_with_financial_constraints(self):
        """RiskConfig should validate financial risk parameters"""
        risk_config = RiskConfig(
            max_portfolio_risk_pct=Decimal('0.05'),
            max_position_risk_pct=Decimal('0.02'),
            max_drawdown_pct=Decimal('0.10'),
            max_leverage=Decimal('5.0'),
            var_confidence_level=Decimal('0.95'),
            kelly_fraction_limit=Decimal('0.25')
        )

        assert risk_config.max_portfolio_risk_pct == Decimal('0.05')
        assert risk_config.max_leverage == Decimal('5.0')
        assert risk_config.var_confidence_level == Decimal('0.95')

        # Invalid percentage should raise error
        with pytest.raises(ValueError):
            RiskConfig(max_drawdown_pct=Decimal('1.5'))  # > 100%

        # Invalid leverage should raise error
        with pytest.raises(ValueError):
            RiskConfig(max_leverage=Decimal('0'))  # Must be positive

    def test_should_create_trading_config_with_strategy_settings(self):
        """TradingConfig should handle strategy and trading parameters"""
        trading_config = TradingConfig(
            base_currency='USDT',
            trading_pairs=['BTCUSDT', 'ETHUSDT'],
            position_sizing_method='KELLY',
            default_order_type='LIMIT',
            slippage_tolerance_pct=Decimal('0.001'),
            enabled_strategies=['TrendFollowing', 'MeanReversion']
        )

        assert trading_config.base_currency == 'USDT'
        assert 'BTCUSDT' in trading_config.trading_pairs
        assert trading_config.position_sizing_method == 'KELLY'
        assert len(trading_config.enabled_strategies) == 2

    def test_should_create_monitoring_config_with_alerting(self):
        """MonitoringConfig should handle monitoring and alerting settings"""
        monitoring_config = MonitoringConfig(
            log_level='INFO',
            metrics_port=9090,
            health_check_port=8080,
            enable_telegram_alerts=True,
            telegram_bot_token='bot_token_123',
            telegram_chat_id='chat_456',
            alert_thresholds={
                'max_drawdown': 0.08,
                'var_breach': 3,
                'consecutive_losses': 5
            }
        )

        assert monitoring_config.log_level == 'INFO'
        assert monitoring_config.enable_telegram_alerts is True
        assert monitoring_config.alert_thresholds['max_drawdown'] == 0.08

    def test_should_create_system_config_with_performance_settings(self):
        """SystemConfig should handle system-level configuration"""
        system_config = SystemConfig(
            environment='production',
            debug=False,
            max_workers=8,
            memory_limit_mb=2048,
            cpu_limit_pct=80.0,
            data_retention_days=730
        )

        assert system_config.environment == 'production'
        assert system_config.debug is False
        assert system_config.max_workers == 8
        assert system_config.data_retention_days == 730


class TestAppConfig:
    """Test main application configuration"""

    def test_should_create_complete_app_config(self):
        """AppConfig should combine all configuration sections"""
        app_config = AppConfig(
            database=DatabaseConfig(),
            exchanges={
                'binance': ExchangeConfig(
                    name='BINANCE',
                    api_key='test_api_key_123',
                    api_secret='test_secret_456'
                )
            },
            risk=RiskConfig(),
            trading=TradingConfig(),
            monitoring=MonitoringConfig(),
            system=SystemConfig()
        )

        assert isinstance(app_config.database, DatabaseConfig)
        assert 'binance' in app_config.exchanges
        assert isinstance(app_config.risk, RiskConfig)
        assert isinstance(app_config.trading, TradingConfig)

    def test_should_validate_config_relationships(self):
        """AppConfig should validate relationships between config sections"""
        # Valid configuration
        app_config = AppConfig(
            database=DatabaseConfig(),
            exchanges={},
            risk=RiskConfig(max_leverage=Decimal('3.0')),
            trading=TradingConfig(),
            monitoring=MonitoringConfig(),
            system=SystemConfig(environment='development')
        )

        assert app_config.system.environment == 'development'
        assert app_config.risk.max_leverage == Decimal('3.0')


class TestConfigLoader:
    """Test configuration loading from various sources"""

    def test_should_load_config_from_environment_variables(self):
        """ConfigLoader should load from environment variables"""
        # Set environment variables
        os.environ['DB_HOST'] = 'test-db.example.com'
        os.environ['DB_PORT'] = '5433'
        os.environ['TRADING_BASE_CURRENCY'] = 'BTC'

        try:
            loader = ConfigLoader()
            config = loader.load_from_env()

            assert config.database.host == 'test-db.example.com'
            assert config.database.port == 5433
            assert config.trading.base_currency == 'BTC'

        finally:
            # Clean up environment variables
            os.environ.pop('DB_HOST', None)
            os.environ.pop('DB_PORT', None)
            os.environ.pop('TRADING_BASE_CURRENCY', None)

    def test_should_load_config_from_yaml_file(self):
        """ConfigLoader should load from YAML configuration file"""
        # Create temporary YAML config file
        yaml_content = '''
database:
  host: yaml-db.example.com
  port: 5434
  database: yaml_trading

risk:
  max_portfolio_risk_pct: 0.03
  max_leverage: 4.0

trading:
  base_currency: ETH
  trading_pairs:
    - ETHUSDT
    - BTCETH
'''

        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_file = f.name

        try:
            loader = ConfigLoader()
            config = loader.load_from_file(temp_file)

            assert config.database.host == 'yaml-db.example.com'
            assert config.database.port == 5434
            assert config.risk.max_leverage == Decimal('4.0')
            assert config.trading.base_currency == 'ETH'

        finally:
            os.unlink(temp_file)

    def test_should_merge_configs_with_precedence(self):
        """ConfigLoader should merge configs with proper precedence"""
        # Environment should override file config
        os.environ['DB_HOST'] = 'env-override.example.com'

        yaml_content = '''
database:
  host: file-db.example.com
  port: 5435
'''

        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_file = f.name

        try:
            loader = ConfigLoader()
            config = loader.load_merged(file_path=temp_file)

            # Environment variable should override file
            assert config.database.host == 'env-override.example.com'
            # File value should be used where no env override
            assert config.database.port == 5435

        finally:
            os.unlink(temp_file)
            os.environ.pop('DB_HOST', None)


class TestConfigValidator:
    """Test configuration validation and security checks"""

    def test_should_validate_production_config_requirements(self):
        """Validator should enforce production configuration requirements"""
        # Test with development environment first, then test production validation
        config = AppConfig(
            database=DatabaseConfig(),
            exchanges={
                'binance': ExchangeConfig(
                    name='BINANCE',
                    api_key='test_api_key_123',
                    api_secret='test_secret_456',
                    testnet=True  # Only testnet exchange
                )
            },
            risk=RiskConfig(),
            trading=TradingConfig(),
            monitoring=MonitoringConfig(),
            system=SystemConfig(environment='development')  # Create as dev first
        )

        # Manually set environment to production to test validator logic
        config.system.environment = 'production'

        validator = ConfigValidator()
        is_valid, errors = validator.validate_production_config(config)

        assert not is_valid
        assert len(errors) > 0
        assert any('non-testnet exchange' in error for error in errors)

    def test_should_validate_security_requirements(self):
        """Validator should check security requirements"""
        # Create config that passes model validation but should fail security checks
        config = AppConfig(
            database=DatabaseConfig(password='weak123'),  # Weak but long enough
            exchanges={
                'binance': ExchangeConfig(
                    name='BINANCE',
                    api_key='test_api_key_123',
                    api_secret='test_secret_456',
                    testnet=False  # Production with test keys
                )
            },
            risk=RiskConfig(),
            trading=TradingConfig(),
            monitoring=MonitoringConfig(),
            system=SystemConfig(environment='production', debug=False)  # Valid for model
        )

        validator = ConfigValidator()
        is_valid, security_issues = validator.validate_security(config)

        assert not is_valid
        assert len(security_issues) > 0

    def test_should_validate_risk_parameter_consistency(self):
        """Validator should check risk parameter consistency"""
        # Create config with valid individual fields but questionable combinations
        config = AppConfig(
            database=DatabaseConfig(),
            exchanges={},
            risk=RiskConfig(
                max_portfolio_risk_pct=Decimal('0.10'),  # 10%
                max_position_risk_pct=Decimal('0.08'),   # 8% < portfolio risk (valid)
                max_leverage=Decimal('20.0')              # Very high leverage
            ),
            trading=TradingConfig(),
            monitoring=MonitoringConfig(),
            system=SystemConfig()
        )

        validator = ConfigValidator()
        is_valid, risk_issues = validator.validate_risk_consistency(config)

        assert not is_valid
        assert len(risk_issues) > 0