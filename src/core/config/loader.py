"""
Configuration loading utilities for the AutoTrading system.
Supports loading from environment variables, YAML files, and merging configs.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from decimal import Decimal

from .models import AppConfig, DatabaseConfig, ExchangeConfig, RiskConfig
from .models import TradingConfig, MonitoringConfig, SystemConfig


class ConfigLoader:
    """Configuration loader with support for multiple sources"""

    def __init__(self):
        self._env_mapping = self._create_env_mapping()

    def _create_env_mapping(self) -> Dict[str, str]:
        """Create mapping from environment variables to config paths"""
        return {
            # Database config
            'DB_HOST': 'database.host',
            'DB_PORT': 'database.port',
            'DB_DATABASE': 'database.database',
            'DB_USERNAME': 'database.username',
            'DB_PASSWORD': 'database.password',
            'DB_MAX_CONNECTIONS': 'database.max_connections',

            # Risk config
            'RISK_MAX_PORTFOLIO_RISK_PCT': 'risk.max_portfolio_risk_pct',
            'RISK_MAX_POSITION_RISK_PCT': 'risk.max_position_risk_pct',
            'RISK_MAX_DRAWDOWN_PCT': 'risk.max_drawdown_pct',
            'RISK_MAX_LEVERAGE': 'risk.max_leverage',

            # Trading config
            'TRADING_BASE_CURRENCY': 'trading.base_currency',
            'TRADING_POSITION_SIZING_METHOD': 'trading.position_sizing_method',
            'TRADING_DEFAULT_ORDER_TYPE': 'trading.default_order_type',

            # Monitoring config
            'MONITORING_LOG_LEVEL': 'monitoring.log_level',
            'MONITORING_METRICS_PORT': 'monitoring.metrics_port',
            'MONITORING_ENABLE_TELEGRAM_ALERTS': 'monitoring.enable_telegram_alerts',
            'MONITORING_TELEGRAM_BOT_TOKEN': 'monitoring.telegram_bot_token',
            'MONITORING_TELEGRAM_CHAT_ID': 'monitoring.telegram_chat_id',

            # System config
            'SYSTEM_ENVIRONMENT': 'system.environment',
            'SYSTEM_DEBUG': 'system.debug',
            'SYSTEM_MAX_WORKERS': 'system.max_workers',

            # Exchange configs (common patterns)
            'BINANCE_API_KEY': 'exchanges.binance.api_key',
            'BINANCE_API_SECRET': 'exchanges.binance.api_secret',
            'BINANCE_TESTNET': 'exchanges.binance.testnet',
        }

    def load_from_env(self) -> AppConfig:
        """Load configuration from environment variables"""
        config_dict = self._build_config_dict_from_env()
        return AppConfig(**config_dict)

    def load_from_file(self, file_path: Union[str, Path]) -> AppConfig:
        """Load configuration from YAML file"""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        # Convert string decimals to Decimal objects
        config_dict = self._convert_decimal_values(config_dict)

        return AppConfig(**config_dict)

    def load_merged(self, file_path: Optional[Union[str, Path]] = None) -> AppConfig:
        """Load configuration with environment variables overriding file config"""
        config_dict = {}

        # Start with file config if provided
        if file_path:
            file_config = self.load_from_file(file_path)
            config_dict = file_config.model_dump()

        # Override with environment variables
        env_dict = self._build_config_dict_from_env()
        config_dict = self._deep_merge_dicts(config_dict, env_dict)

        return AppConfig(**config_dict)

    def _build_config_dict_from_env(self) -> Dict[str, Any]:
        """Build configuration dictionary from environment variables"""
        config_dict = {
            'database': {},
            'exchanges': {},
            'risk': {},
            'trading': {},
            'monitoring': {},
            'system': {}
        }

        for env_var, config_path in self._env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert value to appropriate type
                converted_value = self._convert_env_value(value, config_path)
                self._set_nested_dict_value(config_dict, config_path, converted_value)

        # Handle special cases for exchanges
        self._load_exchange_configs_from_env(config_dict)

        # Handle trading pairs from environment
        trading_pairs = os.getenv('TRADING_PAIRS')
        if trading_pairs:
            config_dict['trading']['trading_pairs'] = [
                pair.strip() for pair in trading_pairs.split(',')
            ]

        # Handle enabled strategies from environment
        enabled_strategies = os.getenv('TRADING_ENABLED_STRATEGIES')
        if enabled_strategies:
            config_dict['trading']['enabled_strategies'] = [
                strategy.strip() for strategy in enabled_strategies.split(',')
            ]

        return config_dict

    def _load_exchange_configs_from_env(self, config_dict: Dict[str, Any]):
        """Load exchange configurations from environment variables"""
        # Look for exchange-specific environment variables
        for env_var, env_value in os.environ.items():
            if env_var.endswith('_API_KEY'):
                exchange_name = env_var.replace('_API_KEY', '').lower()
                secret_var = f"{exchange_name.upper()}_API_SECRET"
                testnet_var = f"{exchange_name.upper()}_TESTNET"

                if secret_var in os.environ:
                    if exchange_name not in config_dict['exchanges']:
                        config_dict['exchanges'][exchange_name] = {}

                    config_dict['exchanges'][exchange_name].update({
                        'name': exchange_name.upper(),
                        'api_key': env_value,
                        'api_secret': os.environ[secret_var],
                        'testnet': self._convert_boolean(os.getenv(testnet_var, 'true'))
                    })

    def _convert_env_value(self, value: str, config_path: str) -> Any:
        """Convert environment variable value to appropriate type"""
        # Boolean conversion
        if any(key in config_path.lower() for key in ['debug', 'testnet', 'enable', 'ssl']):
            return self._convert_boolean(value)

        # Integer conversion
        if any(key in config_path.lower() for key in ['port', 'connections', 'workers', 'days', 'attempts']):
            try:
                return int(value)
            except ValueError:
                return value

        # Float conversion
        if 'pct' in config_path.lower() or 'cpu_limit' in config_path.lower():
            try:
                return float(value)
            except ValueError:
                return value

        # Decimal conversion
        if any(key in config_path.lower() for key in ['risk', 'leverage', 'slippage', 'kelly']):
            try:
                return Decimal(value)
            except:
                return value

        return value

    def _convert_boolean(self, value: str) -> bool:
        """Convert string to boolean"""
        return value.lower() in ('true', '1', 'yes', 'on')

    def _convert_decimal_values(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert string values to Decimal where appropriate"""
        if not isinstance(config_dict, dict):
            return config_dict

        converted = {}
        for key, value in config_dict.items():
            if isinstance(value, dict):
                converted[key] = self._convert_decimal_values(value)
            elif isinstance(value, str):
                # Try to convert decimal-like values
                if any(decimal_key in key.lower() for decimal_key in
                       ['pct', 'leverage', 'fraction', 'slippage']):
                    try:
                        converted[key] = Decimal(value)
                    except:
                        converted[key] = value
                else:
                    converted[key] = value
            else:
                converted[key] = value

        return converted

    def _set_nested_dict_value(self, config_dict: Dict[str, Any], path: str, value: Any):
        """Set value in nested dictionary using dot notation path"""
        keys = path.split('.')
        current = config_dict

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _deep_merge_dicts(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries, with override values taking precedence"""
        result = base.copy()

        for key, value in override.items():
            if (key in result and
                isinstance(result[key], dict) and
                isinstance(value, dict)):
                result[key] = self._deep_merge_dicts(result[key], value)
            else:
                result[key] = value

        return result