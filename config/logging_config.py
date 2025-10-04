"""
Logging Configuration Loader and Manager

Provides centralized loading and management of logging configurations
for different trading modes (paper/live) with environment detection.

Key Features:
- Automatic mode detection
- Environment-specific overrides
- Configuration validation
- Runtime configuration updates
- Integration with LoggerFactory
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass

from src.core.patterns.logging import LoggerFactory


@dataclass
class LoggingConfig:
    """Logging configuration data class"""
    mode: str
    level: str
    enable_structured: bool
    paper_trading: bool
    testnet: bool

    # File logging
    file_logging_enabled: bool
    log_dir: str
    max_file_size: str
    backup_count: int

    # Trading specific
    trade_journal_enabled: bool
    performance_analytics_enabled: bool
    compliance_logging_enabled: bool

    # Component levels
    component_levels: Dict[str, str]

    # Advanced features
    advanced_features: Dict[str, Any]

    # Alerting
    alerting_config: Dict[str, Any]

    # Raw config for custom access
    raw_config: Dict[str, Any]


class LoggingConfigLoader:
    """
    Centralized logging configuration loader and manager

    Handles loading, validation, and application of logging configurations
    with support for different trading modes and environments.
    """

    def __init__(self, config_dir: str = "config"):
        """
        Initialize config loader

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.current_config: Optional[LoggingConfig] = None
        self.logger = logging.getLogger("logging_config_loader")

    def load_config(
        self,
        mode: Optional[str] = None,
        environment: Optional[str] = None,
        config_file: Optional[str] = None
    ) -> LoggingConfig:
        """
        Load logging configuration

        Args:
            mode: Trading mode (paper/live) - auto-detect if None
            environment: Environment (development/staging/production)
            config_file: Specific config file path

        Returns:
            LoggingConfig object
        """
        try:
            # Auto-detect mode if not specified
            if mode is None:
                mode = self._detect_trading_mode()

            # Auto-detect environment if not specified
            if environment is None:
                environment = self._detect_environment()

            # Determine config file
            if config_file is None:
                config_file = self._get_config_file_path(mode)

            # Load configuration
            raw_config = self._load_yaml_config(config_file)

            # Apply environment overrides
            if environment in raw_config.get('environment_overrides', {}):
                raw_config = self._apply_environment_overrides(
                    raw_config, environment
                )

            # Parse configuration
            config = self._parse_config(raw_config, mode)

            # Validate configuration
            self._validate_config(config)

            # Store current config
            self.current_config = config

            self.logger.info(f"Logging configuration loaded: mode={mode}, env={environment}")
            return config

        except Exception as e:
            self.logger.error(f"Failed to load logging configuration: {e}")
            # Return fallback configuration
            return self._get_fallback_config(mode or "paper")

    def apply_config(self, config: LoggingConfig) -> bool:
        """
        Apply logging configuration to the system

        Args:
            config: LoggingConfig to apply

        Returns:
            True if successful
        """
        try:
            # Initialize LoggerFactory with configuration
            LoggerFactory.initialize(
                log_level=getattr(logging, config.level.upper()),
                enable_structured=config.enable_structured,
                trading_mode=config.mode,
                paper_trading=config.paper_trading
            )

            # Create log directories
            if config.file_logging_enabled:
                os.makedirs(config.log_dir, exist_ok=True)

            # Apply component-specific log levels
            for component, level in config.component_levels.items():
                try:
                    component_logger = logging.getLogger(component)
                    component_logger.setLevel(getattr(logging, level.upper()))
                except AttributeError:
                    self.logger.warning(f"Invalid log level for {component}: {level}")

            self.logger.info(f"Logging configuration applied: {config.mode}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to apply logging configuration: {e}")
            return False

    def update_component_level(self, component: str, level: str) -> bool:
        """
        Update log level for a specific component

        Args:
            component: Component name
            level: New log level

        Returns:
            True if successful
        """
        try:
            # Update logger
            component_logger = logging.getLogger(component)
            component_logger.setLevel(getattr(logging, level.upper()))

            # Update current config
            if self.current_config:
                self.current_config.component_levels[component] = level

            self.logger.info(f"Updated log level for {component}: {level}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to update component log level: {e}")
            return False

    def get_current_config(self) -> Optional[LoggingConfig]:
        """Get current logging configuration"""
        return self.current_config

    def _detect_trading_mode(self) -> str:
        """Detect trading mode from environment"""
        # Check environment variables
        mode = os.getenv('TRADING_MODE')
        if mode:
            return mode.lower()

        # Check for paper trading indicators
        paper_trading = os.getenv('PAPER_TRADING', '').lower()
        testnet = os.getenv('TESTNET', '').lower()

        if paper_trading in ['true', '1', 'yes'] or testnet in ['true', '1', 'yes']:
            return 'paper'

        # Check for live trading indicators
        live_trading = os.getenv('LIVE_TRADING', '').lower()
        if live_trading in ['true', '1', 'yes']:
            return 'live'

        # Default to paper trading for safety
        return 'paper'

    def _detect_environment(self) -> str:
        """Detect environment from various sources"""
        # Check explicit environment variable
        env = os.getenv('ENVIRONMENT')
        if env:
            return env.lower()

        # Check common environment indicators
        if os.getenv('PRODUCTION') == 'true':
            return 'production'
        if os.getenv('STAGING') == 'true':
            return 'staging'
        if os.getenv('DEVELOPMENT') == 'true':
            return 'development'

        # Check Python environment
        if os.getenv('PYTHONOPTIMIZE'):
            return 'production'

        # Default to development
        return 'development'

    def _get_config_file_path(self, mode: str) -> str:
        """Get configuration file path for mode"""
        filename = f"logging_{mode}.yaml"
        return str(self.config_dir / filename)

    def _load_yaml_config(self, config_file: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            if not config:
                raise ValueError("Empty configuration file")

            return config

        except FileNotFoundError:
            self.logger.warning(f"Config file not found: {config_file}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"YAML parsing error: {e}")
            raise

    def _apply_environment_overrides(
        self,
        config: Dict[str, Any],
        environment: str
    ) -> Dict[str, Any]:
        """Apply environment-specific overrides"""
        overrides = config.get('environment_overrides', {}).get(environment, {})

        if not overrides:
            return config

        # Deep merge overrides
        def deep_merge(base: Dict, override: Dict) -> Dict:
            result = base.copy()
            for key, value in override.items():
                if '.' in key:
                    # Handle nested keys like "logging.level"
                    keys = key.split('.')
                    target = result
                    for k in keys[:-1]:
                        if k == '*':
                            # Wildcard support for component levels
                            continue
                        if k not in target:
                            target[k] = {}
                        target = target[k]

                    if keys[-1] == '*' and 'components' in result:
                        # Apply to all components
                        for comp in result['components']:
                            if isinstance(result['components'][comp], dict):
                                result['components'][comp]['level'] = value
                    else:
                        target[keys[-1]] = value
                else:
                    if isinstance(value, dict) and key in result:
                        result[key] = deep_merge(result[key], value)
                    else:
                        result[key] = value
            return result

        return deep_merge(config, overrides)

    def _parse_config(self, raw_config: Dict[str, Any], mode: str) -> LoggingConfig:
        """Parse raw configuration into LoggingConfig object"""
        # Main logging settings
        logging_config = raw_config.get('logging', {})

        # Component levels
        components = raw_config.get('components', {})
        component_levels = {}
        for comp_name, comp_config in components.items():
            if isinstance(comp_config, dict):
                component_levels[comp_name] = comp_config.get('level', 'INFO')
            else:
                component_levels[comp_name] = 'INFO'

        # Trading logging settings
        trading_logging = raw_config.get('trading_logging', {})
        unified_logger = trading_logging.get('unified_logger', {})

        # File logging settings
        file_logging = logging_config.get('file_logging', {})

        return LoggingConfig(
            mode=mode,
            level=logging_config.get('level', 'INFO'),
            enable_structured=logging_config.get('enable_structured', True),
            paper_trading=logging_config.get('paper_trading', mode == 'paper'),
            testnet=logging_config.get('testnet', mode in ['paper', 'demo']),

            # File logging
            file_logging_enabled=file_logging.get('enabled', True),
            log_dir=file_logging.get('log_dir', f'logs/{mode}'),
            max_file_size=file_logging.get('max_file_size', '100MB'),
            backup_count=file_logging.get('backup_count', 10),

            # Trading specific
            trade_journal_enabled=unified_logger.get('enable_trade_journal', True),
            performance_analytics_enabled=trading_logging.get('performance_analytics', {}).get('enabled', True),
            compliance_logging_enabled=unified_logger.get('enable_compliance_logging', mode == 'live'),

            # Component levels
            component_levels=component_levels,

            # Advanced features
            advanced_features=raw_config.get('advanced_features', {}),

            # Alerting
            alerting_config=raw_config.get('alerting', {}),

            # Raw config
            raw_config=raw_config
        )

    def _validate_config(self, config: LoggingConfig) -> None:
        """Validate configuration"""
        # Check required fields
        if not config.mode:
            raise ValueError("Trading mode is required")

        if not config.level:
            raise ValueError("Log level is required")

        # Validate log level
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if config.level.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {config.level}")

        # Validate component levels
        for component, level in config.component_levels.items():
            if level.upper() not in valid_levels:
                self.logger.warning(f"Invalid log level for {component}: {level}")

        # Mode-specific validation
        if config.mode == 'live':
            if not config.compliance_logging_enabled:
                self.logger.warning("Compliance logging disabled for live trading")

    def _get_fallback_config(self, mode: str) -> LoggingConfig:
        """Get fallback configuration"""
        return LoggingConfig(
            mode=mode,
            level='INFO',
            enable_structured=True,
            paper_trading=mode == 'paper',
            testnet=mode in ['paper', 'demo'],
            file_logging_enabled=True,
            log_dir=f'logs/{mode}',
            max_file_size='100MB',
            backup_count=10,
            trade_journal_enabled=True,
            performance_analytics_enabled=True,
            compliance_logging_enabled=mode == 'live',
            component_levels={},
            advanced_features={},
            alerting_config={},
            raw_config={}
        )


# Global instance
_config_loader = None


def get_config_loader() -> LoggingConfigLoader:
    """Get global logging configuration loader"""
    global _config_loader
    if _config_loader is None:
        _config_loader = LoggingConfigLoader()
    return _config_loader


def initialize_logging(
    mode: Optional[str] = None,
    environment: Optional[str] = None,
    config_file: Optional[str] = None
) -> LoggingConfig:
    """
    Initialize logging system with configuration

    Args:
        mode: Trading mode (paper/live)
        environment: Environment (development/staging/production)
        config_file: Specific config file

    Returns:
        Applied LoggingConfig
    """
    loader = get_config_loader()
    config = loader.load_config(mode, environment, config_file)
    loader.apply_config(config)
    return config


def update_component_log_level(component: str, level: str) -> bool:
    """
    Update log level for a specific component

    Args:
        component: Component name
        level: New log level

    Returns:
        True if successful
    """
    loader = get_config_loader()
    return loader.update_component_level(component, level)


def get_current_logging_config() -> Optional[LoggingConfig]:
    """Get current logging configuration"""
    loader = get_config_loader()
    return loader.get_current_config()