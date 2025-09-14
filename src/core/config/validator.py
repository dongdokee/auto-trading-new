"""
Configuration validation utilities for the AutoTrading system.
Provides security and consistency validation for configuration objects.
"""

from typing import List, Tuple, Dict, Any
from decimal import Decimal
import re

from .models import AppConfig


class ConfigValidator:
    """Configuration validator for security and consistency checks"""

    def __init__(self):
        self._security_patterns = {
            'weak_passwords': [
                r'^(password|123456|admin|test)$',
                r'^.{1,7}$',  # Too short
            ],
            'test_api_keys': [
                r'^(test|demo|sample)',
                r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}',  # UUID-like test keys
            ]
        }

    def validate_production_config(self, config: AppConfig) -> Tuple[bool, List[str]]:
        """Validate configuration for production environment"""
        errors = []

        if config.system.environment == 'production':
            # Check for required production settings
            if config.system.debug:
                errors.append("Debug mode cannot be enabled in production")

            # Check for exchange configurations
            if not config.exchanges:
                errors.append("Production environment requires at least one exchange configuration")
            else:
                production_exchanges = [
                    ex for ex in config.exchanges.values() if not ex.testnet
                ]
                if not production_exchanges:
                    errors.append("Production environment requires at least one non-testnet exchange")

            # Check for API keys
            for name, exchange in config.exchanges.items():
                if not exchange.api_key or not exchange.api_secret:
                    errors.append(f"Exchange '{name}' missing API credentials")

            # Check database credentials
            if not config.database.username or not config.database.password:
                errors.append("Database credentials required for production")

            # Check monitoring configuration
            if config.monitoring.enable_telegram_alerts:
                if not config.monitoring.telegram_bot_token:
                    errors.append("Telegram bot token required when alerts enabled")

        return len(errors) == 0, errors

    def validate_security(self, config: AppConfig) -> Tuple[bool, List[str]]:
        """Validate security aspects of configuration"""
        issues = []

        # Check database password strength
        if config.database.password:
            if self._is_weak_password(config.database.password):
                issues.append("Database password is too weak")

        # Check for test API keys in production
        if config.system.environment == 'production':
            for name, exchange in config.exchanges.items():
                if self._is_test_api_key(exchange.api_key):
                    issues.append(f"Exchange '{name}' appears to use test API key in production")

                if not exchange.testnet and config.system.debug:
                    issues.append(f"Exchange '{name}' in production mode with debug enabled")

        # Check for insecure settings
        if config.system.environment == 'production' and config.system.debug:
            issues.append("Debug mode enabled in production environment")

        # Check SSL settings for production
        if (config.system.environment == 'production' and
            not config.database.ssl_enabled):
            issues.append("SSL should be enabled for production database connections")

        return len(issues) == 0, issues

    def validate_risk_consistency(self, config: AppConfig) -> Tuple[bool, List[str]]:
        """Validate risk management parameter consistency"""
        issues = []
        risk = config.risk

        # Check risk parameter relationships
        if risk.max_position_risk_pct > risk.max_portfolio_risk_pct:
            issues.append(
                f"Position risk ({risk.max_position_risk_pct}) cannot exceed "
                f"portfolio risk ({risk.max_portfolio_risk_pct})"
            )

        # Check leverage vs risk parameters
        if risk.max_leverage > Decimal('10.0') and risk.max_drawdown_pct < Decimal('0.05'):
            issues.append(
                f"High leverage ({risk.max_leverage}) with low drawdown limit "
                f"({risk.max_drawdown_pct}) may be inconsistent"
            )

        # Check Kelly fraction vs leverage
        if risk.kelly_fraction_limit > Decimal('0.5') and risk.max_leverage > Decimal('5.0'):
            issues.append(
                f"High Kelly fraction ({risk.kelly_fraction_limit}) with high leverage "
                f"({risk.max_leverage}) creates excessive risk"
            )

        # Warn about very conservative settings
        if (risk.max_portfolio_risk_pct < Decimal('0.005') and
            risk.max_leverage < Decimal('2.0')):
            issues.append(
                "Risk settings are very conservative - may limit trading opportunities"
            )

        # Warn about very aggressive settings
        if (risk.max_portfolio_risk_pct > Decimal('0.05') or
            risk.max_leverage > Decimal('20.0')):
            issues.append(
                "Risk settings are very aggressive - consider reducing for safety"
            )

        return len(issues) == 0, issues

    def validate_trading_consistency(self, config: AppConfig) -> Tuple[bool, List[str]]:
        """Validate trading configuration consistency"""
        issues = []
        trading = config.trading

        # Check trading pairs format
        for pair in trading.trading_pairs:
            if not self._is_valid_trading_pair(pair):
                issues.append(f"Invalid trading pair format: {pair}")

        # Check base currency consistency
        base_currency_pairs = [
            pair for pair in trading.trading_pairs
            if pair.endswith(trading.base_currency)
        ]

        if not base_currency_pairs:
            issues.append(
                f"No trading pairs found for base currency {trading.base_currency}"
            )

        # Check strategy configuration
        if not trading.enabled_strategies:
            issues.append("No trading strategies enabled")

        # Check position limits vs risk settings
        max_positions = trading.max_open_positions
        position_risk = config.risk.max_position_risk_pct

        if max_positions * position_risk > Decimal('0.50'):
            issues.append(
                f"Maximum positions ({max_positions}) × position risk ({position_risk}) "
                f"could exceed 50% portfolio exposure"
            )

        return len(issues) == 0, issues

    def validate_monitoring_setup(self, config: AppConfig) -> Tuple[bool, List[str]]:
        """Validate monitoring and alerting configuration"""
        issues = []
        monitoring = config.monitoring

        # Check port conflicts
        if monitoring.metrics_port == monitoring.health_check_port:
            issues.append(
                f"Metrics port and health check port cannot be the same "
                f"({monitoring.metrics_port})"
            )

        # Check alert thresholds
        thresholds = monitoring.alert_thresholds

        if 'max_drawdown' in thresholds:
            dd_threshold = thresholds['max_drawdown']
            if dd_threshold > config.risk.max_drawdown_pct:
                issues.append(
                    f"Drawdown alert threshold ({dd_threshold}) exceeds risk limit "
                    f"({config.risk.max_drawdown_pct})"
                )

        # Check Telegram configuration
        if monitoring.enable_telegram_alerts:
            if not monitoring.telegram_bot_token:
                issues.append("Telegram alerts enabled but no bot token provided")
            if not monitoring.telegram_chat_id:
                issues.append("Telegram alerts enabled but no chat ID provided")

        return len(issues) == 0, issues

    def get_validation_report(self, config: AppConfig) -> Dict[str, Any]:
        """Get comprehensive validation report"""
        report = {
            'overall_valid': True,
            'sections': {}
        }

        # Validate each section
        validations = [
            ('production', self.validate_production_config),
            ('security', self.validate_security),
            ('risk_consistency', self.validate_risk_consistency),
            ('trading_consistency', self.validate_trading_consistency),
            ('monitoring_setup', self.validate_monitoring_setup),
        ]

        for section_name, validator_func in validations:
            is_valid, issues = validator_func(config)
            report['sections'][section_name] = {
                'valid': is_valid,
                'issues': issues
            }

            if not is_valid:
                report['overall_valid'] = False

        return report

    def _is_weak_password(self, password: str) -> bool:
        """Check if password is weak"""
        for pattern in self._security_patterns['weak_passwords']:
            if re.search(pattern, password, re.IGNORECASE):
                return True
        return False

    def _is_test_api_key(self, api_key: str) -> bool:
        """Check if API key appears to be a test key"""
        for pattern in self._security_patterns['test_api_keys']:
            if re.search(pattern, api_key, re.IGNORECASE):
                return True
        return False

    def _is_valid_trading_pair(self, pair: str) -> bool:
        """Validate trading pair format"""
        # Basic format check: should be like BTCUSDT, ETHBTC, etc.
        if len(pair) < 6 or len(pair) > 12:
            return False

        # Should contain only alphanumeric characters
        if not pair.isalnum():
            return False

        # Should be uppercase (convention)
        if not pair.isupper():
            return False

        return True