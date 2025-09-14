"""
Pydantic-based configuration models for the AutoTrading system.
Provides type-safe configuration with validation and serialization.
"""

from decimal import Decimal
from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from pathlib import Path


class DatabaseConfig(BaseModel):
    """Database connection and pool configuration"""
    host: str = Field(default='localhost', description="Database host")
    port: int = Field(default=5432, ge=1024, le=65535, description="Database port")
    database: str = Field(default='autotrading', description="Database name")
    username: Optional[str] = Field(default=None, description="Database username")
    password: Optional[str] = Field(default=None, description="Database password")
    max_connections: int = Field(default=20, ge=1, le=100, description="Maximum connection pool size")
    connection_timeout: int = Field(default=30, ge=1, le=300, description="Connection timeout in seconds")
    query_timeout: int = Field(default=60, ge=1, le=600, description="Query timeout in seconds")
    ssl_enabled: bool = Field(default=False, description="Enable SSL connection")

    model_config = ConfigDict(env_prefix='DB_')


class ExchangeConfig(BaseModel):
    """Exchange API configuration"""
    name: Literal['BINANCE', 'BYBIT', 'OKX'] = Field(..., description="Exchange name")
    api_key: str = Field(..., description="API key")
    api_secret: str = Field(..., description="API secret")
    passphrase: Optional[str] = Field(default=None, description="API passphrase (required for some exchanges)")
    testnet: bool = Field(default=True, description="Use testnet/sandbox")
    rate_limit_requests: int = Field(default=1200, ge=1, description="Rate limit requests per window")
    rate_limit_window: int = Field(default=60, ge=1, description="Rate limit window in seconds")
    timeout: int = Field(default=30, ge=1, le=300, description="Request timeout in seconds")
    retry_attempts: int = Field(default=3, ge=0, le=10, description="Number of retry attempts")

    @field_validator('api_key', 'api_secret')
    @classmethod
    def validate_api_credentials(cls, v):
        if len(v) < 8:
            raise ValueError('API credentials must be at least 8 characters long')
        return v


class RiskConfig(BaseModel):
    """Risk management configuration"""
    max_portfolio_risk_pct: Decimal = Field(
        default=Decimal('0.02'),
        ge=Decimal('0.001'),
        le=Decimal('0.20'),
        description="Maximum portfolio risk as percentage (daily VaR)"
    )
    max_position_risk_pct: Decimal = Field(
        default=Decimal('0.01'),
        ge=Decimal('0.001'),
        le=Decimal('0.50'),
        description="Maximum single position risk as percentage"
    )
    max_drawdown_pct: Decimal = Field(
        default=Decimal('0.10'),
        ge=Decimal('0.01'),
        le=Decimal('0.50'),
        description="Maximum allowed drawdown percentage"
    )
    max_leverage: Decimal = Field(
        default=Decimal('3.0'),
        ge=Decimal('1.0'),
        le=Decimal('100.0'),
        description="Maximum leverage allowed"
    )
    var_confidence_level: Decimal = Field(
        default=Decimal('0.95'),
        ge=Decimal('0.90'),
        le=Decimal('0.99'),
        description="VaR confidence level"
    )
    kelly_fraction_limit: Decimal = Field(
        default=Decimal('0.25'),
        ge=Decimal('0.01'),
        le=Decimal('1.0'),
        description="Maximum Kelly fraction"
    )
    max_consecutive_loss_days: int = Field(
        default=5,
        ge=1,
        le=30,
        description="Maximum consecutive loss days before halt"
    )

    @model_validator(mode='after')
    def validate_risk_relationships(self):
        if self.max_position_risk_pct > self.max_portfolio_risk_pct:
            raise ValueError('Position risk cannot exceed portfolio risk')
        return self

    model_config = ConfigDict(env_prefix='RISK_')


class TradingConfig(BaseModel):
    """Trading system configuration"""
    base_currency: str = Field(default='USDT', description="Base trading currency")
    trading_pairs: List[str] = Field(
        default_factory=lambda: ['BTCUSDT', 'ETHUSDT'],
        description="List of trading pairs"
    )
    position_sizing_method: Literal['FIXED', 'KELLY', 'ATR'] = Field(
        default='KELLY',
        description="Position sizing method"
    )
    default_order_type: Literal['MARKET', 'LIMIT'] = Field(
        default='LIMIT',
        description="Default order type"
    )
    slippage_tolerance_pct: Decimal = Field(
        default=Decimal('0.001'),
        ge=Decimal('0.0001'),
        le=Decimal('0.01'),
        description="Slippage tolerance percentage"
    )
    enabled_strategies: List[str] = Field(
        default_factory=lambda: ['TrendFollowing'],
        description="List of enabled strategy names"
    )
    min_trade_size_usdt: Decimal = Field(
        default=Decimal('10.0'),
        ge=Decimal('1.0'),
        description="Minimum trade size in USDT"
    )
    max_open_positions: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of open positions"
    )

    @field_validator('trading_pairs')
    @classmethod
    def validate_trading_pairs(cls, v):
        if not v:
            raise ValueError('At least one trading pair must be specified')

        for pair in v:
            if not pair or len(pair) < 6:
                raise ValueError(f'Invalid trading pair format: {pair}')

        return v

    model_config = ConfigDict(env_prefix='TRADING_')


class MonitoringConfig(BaseModel):
    """Monitoring and alerting configuration"""
    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = Field(
        default='INFO',
        description="Logging level"
    )
    metrics_port: int = Field(
        default=9090,
        ge=1024,
        le=65535,
        description="Prometheus metrics port"
    )
    health_check_port: int = Field(
        default=8080,
        ge=1024,
        le=65535,
        description="Health check endpoint port"
    )
    enable_telegram_alerts: bool = Field(
        default=False,
        description="Enable Telegram notifications"
    )
    telegram_bot_token: Optional[str] = Field(
        default=None,
        description="Telegram bot token"
    )
    telegram_chat_id: Optional[str] = Field(
        default=None,
        description="Telegram chat ID"
    )
    alert_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            'max_drawdown': 0.05,
            'var_breach': 3,
            'consecutive_losses': 3
        },
        description="Alert threshold configuration"
    )

    @model_validator(mode='after')
    def validate_telegram_config(self):
        if self.enable_telegram_alerts and (not self.telegram_bot_token or not self.telegram_chat_id):
            raise ValueError('Telegram bot token and chat ID required when alerts enabled')
        return self

    model_config = ConfigDict(env_prefix='MONITORING_')


class SystemConfig(BaseModel):
    """System-level configuration"""
    environment: Literal['development', 'staging', 'production'] = Field(
        default='development',
        description="Application environment"
    )
    debug: bool = Field(default=False, description="Enable debug mode")
    max_workers: int = Field(default=4, ge=1, le=32, description="Maximum worker processes")
    memory_limit_mb: int = Field(
        default=1024,
        ge=256,
        le=8192,
        description="Memory limit in MB"
    )
    cpu_limit_pct: float = Field(
        default=70.0,
        ge=10.0,
        le=95.0,
        description="CPU usage limit percentage"
    )
    data_retention_days: int = Field(
        default=365,
        ge=30,
        le=2555,  # 7 years
        description="Data retention period in days"
    )
    backup_enabled: bool = Field(default=True, description="Enable automatic backups")

    @model_validator(mode='after')
    def validate_production_settings(self):
        if self.environment == 'production' and self.debug:
            raise ValueError('Debug mode cannot be enabled in production')
        return self

    model_config = ConfigDict(env_prefix='SYSTEM_')


class AppConfig(BaseModel):
    """Main application configuration combining all subsystem configs"""
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    exchanges: Dict[str, ExchangeConfig] = Field(
        default_factory=dict,
        description="Exchange configurations by name"
    )
    risk: RiskConfig = Field(default_factory=RiskConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)

    @model_validator(mode='after')
    def validate_overall_consistency(self):
        """Validate relationships between different config sections"""
        # Production environment should have at least one exchange configured
        if (self.system.environment == 'production' and
            (not self.exchanges or not any(not ex.testnet for ex in self.exchanges.values()))):
            raise ValueError('Production environment requires at least one non-testnet exchange')
        return self

    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid'
    )