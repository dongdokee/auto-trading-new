# src/integration/orchestrator/models.py
"""
Trading Orchestrator Core Models

This module contains the core data models and enums used by the trading orchestrator.
"""

from dataclasses import dataclass
from enum import Enum


class TradingState(Enum):
    """Trading system state"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class OrchestratorConfig:
    """Configuration for trading orchestrator"""
    enable_paper_trading: bool = True
    max_concurrent_orders: int = 10
    risk_check_interval_seconds: int = 30
    portfolio_rebalance_interval_seconds: int = 300  # 5 minutes
    health_check_interval_seconds: int = 60
    enable_auto_recovery: bool = True
    max_recovery_attempts: int = 3
    emergency_stop_on_risk_breach: bool = True