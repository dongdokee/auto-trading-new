# src/integration/trading_orchestrator_new.py
"""
Trading Orchestrator - Backward Compatibility Wrapper

This is a backward compatibility wrapper for the refactored trading orchestrator.
All classes and functionality are imported from the new modular structure.

DEPRECATION WARNING: This wrapper will be removed in a future version.
Please update imports to use:
  from src.integration.orchestrator import TradingOrchestrator, TradingState, OrchestratorConfig
"""

import warnings
from .orchestrator import TradingOrchestrator, TradingState, OrchestratorConfig

# Issue deprecation warning
warnings.warn(
    "src.integration.trading_orchestrator is deprecated. "
    "Use 'from src.integration.orchestrator import TradingOrchestrator' instead. "
    "This compatibility layer will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export all classes for backward compatibility
__all__ = [
    "TradingState",
    "OrchestratorConfig",
    "TradingOrchestrator"
]