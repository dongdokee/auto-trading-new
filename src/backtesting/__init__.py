"""
백테스팅 프레임워크 모듈

이 모듈은 거래 전략의 백테스트 및 성과 분석을 위한 종합적인 프레임워크를 제공합니다.
TDD 방법론을 사용하여 개발되었으며, 과적합을 방지하기 위한 Walk-Forward 검증을 지원합니다.

주요 컴포넌트:
- DataLoader: 다양한 형태의 히스토리 데이터 로딩
- DataValidator: 데이터 품질 검증 및 전처리
- BacktestEngine: Walk-forward 백테스트 실행 엔진

Phase 2.1 구현 완료 - 2025-09-14
"""

from typing import Optional

# Data Loading and Validation
from .data_loader import DataLoader, DataSource, LoaderConfig
from .data_validator import DataValidator, ValidationConfig, ValidationResult, ValidationError

# Backtesting Engine
from .backtest_engine import (
    BacktestEngine, BacktestConfig, BacktestResult,
    StrategyInterface, Portfolio, Position, Trade,
    WalkForwardResult
)

__all__ = [
    # Data Processing
    'DataLoader',
    'DataSource',
    'LoaderConfig',

    # Data Validation
    'DataValidator',
    'ValidationConfig',
    'ValidationResult',
    'ValidationError',

    # Backtesting Engine
    'BacktestEngine',
    'BacktestConfig',
    'BacktestResult',
    'StrategyInterface',
    'Portfolio',
    'Position',
    'Trade',
    'WalkForwardResult'
]

# Version info
__version__ = '0.2.0'
__phase__ = 'Phase 2.1 - Backtesting Framework Completed'