"""
DataValidator 클래스 단위 테스트

TDD 방법론을 사용하여 데이터 품질 검증기의 모든 기능을 테스트합니다.
Red-Green-Refactor 사이클을 따라 개발합니다.

테스트 대상:
- OHLCV 데이터 검증
- 가격 이상치 감지
- 시계열 연속성 확인
- 데이터 누락 처리
- 볼륨 이상값 감지

Created: 2025-09-14 (Phase 2.1)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# 아직 구현되지 않은 클래스 - 테스트를 먼저 작성 (TDD Red phase)
from src.backtesting.data_validator import DataValidator, ValidationConfig, ValidationResult


class TestDataValidatorInitialization:
    """DataValidator 초기화 테스트"""

    def test_should_create_validator_with_default_config(self):
        """기본 설정으로 DataValidator를 생성할 수 있어야 함"""
        validator = DataValidator()

        assert validator is not None
        assert isinstance(validator.config, ValidationConfig)
        assert validator.config.outlier_threshold == 3.0
        assert validator.config.volume_threshold == 5.0
        assert validator.config.price_change_threshold == 0.2

    def test_should_create_validator_with_custom_config(self):
        """커스텀 설정으로 DataValidator를 생성할 수 있어야 함"""
        config = ValidationConfig(
            outlier_threshold=2.5,
            volume_threshold=4.0,
            price_change_threshold=0.15,
            min_volume=100,
            max_gap_minutes=60
        )

        validator = DataValidator(config)

        assert validator.config.outlier_threshold == 2.5
        assert validator.config.volume_threshold == 4.0
        assert validator.config.price_change_threshold == 0.15
        assert validator.config.min_volume == 100
        assert validator.config.max_gap_minutes == 60


class TestOHLCVValidation:
    """OHLCV 데이터 검증 테스트"""

    def _create_valid_ohlcv_data(self) -> pd.DataFrame:
        """유효한 OHLCV 테스트 데이터 생성"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
        n = len(dates)

        np.random.seed(42)
        base_price = 50000
        returns = np.random.normal(0, 0.01, n)
        prices = base_price * np.exp(np.cumsum(returns))

        data = pd.DataFrame({
            'datetime': dates,
            'open': prices * (1 + np.random.normal(0, 0.001, n)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n))),
            'close': prices,
            'volume': np.random.lognormal(10, 0.5, n)
        })

        # OHLC 관계 보장
        for i in range(n):
            high_val = max(data.iloc[i]['open'], data.iloc[i]['close'], data.iloc[i]['high'])
            low_val = min(data.iloc[i]['open'], data.iloc[i]['close'], data.iloc[i]['low'])
            data.iloc[i, data.columns.get_loc('high')] = high_val
            data.iloc[i, data.columns.get_loc('low')] = low_val

        return data

    def test_should_validate_correct_ohlcv_data(self):
        """올바른 OHLCV 데이터를 검증할 수 있어야 함"""
        validator = DataValidator()
        data = self._create_valid_ohlcv_data()

        result = validator.validate_ohlcv_data(data)

        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert result.total_errors == 0
        assert len(result.errors) == 0
        assert result.data_quality_score > 0.95

    def test_should_detect_invalid_ohlc_relationships(self):
        """잘못된 OHLC 관계를 감지해야 함"""
        validator = DataValidator()
        data = self._create_valid_ohlcv_data()

        # high < low 인 잘못된 관계 생성
        data.iloc[10, data.columns.get_loc('high')] = 100
        data.iloc[10, data.columns.get_loc('low')] = 200

        result = validator.validate_ohlcv_data(data)

        assert result.is_valid is False
        assert result.total_errors > 0
        assert any('OHLC relationship' in error['type'] for error in result.errors)
        assert result.data_quality_score < 0.95

    def test_should_detect_negative_prices(self):
        """음수 가격을 감지해야 함"""
        validator = DataValidator()
        data = self._create_valid_ohlcv_data()

        # 음수 가격 생성
        data.iloc[5, data.columns.get_loc('close')] = -100

        result = validator.validate_ohlcv_data(data)

        assert result.is_valid is False
        assert any('negative price' in error['type'] for error in result.errors)
        assert result.data_quality_score < 0.95

    def test_should_detect_zero_volume(self):
        """0 볼륨을 감지해야 함"""
        validator = DataValidator()
        data = self._create_valid_ohlcv_data()

        # 0 볼륨 생성
        data.iloc[15, data.columns.get_loc('volume')] = 0

        result = validator.validate_ohlcv_data(data)

        assert result.is_valid is False
        assert any('zero volume' in error['type'] for error in result.errors)


class TestOutlierDetection:
    """이상치 감지 테스트"""

    def test_should_detect_price_outliers(self):
        """가격 이상치를 감지해야 함"""
        validator = DataValidator()
        data = self._create_valid_ohlcv_data()

        # 극단적인 가격 이상치 생성
        data.iloc[20, data.columns.get_loc('close')] = data['close'].mean() * 10

        result = validator.validate_ohlcv_data(data)

        assert result.is_valid is False
        assert any('price outlier' in error['type'] for error in result.errors)

    def test_should_detect_volume_outliers(self):
        """볼륨 이상치를 감지해야 함"""
        validator = DataValidator()
        data = self._create_valid_ohlcv_data()

        # 극단적인 볼륨 이상치 생성
        data.iloc[25, data.columns.get_loc('volume')] = data['volume'].mean() * 100

        result = validator.validate_ohlcv_data(data)

        outlier_errors = [e for e in result.errors if 'volume outlier' in e['type']]
        assert len(outlier_errors) > 0

    def test_should_handle_multiple_outliers(self):
        """다중 이상치를 처리해야 함"""
        validator = DataValidator()
        data = self._create_valid_ohlcv_data()

        # 여러 이상치 생성
        data.iloc[10, data.columns.get_loc('close')] = data['close'].mean() * 5
        data.iloc[20, data.columns.get_loc('volume')] = data['volume'].mean() * 50
        data.iloc[30, data.columns.get_loc('high')] = data['high'].mean() * 3

        result = validator.validate_ohlcv_data(data)

        assert result.total_errors >= 3
        assert result.data_quality_score < 0.9


class TestTimeSeriesContinuity:
    """시계열 연속성 테스트"""

    def test_should_validate_continuous_timestamps(self):
        """연속된 타임스탬프를 검증해야 함"""
        validator = DataValidator()
        data = self._create_continuous_time_data()

        result = validator.validate_time_continuity(data)

        assert result.is_valid is True
        assert result.total_errors == 0

    def test_should_detect_missing_timestamps(self):
        """누락된 타임스탬프를 감지해야 함"""
        validator = DataValidator()
        data = self._create_continuous_time_data()

        # 중간 데이터 제거하여 갭 생성
        data = data.drop(data.index[10:15])

        result = validator.validate_time_continuity(data)

        assert result.is_valid is False
        assert any('time gap' in error['type'] for error in result.errors)

    def test_should_detect_duplicate_timestamps(self):
        """중복된 타임스탬프를 감지해야 함"""
        validator = DataValidator()
        data = self._create_continuous_time_data()

        # 중복 타임스탬프 생성
        duplicate_row = data.iloc[20:21].copy()
        data = pd.concat([data, duplicate_row], ignore_index=True)

        result = validator.validate_time_continuity(data)

        assert result.is_valid is False
        assert any('duplicate timestamp' in error['type'] for error in result.errors)

    def test_should_detect_out_of_order_timestamps(self):
        """순서가 잘못된 타임스탬프를 감지해야 함"""
        validator = DataValidator()
        data = self._create_continuous_time_data()

        # 순서 뒤바꾸기
        data.iloc[30], data.iloc[31] = data.iloc[31].copy(), data.iloc[30].copy()

        result = validator.validate_time_continuity(data)

        assert result.is_valid is False
        assert any('timestamp order' in error['type'] for error in result.errors)


class TestDataQualityMetrics:
    """데이터 품질 메트릭 테스트"""

    def test_should_calculate_data_quality_score(self):
        """데이터 품질 점수를 계산해야 함"""
        validator = DataValidator()
        perfect_data = self._create_valid_ohlcv_data()

        result = validator.validate_ohlcv_data(perfect_data)

        assert 0.0 <= result.data_quality_score <= 1.0
        assert result.data_quality_score > 0.95  # 완벽한 데이터는 높은 점수

    def test_should_provide_detailed_metrics(self):
        """상세한 메트릭을 제공해야 함"""
        validator = DataValidator()
        data = self._create_valid_ohlcv_data()

        result = validator.validate_ohlcv_data(data)

        assert hasattr(result, 'metrics')
        assert 'outlier_percentage' in result.metrics
        assert 'missing_data_percentage' in result.metrics
        assert 'data_consistency_score' in result.metrics

    def test_should_calculate_completeness_ratio(self):
        """완성도 비율을 계산해야 함"""
        validator = DataValidator()
        data = self._create_valid_ohlcv_data()

        # 일부 데이터를 NaN으로 설정
        data.iloc[5:10, data.columns.get_loc('volume')] = np.nan

        result = validator.validate_ohlcv_data(data)

        assert result.metrics['missing_data_percentage'] > 0
        assert result.metrics['completeness_ratio'] < 1.0


class TestValidationPerformance:
    """검증 성능 테스트"""

    def test_should_handle_large_dataset_efficiently(self):
        """대용량 데이터셋을 효율적으로 처리해야 함"""
        validator = DataValidator()
        large_data = self._create_large_ohlcv_data(10000)  # 10k rows

        import time
        start_time = time.time()
        result = validator.validate_ohlcv_data(large_data)
        end_time = time.time()

        # 10k rows를 5초 이내에 처리
        assert (end_time - start_time) < 5.0
        assert isinstance(result, ValidationResult)

    def test_should_provide_progress_updates(self):
        """진행 상황 업데이트를 제공해야 함"""
        config = ValidationConfig(show_progress=True)
        validator = DataValidator(config)
        data = self._create_valid_ohlcv_data()

        result = validator.validate_ohlcv_data(data)

        # 프로그레스 기능이 정상 작동
        assert result.validation_time > 0

    def _create_valid_ohlcv_data(self) -> pd.DataFrame:
        """유효한 OHLCV 테스트 데이터 생성"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1H')
        n = len(dates)

        np.random.seed(42)
        base_price = 50000
        returns = np.random.normal(0, 0.01, n)
        prices = base_price * np.exp(np.cumsum(returns))

        data = pd.DataFrame({
            'datetime': dates,
            'open': prices * (1 + np.random.normal(0, 0.001, n)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n))),
            'close': prices,
            'volume': np.random.lognormal(10, 0.5, n)
        })

        # OHLC 관계 보장
        for i in range(n):
            high_val = max(data.iloc[i]['open'], data.iloc[i]['close'], data.iloc[i]['high'])
            low_val = min(data.iloc[i]['open'], data.iloc[i]['close'], data.iloc[i]['low'])
            data.iloc[i, data.columns.get_loc('high')] = high_val
            data.iloc[i, data.columns.get_loc('low')] = low_val

        return data

    def _create_continuous_time_data(self) -> pd.DataFrame:
        """연속된 시간 데이터 생성"""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='15T')
        n = len(dates)

        data = pd.DataFrame({
            'datetime': dates,
            'open': 50000 + np.random.normal(0, 100, n),
            'high': 50100 + np.random.normal(0, 100, n),
            'low': 49900 + np.random.normal(0, 100, n),
            'close': 50000 + np.random.normal(0, 100, n),
            'volume': np.random.lognormal(8, 0.5, n)
        })

        return data

    def _create_large_ohlcv_data(self, rows: int) -> pd.DataFrame:
        """대용량 OHLCV 데이터 생성"""
        dates = pd.date_range(start='2020-01-01', periods=rows, freq='1H')
        n = len(dates)

        np.random.seed(42)
        base_price = 50000
        returns = np.random.normal(0.0001, 0.02, n)
        prices = base_price * np.exp(np.cumsum(returns))

        data = pd.DataFrame({
            'datetime': dates,
            'open': prices * (1 + np.random.normal(0, 0.001, n)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n))),
            'close': prices,
            'volume': np.random.lognormal(8, 0.5, n)
        })

        # OHLC 관계 보장 (샘플링으로 성능 고려)
        data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
        data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))

        return data