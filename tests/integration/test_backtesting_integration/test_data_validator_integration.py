"""
DataValidator 통합 테스트

DataValidator의 핵심 기능을 종합적으로 테스트합니다.
실제 사용 시나리오를 기반으로 한 통합 테스트입니다.

Created: 2025-09-14 (Phase 2.1)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.backtesting.data_validator import DataValidator, ValidationConfig, ValidationResult


class TestDataValidatorIntegration:
    """DataValidator 통합 테스트"""

    def test_should_validate_perfect_ohlcv_data(self):
        """완벽한 OHLCV 데이터 검증 테스트"""
        validator = DataValidator()
        perfect_data = self._create_perfect_ohlcv_data()

        result = validator.validate_ohlcv_data(perfect_data)

        # 기본 검증
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert result.total_errors == 0
        assert result.data_quality_score >= 0.95
        assert result.validated_rows == len(perfect_data)

        # 메트릭 검증
        assert 'missing_data_percentage' in result.metrics
        assert 'completeness_ratio' in result.metrics
        assert 'outlier_percentage' in result.metrics
        assert result.metrics['completeness_ratio'] >= 0.95

    def test_should_detect_multiple_data_issues(self):
        """다중 데이터 문제 감지 테스트"""
        validator = DataValidator()
        problematic_data = self._create_problematic_ohlcv_data()

        result = validator.validate_ohlcv_data(problematic_data)

        # 문제가 있는 데이터는 invalid
        assert result.is_valid is False
        assert result.total_errors > 0

        # 다양한 종류의 에러가 감지되어야 함
        error_types = {error.type for error in result.errors}
        expected_types = {'OHLC relationship', 'negative price', 'zero volume'}
        assert len(error_types.intersection(expected_types)) > 0

        # 품질 점수가 낮아야 함 (많은 에러로 인해)
        assert result.data_quality_score < 0.9

    def test_should_validate_time_continuity_properly(self):
        """시계열 연속성 검증 테스트"""
        validator = DataValidator()

        # 연속된 데이터
        continuous_data = self._create_continuous_time_data()
        result = validator.validate_time_continuity(continuous_data)
        assert result.is_valid is True

        # 갭이 있는 데이터
        gapped_data = self._create_gapped_time_data()
        result = validator.validate_time_continuity(gapped_data)
        assert result.is_valid is False
        assert any('time gap' in error.type for error in result.errors + result.warnings)

    def test_should_handle_custom_configuration(self):
        """커스텀 설정 처리 테스트"""
        config = ValidationConfig(
            outlier_threshold=2.0,  # 더 엄격한 이상치 감지
            volume_threshold=3.0,   # 더 엄격한 볼륨 체크
            max_gap_minutes=5,      # 5분 이상 갭을 경고
            show_progress=True
        )

        validator = DataValidator(config)
        data = self._create_perfect_ohlcv_data()

        result = validator.validate_ohlcv_data(data)

        assert result.validation_time > 0
        assert validator.config.outlier_threshold == 2.0
        assert validator.config.max_gap_minutes == 5

    def test_should_calculate_comprehensive_metrics(self):
        """종합적인 메트릭 계산 테스트"""
        validator = DataValidator()

        # 일부 데이터에 결측치 포함
        data_with_missing = self._create_perfect_ohlcv_data()
        data_with_missing.iloc[10:15, data_with_missing.columns.get_loc('volume')] = np.nan

        result = validator.validate_ohlcv_data(data_with_missing)

        # 메트릭이 모두 계산되었는지 확인
        required_metrics = [
            'missing_data_percentage',
            'completeness_ratio',
            'outlier_percentage',
            'data_consistency_score',
            'total_cells',
            'missing_cells'
        ]

        for metric in required_metrics:
            assert metric in result.metrics
            assert isinstance(result.metrics[metric], (int, float, np.integer, np.floating))

        # 결측치가 있으므로 완성도가 1.0 미만이어야 함
        assert result.metrics['completeness_ratio'] < 1.0
        assert result.metrics['missing_data_percentage'] > 0

    def test_should_handle_edge_cases_gracefully(self):
        """경계 조건을 우아하게 처리해야 함"""
        validator = DataValidator()

        # 빈 데이터프레임
        empty_data = pd.DataFrame()
        result = validator.validate_ohlcv_data(empty_data)
        assert isinstance(result, ValidationResult)

        # 단일 행 데이터
        single_row = self._create_perfect_ohlcv_data().iloc[:1]
        result = validator.validate_ohlcv_data(single_row)
        assert isinstance(result, ValidationResult)
        assert result.validated_rows == 1

        # 필수 컬럼 누락
        incomplete_data = self._create_perfect_ohlcv_data().drop(columns=['volume'])
        result = validator.validate_ohlcv_data(incomplete_data)
        assert result.is_valid is False
        assert any('missing columns' in error.type for error in result.errors)

    def test_should_provide_actionable_error_information(self):
        """실행 가능한 에러 정보를 제공해야 함"""
        validator = DataValidator()
        problematic_data = self._create_problematic_ohlcv_data()

        result = validator.validate_ohlcv_data(problematic_data)

        # 에러에 충분한 정보가 포함되어야 함
        for error in result.errors:
            assert hasattr(error, 'type')
            assert hasattr(error, 'severity')
            assert hasattr(error, 'message')
            assert error.message is not None and len(error.message) > 0

            # 에러에 위치 정보가 포함되어야 함 (가능한 경우)
            if error.row_index is not None:
                assert isinstance(error.row_index, (int, np.integer))

    def _create_perfect_ohlcv_data(self) -> pd.DataFrame:
        """완벽한 OHLCV 데이터 생성"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
        n = len(dates)

        np.random.seed(42)
        base_price = 50000
        returns = np.random.normal(0, 0.005, n)  # 작은 변동성
        prices = base_price * np.exp(np.cumsum(returns))

        data = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': prices * 1.01,   # 항상 약간 높게
            'low': prices * 0.99,    # 항상 약간 낮게
            'close': prices,
            'volume': np.random.lognormal(10, 0.3, n)  # 일관된 볼륨
        })

        # OHLC 관계를 완벽하게 보장
        for i in range(n):
            o, c = data.iloc[i]['open'], data.iloc[i]['close']
            data.iloc[i, data.columns.get_loc('high')] = max(o, c) * 1.002
            data.iloc[i, data.columns.get_loc('low')] = min(o, c) * 0.998

        return data

    def _create_problematic_ohlcv_data(self) -> pd.DataFrame:
        """문제가 있는 OHLCV 데이터 생성"""
        data = self._create_perfect_ohlcv_data()

        # 다양한 문제 삽입
        # 1. 잘못된 OHLC 관계
        data.iloc[10, data.columns.get_loc('high')] = 100
        data.iloc[10, data.columns.get_loc('low')] = 200

        # 2. 음수 가격
        data.iloc[20, data.columns.get_loc('close')] = -100

        # 3. 0 볼륨
        data.iloc[30, data.columns.get_loc('volume')] = 0

        # 4. 극단적 이상치
        data.iloc[40, data.columns.get_loc('close')] = data['close'].mean() * 10

        # 5. 결측치
        data.iloc[50:55, data.columns.get_loc('open')] = np.nan

        return data

    def _create_continuous_time_data(self) -> pd.DataFrame:
        """연속된 시간 데이터 생성"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
        n = len(dates)

        return pd.DataFrame({
            'datetime': dates,
            'open': 50000 + np.random.normal(0, 50, n),
            'high': 50050 + np.random.normal(0, 50, n),
            'low': 49950 + np.random.normal(0, 50, n),
            'close': 50000 + np.random.normal(0, 50, n),
            'volume': np.random.lognormal(8, 0.3, n)
        })

    def _create_gapped_time_data(self) -> pd.DataFrame:
        """시간 갭이 있는 데이터 생성"""
        dates1 = pd.date_range(start='2023-01-01 00:00', periods=50, freq='1h')
        dates2 = pd.date_range(start='2023-01-01 10:00', periods=50, freq='1h')  # 7시간 갭
        dates = pd.concat([pd.Series(dates1), pd.Series(dates2)]).reset_index(drop=True)

        n = len(dates)

        return pd.DataFrame({
            'datetime': dates,
            'open': 50000 + np.random.normal(0, 50, n),
            'high': 50050 + np.random.normal(0, 50, n),
            'low': 49950 + np.random.normal(0, 50, n),
            'close': 50000 + np.random.normal(0, 50, n),
            'volume': np.random.lognormal(8, 0.3, n)
        })