"""
데이터 품질 검증기

금융 시계열 데이터의 품질을 종합적으로 검증하는 클래스입니다.
OHLCV 데이터의 무결성, 이상치 감지, 시계열 연속성 등을 확인합니다.

주요 기능:
- OHLCV 관계 검증
- 가격/볼륨 이상치 감지
- 시계열 연속성 확인
- 데이터 완성도 평가
- 품질 점수 계산

TDD 방법론으로 구현 - Phase 2.1
Created: 2025-09-14
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
import time
from scipy import stats


@dataclass
class ValidationConfig:
    """데이터 검증 설정"""
    outlier_threshold: float = 3.0
    volume_threshold: float = 5.0
    price_change_threshold: float = 0.2
    min_volume: float = 0.0
    max_gap_minutes: int = 30
    show_progress: bool = False
    enable_outlier_detection: bool = True
    enable_time_continuity_check: bool = True
    enable_relationship_check: bool = True


@dataclass
class ValidationError:
    """검증 에러 정보"""
    type: str
    severity: str  # 'ERROR', 'WARNING', 'INFO'
    message: str
    row_index: Optional[int] = None
    column: Optional[str] = None
    value: Optional[Any] = None
    expected_range: Optional[Tuple[float, float]] = None


@dataclass
class ValidationResult:
    """검증 결과"""
    is_valid: bool
    total_errors: int
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    data_quality_score: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    validation_time: float = 0.0
    validated_rows: int = 0


class DataValidator:
    """데이터 품질 검증기"""

    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        DataValidator 초기화

        Args:
            config: 검증 설정
        """
        self.config = config or ValidationConfig()
        self.logger = logging.getLogger(__name__)

    def validate_ohlcv_data(self, data: pd.DataFrame) -> ValidationResult:
        """
        OHLCV 데이터 종합 검증

        Args:
            data: 검증할 OHLCV 데이터

        Returns:
            ValidationResult: 검증 결과
        """
        start_time = time.time()

        result = ValidationResult(
            is_valid=True,
            total_errors=0,
            validated_rows=len(data)
        )

        # 기본 구조 검증
        self._validate_basic_structure(data, result)

        # OHLC 관계 검증
        if self.config.enable_relationship_check:
            self._validate_ohlc_relationships(data, result)

        # 이상치 감지
        if self.config.enable_outlier_detection:
            self._detect_outliers(data, result)

        # 가격/볼륨 기본 검증
        self._validate_price_volume_basics(data, result)

        # 시계열 연속성 검증
        if self.config.enable_time_continuity_check:
            self._validate_time_continuity_in_ohlcv(data, result)

        # 품질 메트릭 계산
        self._calculate_quality_metrics(data, result)

        # 최종 결과 설정
        result.total_errors = len([e for e in result.errors if e.severity == 'ERROR'])
        result.is_valid = result.total_errors == 0

        result.validation_time = time.time() - start_time

        if self.config.show_progress:
            self.logger.info(f"Validation completed in {result.validation_time:.2f}s")

        return result

    def validate_time_continuity(self, data: pd.DataFrame) -> ValidationResult:
        """
        시계열 연속성 전용 검증

        Args:
            data: 검증할 데이터

        Returns:
            ValidationResult: 시계열 검증 결과
        """
        start_time = time.time()

        result = ValidationResult(
            is_valid=True,
            total_errors=0,
            validated_rows=len(data)
        )

        self._validate_time_continuity(data, result)

        result.total_errors = len([e for e in result.errors if e.severity == 'ERROR'])
        result.is_valid = result.total_errors == 0
        result.validation_time = time.time() - start_time

        return result

    def _validate_basic_structure(self, data: pd.DataFrame, result: ValidationResult):
        """기본 데이터 구조 검증"""
        required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']

        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            error = ValidationError(
                type='missing columns',
                severity='ERROR',
                message=f"Missing required columns: {missing_columns}"
            )
            result.errors.append(error)

        # 데이터 타입 검증
        if 'datetime' in data.columns:
            if not pd.api.types.is_datetime64_any_dtype(data['datetime']):
                error = ValidationError(
                    type='datetime type',
                    severity='ERROR',
                    message="datetime column must be datetime type",
                    column='datetime'
                )
                result.errors.append(error)

        # 숫자 컬럼 검증
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    error = ValidationError(
                        type='numeric type',
                        severity='ERROR',
                        message=f"{col} column must be numeric type",
                        column=col
                    )
                    result.errors.append(error)

    def _validate_ohlc_relationships(self, data: pd.DataFrame, result: ValidationResult):
        """OHLC 관계 검증"""
        ohlc_cols = ['open', 'high', 'low', 'close']

        if not all(col in data.columns for col in ohlc_cols):
            return

        for idx, row in data.iterrows():
            try:
                o, h, l, c = row['open'], row['high'], row['low'], row['close']

                # NaN 체크
                if pd.isna(o) or pd.isna(h) or pd.isna(l) or pd.isna(c):
                    continue

                # high >= max(open, close)
                max_oc = max(o, c)
                if h < max_oc:
                    error = ValidationError(
                        type='OHLC relationship',
                        severity='ERROR',
                        message=f"High ({h}) must be >= max(open, close) ({max_oc})",
                        row_index=idx,
                        value=h,
                        expected_range=(max_oc, float('inf'))
                    )
                    result.errors.append(error)

                # low <= min(open, close)
                min_oc = min(o, c)
                if l > min_oc:
                    error = ValidationError(
                        type='OHLC relationship',
                        severity='ERROR',
                        message=f"Low ({l}) must be <= min(open, close) ({min_oc})",
                        row_index=idx,
                        value=l,
                        expected_range=(0, min_oc)
                    )
                    result.errors.append(error)

                # high >= low
                if h < l:
                    error = ValidationError(
                        type='OHLC relationship',
                        severity='ERROR',
                        message=f"High ({h}) must be >= Low ({l})",
                        row_index=idx
                    )
                    result.errors.append(error)

            except Exception as e:
                error = ValidationError(
                    type='OHLC validation error',
                    severity='WARNING',
                    message=f"Error validating OHLC relationships at row {idx}: {str(e)}",
                    row_index=idx
                )
                result.warnings.append(error)

    def _detect_outliers(self, data: pd.DataFrame, result: ValidationResult):
        """이상치 감지"""

        # 가격 이상치 감지
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in data.columns:
                self._detect_column_outliers(
                    data, col, self.config.outlier_threshold,
                    'price outlier', result
                )

        # 볼륨 이상치 감지 (로그 스케일)
        if 'volume' in data.columns:
            try:
                log_volume = np.log1p(data['volume'].replace(0, np.nan))
                z_scores = np.abs(stats.zscore(log_volume, nan_policy='omit'))

                outlier_indices = np.where(z_scores > self.config.volume_threshold)[0]

                for idx in outlier_indices:
                    if not pd.isna(data.iloc[idx]['volume']):
                        error = ValidationError(
                            type='volume outlier',
                            severity='WARNING',
                            message=f"Volume outlier detected: {data.iloc[idx]['volume']}",
                            row_index=data.index[idx],
                            column='volume',
                            value=data.iloc[idx]['volume']
                        )
                        result.warnings.append(error)

            except Exception as e:
                error = ValidationError(
                    type='volume outlier detection error',
                    severity='WARNING',
                    message=f"Error detecting volume outliers: {str(e)}"
                )
                result.warnings.append(error)

    def _detect_column_outliers(self, data: pd.DataFrame, column: str,
                               threshold: float, error_type: str,
                               result: ValidationResult):
        """특정 컬럼의 이상치 감지"""
        try:
            if column not in data.columns:
                return

            values = data[column].dropna()
            if len(values) < 3:
                return

            z_scores = np.abs(stats.zscore(values))
            outlier_indices = values.index[z_scores > threshold]

            for idx in outlier_indices:
                error = ValidationError(
                    type=error_type,
                    severity='WARNING',
                    message=f"{column.title()} outlier detected: {data.loc[idx, column]}",
                    row_index=idx,
                    column=column,
                    value=data.loc[idx, column]
                )
                result.warnings.append(error)

        except Exception as e:
            error = ValidationError(
                type=f'{error_type} detection error',
                severity='WARNING',
                message=f"Error detecting {column} outliers: {str(e)}"
            )
            result.warnings.append(error)

    def _validate_price_volume_basics(self, data: pd.DataFrame, result: ValidationResult):
        """가격과 볼륨 기본 검증"""

        # 음수 가격 체크
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in data.columns:
                negative_prices = data[data[col] <= 0]
                for idx, row in negative_prices.iterrows():
                    error = ValidationError(
                        type='negative price',
                        severity='ERROR',
                        message=f"Negative or zero price in {col}: {row[col]}",
                        row_index=idx,
                        column=col,
                        value=row[col]
                    )
                    result.errors.append(error)

        # 제로 볼륨 체크
        if 'volume' in data.columns:
            zero_volume = data[data['volume'] == 0]
            for idx, row in zero_volume.iterrows():
                error = ValidationError(
                    type='zero volume',
                    severity='WARNING',
                    message=f"Zero volume detected at row {idx}",
                    row_index=idx,
                    column='volume',
                    value=0
                )
                result.warnings.append(error)

            # 최소 볼륨 체크
            if self.config.min_volume > 0:
                low_volume = data[(data['volume'] > 0) & (data['volume'] < self.config.min_volume)]
                for idx, row in low_volume.iterrows():
                    error = ValidationError(
                        type='low volume',
                        severity='INFO',
                        message=f"Volume below minimum threshold: {row['volume']}",
                        row_index=idx,
                        column='volume',
                        value=row['volume']
                    )
                    result.warnings.append(error)

    def _validate_time_continuity_in_ohlcv(self, data: pd.DataFrame, result: ValidationResult):
        """OHLCV 데이터의 시계열 연속성 검증"""
        if 'datetime' in data.columns:
            self._validate_time_continuity(data, result)

    def _validate_time_continuity(self, data: pd.DataFrame, result: ValidationResult):
        """시계열 연속성 검증"""
        if 'datetime' not in data.columns:
            error = ValidationError(
                type='missing datetime column',
                severity='ERROR',
                message="datetime column required for time continuity check"
            )
            result.errors.append(error)
            return

        datetime_col = data['datetime']

        # 중복 타임스탬프 체크
        duplicates = datetime_col.duplicated()
        for idx in data.index[duplicates]:
            error = ValidationError(
                type='duplicate timestamp',
                severity='ERROR',
                message=f"Duplicate timestamp: {datetime_col[idx]}",
                row_index=idx,
                column='datetime',
                value=datetime_col[idx]
            )
            result.errors.append(error)

        # 시간 순서 체크
        if not datetime_col.is_monotonic_increasing:
            # 순서가 잘못된 지점 찾기
            for i in range(1, len(datetime_col)):
                if datetime_col.iloc[i] <= datetime_col.iloc[i-1]:
                    error = ValidationError(
                        type='timestamp order',
                        severity='ERROR',
                        message=f"Timestamp out of order at row {data.index[i]}: {datetime_col.iloc[i]} <= {datetime_col.iloc[i-1]}",
                        row_index=data.index[i],
                        column='datetime'
                    )
                    result.errors.append(error)

        # 시간 갭 체크 (설정된 경우)
        if self.config.max_gap_minutes > 0:
            time_diffs = datetime_col.diff()
            max_gap = timedelta(minutes=self.config.max_gap_minutes)

            large_gaps = time_diffs > max_gap
            for idx in data.index[large_gaps]:
                gap_minutes = time_diffs[idx].total_seconds() / 60
                error = ValidationError(
                    type='time gap',
                    severity='WARNING',
                    message=f"Large time gap: {gap_minutes:.1f} minutes",
                    row_index=idx,
                    column='datetime'
                )
                result.warnings.append(error)

    def _calculate_quality_metrics(self, data: pd.DataFrame, result: ValidationResult):
        """품질 메트릭 계산"""
        total_rows = len(data)
        total_cells = total_rows * len(data.columns)

        # 누락 데이터 계산
        missing_cells = data.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0

        # 완성도 비율
        completeness_ratio = 1.0 - (missing_cells / total_cells) if total_cells > 0 else 0

        # 이상치 비율
        outlier_warnings = len([w for w in result.warnings if 'outlier' in w.type])
        outlier_percentage = (outlier_warnings / total_rows) * 100 if total_rows > 0 else 0

        # 일관성 점수 (에러가 적을수록 높음)
        total_errors = len([e for e in result.errors if e.severity == 'ERROR'])
        error_ratio = total_errors / total_rows if total_rows > 0 else 0
        consistency_score = max(0.0, 1.0 - (error_ratio * 5))  # 에러에 더 큰 페널티

        # 전체 품질 점수 (가중평균)
        quality_score = (
            completeness_ratio * 0.4 +      # 완성도 40%
            consistency_score * 0.4 +       # 일관성 40%
            max(0.0, 1.0 - outlier_percentage/100) * 0.2  # 이상치 적음 20%
        )

        result.data_quality_score = max(0.0, min(1.0, quality_score))

        result.metrics = {
            'missing_data_percentage': missing_percentage,
            'completeness_ratio': completeness_ratio,
            'outlier_percentage': outlier_percentage,
            'data_consistency_score': consistency_score,
            'total_cells': total_cells,
            'missing_cells': missing_cells
        }