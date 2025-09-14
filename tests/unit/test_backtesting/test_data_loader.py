"""
DataLoader 클래스 단위 테스트

TDD 방법론을 사용하여 히스토리 데이터 로더의 모든 기능을 검증합니다.
Red-Green-Refactor 사이클을 따라 개발합니다.

테스트 대상:
- CSV/Parquet 데이터 로딩
- 데이터 형식 검증
- 메모리 효율적 처리
- 다양한 데이터 소스 지원

Created: 2025-09-14 (Phase 2.1)
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta

# 아직 구현되지 않은 클래스 - 테스트를 먼저 작성 (TDD Red phase)
from src.backtesting.data_loader import DataLoader, DataSource, LoaderConfig


class TestDataLoaderInitialization:
    """DataLoader 초기화 테스트"""

    def test_should_create_data_loader_with_default_config(self):
        """기본 설정으로 DataLoader를 생성할 수 있어야 함"""
        loader = DataLoader()

        assert loader is not None
        assert isinstance(loader.config, LoaderConfig)
        assert loader.supported_formats == ['.csv', '.parquet', '.json']

    def test_should_create_data_loader_with_custom_config(self):
        """커스텀 설정으로 DataLoader를 생성할 수 있어야 함"""
        config = LoaderConfig(
            chunk_size=5000,
            memory_limit_mb=256,
            validate_data=True,
            date_column='timestamp'
        )

        loader = DataLoader(config)

        assert loader.config.chunk_size == 5000
        assert loader.config.memory_limit_mb == 256
        assert loader.config.validate_data is True
        assert loader.config.date_column == 'timestamp'


class TestDataSourceHandling:
    """데이터 소스 처리 테스트"""

    def test_should_detect_csv_data_source(self):
        """CSV 파일을 올바르게 감지해야 함"""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            tmp_path = tmp.name

            # 샘플 CSV 데이터 생성
            sample_data = self._create_sample_ohlcv_data()
            sample_data.to_csv(tmp_path, index=False)

            loader = DataLoader()
            source = loader.detect_source(tmp_path)

            assert source.format == 'csv'
            assert source.path == tmp_path
            assert source.is_valid is True

            # 정리 (Windows 파일 잠금 처리)
            try:
                os.unlink(tmp_path)
            except PermissionError:
                # Windows에서 파일이 아직 사용 중인 경우 무시
                pass

    @pytest.mark.skip(reason="Parquet engine not available - will implement after pyarrow installation")
    def test_should_detect_parquet_data_source(self):
        """Parquet 파일을 올바르게 감지해야 함"""
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            tmp_path = tmp.name

            # 샘플 Parquet 데이터 생성
            sample_data = self._create_sample_ohlcv_data()
            sample_data.to_parquet(tmp_path, index=False)

            loader = DataLoader()
            source = loader.detect_source(tmp_path)

            assert source.format == 'parquet'
            assert source.path == tmp_path
            assert source.is_valid is True

            # 정리
            os.unlink(tmp_path)

    def test_should_reject_unsupported_format(self):
        """지원하지 않는 형식의 파일을 거부해야 함"""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp_path = tmp.name

            loader = DataLoader()

            with pytest.raises(ValueError, match="Unsupported file format"):
                loader.detect_source(tmp_path)

            # 정리
            os.unlink(tmp_path)

    def test_should_handle_nonexistent_file(self):
        """존재하지 않는 파일을 처리해야 함"""
        loader = DataLoader()

        with pytest.raises(FileNotFoundError):
            loader.detect_source("nonexistent_file.csv")

    def _create_sample_ohlcv_data(self) -> pd.DataFrame:
        """테스트용 샘플 OHLCV 데이터 생성"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1D')
        n = len(dates)

        # 가격 데이터 시뮬레이션 (GBM)
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.02, n)
        prices = 50000 * np.exp(np.cumsum(returns))

        data = pd.DataFrame({
            'datetime': dates,
            'open': prices * (1 + np.random.normal(0, 0.001, n)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, n)
        })

        # high >= low >= 0, high >= close, low <= close 보장
        data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
        data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))

        return data.round(6)


class TestDataLoading:
    """실제 데이터 로딩 테스트"""

    def test_should_load_csv_data_completely(self):
        """CSV 데이터를 완전히 로드할 수 있어야 함"""
        # 테스트 데이터 준비
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            tmp_path = tmp.name
            sample_data = self._create_sample_ohlcv_data()
            sample_data.to_csv(tmp_path, index=False)

            loader = DataLoader()
            loaded_data = loader.load(tmp_path)

            # 데이터 검증
            assert isinstance(loaded_data, pd.DataFrame)
            assert len(loaded_data) == len(sample_data)
            assert list(loaded_data.columns) == list(sample_data.columns)

            # 날짜 컬럼이 datetime 타입으로 변환되었는지 확인
            assert pd.api.types.is_datetime64_any_dtype(loaded_data['datetime'])

            # 정리
            os.unlink(tmp_path)

    def test_should_load_data_in_chunks_when_large(self):
        """큰 데이터를 청크 단위로 로드할 수 있어야 함"""
        # 작은 청크 사이즈로 설정
        config = LoaderConfig(chunk_size=100)
        loader = DataLoader(config)

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            tmp_path = tmp.name
            # 청크보다 큰 데이터 생성 (365 rows > 100 chunk_size)
            sample_data = self._create_large_sample_data(rows=365)
            sample_data.to_csv(tmp_path, index=False)

            loaded_data = loader.load(tmp_path)

            # 전체 데이터가 정확히 로드되었는지 확인
            assert len(loaded_data) == 365
            assert isinstance(loaded_data, pd.DataFrame)

            # 정리
            os.unlink(tmp_path)

    def test_should_validate_ohlcv_data_structure(self):
        """OHLCV 데이터 구조를 검증해야 함"""
        loader = DataLoader()

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            tmp_path = tmp.name
            sample_data = self._create_sample_ohlcv_data()
            sample_data.to_csv(tmp_path, index=False)

            loaded_data = loader.load(tmp_path)
            validation_result = loader.validate_ohlcv_structure(loaded_data)

            assert validation_result['is_valid'] is True
            assert validation_result['has_required_columns'] is True
            assert validation_result['datetime_column_valid'] is True
            assert validation_result['numeric_columns_valid'] is True

            # 정리
            os.unlink(tmp_path)

    def test_should_handle_malformed_csv_data(self):
        """잘못된 형식의 CSV 데이터를 처리해야 함"""
        loader = DataLoader()

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as tmp:
            tmp_path = tmp.name
            # 잘못된 CSV 데이터 작성
            tmp.write("invalid,csv,data\n")
            tmp.write("with,inconsistent,columns,extra\n")
            tmp.write("and,missing\n")

        with pytest.raises(pd.errors.ParserError):
            loader.load(tmp_path)

        # 정리 (Windows 파일 잠금 처리)
        try:
            os.unlink(tmp_path)
        except PermissionError:
            # Windows에서 파일이 아직 사용 중인 경우 무시
            pass

    def test_should_handle_memory_limits(self):
        """메모리 제한을 적절히 처리해야 함"""
        # 매우 작은 메모리 제한 설정 (1MB)
        config = LoaderConfig(memory_limit_mb=1, chunk_size=50)
        loader = DataLoader(config)

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            tmp_path = tmp.name
            # 큰 데이터 생성
            large_data = self._create_large_sample_data(rows=1000)
            large_data.to_csv(tmp_path, index=False)

            # 메모리 제한 때문에 청크로 처리되어야 함
            loaded_data = loader.load(tmp_path)

            # 데이터가 여전히 완전히 로드되어야 함
            assert len(loaded_data) == 1000
            assert isinstance(loaded_data, pd.DataFrame)

            # 정리
            os.unlink(tmp_path)

    def _create_sample_ohlcv_data(self) -> pd.DataFrame:
        """테스트용 샘플 OHLCV 데이터 생성"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1D')
        n = len(dates)

        # 가격 데이터 시뮬레이션
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.02, n)
        prices = 50000 * np.exp(np.cumsum(returns))

        data = pd.DataFrame({
            'datetime': dates,
            'open': prices * (1 + np.random.normal(0, 0.001, n)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, n)
        })

        # OHLC 관계 보장
        data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
        data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))

        return data.round(6)

    def _create_large_sample_data(self, rows: int) -> pd.DataFrame:
        """대용량 테스트 데이터 생성"""
        dates = pd.date_range(start='2020-01-01', periods=rows, freq='1h')
        n = len(dates)

        np.random.seed(42)
        returns = np.random.normal(0.00001, 0.003, n)
        prices = 50000 * np.exp(np.cumsum(returns))

        data = pd.DataFrame({
            'datetime': dates,
            'open': prices * (1 + np.random.normal(0, 0.0005, n)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.001, n))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.001, n))),
            'close': prices,
            'volume': np.random.lognormal(8, 0.5, n)
        })

        # OHLC 관계 보장
        data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
        data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))

        return data.round(6)


class TestDataCaching:
    """데이터 캐싱 기능 테스트"""

    def test_should_cache_loaded_data_when_enabled(self):
        """캐싱이 활성화되어 있을 때 데이터를 캐시해야 함"""
        config = LoaderConfig(enable_caching=True)
        loader = DataLoader(config)

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            tmp_path = tmp.name
            sample_data = self._create_sample_ohlcv_data()
            sample_data.to_csv(tmp_path, index=False)

            # 첫 번째 로드
            data1 = loader.load(tmp_path)

            # 두 번째 로드 (캐시에서 가져와야 함)
            data2 = loader.load(tmp_path)

            # 데이터가 동일해야 함
            pd.testing.assert_frame_equal(data1, data2)

            # 캐시가 작동했는지 확인
            assert loader.cache_hits > 0

            # 정리
            os.unlink(tmp_path)

    def test_should_not_cache_when_disabled(self):
        """캐싱이 비활성화되어 있을 때 캐시하지 않아야 함"""
        config = LoaderConfig(enable_caching=False)
        loader = DataLoader(config)

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            tmp_path = tmp.name
            sample_data = self._create_sample_ohlcv_data()
            sample_data.to_csv(tmp_path, index=False)

            # 로드
            data1 = loader.load(tmp_path)
            data2 = loader.load(tmp_path)

            # 캐시 히트가 없어야 함
            assert loader.cache_hits == 0

            # 정리
            os.unlink(tmp_path)

    def _create_sample_ohlcv_data(self) -> pd.DataFrame:
        """테스트용 샘플 OHLCV 데이터 생성"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1D')
        n = len(dates)

        np.random.seed(42)
        prices = 50000 + np.cumsum(np.random.normal(0, 100, n))

        return pd.DataFrame({
            'datetime': dates,
            'open': prices * (1 + np.random.normal(0, 0.001, n)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, n)
        })