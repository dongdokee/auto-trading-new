"""
히스토리 데이터 로더

다양한 형태의 금융 데이터를 효율적으로 로드하고 검증하는 클래스입니다.
메모리 효율성과 데이터 품질을 고려하여 설계되었습니다.

지원 기능:
- CSV, Parquet, JSON 형식 지원
- 청크 단위 대용량 데이터 처리
- 데이터 유효성 검증
- 캐싱 지원
- OHLCV 데이터 구조 검증

TDD 방법론으로 구현 - Phase 2.1
Created: 2025-09-14
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import os
from dataclasses import dataclass
import logging
from datetime import datetime
import hashlib
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class LoaderConfig:
    """데이터 로더 설정"""
    chunk_size: int = 10000
    memory_limit_mb: int = 512
    validate_data: bool = True
    date_column: str = 'datetime'
    enable_caching: bool = False
    cache_dir: Optional[str] = None
    required_columns: List[str] = None

    def __post_init__(self):
        if self.required_columns is None:
            self.required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']


@dataclass
class DataSource:
    """데이터 소스 정보"""
    path: str
    format: str
    size_bytes: int
    is_valid: bool
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DataLoader:
    """히스토리 데이터 로더"""

    def __init__(self, config: Optional[LoaderConfig] = None):
        """
        DataLoader 초기화

        Args:
            config: 로더 설정
        """
        self.config = config or LoaderConfig()
        self.supported_formats = ['.csv', '.parquet', '.json']
        self.cache = {} if self.config.enable_caching else None
        self.cache_hits = 0

        # 로거 설정
        self.logger = logging.getLogger(__name__)

    def detect_source(self, file_path: str) -> DataSource:
        """
        데이터 소스 감지 및 검증

        Args:
            file_path: 데이터 파일 경로

        Returns:
            DataSource: 감지된 소스 정보

        Raises:
            FileNotFoundError: 파일이 존재하지 않을 때
            ValueError: 지원하지 않는 형식일 때
        """
        path = Path(file_path)

        # 파일 존재 확인
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # 파일 형식 확인
        file_extension = path.suffix.lower()
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")

        # 파일 정보 수집
        file_stats = os.stat(file_path)
        size_bytes = file_stats.st_size

        # 형식별 처리
        format_name = file_extension[1:]  # .csv -> csv

        return DataSource(
            path=file_path,
            format=format_name,
            size_bytes=size_bytes,
            is_valid=True,
            metadata={
                'modified_time': datetime.fromtimestamp(file_stats.st_mtime),
                'size_mb': size_bytes / (1024 * 1024)
            }
        )

    def load(self, file_path: str) -> pd.DataFrame:
        """
        데이터 파일 로드

        Args:
            file_path: 로드할 파일 경로

        Returns:
            pd.DataFrame: 로드된 데이터

        Raises:
            FileNotFoundError: 파일이 존재하지 않을 때
            pd.errors.ParserError: 파일 파싱 에러
        """
        # 캐시 확인
        if self.cache is not None:
            cache_key = self._get_cache_key(file_path)
            if cache_key in self.cache:
                self.cache_hits += 1
                return self.cache[cache_key].copy()

        # 데이터 소스 감지
        source = self.detect_source(file_path)

        # 메모리 사용량 체크
        available_memory_mb = self._get_available_memory_mb()
        estimated_memory_mb = source.size_bytes / (1024 * 1024) * 3  # rough estimate

        # 청크 처리 여부 결정
        use_chunks = (
            estimated_memory_mb > self.config.memory_limit_mb or
            estimated_memory_mb > available_memory_mb * 0.5
        )

        # 데이터 로드
        if use_chunks:
            data = self._load_in_chunks(source)
        else:
            data = self._load_complete(source)

        # 데이터 후처리
        data = self._postprocess_data(data)

        # 캐시 저장
        if self.cache is not None:
            cache_key = self._get_cache_key(file_path)
            self.cache[cache_key] = data.copy()

        return data

    def _load_complete(self, source: DataSource) -> pd.DataFrame:
        """전체 데이터 로드"""
        if source.format == 'csv':
            return pd.read_csv(source.path)
        elif source.format == 'parquet':
            return pd.read_parquet(source.path)
        elif source.format == 'json':
            return pd.read_json(source.path)
        else:
            raise ValueError(f"Unsupported format: {source.format}")

    def _load_in_chunks(self, source: DataSource) -> pd.DataFrame:
        """청크 단위로 데이터 로드"""
        chunks = []

        if source.format == 'csv':
            chunk_reader = pd.read_csv(source.path, chunksize=self.config.chunk_size)
            for chunk in chunk_reader:
                chunks.append(chunk)
        elif source.format == 'parquet':
            # Parquet은 행 단위 청크 읽기가 제한적이므로 전체 로드
            return pd.read_parquet(source.path)
        elif source.format == 'json':
            # JSON도 청크 읽기가 제한적
            return pd.read_json(source.path)
        else:
            raise ValueError(f"Unsupported format: {source.format}")

        # 청크 결합
        return pd.concat(chunks, ignore_index=True)

    def _postprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """데이터 후처리"""
        # 날짜 컬럼 변환
        if self.config.date_column in data.columns:
            if not pd.api.types.is_datetime64_any_dtype(data[self.config.date_column]):
                data[self.config.date_column] = pd.to_datetime(data[self.config.date_column])

        # 데이터 정렬 (날짜 기준)
        if self.config.date_column in data.columns:
            data = data.sort_values(self.config.date_column).reset_index(drop=True)

        return data

    def validate_ohlcv_structure(self, data: pd.DataFrame) -> Dict[str, bool]:
        """
        OHLCV 데이터 구조 검증

        Args:
            data: 검증할 데이터

        Returns:
            Dict: 검증 결과
        """
        result = {
            'is_valid': True,
            'has_required_columns': True,
            'datetime_column_valid': True,
            'numeric_columns_valid': True,
            'price_relationships_valid': True
        }

        # 필수 컬럼 확인
        missing_columns = set(self.config.required_columns) - set(data.columns)
        if missing_columns:
            result['has_required_columns'] = False
            result['is_valid'] = False

        # 날짜 컬럼 검증
        if self.config.date_column in data.columns:
            if not pd.api.types.is_datetime64_any_dtype(data[self.config.date_column]):
                result['datetime_column_valid'] = False
                result['is_valid'] = False
        else:
            result['datetime_column_valid'] = False
            result['is_valid'] = False

        # 숫자 컬럼 검증
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    result['numeric_columns_valid'] = False
                    result['is_valid'] = False

        # OHLC 관계 검증 (샘플링으로 성능 고려)
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            try:
                sample_size = min(1000, len(data))
                sample_data = data.sample(sample_size) if len(data) > sample_size else data

                # 모든 OHLC 컬럼이 숫자형인지 확인
                ohlc_cols = ['open', 'high', 'low', 'close']
                numeric_ohlc = all(pd.api.types.is_numeric_dtype(sample_data[col]) for col in ohlc_cols)

                if numeric_ohlc:
                    # high >= max(open, close)
                    high_valid = (sample_data['high'] >= sample_data[['open', 'close']].max(axis=1)).all()
                    # low <= min(open, close)
                    low_valid = (sample_data['low'] <= sample_data[['open', 'close']].min(axis=1)).all()
                    # 모든 가격 > 0
                    positive_prices = (sample_data[['open', 'high', 'low', 'close']] > 0).all().all()

                    if not (high_valid and low_valid and positive_prices):
                        result['price_relationships_valid'] = False
                        result['is_valid'] = False
                else:
                    # 숫자가 아닌 경우 관계 검증 실패
                    result['price_relationships_valid'] = False
                    result['is_valid'] = False
            except Exception:
                # 예외 발생 시 관계 검증 실패로 처리
                result['price_relationships_valid'] = False
                result['is_valid'] = False

        return result

    def _get_cache_key(self, file_path: str) -> str:
        """캐시 키 생성"""
        # 파일 경로와 수정 시간 기반으로 해시 생성
        file_stats = os.stat(file_path)
        key_data = f"{file_path}_{file_stats.st_mtime}_{file_stats.st_size}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_available_memory_mb(self) -> float:
        """사용 가능한 메모리 크기 (MB) 반환"""
        if PSUTIL_AVAILABLE:
            try:
                memory = psutil.virtual_memory()
                return memory.available / (1024 * 1024)
            except:
                pass

        # psutil을 사용할 수 없는 경우 기본값 반환
        return 1024.0

    def clear_cache(self):
        """캐시 초기화"""
        if self.cache is not None:
            self.cache.clear()
            self.cache_hits = 0

    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        if self.cache is None:
            return {'enabled': False}

        return {
            'enabled': True,
            'entries': len(self.cache),
            'hits': self.cache_hits,
            'hit_rate': self.cache_hits / max(1, len(self.cache))
        }