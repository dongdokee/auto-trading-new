# 🔄 AutoTrading 시스템 리팩토링 계획서

**목적**: 코드 품질 향상, 유지보수성 개선, 성능 최적화
**작성일**: 2025-10-03
**대상 시스템**: 한국 암호화폐 선물 자동매매 시스템
**현재 상태**: Phase 6.1 완료 (100% 구현)

## 📊 현재 시스템 분석

### 시스템 규모
| 구분 | 수량 | 상세 |
|------|------|------|
| 소스 코드 | 95개 파일 | 29,452 라인 |
| 테스트 코드 | 81개 파일 | 23,738 라인 |
| 모듈 수 | 15개 | api, backtesting, core, execution 등 |
| 테스트 수 | 824+ | 100% 성공률 |

### 모듈별 파일 수
```
integration: 18 files     (가장 복잡)
core: 13 files
strategy_engine: 10 files
optimization: 9 files      (대형 파일 다수)
market_data: 9 files
execution: 7 files
api: 7 files
portfolio: 5 files
backtesting: 4 files
risk_management: 4 files
utils: 4 files
```

### 식별된 문제점

#### 1. 코드 복잡도 문제
**대형 파일 현황** (500라인 이상):
- `optimization/analytics_system.py`: 990 라인
- `optimization/db_optimizer.py`: 871 라인
- `optimization/deployment_tools.py`: 789 라인
- `optimization/cache_manager.py`: 772 라인
- `optimization/performance_enhancer.py`: 757 라인
- `portfolio/adaptive_allocator.py`: 705 라인
- `optimization/hyperparameter_tuner.py`: 681 라인
- `execution/execution_algorithms.py`: 655 라인

**권장 사항**: 파일당 최대 300라인, 함수당 최대 50라인

#### 2. 코드 중복 문제
- **Logger 패턴**: 34개 파일에서 동일한 logger 초기화
- **Connection 패턴**: connect/disconnect 메소드 중복 구현
- **Manager 클래스**: 10개의 유사한 Manager 클래스 존재

#### 3. 의존성 복잡도
- **상대 임포트**: 24개 파일에서 `from ..` 패턴 사용
- **순환 의존성**: 모듈 간 강한 결합 의심
- **인터페이스 부재**: 명확한 추상화 레이어 부족

## 🎯 리팩토링 목표

### 정량적 목표
| 메트릭 | 현재 | 목표 | 개선율 |
|--------|------|------|--------|
| 평균 파일 크기 | 310 라인 | 200 라인 | 35% 감소 |
| 최대 파일 크기 | 990 라인 | 300 라인 | 70% 감소 |
| 코드 중복률 | 추정 30% | 15% | 50% 감소 |
| 테스트 커버리지 | 100% | 100% | 유지 |

### 정성적 목표
- **가독성**: 코드 이해도 향상
- **유지보수성**: 수정 및 확장 용이성
- **테스트 용이성**: Mock 및 단위 테스트 개선
- **성능**: 메모리 사용량 및 실행 속도 최적화

## 📋 단계별 실행 계획

### Phase 1: 문서화 및 기준선 설정 (Days 1-2)

#### Day 1: 분석 및 문서화
- [x] 현재 시스템 분석 완료
- [x] 리팩토링 계획 문서 작성
- [ ] 코드 메트릭 기준선 측정
- [ ] 의존성 그래프 생성

#### Day 2: 테스트 환경 준비
- [ ] 리팩토링 전 전체 테스트 실행
- [ ] 성능 벤치마크 기준 측정
- [ ] 백업 브랜치 생성 (`refactor/baseline`)

### Phase 2: 대형 모듈 분해 (Days 3-8)

#### 2.1 Optimization 모듈 리팩토링 (Days 3-5)

**우선순위 1: analytics_system.py (990라인)**
```
현재 구조:
analytics_system.py (990라인)
├── AnalyticsResult
├── TimeSeriesData
├── StatisticalAnalyzer
├── MachineLearningPipeline
├── PerformanceAnalyzer
├── RiskAnalyzer
└── AdvancedAnalyticsSystem

목표 구조:
analytics/
├── __init__.py
├── core.py (AnalyticsResult, TimeSeriesData)
├── statistical.py (StatisticalAnalyzer)
├── ml_pipeline.py (MachineLearningPipeline)
├── performance.py (PerformanceAnalyzer)
├── risk.py (RiskAnalyzer)
└── system.py (AdvancedAnalyticsSystem)
```

**우선순위 2: db_optimizer.py (871라인)**
```
현재 구조:
db_optimizer.py (871라인)
├── QueryPlan, QueryStats
├── QueryOptimizer
├── ConnectionPoolManager
└── DatabaseOptimizer

목표 구조:
database/
├── __init__.py
├── models.py (QueryPlan, QueryStats)
├── query_optimizer.py (QueryOptimizer)
├── connection_pool.py (ConnectionPoolManager)
└── optimizer.py (DatabaseOptimizer)
```

**우선순위 3: deployment_tools.py (789라인)**
```
현재 구조:
deployment_tools.py (789라인)
├── DeploymentResult, ContainerInfo
├── ContainerManager
├── BackupManager
├── RollingDeploymentStrategy
└── ProductionDeploymentTools

목표 구조:
deployment/
├── __init__.py
├── models.py (DeploymentResult, ContainerInfo)
├── container.py (ContainerManager)
├── backup.py (BackupManager)
├── strategies.py (RollingDeploymentStrategy)
└── tools.py (ProductionDeploymentTools)
```

#### 2.2 Integration 모듈 재구조화 (Days 6-8)

**현재 구조 분석:**
```
integration/ (18 files)
├── adapters/ (4 files)
├── events/ (3 files)
├── monitoring/ (3 files)
├── state/ (2 files)
└── trading_orchestrator.py (598라인)
```

**목표 구조:**
```
integration/
├── __init__.py
├── orchestrator/
│   ├── __init__.py
│   ├── coordinator.py (핵심 조정 로직)
│   ├── scheduler.py (스케줄링)
│   └── lifecycle.py (생명주기 관리)
├── adapters/
├── events/
├── monitoring/
└── state/
```

### Phase 3: 공통 패턴 추출 (Days 9-12)

#### 3.1 Connection Management 패턴 통합 (Day 9)

**문제**: 15개 파일에서 connect/disconnect 메소드 중복 구현

**해결 방안:**
```python
# src/core/patterns/connection.py
from abc import ABC, abstractmethod
from typing import Optional, Any
import asyncio

class BaseConnectionManager(ABC):
    """공통 연결 관리 추상 클래스"""

    def __init__(self):
        self._connected = False
        self._connection: Optional[Any] = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5

    @abstractmethod
    async def _create_connection(self) -> Any:
        """실제 연결 생성 (구현체별 정의)"""
        pass

    @abstractmethod
    async def _close_connection(self, connection: Any) -> None:
        """연결 정리 (구현체별 정의)"""
        pass

    async def connect(self) -> None:
        """표준화된 연결 메소드"""
        if self._connected:
            return

        try:
            self._connection = await self._create_connection()
            self._connected = True
            self._reconnect_attempts = 0
        except Exception as e:
            await self._handle_connection_error(e)

    async def disconnect(self) -> None:
        """표준화된 연결 해제 메소드"""
        if not self._connected:
            return

        try:
            if self._connection:
                await self._close_connection(self._connection)
        finally:
            self._connected = False
            self._connection = None

    async def _handle_connection_error(self, error: Exception) -> None:
        """연결 오류 처리 및 재시도"""
        if self._reconnect_attempts < self._max_reconnect_attempts:
            self._reconnect_attempts += 1
            await asyncio.sleep(2 ** self._reconnect_attempts)
            await self.connect()
        else:
            raise error
```

**적용 대상:**
- `api/binance/client.py`
- `api/binance/websocket.py`
- `api/binance/executor.py`
- `optimization/db_optimizer.py`

#### 3.2 Logger 패턴 중앙화 (Day 10)

**문제**: 34개 파일에서 동일한 logger 초기화 패턴

**해결 방안:**
```python
# src/core/patterns/logging.py
import logging
import structlog
from typing import Optional, Dict, Any
from functools import lru_cache

class LoggerFactory:
    """중앙화된 로거 팩토리"""

    @staticmethod
    @lru_cache(maxsize=128)
    def get_logger(name: str, context: Optional[Dict[str, Any]] = None) -> structlog.BoundLogger:
        """구조화된 로거 생성"""
        logger = structlog.get_logger(name)
        if context:
            logger = logger.bind(**context)
        return logger

    @staticmethod
    def get_trading_logger(component: str, symbol: Optional[str] = None) -> structlog.BoundLogger:
        """거래 특화 로거"""
        context = {"component": component}
        if symbol:
            context["symbol"] = symbol
        return LoggerFactory.get_logger("trading", context)

    @staticmethod
    def get_performance_logger() -> structlog.BoundLogger:
        """성능 측정 로거"""
        return LoggerFactory.get_logger("performance")

# 사용 예시
# AS-IS: logger = logging.getLogger(__name__)
# TO-BE: logger = LoggerFactory.get_trading_logger("risk_management")
```

#### 3.3 Manager 클래스 표준화 (Days 11-12)

**문제**: 10개의 Manager 클래스가 서로 다른 패턴 사용

**해결 방안:**
```python
# src/core/patterns/manager.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import asyncio

class BaseManager(ABC):
    """표준화된 매니저 기본 클래스"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._initialized = False
        self._running = False
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)

    async def initialize(self) -> None:
        """초기화 템플릿 메소드"""
        if self._initialized:
            return

        await self._before_initialize()
        await self._do_initialize()
        await self._after_initialize()
        self._initialized = True

    async def start(self) -> None:
        """시작 템플릿 메소드"""
        if not self._initialized:
            await self.initialize()

        if self._running:
            return

        await self._before_start()
        await self._do_start()
        await self._after_start()
        self._running = True

    async def stop(self) -> None:
        """중지 템플릿 메소드"""
        if not self._running:
            return

        await self._before_stop()
        await self._do_stop()
        await self._after_stop()
        self._running = False

    @abstractmethod
    async def _do_initialize(self) -> None:
        """실제 초기화 로직 (구현체별 정의)"""
        pass

    @abstractmethod
    async def _do_start(self) -> None:
        """실제 시작 로직 (구현체별 정의)"""
        pass

    @abstractmethod
    async def _do_stop(self) -> None:
        """실제 중지 로직 (구현체별 정의)"""
        pass

    # Hook 메소드들 (선택적 오버라이드)
    async def _before_initialize(self) -> None: pass
    async def _after_initialize(self) -> None: pass
    async def _before_start(self) -> None: pass
    async def _after_start(self) -> None: pass
    async def _before_stop(self) -> None: pass
    async def _after_stop(self) -> None: pass
```

### Phase 4: 의존성 개선 (Days 13-16)

#### 4.1 순환 의존성 제거 (Days 13-14)

**분석 계획:**
1. 의존성 그래프 생성
2. 순환 의존성 식별
3. 인터페이스 분리 적용
4. 의존성 역전 구현

**도구 사용:**
```bash
# 의존성 분석
"/c/Users/dongd/anaconda3/envs/autotrading/Scripts/pip.exe" install pydeps
pydeps src --show-deps --max-cluster-size 20
```

#### 4.2 모듈 경계 명확화 (Days 15-16)

**각 모듈의 public API 정의:**
```python
# 예시: src/execution/__init__.py
"""
Order Execution Module

Public API:
- OrderManager: 주문 생명주기 관리
- SmartOrderRouter: 지능형 주문 라우팅
- ExecutionAlgorithms: 실행 알고리즘
"""

from .order_manager import OrderManager
from .order_router import SmartOrderRouter
from .execution_algorithms import ExecutionAlgorithms
from .models import Order, ExecutionResult

__all__ = [
    "OrderManager",
    "SmartOrderRouter",
    "ExecutionAlgorithms",
    "Order",
    "ExecutionResult"
]

# 내부 구현은 노출하지 않음
# from .slippage_controller import SlippageController  # 내부 사용만
```

### Phase 5: 성능 최적화 (Days 17-20)

#### 5.1 비동기 패턴 최적화 (Days 17-18)

**문제점 식별:**
- 불필요한 await 사용
- 동시성 처리 미흡
- 비동기 컨텍스트 매니저 부재

**최적화 방안:**
```python
# AS-IS: 순차 처리
async def process_orders(orders):
    results = []
    for order in orders:
        result = await process_single_order(order)
        results.append(result)
    return results

# TO-BE: 병렬 처리
async def process_orders(orders):
    tasks = [process_single_order(order) for order in orders]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

#### 5.2 메모리 최적화 (Days 19-20)

**Generator 패턴 적용:**
```python
# AS-IS: 메모리 집약적
def load_all_data():
    return [process_record(r) for r in large_dataset]

# TO-BE: 메모리 효율적
def load_data_stream():
    for record in large_dataset:
        yield process_record(record)
```

### Phase 6: 테스트 및 검증 (Days 21-24)

#### 6.1 테스트 업데이트 (Days 21-22)
- 리팩토링된 모듈별 단위 테스트 작성
- 통합 테스트 업데이트
- Mock 객체 개선

#### 6.2 성능 검증 (Days 23-24)
- 전체 테스트 스위트 실행 (824+ 테스트)
- 성능 벤치마크 비교
- 메모리 사용량 측정

## 🔄 리팩토링 실행 원칙

### 1. 점진적 개선 (Incremental Improvement)
- 한 번에 하나의 모듈만 수정
- 각 단계별 테스트 실행
- 롤백 가능한 상태 유지

### 2. 테스트 주도 (Test-Driven)
- 리팩토링 전 테스트 작성
- 기능 변경 없이 구조만 개선
- Red-Green-Refactor 사이클 준수

### 3. 하위 호환성 (Backward Compatibility)
- 기존 API 인터페이스 유지
- 점진적 마이그레이션 경로 제공
- Deprecation 경고 활용

### 4. 성능 모니터링 (Performance Monitoring)
- 각 단계별 성능 측정
- 메모리 사용량 추적
- 실행 시간 비교

## 📊 성공 지표

### 정량적 지표
| 지표 | 현재 | 목표 | 측정 방법 |
|------|------|------|----------|
| 평균 파일 크기 | 310 라인 | 200 라인 | wc -l 명령어 |
| 최대 파일 크기 | 990 라인 | 300 라인 | find + sort |
| 순환 의존성 | 미측정 | 0개 | pydeps 도구 |
| 테스트 커버리지 | 100% | 100% | pytest-cov |
| 메모리 사용량 | 기준선 | -15% | memory_profiler |

### 정성적 지표
- **코드 가독성**: 팀원 코드 리뷰 점수
- **개발 속도**: 새 기능 추가 시간
- **버그 발생률**: 프로덕션 이슈 수
- **문서화 품질**: API 문서 완성도

## 🚨 위험 관리

### 주요 위험 요소
1. **기능 저하**: 리팩토링 중 기능 손실
2. **성능 저하**: 추상화로 인한 성능 오버헤드
3. **일정 지연**: 예상보다 복잡한 의존성
4. **테스트 실패**: 리팩토링으로 인한 테스트 불안정

### 위험 완화 방안
1. **기능 저하 방지**
   - 모든 변경사항에 대한 테스트 실행
   - 기능 변경 없는 구조적 리팩토링만 진행
   - 단계별 검증 체크포인트 설정

2. **성능 저하 방지**
   - 각 단계별 성능 벤치마크 실행
   - 추상화 레벨 최적화
   - 핫패스 코드 성능 우선 고려

3. **일정 관리**
   - 버퍼 시간 포함한 여유있는 계획
   - 데일리 진행 상황 점검
   - 우선순위별 단계적 접근

4. **테스트 안정성**
   - 리팩토링 전 전체 테스트 실행
   - Mock 객체 표준화
   - 통합 테스트 강화

## 📅 마일스톤 및 체크포인트

### Week 1: 기반 구축
- [ ] Day 2: 리팩토링 환경 준비 완료
- [ ] Day 4: optimization 모듈 분석 완료
- [ ] Day 6: 첫 번째 대형 파일 리팩토링 완료

### Week 2: 핵심 리팩토링
- [ ] Day 8: optimization 모듈 리팩토링 완료
- [ ] Day 10: Connection 패턴 통합 완료
- [ ] Day 12: Logger 패턴 중앙화 완료

### Week 3: 품질 개선
- [ ] Day 14: Manager 클래스 표준화 완료
- [ ] Day 16: 의존성 개선 완료
- [ ] Day 18: 비동기 패턴 최적화 완료

### Week 4: 검증 및 마무리
- [ ] Day 20: 메모리 최적화 완료
- [ ] Day 22: 테스트 업데이트 완료
- [ ] Day 24: 최종 검증 및 문서화 완료

## 📋 체크리스트

### 리팩토링 시작 전
- [ ] 현재 코드 백업 생성
- [ ] 전체 테스트 스위트 실행 (824+ 테스트)
- [ ] 성능 기준선 측정
- [ ] 의존성 그래프 생성

### 각 모듈 리팩토링 시
- [ ] 리팩토링 대상 모듈 테스트 실행
- [ ] 단계별 변경사항 커밋
- [ ] 리팩토링 후 테스트 실행
- [ ] 성능 영향 확인

### 리팩토링 완료 후
- [ ] 전체 테스트 스위트 실행
- [ ] 성능 벤치마크 비교
- [ ] 코드 메트릭 측정
- [ ] 문서 업데이트

## 🔗 관련 문서

- **프로젝트 현황**: `PROJECT_STATUS.md`
- **기술 스택**: `PROJECT_STRUCTURE.md`
- **개발 가이드**: `CLAUDE.md`
- **아키텍처 결정**: `docs/ARCHITECTURE_DECISIONS.md`
- **TDD 방법론**: `docs/augmented-coding.md`

---

**작성자**: Claude Code Assistant
**승인자**: Project Owner
**최종 업데이트**: 2025-10-03
**다음 리뷰**: 리팩토링 완료 후 (예상 2025-10-27)