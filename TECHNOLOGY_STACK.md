# 기술 스택 및 아키텍처 사양서

**프로젝트**: 코인 선물 자동매매 시스템
**문서 타입**: 기술 사양서 (Technology Stack Specification)
**마지막 업데이트**: 2025-09-15
**관련 문서**: 📋 `@PROJECT_STRUCTURE.md`, 📋 `@ENVIRONMENT.md`

## 📋 문서 목적

이 문서는 코인 선물 자동매매 시스템의 모든 기술적 구성 요소, 버전, 아키텍처 선택 사항의 **단일 정보원(Single Source of Truth)**입니다.

---

## 🏗️ 핵심 기술 스택

### 프로그래밍 언어 및 런타임
- **Python**: 3.10.18 (Anaconda 환경)
- **가상환경**: Anaconda `autotrading` 환경
- **패키지 관리**: pip + conda (hybrid approach)

### 아키텍처 패턴
- **Clean Architecture**: 도메인 중심 설계
- **Hexagonal Architecture**: 포트 어댑터 패턴
- **Event-Driven Architecture**: 비동기 이벤트 처리
- **CQRS Pattern**: 명령과 쿼리 분리
- **Repository Pattern**: 데이터 접근 추상화

### 데이터베이스 및 스토리지
- **주 데이터베이스**: PostgreSQL 15+
- **시계열 데이터**: TimescaleDB (PostgreSQL 확장)
- **마이그레이션**: Alembic 1.13.0+
- **ORM**: SQLAlchemy 2.0+
- **연결 풀링**: asyncpg + SQLAlchemy async

### 비동기 처리 및 네트워킹
- **비동기 런타임**: asyncio (Python 표준 라이브러리)
- **HTTP 클라이언트**: aiohttp 3.9.0+
- **WebSocket**: websockets 12.0+
- **동시성 제어**: asyncio.Queue, asyncio.Lock

### 데이터 분석 및 수치 계산
- **수치 계산**: numpy 2.2.5
- **데이터 조작**: pandas 2.3.2
- **과학 계산**: scipy 1.15.3
- **통계 분석**: statsmodels 0.14.0+
- **머신러닝**: scikit-learn 1.7.1

### 금융 데이터 및 API 연동
- **거래소 API**: ccxt 4.4.82 (다중 거래소 지원)
- **REST API**: aiohttp 기반 비동기 호출
- **WebSocket 스트림**: 실시간 시장 데이터
- **레이트 리미팅**: asyncio 기반 토큰 버킷 구현

### 테스팅 프레임워크
- **단위 테스트**: pytest 8.0+
- **비동기 테스트**: pytest-asyncio 0.23.0+
- **모킹**: pytest-mock, unittest.mock
- **테스트 커버리지**: pytest-cov

### 로깅 및 모니터링
- **구조화 로깅**: structlog 24.2.0
- **로그 포맷**: JSON 구조화 로그
- **민감 데이터 필터링**: 커스텀 필터 구현
- **성능 모니터링**: 커스텀 메트릭 수집

### 설정 관리
- **설정 모델**: Pydantic 2.0+ (타입 안전 설정)
- **환경 변수**: python-dotenv 1.0.0+
- **YAML 설정**: PyYAML 6.0+
- **설정 검증**: Pydantic 기반 스키마 검증

---

## 🏛️ 시스템 아키텍처 계층

### 1. 프레젠테이션 계층 (Presentation Layer)
- **CLI 인터페이스**: argparse 기반 명령행 도구
- **설정 인터페이스**: 환경 변수 + YAML 파일
- **모니터링 대시보드**: (향후 구현 예정)

### 2. 응용 계층 (Application Layer)
- **전략 관리자**: Strategy Engine 오케스트레이션
- **포트폴리오 관리자**: 자산 배분 및 최적화
- **주문 실행기**: (Phase 4 구현 예정)
- **리스크 컨트롤러**: 위험 관리 및 모니터링

### 3. 도메인 계층 (Domain Layer)
- **거래 전략**: 순수 비즈니스 로직
- **리스크 모델**: Kelly Criterion, VaR 계산
- **포지션 관리**: 포지션 생명주기 관리
- **시장 데이터**: OHLCV 및 메타데이터 모델

### 4. 인프라 계층 (Infrastructure Layer)
- **데이터 저장소**: PostgreSQL + TimescaleDB
- **외부 API**: Binance Futures API 연동
- **실시간 데이터**: WebSocket 스트림 처리
- **로깅 시스템**: 구조화 로그 및 감사 추적

---

## 🔧 개발 도구 및 환경

### 개발 환경
- **IDE 지원**: VS Code, PyCharm 호환
- **린팅**: flake8, black (코드 포매팅)
- **타입 체킹**: mypy (점진적 타입 체킹)
- **의존성 관리**: requirements.txt + environment.yml

### 버전 관리 및 CI/CD
- **VCS**: Git (로컬 개발)
- **브랜치 전략**: Feature branch workflow
- **커밋 컨벤션**: TDD 기반 구조적/행동적 변경 구분

### 보안 및 암호화
- **API 키 관리**: 환경 변수 + .env 파일
- **민감 데이터**: 로그에서 자동 마스킹
- **HTTPS**: aiohttp 기반 안전한 API 통신
- **인증**: Binance API 서명 및 인증 처리

---

## 📊 성능 및 확장성 고려사항

### 성능 요구사항
- **주문 실행 지연시간**: <100ms (전체 워크플로)
- **데이터 처리 처리량**: 1000+ 틱/초
- **동시 연결**: 다중 거래소 WebSocket 연결
- **메모리 효율성**: 대용량 시계열 데이터 처리

### 확장성 설계
- **비동기 처리**: asyncio 기반 논블로킹 I/O
- **데이터베이스 최적화**: TimescaleDB 시계열 최적화
- **연결 풀링**: 데이터베이스 연결 재사용
- **리소스 관리**: 컨텍스트 매니저 기반 리소스 정리

### 안정성 및 복원력
- **에러 처리**: 계층별 예외 처리 전략
- **재시도 로직**: Exponential backoff 구현
- **서킷 브레이커**: 외부 서비스 장애 대응
- **자동 복구**: 네트워크 연결 끊김 자동 복구

---

## 🔄 의존성 관리

### 핵심 Python 패키지 (requirements.txt)
```
# 데이터 분석 및 수치 계산
pandas==2.3.2
numpy==2.2.5
scipy==1.15.3
scikit-learn==1.7.1
statsmodels>=0.14.0

# 비동기 처리 및 네트워킹
aiohttp>=3.9.0
websockets==12.0
asyncio-throttle>=1.0.0

# 데이터베이스 및 ORM
sqlalchemy>=2.0.0
alembic>=1.13.0
asyncpg>=0.29.0
psycopg2-binary>=2.9.0

# 거래소 API 및 금융 데이터
ccxt==4.4.82
python-dotenv>=1.0.0

# 설정 및 검증
pydantic>=2.0.0
PyYAML>=6.0

# 로깅 및 모니터링
structlog==24.2.0

# 테스팅 프레임워크
pytest>=8.0.0
pytest-asyncio>=0.23.0
pytest-mock>=3.12.0
pytest-cov>=4.0.0
```

### Anaconda 환경 패키지 (environment.yml)
```yaml
name: autotrading
dependencies:
  - python=3.10.18
  - pandas=2.3.2
  - numpy=2.2.5
  - scipy=1.15.3
  - scikit-learn=1.7.1
  - pip
  - pip:
    - -r requirements.txt
```

---

## 🎯 아키텍처 결정 기록 (Architecture Decision Records)

### ADR-001: Python 3.10.18 선택
- **결정**: Python 3.10.18 사용
- **이유**: 안정성, 성능, asyncio 개선사항, 타입 힌트 지원
- **대안**: Python 3.11/3.12 (너무 새로움), Python 3.9 (기능 부족)

### ADR-002: PostgreSQL + TimescaleDB 선택
- **결정**: PostgreSQL을 TimescaleDB 확장과 함께 사용
- **이유**: 시계열 데이터 최적화, ACID 속성, 고성능 쿼리
- **대안**: MongoDB (스키마 유연성 부족), InfluxDB (관계형 데이터 부족)

### ADR-003: asyncio 기반 비동기 아키텍처
- **결정**: asyncio 기반 완전 비동기 처리
- **이유**: 높은 동시성, I/O 집약적 작업 최적화, 메모리 효율성
- **대안**: 멀티스레딩 (GIL 제약), 멀티프로세싱 (메모리 오버헤드)

### ADR-004: Clean Architecture + Hexagonal Pattern
- **결정**: Clean Architecture와 Hexagonal Architecture 결합
- **이유**: 테스트 가능성, 의존성 역전, 외부 시스템 분리
- **대안**: 레이어드 아키텍처 (결합도 높음), MVC (금융 도메인 부적합)

---

## 📚 관련 문서 참조

### 환경 설정 및 구성
- **📋 환경 설정 가이드**: `@ENVIRONMENT.md`
- **📋 프로젝트 구조**: `@PROJECT_STRUCTURE.md`

### 아키텍처 및 설계
- **📋 시스템 아키텍처**: `@docs/project-system-architecture.md`
- **📋 금융공학 모델**: `@docs/project-system-design/2-financial-engineering.md`

### 구현 가이드
- **📋 TDD 방법론**: `@docs/augmented-coding.md`
- **📋 엔지니어링 가이드**: `@docs/software-engineering-guide.md`

---

**문서 관리 규칙**: 이 문서는 기술 스택 정보의 단일 정보원입니다. 다른 문서에서 기술 스택 정보가 필요한 경우 이 문서를 참조해야 합니다.

**업데이트 정책**: 새로운 패키지 추가, 버전 업그레이드, 아키텍처 변경 시 이 문서를 먼저 업데이트해야 합니다.