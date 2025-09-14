# 코인 선물 자동매매 시스템 (Cryptocurrency Futures Auto Trading System)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Architecture](https://img.shields.io/badge/Architecture-Microservices-green)
![Exchange](https://img.shields.io/badge/Exchange-Binance%20Futures-yellow)
![Status](https://img.shields.io/badge/Status-Design%20%26%20Planning-orange)

## 📋 프로젝트 개요

**고급 정량적 거래 전략을 구현한 한국어 암호화폐 선물 자동매매 시스템**입니다. 정교한 리스크 관리와 포트폴리오 최적화를 통해 안정적이고 지속 가능한 수익을 목표로 합니다.

### 🎯 핵심 목표
- **목표 Sharpe Ratio**: ≥ 1.5
- **최대 허용 드로다운**: -12%
- **연간 목표 수익률**: 30-50%
- **파산 확률**: < 1%
- **청산 확률 (24시간)**: < 0.5%

### 💡 시스템 철학
- **리스크 우선**: 수익 극대화보다 자본 보존 우선
- **과학적 접근**: 검증된 금융공학 모델 적용
- **실전 최적화**: 거래소 특성과 실제 비용 완벽 반영
- **지속가능성**: 장기 운영 가능한 안정적 시스템

## 🏗️ 시스템 아키텍처

### 마이크로서비스 아키텍처
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Trading       │    │   Risk          │    │   Strategy      │
│   Engine        │◄──►│   Manager       │◄──►│   Engine        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Order         │    │   Data Feed     │    │   Portfolio     │
│   Executor      │    │   Service       │    │   Manager       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 핵심 컴포넌트
- **Trading Engine**: Python/asyncio 기반 메인 조정 허브
- **Risk Manager**: Kelly 최적화 및 VaR 기반 리스크 제어
- **Strategy Engine**: 멀티 전략 실행 및 시장 상태 감지
- **Order Executor**: 스마트 주문 라우팅 및 실행
- **Data Feed Service**: 실시간 시장 데이터 수집
- **Portfolio Manager**: 동적 포트폴리오 할당 및 최적화

### 아키텍처 패턴
- **Event-driven Architecture**: 느슨한 결합을 위한 이벤트 기반 아키텍처
- **CQRS**: 명령-쿼리 책임 분리 패턴
- **Hexagonal Architecture**: 관심사의 명확한 분리

## 🔧 기술 스택

### Backend & Runtime
- **언어**: Python 3.10+
- **비동기 처리**: asyncio
- **통신**: gRPC (서비스 간), WebSocket (시장 데이터)

### 데이터베이스
- **PostgreSQL**: 트랜잭션 데이터
- **TimescaleDB**: 시계열 데이터
- **Redis**: 캐싱 및 상태 관리

### 인프라
- **컨테이너화**: Docker
- **오케스트레이션**: Kubernetes (선택사항)
- **모니터링**: Prometheus + Grafana, AlertManager

### 거래소
- **Primary**: Binance Futures (USDT-M)
- **지원 모드**: PAPER, TESTNET, LIVE

## 📊 금융공학 모델

### 리스크 관리
- **Kelly Optimization**: Fractional Kelly Criterion을 통한 포지션 사이징
- **VaR Models**: Value at Risk 기반 리스크 측정
- **Drawdown Monitoring**: 실시간 손실 추적 및 제어
- **Liquidation Probability**: 청산 위험 확률 계산

### 전략 시스템
- **Regime Detection**: HMM 및 GARCH 모델을 통한 시장 상태 식별
- **Multi-Strategy Matrix**:
  - Trend Following
  - Mean Reversion
  - Funding Rate Arbitrage
- **Dynamic Allocation**: 리스크 제약 조건 하의 동적 자본 배분

### 포트폴리오 최적화
- **Kelly-based Position Sizing**: 수학적으로 최적화된 포지션 크기
- **Risk Parity**: 리스크 균형 기반 자본 배분
- **Correlation Management**: 상관관계를 고려한 다각화

## 🧪 개발 방법론

### TDD (Test-Driven Development)
Kent Beck의 TDD 원칙 및 Tidy First 접근법 준수:

1. **Red**: 실패하는 테스트 먼저 작성
2. **Green**: 테스트를 통과하는 최소한의 코드 작성
3. **Refactor**: 테스트 통과 후 코드 구조 개선

### 핵심 개발 규칙
- 구조적 변경과 행동 변경을 같은 커밋에 혼합하지 않음
- 모든 테스트 통과 후 커밋
- 구조적 변경과 행동적 변경은 별도 커밋
- 의미있는 테스트 이름으로 행동 설명
- 한 번에 하나의 리팩토링 변경

### 금융 컴포넌트 TDD 워크플로우
1. **포괄적 테스트**: 알려진 예상 출력으로 수학적 함수 테스트
2. **엣지 케이스 테스트**: 제로 포지션, 극한 시장 조건, 경계값
3. **벤치마크 검증**: 기존 금융 모델과 리스크 메트릭 비교
4. **지속적 리팩토링**: 금융 로직의 깔끔함과 유지보수성 유지

## 📁 프로젝트 구조

```
AutoTradingNew/
├── docs/                           # 설계 문서
│   ├── project-system-architecture.md
│   ├── augmented-coding.md
│   ├── software-engineering-guide.md
│   └── project-system-design/
│       ├── 1-core-system.md
│       ├── 2-financial-engineering.md
│       ├── 3-strategy-engine.md
│       ├── 4-risk-management.md
│       ├── 5-portfolio-optimization.md
│       ├── 6-execution-engine.md
│       ├── 7-market-microstructure.md
│       ├── 8-backtesting.md
│       ├── 9-monitoring.md
│       ├── 10-infrastructure.md
│       ├── 11-data-quality.md
│       ├── 12-main-system.md
│       ├── 13-validation-checklist.md
│       └── 14-implementation-guide.md
├── src/                            # 소스 코드 (구현 예정)
├── tests/                          # 테스트 코드 (구현 예정)
├── config/                         # 설정 파일 (구현 예정)
├── scripts/                        # 유틸리티 스크립트 (구현 예정)
├── CLAUDE.md                       # Claude Code 개발 가이드
└── README.md                       # 이 파일
```

## 🚀 시작하기

> **참고**: 현재 프로젝트는 설계 및 기획 단계에 있으며, 구현은 아직 시작되지 않았습니다.

### 필수 요구사항
- Python 3.10+
- Docker & Docker Compose
- PostgreSQL 13+
- Redis 6+
- 충분한 시스템 리소스 (메모리 8GB+, SSD 권장)

### 성능 요구사항
- 주문 실행 경로 100ms 미만 지연시간
- 실시간 시장 데이터 피드 처리
- 대용량 시계열 데이터셋 효율적 처리
- 다중 거래 쌍 확장 가능한 아키텍처

## 🧪 테스팅 전략

### 테스팅 피라미드
- **단위 테스트**: 알려진 예상 출력으로 모든 수학적/금융 함수
- **통합 테스트**: 거래소 연결성 및 데이터 파이프라인
- **백테스팅**: 과거 데이터로 전략 검증
- **페이퍼 트레이딩**: 실제 돈 없이 라이브 시스템 검증
- **카오스 엔지니어링**: 복원력 테스팅

## 📈 모니터링 및 관찰성

### 핵심 메트릭
- **성능 메트릭**: Sharpe Ratio, Drawdown, PnL
- **시스템 헬스**: 지연시간, 에러율, 처리량
- **리스크 메트릭**: VaR 위반, 포지션 크기, 레버리지
- **알림 임계값**: 중요한 시스템 및 금융 이벤트

### 인프라 모니터링
- Prometheus + Grafana 스택
- 실시간 대시보드
- 자동화된 알림 시스템
- 포괄적인 로깅 및 추적

## 🔒 보안 고려사항

- 환경 변수 또는 안전한 키 관리 시스템에 API 키 저장
- 코드나 버전 관리에 자격 증명 없음
- 민감한 데이터에 대한 암호화 통신
- 모든 거래 활동에 대한 포괄적인 감사 로깅
- 극한 시장 상황을 위한 회로 차단기
- 긴급 종료를 위한 킬 스위치

## 🏢 규정 준수

- 자동 거래 시스템에 대한 규제 준수
- 모든 거래 결정에 대한 포괄적인 감사 추적
- 포지션 크기 제한 및 최대 드로다운 임계값
- 극한 시장 조건을 위한 회로 차단기

## 📚 문서 참조

### 핵심 시스템 문서
- [시스템 아키텍처](docs/project-system-architecture.md) - C4 모델 완전 문서
- [핵심 시스템](docs/project-system-design/1-core-system.md) - 메인 시스템 설계
- [금융공학 모델](docs/project-system-design/2-financial-engineering.md) - Kelly 최적화 및 리스크 모델
- [전략 엔진](docs/project-system-design/3-strategy-engine.md) - 멀티 전략 시스템
- [리스크 관리](docs/project-system-design/4-risk-management.md) - VaR 및 드로다운 제어

### 개발 가이드
- [개발 방법론](docs/augmented-coding.md) - Kent Beck TDD 및 Tidy First 원칙
- [소프트웨어 엔지니어링](docs/software-engineering-guide.md) - 포괄적 엔지니어링 가이드라인
- [Claude 개발 가이드](CLAUDE.md) - Claude Code 작업을 위한 가이드

## 🤝 기여하기

현재 프로젝트는 설계 단계이며, 기여 가이드라인은 구현이 시작되면 업데이트될 예정입니다.

## ⚠️ 면책 조항

이 소프트웨어는 교육 및 연구 목적으로 제공됩니다. 실제 거래에서의 사용은 본인의 책임이며, 금융 손실에 대한 책임을 지지 않습니다. 모든 거래는 위험을 수반하며, 투자한 자본을 잃을 수 있습니다.

## 📄 라이선스

TBD (구현 단계에서 결정 예정)

---

**프로젝트 상태**: 설계 및 기획 단계
**마지막 업데이트**: 2025년 1월
**문의**: 프로젝트 이슈 또는 문의사항은 GitHub Issues를 통해 제출해 주세요.