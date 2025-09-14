# 코인 선물 자동매매 시스템 - 핵심 시스템

## 1. 시스템 개요

### 1.1 기본 정보
```yaml
exchange: Binance Futures (USDT-M)
capital_stages: [230, 770, 7700]  # USDT
universe_size: [10, 25]  # min, max symbols
environment: Docker/Linux → VPS (production)
operation_modes: [PAPER, TESTNET, LIVE]

# 단위 정의 (USDT 통일)
units:
  base_currency: "USDT"
  price: "USDT per coin"
  quantity: "coin units"
  equity: "USDT"
  pnl: "USDT"
  margin: "USDT"
  leverage: "multiplier (e.g., 3 = 3x)"
  returns: "decimal (e.g., 0.01 = 1%)"
  volatility: "decimal per period"
```

### 1.2 핵심 목표
- **목표 Sharpe Ratio**: ≥ 1.5
- **최대 허용 드로다운**: -12%
- **연간 목표 수익률**: 30-50%
- **파산 확률**: < 1%
- **청산 확률 (24h)**: < 0.5%

### 1.3 시스템 철학
- **리스크 우선**: 수익 극대화보다 자본 보존 우선
- **과학적 접근**: 검증된 금융공학 모델 적용
- **실전 최적화**: 거래소 특성과 실제 비용 완벽 반영
- **지속가능성**: 장기 운영 가능한 안정적 시스템

### 1.4 핵심 컴포넌트 구조
```
Trading System
├── Financial Engineering Layer
│   ├── Kelly Criterion Optimizer
│   ├── Risk Metrics Calculator
│   └── Portfolio Optimizer
├── Strategy Layer
│   ├── Regime Detector
│   ├── Strategy Matrix
│   └── Signal Generator
├── Risk Management Layer
│   ├── Risk Controller
│   ├── Position Sizer
│   └── Liquidation Monitor
├── Execution Layer
│   ├── Smart Order Router
│   ├── Market Impact Model
│   └── Slippage Controller
└── Monitoring Layer
    ├── Real-time Dashboard
    ├── Alert System
    └── Performance Tracker
```

### 1.5 데이터 흐름
```
Market Data → Regime Detection → Strategy Selection → Signal Generation
    ↓                                                      ↓
Data Quality Check                                   Risk Assessment
    ↓                                                      ↓
Cleaned Data                                        Position Sizing
    ↓                                                      ↓
Feature Engineering                                  Order Execution
    ↓                                                      ↓
Model Input                                        Portfolio Update
```

### 1.6 운영 모드별 설정

#### Paper Trading Mode
```yaml
capital: 100  # USDT (가상)
max_leverage: 2
risk_limits:
  var_daily: 0.05
  max_drawdown: 0.20
execution:
  simulated: true
  latency: 0
```

#### Staging Mode
```yaml
capital: 230  # USDT (실제)
max_leverage: 3
risk_limits:
  var_daily: 0.03
  max_drawdown: 0.15
execution:
  real: true
  size_limit: 0.1  # 10% of normal size
```

#### Production Mode
```yaml
capital: 7700  # USDT (실제)
max_leverage: 10
risk_limits:
  var_daily: 0.02
  max_drawdown: 0.12
execution:
  real: true
  size_limit: 1.0  # full size
```

### 1.7 거래소 연동 사양

#### Binance Futures USDT-M
- **API Rate Limits**: 
  - Order: 300/min, 1200/min (IP)
  - Weight: 6000/min
- **WebSocket Streams**:
  - Trade streams
  - Depth streams
  - Mark price streams
  - Account streams
- **Latency Requirements**:
  - p50: < 50ms
  - p99: < 200ms

### 1.8 시스템 요구사항

#### 하드웨어
- CPU: 4+ cores
- RAM: 8GB minimum
- Storage: 100GB SSD
- Network: 100Mbps+ dedicated

#### 소프트웨어
- OS: Ubuntu 20.04 LTS or later
- Python: 3.10+
- Docker: 20.10+
- PostgreSQL: 14+

### 1.9 성능 목표

#### 실행 품질
- Fill Rate: > 95%
- Slippage: < 5 bps
- Post-Only Success: > 70%

#### 시스템 안정성
- Uptime: > 99.9%
- API Error Rate: < 0.1%
- Data Loss: 0%

#### 리스크 관리
- Risk Check Latency: < 10ms
- Position Update Latency: < 100ms
- Alert Delivery: < 1s