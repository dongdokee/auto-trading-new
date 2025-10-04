# Binance Testnet Paper Trading Setup Guide
# Binance 테스트넷 모의매매 설정 가이드

이 가이드는 Binance Testnet을 사용하여 실제 돈 없이 안전하게 트레이딩 전략을 테스트할 수 있는 완전한 paper trading 환경을 설정하는 방법을 설명합니다.

## 📋 목차

1. [개요](#개요)
2. [사전 준비사항](#사전-준비사항)
3. [Binance Testnet 계정 설정](#binance-testnet-계정-설정)
4. [API 키 생성](#api-키-생성)
5. [환경 설정](#환경-설정)
6. [Paper Trading 실행](#paper-trading-실행)
7. [모니터링 및 분석](#모니터링-및-분석)
8. [문제 해결](#문제-해결)
9. [고급 설정](#고급-설정)

## 🎯 개요

### Paper Trading이란?

Paper Trading은 실제 돈을 사용하지 않고 가상의 자본으로 거래를 시뮬레이션하는 것입니다. 이를 통해:

- **위험 없는 전략 테스트**: 실제 돈 없이 트레이딩 전략 검증
- **실시간 시장 데이터**: Binance의 실제 시장 데이터 사용
- **완전한 거래 시뮬레이션**: 실제와 동일한 거래 환경
- **성능 분석**: 상세한 수익률 및 위험 분석

### 시스템 특징

- ✅ **Binance Testnet 통합**: 실제 거래소 환경과 동일
- ✅ **실시간 마켓 데이터**: WebSocket을 통한 실시간 데이터
- ✅ **다중 전략 실행**: 모멘텀, 평균회귀, 돌파 전략 등
- ✅ **리스크 관리**: 실제와 동일한 리스크 관리 시스템
- ✅ **성능 추적**: 실시간 수익률 및 거래 통계
- ✅ **포괄적 로깅**: 모든 거래 활동 기록

## 🛠️ 사전 준비사항

### 시스템 요구사항

- **Python**: 3.8 이상
- **운영체제**: Windows, macOS, Linux
- **메모리**: 최소 4GB RAM 권장
- **인터넷**: 안정적인 인터넷 연결 (WebSocket 통신용)

### 필요한 패키지

```bash
# 가상환경 활성화 (이미 설정되어 있음)
conda activate autotrading

# 필요한 패키지 설치 (이미 설치되어 있어야 함)
pip install aiohttp websockets pandas numpy
```

## 🏦 Binance Testnet 계정 설정

### 1단계: Testnet 웹사이트 접속

1. 브라우저에서 [Binance Testnet](https://testnet.binancefuture.com/) 접속
2. 우측 상단의 **"로그인"** 클릭

### 2단계: 계정 생성

1. **"회원가입"** 클릭
2. 이메일 주소 입력 (실제 이메일 주소 사용 권장)
3. 비밀번호 설정 (강력한 비밀번호 사용)
4. 이메일 인증 완료

### 3단계: 테스트 자금 확보

1. 로그인 후 우측 상단의 **"지갑"** 클릭
2. **"테스트넷 자금 받기"** 또는 **"Faucet"** 클릭
3. 가상 USDT 받기 (보통 10,000 USDT 제공)

> **참고**: Testnet의 모든 자금은 가상이며 실제 가치가 없습니다.

## 🔑 API 키 생성

### 1단계: API 관리 페이지 접속

1. Testnet에 로그인한 상태에서 우측 상단 프로필 아이콘 클릭
2. **"API Management"** 선택

### 2단계: 새 API 키 생성

1. **"Create API"** 버튼 클릭
2. API 키 이름 입력 (예: "Paper Trading Bot")
3. **"Create"** 클릭

### 3단계: API 키 정보 확인

생성 후 다음 정보를 안전한 곳에 저장하세요:

- **API Key**: 공개 키 (예: `abcd1234...`)
- **Secret Key**: 비밀 키 (⚠️ 절대 공유하지 마세요!)

### 4단계: API 권한 설정

1. 생성된 API 키 옆의 **"Edit"** 클릭
2. 다음 권한 활성화:
   - ✅ **"Enable Reading"** (읽기 권한)
   - ✅ **"Enable Futures"** (선물 거래 권한)
   - ❌ **"Enable Withdrawal"** (출금 권한 - 비활성화 권장)

## ⚙️ 환경 설정

### 1단계: 환경변수 파일 생성

```bash
# 템플릿 파일을 복사하여 실제 환경변수 파일 생성
cp .env.template .env
```

### 2단계: API 키 설정

`.env` 파일을 편집기로 열고 다음 값들을 수정하세요:

```bash
# Binance Testnet API 키 입력
BINANCE_TESTNET_API_KEY=your_actual_api_key_here
BINANCE_TESTNET_API_SECRET=your_actual_secret_key_here

# 환경 설정
ENVIRONMENT=paper_trading
LOG_LEVEL=INFO
TZ=Asia/Seoul
```

### 3단계: 설정 파일 확인

`config/trading.yaml` 파일이 존재하는지 확인하고, 필요시 수정:

```yaml
# 주요 설정 확인
trading:
  mode: "paper"  # 반드시 paper 모드

exchanges:
  binance:
    testnet: true  # 반드시 true
    paper_trading: true  # 반드시 true

paper_trading:
  initial_balance: 100000.0  # 시작 자금 ($100,000)
  commission_rate: 0.001     # 수수료 0.1%
```

## 🚀 Paper Trading 실행

### 기본 실행

```bash
# Paper trading 시스템 시작
python scripts/paper_trading.py
```

### 커스텀 설정으로 실행

```bash
# 특정 설정 파일 사용
python scripts/paper_trading.py config/my_custom_config.yaml
```

### 실행 시 나타나는 화면

```
📝 AutoTrading Paper Trading System
🏦 Using Binance Testnet - No real money at risk!
==================================================
🚀 Initializing Paper Trading System...
📋 Configuration loaded - Starting balance: $100,000.00
📝 Logging system configured
🔗 Connected to Binance Testnet
📊 Subscribed to market data for BTCUSDT
📊 Subscribed to market data for ETHUSDT
📊 Subscribed to market data for BNBUSDT
🧩 All components initialized
✅ Paper Trading System initialized successfully!
🎯 Starting paper trading session...
💰 Initial Balance: $100,000.00
📈 Monitoring market for trading opportunities...
```

### 거래 실행 예시

```
✅ BUY executed: 0.500000 BTCUSDT @ $43,250.00
💰 Balance: $78,375.00 | Position: 0.500000

✅ SELL executed: 0.250000 ETHUSDT @ $2,680.00
💰 Balance: $79,045.00 | Position: 0.750000
```

### 성능 리포트 예시

```
============================================================
📊 PAPER TRADING PERFORMANCE REPORT
============================================================
🕒 Session Duration: 2:34:15
💰 Initial Balance: $100,000.00
💰 Current Balance: $98,750.00
📈 Total Portfolio Value: $102,340.00
💵 Total PnL: $2,340.00 (2.34%)
📊 Trades Executed: 24

🎯 Current Positions:
   BTCUSDT: 0.750000
   ETHUSDT: 1.250000
   BNBUSDT: 5.000000
============================================================
```

## 📊 모니터링 및 분석

### 실시간 모니터링

시스템이 실행되는 동안 다음과 같은 정보를 실시간으로 확인할 수 있습니다:

1. **거래 실행**: 매매 신호 발생 시 즉시 표시
2. **잔고 변화**: 각 거래 후 잔고 및 포지션 업데이트
3. **성능 리포트**: 15분마다 자동 생성

### 로그 파일 확인

```bash
# 거래 로그 확인
tail -f logs/paper_trading.log

# 특정 날짜의 로그 확인
grep "2024-01-15" logs/paper_trading.log
```

### 데이터베이스 분석

시스템은 모든 거래 데이터를 SQLite 데이터베이스에 저장합니다:

```bash
# 데이터베이스 파일 위치
ls -la data/paper_trading.db

# SQL로 데이터 분석 (선택사항)
sqlite3 data/paper_trading.db
```

### 성능 지표

시스템이 추적하는 주요 성능 지표:

- **총 수익률**: (현재 가치 - 초기 자본) / 초기 자본
- **샤프 비율**: 위험 대비 수익률
- **최대 낙폭**: 고점 대비 최대 하락폭
- **승률**: 수익 거래 / 전체 거래
- **평균 거래 크기**: 평균 포지션 크기
- **거래 빈도**: 시간당 거래 횟수

## 🔧 문제 해결

### 자주 발생하는 문제들

#### 1. API 연결 실패

**증상**: `Failed to connect to Binance API` 오류

**해결 방법**:
```bash
# 1. API 키 확인
echo $BINANCE_TESTNET_API_KEY
echo $BINANCE_TESTNET_API_SECRET

# 2. 인터넷 연결 확인
ping testnet.binancefuture.com

# 3. API 키 권한 확인 (Testnet 웹사이트에서)
```

#### 2. 환경변수 로드 실패

**증상**: `Environment variable not found` 오류

**해결 방법**:
```bash
# 1. .env 파일 존재 확인
ls -la .env

# 2. .env 파일 내용 확인
cat .env

# 3. 환경변수 직접 설정
export BINANCE_TESTNET_API_KEY="your_key_here"
export BINANCE_TESTNET_API_SECRET="your_secret_here"
```

#### 3. WebSocket 연결 실패

**증상**: `WebSocket connection failed` 오류

**해결 방법**:
```bash
# 1. 방화벽 설정 확인
# 2. 프록시 설정 확인
# 3. 다른 네트워크에서 테스트
```

#### 4. 거래 신호가 생성되지 않음

**증상**: 시스템이 실행되지만 거래가 발생하지 않음

**해결 방법**:
```bash
# 1. 디버그 모드로 실행
LOG_LEVEL=DEBUG python scripts/paper_trading.py

# 2. 전략 설정 확인
grep -A 10 "strategies:" config/trading.yaml

# 3. 시장 데이터 수신 확인
grep "Market data" logs/paper_trading.log
```

### 로그 레벨 조정

문제 해결을 위해 로그 레벨을 조정할 수 있습니다:

```bash
# .env 파일에서
LOG_LEVEL=DEBUG  # 상세한 디버그 정보
LOG_LEVEL=INFO   # 일반 정보 (기본값)
LOG_LEVEL=WARNING # 경고만
LOG_LEVEL=ERROR  # 오류만
```

## ⚡ 고급 설정

### 전략 커스터마이징

`config/trading.yaml`에서 전략 파라미터를 조정할 수 있습니다:

```yaml
strategies:
  momentum:
    enabled: true
    allocation: 0.4  # 40% 할당
    timeframe: "5m"
    parameters:
      lookback_period: 20        # 과거 20개 봉 참조
      momentum_threshold: 0.02   # 2% 이상 모멘텀 시 거래

  mean_reversion:
    enabled: true
    allocation: 0.3  # 30% 할당
    parameters:
      deviation_threshold: 2.0   # 표준편차 2배 이상 시 거래
```

### 리스크 관리 설정

```yaml
risk_management:
  max_position_size: 0.1         # 최대 포지션 크기 10%
  max_daily_loss: 0.05          # 일일 최대 손실 5%
  default_stop_loss_pct: 0.02   # 기본 손절 2%
  default_take_profit_pct: 0.04 # 기본 익절 4%
```

### 성능 최적화

```yaml
paper_trading:
  # 실행 빈도 조정
  report_interval_minutes: 5     # 5분마다 리포트

  # 지연 시뮬레이션 조정
  min_latency_ms: 5             # 최소 지연 5ms
  max_latency_ms: 25            # 최대 지연 25ms

  # 슬리피지 조정
  max_slippage: 0.001           # 최대 슬리피지 0.1%
```

### 다중 거래쌍 설정

```yaml
trading:
  trading_pairs:
    - "BTC/USDT"   # 비트코인
    - "ETH/USDT"   # 이더리움
    - "BNB/USDT"   # 바이낸스 코인
    - "ADA/USDT"   # 카르다노
    - "SOL/USDT"   # 솔라나
    - "DOT/USDT"   # 폴카닷
    - "LINK/USDT"  # 체인링크
```

### 백테스팅 설정

```yaml
backtesting:
  enabled: true
  start_date: "2024-01-01"
  end_date: "2024-12-31"
  initial_capital: 100000.0
  benchmark: "BTC/USDT"
```

## 📈 실전 운용 가이드

### 권장 실행 순서

1. **단일 전략 테스트**: 먼저 하나의 전략만 활성화하여 테스트
2. **파라미터 최적화**: 백테스팅을 통해 최적 파라미터 찾기
3. **다중 전략 결합**: 검증된 전략들을 조합하여 운용
4. **실시간 모니터링**: 성능 지표를 지속적으로 모니터링
5. **정기적 검토**: 주기적으로 전략 성과 검토 및 조정

### 성공적인 Paper Trading을 위한 팁

1. **충분한 테스트 기간**: 최소 1-2주 이상 연속 운용
2. **다양한 시장 상황**: 상승장, 하락장, 횡보장에서 모두 테스트
3. **리스크 관리**: 항상 보수적인 리스크 관리 적용
4. **성과 분석**: 정기적으로 거래 내역과 성과 분석
5. **지속적 개선**: 결과를 바탕으로 전략 및 설정 개선

### 실전 투입 전 체크리스트

- [ ] 1개월 이상 안정적인 paper trading 운용
- [ ] 일관된 수익 창출 확인
- [ ] 최대 낙폭이 허용 범위 내 유지
- [ ] 모든 리스크 관리 기능 정상 작동 확인
- [ ] 시스템 안정성 검증 완료
- [ ] 실전 투입 자금 계획 수립

## ⚠️ 주의사항

### 보안 관련

- ✅ **API 키 보안**: API 키를 절대 다른 사람과 공유하지 마세요
- ✅ **권한 최소화**: 필요한 권한만 활성화하세요
- ✅ **정기 갱신**: API 키를 정기적으로 갱신하세요
- ✅ **로그 확인**: 의심스러운 활동이 없는지 정기적으로 확인하세요

### 기술적 제한사항

- **Testnet 제한**: 실제 시장과 유동성이 다를 수 있습니다
- **데이터 지연**: 실시간 데이터에도 약간의 지연이 있을 수 있습니다
- **슬리피지 차이**: 실제 거래에서는 더 큰 슬리피지가 발생할 수 있습니다

### 법적 고지

- 이 시스템은 교육 및 테스트 목적으로만 사용하세요
- 실제 투자 시에는 충분한 리스크 관리와 전문가 상담을 받으세요
- 모든 투자 결정에 대한 책임은 사용자에게 있습니다

## 📚 추가 자료

### 참고 문서

- [Binance API 문서](https://binance-docs.github.io/apidocs/futures/en/)
- [Binance Testnet 가이드](https://testnet.binancefuture.com/en/futures/BTCUSDT)
- [시스템 아키텍처](docs/project-system-architecture.md)
- [전략 엔진 가이드](src/strategy_engine/CLAUDE.md)

### 커뮤니티 및 지원

- 📧 **이슈 리포트**: GitHub Issues 사용
- 💬 **질문 및 토론**: GitHub Discussions
- 📖 **문서 개선**: Pull Request 환영

---

**Happy Paper Trading! 🎯**

이 가이드를 따라 안전하고 효과적인 paper trading 환경을 구축하여 트레이딩 실력을 향상시키세요. 실제 돈 없이도 전문적인 트레이딩 경험을 쌓을 수 있습니다!