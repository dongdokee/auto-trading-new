# 코인 선물 자동매매 시스템 - 인프라 아키텍처

## 10.1 시스템 구성

```yaml
architecture:
  components:
    - trading_engine:
        type: "core"
        language: "Python 3.10+"
        framework: "asyncio"
        dependencies:
          - numpy==1.26.4
          - pandas==2.2.2
          - scipy==1.11.4
          - scikit-learn==1.3.2
          - arch==6.3.0  # GARCH
          - hmmlearn==0.3.2  # HMM
        
    - risk_manager:
        type: "core"
        priority: "highest"
        update_frequency: "1s"
        fail_safe: "halt_trading"
        
    - data_feed:
        type: "infrastructure"
        sources: 
          - websocket: "primary"
          - rest_api: "fallback"
        redundancy: "active-standby"
        data_validation: true
        
    - database:
        type: "persistence"
        engine: "PostgreSQL 14+"
        backup: "continuous"
        retention: "2 years"
        
    - monitoring:
        type: "observability"
        tools: 
          - prometheus: "metrics"
          - grafana: "visualization"
          - alertmanager: "alerts"
        
    - backtester:
        type: "research"
        features:
          - walk_forward: true
          - monte_carlo: true
          - no_lookahead: true
```

## 10.2 배포 설정

```yaml
deployment:
  environments:
    paper:
      api: "testnet.binance.vision"
      capital: 100  # USDT (paper money)
      max_leverage: 2
      strategies: ["test"]
      
    staging:
      api: "api.binance.com"
      capital: 230  # USDT
      max_leverage: 3
      strategies: ["conservative"]
      
    production:
      api: "api.binance.com"
      capital: 7700  # USDT
      max_leverage: 10
      strategies: ["all"]
      high_availability: true
      
  health_checks:
    - endpoint: "/health"
      interval: "10s"
      timeout: "5s"
      
  rollback:
    automatic: true
    conditions:
      - "error_rate > 0.05"
      - "latency_p99 > 1000ms"
      - "drawdown > 0.05"
      
  kill_switch:
    triggers:
      - "drawdown > 0.10"
      - "var_breach_count > 3"
      - "api_error_rate > 0.10"
    actions:
      - "halt_new_orders"
      - "cancel_pending_orders"
      - "liquidate_positions"
      - "send_alerts"
```

## 10.3 데이터베이스 스키마

```sql
-- 포지션 테이블
CREATE TABLE positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    size DECIMAL(20, 8) NOT NULL,
    entry_price DECIMAL(20, 8) NOT NULL,
    current_price DECIMAL(20, 8),
    leverage DECIMAL(5, 2),
    margin DECIMAL(20, 8),
    liquidation_price DECIMAL(20, 8),
    unrealized_pnl DECIMAL(20, 8),
    realized_pnl DECIMAL(20, 8),
    open_time TIMESTAMP NOT NULL,
    close_time TIMESTAMP,
    status VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 거래 이력
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    order_id VARCHAR(50) UNIQUE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    order_type VARCHAR(20) NOT NULL,
    size DECIMAL(20, 8) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    notional DECIMAL(20, 8) NOT NULL,
    commission DECIMAL(20, 8),
    slippage DECIMAL(10, 6),
    strategy VARCHAR(50),
    signal_strength DECIMAL(5, 4),
    execution_time TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 성과 메트릭
CREATE TABLE performance_metrics (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    equity DECIMAL(20, 8) NOT NULL,
    daily_return DECIMAL(10, 6),
    sharpe_ratio DECIMAL(10, 4),
    sortino_ratio DECIMAL(10, 4),
    max_drawdown DECIMAL(10, 6),
    var_daily DECIMAL(20, 8),
    trade_count INTEGER,
    win_rate DECIMAL(5, 4),
    total_commission DECIMAL(20, 8),
    funding_pnl DECIMAL(20, 8),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 시장 데이터
CREATE TABLE market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open DECIMAL(20, 8),
    high DECIMAL(20, 8),
    low DECIMAL(20, 8),
    close DECIMAL(20, 8),
    volume DECIMAL(20, 8),
    funding_rate DECIMAL(10, 8),
    open_interest DECIMAL(20, 8),
    UNIQUE(symbol, timestamp)
);

-- 인덱스
CREATE INDEX idx_positions_symbol ON positions(symbol);
CREATE INDEX idx_positions_status ON positions(status);
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_execution_time ON trades(execution_time);
CREATE INDEX idx_performance_date ON performance_metrics(date);
CREATE INDEX idx_market_data_symbol_timestamp ON market_data(symbol, timestamp);
```

## 10.4 Docker 설정

```dockerfile
# Dockerfile
FROM python:3.10-slim

# 시스템 패키지
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드
COPY . .

# 환경 변수
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC

# 실행
CMD ["python", "main.py"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  trading_bot:
    build: .
    container_name: crypto_trading_bot
    restart: unless-stopped
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - BINANCE_API_KEY=${BINANCE_API_KEY}
      - BINANCE_API_SECRET=${BINANCE_API_SECRET}
      - ENVIRONMENT=${ENVIRONMENT:-production}
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
    depends_on:
      - postgres
      - redis
    networks:
      - trading_network

  postgres:
    image: postgres:14
    container_name: trading_postgres
    restart: unless-stopped
    environment:
      - POSTGRES_DB=trading_bot
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - trading_network

  redis:
    image: redis:7
    container_name: trading_redis
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    networks:
      - trading_network

  prometheus:
    image: prom/prometheus
    container_name: trading_prometheus
    restart: unless-stopped
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    networks:
      - trading_network

  grafana:
    image: grafana/grafana
    container_name: trading_grafana
    restart: unless-stopped
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    ports:
      - "3000:3000"
    networks:
      - trading_network

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  trading_network:
    driver: bridge
```

## 10.5 API 연동

```python
import asyncio
import aiohttp
from typing import Dict, List, Optional
import hmac
import hashlib
import time

class BinanceAPIClient:
    """Binance Futures API 클라이언트"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        
        if testnet:
            self.base_url = "https://testnet.binancefuture.com"
            self.ws_url = "wss://stream.binancefuture.com"
        else:
            self.base_url = "https://fapi.binance.com"
            self.ws_url = "wss://fstream.binance.com"
        
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _sign_request(self, params: Dict) -> Dict:
        """요청 서명"""
        params['timestamp'] = int(time.time() * 1000)
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        params['signature'] = signature
        return params
    
    async def get_account(self) -> Dict:
        """계정 정보 조회"""
        endpoint = "/fapi/v2/account"
        params = self._sign_request({})
        
        headers = {'X-MBX-APIKEY': self.api_key}
        
        async with self.session.get(
            f"{self.base_url}{endpoint}",
            params=params,
            headers=headers
        ) as response:
            return await response.json()
    
    async def place_order(self, symbol: str, side: str, 
                          order_type: str, **kwargs) -> Dict:
        """주문 실행"""
        endpoint = "/fapi/v1/order"
        
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            **kwargs
        }
        
        params = self._sign_request(params)
        headers = {'X-MBX-APIKEY': self.api_key}
        
        async with self.session.post(
            f"{self.base_url}{endpoint}",
            params=params,
            headers=headers
        ) as response:
            return await response.json()
    
    async def cancel_order(self, symbol: str, order_id: str) -> Dict:
        """주문 취소"""
        endpoint = "/fapi/v1/order"
        
        params = {
            'symbol': symbol,
            'orderId': order_id
        }
        
        params = self._sign_request(params)
        headers = {'X-MBX-APIKEY': self.api_key}
        
        async with self.session.delete(
            f"{self.base_url}{endpoint}",
            params=params,
            headers=headers
        ) as response:
            return await response.json()
    
    async def get_positions(self) -> List[Dict]:
        """포지션 조회"""
        endpoint = "/fapi/v2/positionRisk"
        params = self._sign_request({})
        headers = {'X-MBX-APIKEY': self.api_key}
        
        async with self.session.get(
            f"{self.base_url}{endpoint}",
            params=params,
            headers=headers
        ) as response:
            positions = await response.json()
            # 활성 포지션만 필터링
            return [p for p in positions if float(p['positionAmt']) != 0]

## 10.6 WebSocket 데이터 피드

class BinanceWebSocketClient:
    """실시간 데이터 피드"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.ws = None
        self.callbacks = {}
        
    async def connect(self):
        """WebSocket 연결"""
        streams = []
        for symbol in self.symbols:
            symbol_lower = symbol.lower()
            streams.extend([
                f"{symbol_lower}@trade",
                f"{symbol_lower}@depth20@100ms",
                f"{symbol_lower}@markPrice@1s"
            ])
        
        stream_url = f"wss://fstream.binance.com/stream?streams={'/'.join(streams)}"
        
        self.ws = await aiohttp.ClientSession().ws_connect(stream_url)
        
        # 메시지 처리 루프
        asyncio.create_task(self._message_loop())
    
    async def _message_loop(self):
        """메시지 처리 루프"""
        async for msg in self.ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = msg.json()
                stream = data.get('stream', '')
                
                if '@trade' in stream:
                    await self._handle_trade(data['data'])
                elif '@depth' in stream:
                    await self._handle_depth(data['data'])
                elif '@markPrice' in stream:
                    await self._handle_mark_price(data['data'])
    
    async def _handle_trade(self, data: Dict):
        """체결 데이터 처리"""
        if 'trade' in self.callbacks:
            await self.callbacks['trade'](data)
    
    async def _handle_depth(self, data: Dict):
        """호가 데이터 처리"""
        if 'depth' in self.callbacks:
            await self.callbacks['depth'](data)
    
    async def _handle_mark_price(self, data: Dict):
        """마크 가격 처리"""
        if 'mark_price' in self.callbacks:
            await self.callbacks['mark_price'](data)
    
    def register_callback(self, event_type: str, callback):
        """콜백 등록"""
        self.callbacks[event_type] = callback
    
    async def disconnect(self):
        """연결 종료"""
        if self.ws:
            await self.ws.close()

## 10.7 로깅 설정

```python
import logging
import logging.handlers
from pathlib import Path

def setup_logging(log_dir: str = "logs"):
    """로깅 설정"""
    
    Path(log_dir).mkdir(exist_ok=True)
    
    # 포맷터
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # 파일 핸들러 (일별 로테이션)
    file_handler = logging.handlers.TimedRotatingFileHandler(
        f"{log_dir}/trading_bot.log",
        when='midnight',
        interval=1,
        backupCount=30
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # 에러 파일 핸들러
    error_handler = logging.handlers.RotatingFileHandler(
        f"{log_dir}/errors.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)
    
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    
    # 특정 로거 설정
    trading_logger = logging.getLogger('trading')
    risk_logger = logging.getLogger('risk')
    execution_logger = logging.getLogger('execution')
    
    return root_logger
```