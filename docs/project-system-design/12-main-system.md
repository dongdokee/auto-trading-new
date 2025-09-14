# 코인 선물 자동매매 시스템 - 메인 시스템

## 메인 트레이딩 봇

```python
class TradingBot:
    """메인 트레이딩 봇"""
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: 설정 파일 경로
        """
        self.config = self._load_config(config_path)
        self.is_running = False
        
        # 핵심 컴포넌트 초기화
        self.risk_controller = RiskController(
            self.config['initial_capital']
        )
        self.kelly_optimizer = ContinuousKellyOptimizer(
            fractional=self.config.get('kelly_fraction', 0.25)
        )
        self.position_sizer = PositionSizer(
            self.risk_controller,
            self.kelly_optimizer
        )
        self.order_router = SmartOrderRouter()
        self.regime_detector = NoLookAheadRegimeDetector()
        self.strategy_matrix = StrategyMatrix()
        self.dashboard = TradingDashboard(self.risk_controller)
        
        # 포트폴리오 상태
        self.portfolio = {
            'equity': self.config['initial_capital'],
            'initial_equity': self.config['initial_capital'],
            'positions': {},
            'returns': [],
            'high_water_mark': self.config['initial_capital']
        }
        
    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    async def start(self):
        """봇 시작"""
        print("Starting Trading Bot...")
        
        # 사전 체크
        checklist = PreOperationChecklist()
        if not checklist.run_all_checks():
            print("Pre-operation checks failed. Aborting.")
            return
        
        self.is_running = True
        
        # 메인 루프
        try:
            await self._main_loop()
        except KeyboardInterrupt:
            print("Shutting down gracefully...")
            await self.shutdown()
        except Exception as e:
            print(f"Fatal error: {e}")
            await self.emergency_shutdown()
    
    async def _main_loop(self):
        """메인 트레이딩 루프"""
        
        while self.is_running:
            try:
                # 시장 데이터 수집
                market_data = await self._fetch_market_data()
                
                # 레짐 감지
                regime_info = self.regime_detector.detect_regime(
                    market_data,
                    len(market_data) - 1
                )
                
                # 전략 가중치 결정
                strategy_weights = self.strategy_matrix.get_strategy_weights(
                    regime_info
                )
                
                # 신호 생성
                signals = await self._generate_signals(
                    market_data,
                    regime_info,
                    strategy_weights
                )
                
                # 리스크 체크
                violations = self.risk_controller.check_all_limits(
                    self.portfolio
                )
                
                if violations:
                    await self._handle_risk_violations(violations)
                    continue
                
                # 신호 실행
                for signal in signals:
                    # 포지션 사이징
                    position_size = self.position_sizer.calculate_position_size(
                        signal,
                        {'price': market_data['close'].iloc[-1], 
                         'regime': regime_info['regime']},
                        self.portfolio
                    )
                    
                    if position_size > 0:
                        # 주문 생성
                        order = Order(
                            symbol=signal['symbol'],
                            side=signal['side'],
                            size=position_size,
                            urgency=signal.get('urgency', 'MEDIUM')
                        )
                        
                        # 주문 실행
                        execution = await self.order_router.route_order(order)
                        
                        # 포트폴리오 업데이트
                        self._update_portfolio(execution)
                
                # 대시보드 업데이트
                self.dashboard.update_metrics(
                    self.portfolio,
                    {'volatility': regime_info['volatility_forecast']}
                )
                
                # 알림 체크
                alerts = self.dashboard.get_alerts()
                for alert in alerts:
                    await self._send_alert(alert)
                
                # 대기
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"Error in main loop: {e}")
                await asyncio.sleep(5)
    
    async def _fetch_market_data(self) -> pd.DataFrame:
        """시장 데이터 수집"""
        # TODO: 실제 구현 필요
        return pd.DataFrame()
    
    async def _generate_signals(self, market_data: pd.DataFrame, 
                               regime_info: Dict, 
                               strategy_weights: Dict) -> List[Dict]:
        """전략 신호 생성"""
        # TODO: 실제 구현 필요
        return []
    
    async def _handle_risk_violations(self, violations: List[Tuple]):
        """리스크 위반 처리"""
        for violation_type, value in violations:
            if violation_type == 'LIQUIDATION_RISK':
                # 긴급 포지션 축소
                await self._reduce_positions(0.5)
            elif violation_type == 'DRAWDOWN':
                # 신규 거래 중단
                self.is_running = False
    
    async def _reduce_positions(self, reduction_factor: float):
        """포지션 축소"""
        # TODO: 실제 구현 필요
        pass
    
    def _update_portfolio(self, execution: Dict):
        """포트폴리오 업데이트"""
        # TODO: 실제 구현 필요
        pass
    
    async def _send_alert(self, alert: Dict):
        """알림 발송"""
        print(f"[{alert['level']}] {alert['message']}")
    
    async def shutdown(self):
        """정상 종료"""
        self.is_running = False
        # 모든 포지션 정리
        # TODO: 실제 구현 필요
    
    async def emergency_shutdown(self):
        """긴급 종료"""
        self.is_running = False
        # 즉시 모든 포지션 청산
        # TODO: 실제 구현 필요

## Entry Point

async def main():
    """메인 진입점"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Crypto Futures Trading Bot')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--mode', choices=['paper', 'staging', 'production'], 
                       default='paper', help='Trading mode')
    
    args = parser.parse_args()
    
    # 봇 초기화 및 실행
    bot = TradingBot(args.config)
    await bot.start()

if __name__ == "__main__":
    asyncio.run(main())

## Configuration File (config.yaml)

# Trading Bot Configuration
trading:
  initial_capital: 230  # USDT
  kelly_fraction: 0.25
  universe:
    - BTCUSDT
    - ETHUSDT
    - BNBUSDT
    - SOLUSDT
    - XRPUSDT
  
risk:
  max_leverage: 10
  max_drawdown: 0.12
  var_limit: 0.02
  liquidation_prob_limit: 0.005
  
execution:
  fee_maker: 0.0002
  fee_taker: 0.0004
  slippage_estimate: 0.0005
  
monitoring:
  dashboard_port: 8080
  prometheus_port: 9090
  alert_webhook: "https://hooks.slack.com/services/..."
  
database:
  host: "localhost"
  port: 5432
  name: "trading_bot"
  user: "trader"
  
api:
  exchange: "binance"
  api_key: "${BINANCE_API_KEY}"
  api_secret: "${BINANCE_API_SECRET}"
  testnet: false
```

## 전체 시스템 통합

```python
class SystemIntegration:
    """전체 시스템 통합 관리"""
    
    def __init__(self):
        self.components = {}
        self.health_status = {}
        
    async def initialize_all_components(self, config: Dict):
        """모든 컴포넌트 초기화"""
        
        # 데이터 레이어
        self.components['data_feed'] = await self._init_data_feed(config)
        self.components['data_quality'] = DataQualityManager()
        
        # 전략 레이어
        self.components['regime_detector'] = NoLookAheadRegimeDetector()
        self.components['strategy_engine'] = await self._init_strategies(config)
        
        # 리스크 레이어
        self.components['risk_controller'] = RiskController(config['initial_capital'])
        self.components['position_sizer'] = PositionSizer(
            self.components['risk_controller'],
            ContinuousKellyOptimizer()
        )
        
        # 실행 레이어
        self.components['order_router'] = SmartOrderRouter()
        self.components['execution_monitor'] = SlippageController()
        
        # 모니터링 레이어
        self.components['dashboard'] = TradingDashboard(
            self.components['risk_controller']
        )
        self.components['alert_system'] = AlertSystem()
        
        return self.components
    
    async def _init_data_feed(self, config: Dict):
        """데이터 피드 초기화"""
        client = BinanceWebSocketClient(config['trading']['universe'])
        await client.connect()
        return client
    
    async def _init_strategies(self, config: Dict):
        """전략 엔진 초기화"""
        strategies = {
            'trend_following': TrendFollowingStrategy(),
            'mean_reversion': MeanReversionStrategy(),
            'range_trading': RangeTradingStrategy(),
            'funding_arb': FundingArbitrage()
        }
        return strategies
    
    async def health_check(self) -> Dict:
        """시스템 상태 체크"""
        
        for name, component in self.components.items():
            try:
                if hasattr(component, 'health_check'):
                    status = await component.health_check()
                else:
                    status = 'OK'
                
                self.health_status[name] = {
                    'status': status,
                    'timestamp': pd.Timestamp.now()
                }
            except Exception as e:
                self.health_status[name] = {
                    'status': 'ERROR',
                    'error': str(e),
                    'timestamp': pd.Timestamp.now()
                }
        
        return self.health_status
    
    async def graceful_shutdown(self):
        """안전한 종료"""
        
        print("Initiating graceful shutdown...")
        
        # 1. 신규 주문 중단
        if 'order_router' in self.components:
            self.components['order_router'].halt_new_orders = True
        
        # 2. 대기 중인 주문 취소
        # TODO: 구현
        
        # 3. 포지션 정리
        # TODO: 구현
        
        # 4. 데이터 피드 종료
        if 'data_feed' in self.components:
            await self.components['data_feed'].disconnect()
        
        # 5. 최종 상태 저장
        # TODO: 구현
        
        print("Shutdown complete")

## 시스템 시작 스크립트

async def start_system():
    """시스템 시작"""
    
    # 설정 로드
    config = load_config('config.yaml')
    
    # 로깅 설정
    setup_logging()
    
    # 시스템 초기화
    system = SystemIntegration()
    components = await system.initialize_all_components(config)
    
    # 트레이딩 봇 생성
    bot = TradingBot(config)
    bot.components = components
    
    # 시스템 시작
    try:
        await bot.start()
    except Exception as e:
        logging.error(f"System error: {e}")
        await system.graceful_shutdown()
        raise

if __name__ == "__main__":
    asyncio.run(start_system())
```