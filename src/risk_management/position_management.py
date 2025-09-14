# src/risk_management/position_management.py
"""
포지션 생명주기 관리 시스템
포지션 오픈/업데이트/클로즈 및 스톱로스/테이크프로핏 관리
"""
from typing import Dict, Optional
from datetime import datetime
from .risk_management import RiskController
from src.utils.logger import TradingLogger, get_trading_logger


class PositionManager:
    """포지션 생명주기 관리"""

    def __init__(self, risk_controller: RiskController, logger: Optional[TradingLogger] = None):
        self.risk_controller = risk_controller
        self.positions = {}
        self.logger = logger or get_trading_logger("position_manager", log_to_file=False)

        # 포지션 매니저 초기화 로그
        self.logger.info(
            "Position manager initialized",
            component="PositionManager",
            initial_positions_count=len(self.positions)
        )

    def open_position(self, symbol: str, side: str, size: float,
                     price: float, leverage: float) -> Dict:
        """포지션 오픈"""

        position = {
            'symbol': symbol,
            'side': side,
            'size': size,
            'entry_price': price,
            'current_price': price,
            'leverage': leverage,
            'margin': (size * price) / leverage,
            'liquidation_price': self._calculate_liquidation_price(
                side, price, leverage
            ),
            'unrealized_pnl': 0.0,
            'realized_pnl': 0.0,
            'open_time': datetime.now(),
            'trailing_stop': None,
            'take_profit': None,
            'stop_loss': None
        }

        self.positions[symbol] = position

        # 포지션 오픈 로깅
        self.logger.log_trade(
            "Position opened",
            symbol=symbol,
            side=side,
            size=size,
            entry_price=price,
            leverage=leverage,
            margin=position['margin'],
            liquidation_price=position['liquidation_price'],
            notional_usdt=size * price,
            liquidation_distance_pct=abs(position['liquidation_price'] - price) / price * 100,
            open_time=position['open_time'].isoformat()
        )

        return position

    def update_position(self, symbol: str, current_price: float) -> Optional[Dict]:
        """포지션 업데이트"""

        if symbol not in self.positions:
            return None

        position = self.positions[symbol]
        position['current_price'] = current_price

        # PnL 계산
        if position['side'] == 'LONG':
            position['unrealized_pnl'] = (
                (current_price - position['entry_price']) * position['size']
            )
        else:  # SHORT
            position['unrealized_pnl'] = (
                (position['entry_price'] - current_price) * position['size']
            )

        # Trailing stop 업데이트
        if position.get('trailing_distance'):
            self._update_trailing_stop(position, current_price)

        return position

    def close_position(self, symbol: str, price: float,
                      reason: str = 'MANUAL') -> Optional[Dict]:
        """포지션 종료"""

        if symbol not in self.positions:
            return None

        position = self.positions[symbol]

        # 최종 PnL 계산
        if position['side'] == 'LONG':
            final_pnl = (price - position['entry_price']) * position['size']
        else:  # SHORT
            final_pnl = (position['entry_price'] - price) * position['size']

        position['realized_pnl'] = final_pnl
        position['close_price'] = price
        position['close_time'] = datetime.now()
        position['close_reason'] = reason

        # 포지션 클로즈 로깅
        holding_duration = (position['close_time'] - position['open_time']).total_seconds()

        self.logger.log_trade(
            "Position closed",
            symbol=symbol,
            side=position['side'],
            size=position['size'],
            entry_price=position['entry_price'],
            close_price=price,
            leverage=position['leverage'],
            realized_pnl=final_pnl,
            return_pct=(final_pnl / (position['size'] * position['entry_price']) * 100),
            holding_duration_seconds=holding_duration,
            holding_duration_minutes=holding_duration / 60,
            close_reason=reason,
            open_time=position['open_time'].isoformat(),
            close_time=position['close_time'].isoformat()
        )

        # 포지션 제거
        del self.positions[symbol]

        return position

    def check_stop_conditions(self, symbol: str,
                            current_price: float) -> Optional[str]:
        """스톱 조건 체크"""

        if symbol not in self.positions:
            return None

        position = self.positions[symbol]

        # Stop loss 체크
        if position.get('stop_loss'):
            if position['side'] == 'LONG' and current_price <= position['stop_loss']:
                return 'STOP_LOSS'
            elif position['side'] == 'SHORT' and current_price >= position['stop_loss']:
                return 'STOP_LOSS'

        # Take profit 체크
        if position.get('take_profit'):
            if position['side'] == 'LONG' and current_price >= position['take_profit']:
                return 'TAKE_PROFIT'
            elif position['side'] == 'SHORT' and current_price <= position['take_profit']:
                return 'TAKE_PROFIT'

        # Trailing stop 체크
        if position.get('trailing_stop'):
            if position['side'] == 'LONG' and current_price <= position['trailing_stop']:
                return 'TRAILING_STOP'
            elif position['side'] == 'SHORT' and current_price >= position['trailing_stop']:
                return 'TRAILING_STOP'

        return None

    def _calculate_liquidation_price(self, side: str,
                                    entry_price: float,
                                    leverage: float) -> float:
        """청산가 계산"""

        # 간소화된 버전 (실제로는 더 복잡)
        mmr = 0.005  # 0.5% maintenance margin

        if side == 'LONG':
            liquidation_price = entry_price * (1 - 1/leverage + mmr)
        else:  # SHORT
            liquidation_price = entry_price * (1 + 1/leverage - mmr)

        return liquidation_price

    def _update_trailing_stop(self, position: Dict, current_price: float):
        """Trailing stop 업데이트"""

        trailing_distance = position.get('trailing_distance', 0.02)  # 2% 기본값

        if position['side'] == 'LONG':
            new_stop = current_price * (1 - trailing_distance)
            if position['trailing_stop'] is None or new_stop > position['trailing_stop']:
                position['trailing_stop'] = new_stop
        else:  # SHORT
            new_stop = current_price * (1 + trailing_distance)
            if position['trailing_stop'] is None or new_stop < position['trailing_stop']:
                position['trailing_stop'] = new_stop