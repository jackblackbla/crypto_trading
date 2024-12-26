import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from .base_strategy import BaseStrategy

class SwingStrategy(BaseStrategy):
    """스윙 트레이딩 전략 (EMA 크로스오버 기반)"""
    
    def __init__(self, params: Dict[str, Any]):
        name = "swing_ema_crossover"
        super().__init__(name, params)
        
        # 기본 파라미터
        self.fast_ema = params.get('fast_ema', 50)
        self.slow_ema = params.get('slow_ema', 200)
        self.rsi_period = params.get('rsi_period', 14)
        self.rsi_threshold = params.get('rsi_threshold', 30)  # RSI 매수 임계값
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """스윙 트레이딩 시그널 생성"""
        if not self.validate_data(data):
            return pd.DataFrame()
        
        df = data.copy()
        
        # EMA 계산
        df[f'ema_{self.fast_ema}'] = df['close'].ewm(
            span=self.fast_ema, adjust=False
        ).mean()
        df[f'ema_{self.slow_ema}'] = df['close'].ewm(
            span=self.slow_ema, adjust=False
        ).mean()
        
        # RSI 계산
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 시그널 생성
        df['trend'] = 0  # 0: 중립, 1: 상승, -1: 하락
        df.loc[df[f'ema_{self.fast_ema}'] > df[f'ema_{self.slow_ema}'], 'trend'] = 1
        df.loc[df[f'ema_{self.fast_ema}'] < df[f'ema_{self.slow_ema}'], 'trend'] = -1
        
        # 매수/매도 시그널
        df['signal'] = 'HOLD'
        
        # 매수 조건:
        # 1) 빠른 EMA가 느린 EMA를 상향 돌파
        # 2) RSI가 과매도 구간(30 이하)에서 반등
        buy_condition = (
            (df['trend'] == 1) &  # 상승 추세
            (df['trend'].shift(1) == -1) &  # 추세 전환
            (df['rsi'] < self.rsi_threshold)  # RSI 과매도
        )
        df.loc[buy_condition, 'signal'] = 'BUY'
        
        # 매도 조건: 하락 추세 전환
        sell_condition = (
            (df['trend'] == -1) &
            (df['trend'].shift(1) == 1)
        )
        df.loc[sell_condition, 'signal'] = 'SELL'
        
        # 시그널 점수 (-100 ~ +100)
        df['score'] = 0
        
        # 상승 추세에서 점수 상향
        df.loc[df['trend'] == 1, 'score'] += 50
        
        # RSI 기반 추가 점수
        df.loc[df['rsi'] < 30, 'score'] += 30  # 과매도 구간
        df.loc[df['rsi'] > 70, 'score'] -= 30  # 과매수 구간
        
        # 시그널 발생 지점에서 추가 점수
        df.loc[df['signal'] == 'BUY', 'score'] += 20
        df.loc[df['signal'] == 'SELL', 'score'] -= 20
        
        self.logger.info(f"시그널 생성 완료: {len(df[(df['signal'] != 'HOLD')])}개 시그널")
        return df