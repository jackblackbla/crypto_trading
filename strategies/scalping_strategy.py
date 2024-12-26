import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Any, Tuple
from .base_strategy import BaseStrategy

class ScalpingStrategy(BaseStrategy):
    """단타 트레이딩 전략 (볼린저 밴드, ATR, Keltner Channels, 거래량, 캔들 패턴 기반)"""
    
    def __init__(self, params: Dict[str, Any]):
        name = "scalping_advanced"
        super().__init__(name, params)
        
        # 기본 파라미터
        self.bb_period = params.get('bb_period', 20)
        self.bb_std = params.get('bb_std', 2.0)
        self.volume_ma_period = params.get('volume_ma_period', 20)
        self.volume_mult = params.get('volume_mult', 2.0)  # 거래량 급증 기준
        self.atr_period = params.get('atr_period', 14)
        self.kc_period = params.get('kc_period', 20)
        self.kc_mult = params.get('kc_mult', 2)
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """단타 매매 시그널 생성"""
        if not self.validate_data(data):
            return pd.DataFrame()
        
        df = data.copy()
        
        # 볼린저 밴드 계산
        df.ta.bbands(length=self.bb_period, std=self.bb_std, append=True)
        
        # ATR 계산
        df.ta.atr(length=self.atr_period, append=True)
        print(df.columns)

        # Keltner Channels 계산
        df.ta.kc(length=self.kc_period, multiplier=self.kc_mult, append=True)
        print(df.columns)
        
        # 거래량 분석
        df['volume_ma'] = df['volume'].rolling(window=self.volume_ma_period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # 캔들 패턴 분석 (망치형, 장악형)
        df['hammer'] = ((df['open'] - df['low']) > (df['high'] - df['open']) * 2) & (df['close'] > df['open'])
        df['engulfing'] = ((df['open'].shift(1) < df['close'].shift(1)) & (df['open'] > df['close']) & (df['open'] > df['close'].shift(1)) & (df['close'] < df['open'].shift(1)))
        
        # 시그널 생성
        df['signal'] = 'HOLD'
        
        # 매수 조건:
        long_condition = (
            (df['close'] > df[f'BBU_{self.bb_period}_{self.bb_std}']) &
            (df[f'ATRr_{self.atr_period}'] > df[f'ATRr_{self.atr_period}'].shift(1)) &
            (df['volume_ratio'] > self.volume_mult) &
            (df['hammer'])
        )
        df.loc[long_condition, 'signal'] = 'BUY'
        
        # 매도 조건:
        # 1) 가격이 볼린저 밴드 하단 돌파
        # 2) ATR 증가 (변동성 증가)
        # 3) 거래량 급증
        # 4) 하락 장악형 캔들 패턴
        bearish_engulfing = ((df['open'].shift(1) > df['close'].shift(1)) & (df['open'] < df['close']) & (df['open'] < df['close'].shift(1)) & (df['close'] > df['open'].shift(1)))
        short_condition = (
            (df['close'] < df[f'BBL_{self.bb_period}_{self.bb_std}']) &
            (df[f'ATRr_{self.atr_period}'] > df[f'ATRr_{self.atr_period}'].shift(1)) &
            (df['volume_ratio'] > self.volume_mult) &
            (bearish_engulfing)
        )
        df.loc[short_condition, 'signal'] = 'SELL'
        
        # 시그널 점수 (-100 ~ +100)
        df['score'] = 0
        
        # 볼린저 밴드 기반 점수
        df.loc[df['close'] > df[f'BBU_{self.bb_period}_{self.bb_std}'], 'score'] += 30
        df.loc[df['close'] < df[f'BBL_{self.bb_period}_{self.bb_std}'], 'score'] -= 30
        
        # ATR 기반 점수
        df.loc[df[f'ATRr_{self.atr_period}'] > df[f'ATRr_{self.atr_period}'].shift(1), 'score'] += 15
        
        # Keltner Channels 기반 점수
        df.loc[df['close'] > df[f'KCUe_{self.kc_period}_{int(self.kc_mult)}'], 'score'] += 20
        df.loc[df['close'] < df[f'KCLe_{self.kc_period}_{int(self.kc_mult)}'], 'score'] -= 20
        
        # 거래량 기반 추가 점수
        volume_score = (df['volume_ratio'] - 1) * 10  # 거래량 비율에 따라 가중치
        volume_score = volume_score.clip(-20, 20)  # -20 ~ +20 범위로 제한
        df['score'] += volume_score
        
        # 캔들 패턴 기반 점수
        df.loc[df['hammer'], 'score'] += 10
        df.loc[df['engulfing'], 'score'] += 10
        df.loc[bearish_engulfing, 'score'] -= 10
        
        self.logger.info(f"고급 단타 시그널 생성 완료: {len(df[(df['signal'] != 'HOLD')])}개 시그널")
        return df