import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from .base_strategy import BaseStrategy

class ScalpingStrategy(BaseStrategy):
    """단타 트레이딩 전략 (볼린저 밴드 + 거래량 기반)"""
    
    def __init__(self, params: Dict[str, Any]):
        name = "scalping_bollinger"
        super().__init__(name, params)
        
        # 기본 파라미터
        self.bb_period = params.get('bb_period', 20)
        self.bb_std = params.get('bb_std', 2.0)
        self.volume_ma_period = params.get('volume_ma_period', 20)
        self.volume_mult = params.get('volume_mult', 2.0)  # 거래량 급증 기준
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """단타 매매 시그널 생성"""
        if not self.validate_data(data):
            return pd.DataFrame()
        
        df = data.copy()
        
        # 볼린저 밴드 계산
        df['bb_middle'] = df['close'].rolling(window=self.bb_period).mean()
        bb_std = df['close'].rolling(window=self.bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * self.bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std * self.bb_std)
        
        # 밴드폭 계산
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_width_ma'] = df['bb_width'].rolling(window=self.bb_period).mean()
        
        # 거래량 분석
        df['volume_ma'] = df['volume'].rolling(window=self.volume_ma_period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # 시그널 생성
        df['signal'] = 'HOLD'
        
        # 매수 조건:
        # 1) 가격이 상단 밴드 돌파
        # 2) 밴드폭 확장 (변동성 증가)
        # 3) 거래량 급증
        long_condition = (
            (df['close'] > df['bb_upper']) &  # 상단 돌파
            (df['bb_width'] > df['bb_width_ma']) &  # 밴드폭 확장
            (df['volume_ratio'] > self.volume_mult)  # 거래량 급증
        )
        df.loc[long_condition, 'signal'] = 'BUY'
        
        # 매도 조건:
        # 1) 가격이 하단 밴드 돌파
        # 2) 밴드폭 확장
        # 3) 거래량 급증
        short_condition = (
            (df['close'] < df['bb_lower']) &
            (df['bb_width'] > df['bb_width_ma']) &
            (df['volume_ratio'] > self.volume_mult)
        )
        df.loc[short_condition, 'signal'] = 'SELL'
        
        # 시그널 점수 (-100 ~ +100)
        df['score'] = 0
        
        # 볼린저 밴드 기반 점수
        df.loc[df['close'] > df['bb_upper'], 'score'] += 40
        df.loc[df['close'] < df['bb_lower'], 'score'] -= 40
        
        # 거래량 기반 추가 점수
        volume_score = (df['volume_ratio'] - 1) * 10  # 거래량 비율에 따라 가중치
        volume_score = volume_score.clip(-30, 30)  # -30 ~ +30 범위로 제한
        df['score'] += volume_score
        
        # 밴드폭 확장 시 추가 점수
        bandwidth_score = ((df['bb_width'] / df['bb_width_ma'] - 1) * 20).clip(-20, 20)
        df['score'] += bandwidth_score
        
        self.logger.info(f"시그널 생성 완료: {len(df[(df['signal'] != 'HOLD')])}개 시그널")
        return df