import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy

class MarketSentimentStrategy(BaseStrategy):
    """시장 심리 전략 (펀딩비 + OI 기반)"""
    
    def __init__(self, params: Dict[str, Any]):
        name = "market_sentiment"
        super().__init__(name, params)
        
        # 펀딩비 관련 파라미터
        self.funding_extreme = params.get('funding_extreme', 0.01)  # 극단적 펀딩비 기준 (±1%)
        self.funding_ma_period = params.get('funding_ma_period', 24)  # 펀딩비 MA 기간 (8시간 * 3)
        
        # OI 관련 파라미터
        self.oi_ma_period = params.get('oi_ma_period', 12)  # OI 이동평균 기간
        self.oi_surge_threshold = params.get('oi_surge_threshold', 1.5)  # OI 급증 기준 (50% 이상)
        self.price_range_threshold = params.get('price_range_threshold', 0.002)  # 가격 횡보 기준 (±0.2%)
        
    def _calculate_funding_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """펀딩비율 기반 시그널"""
        result = df.copy()
        
        # 펀딩비 이동평균
        result['funding_ma'] = result['fundingRate'].rolling(
            window=self.funding_ma_period
        ).mean()
        
        # 극단적 펀딩비 체크
        result['funding_extreme'] = 0
        
        # 극단적 양수 펀딩비 (숏 시그널)
        result.loc[result['fundingRate'] > self.funding_extreme, 'funding_extreme'] = -1
        
        # 극단적 음수 펀딩비 (롱 시그널)
        result.loc[result['fundingRate'] < -self.funding_extreme, 'funding_extreme'] = 1
        
        # 펀딩비 방향성
        result['funding_trend'] = np.where(
            result['fundingRate'] > result['funding_ma'], 1, -1
        )
        
        return result
        
    def _calculate_oi_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """OI(미결제약정) 기반 시그널"""
        result = df.copy()
        
        # OI 이동평균
        result['oi_ma'] = result['sumOpenInterest'].rolling(
            window=self.oi_ma_period
        ).mean()
        
        # OI 변화율
        result['oi_change'] = result['sumOpenInterest'] / result['oi_ma'] - 1
        
        # 가격 변화율 (횡보 구간 체크용)
        result['price_change'] = result['close'].pct_change()
        
        # OI 급증 + 가격 횡보 체크
        result['oi_surge'] = 0
        
        oi_surge_condition = (
            (result['oi_change'] > self.oi_surge_threshold) &  # OI 급증
            (result['price_change'].abs() < self.price_range_threshold)  # 가격 횡보
        )
        result.loc[oi_surge_condition, 'oi_surge'] = 1
        
        return result
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """시장 심리 시그널 생성"""
        if not self.validate_data(data):
            return pd.DataFrame()
            
        if 'fundingRate' not in data.columns or 'sumOpenInterest' not in data.columns:
            self.logger.error("펀딩비율 또는 OI 데이터 누락")
            return pd.DataFrame()
        
        # 펀딩비/OI 시그널 계산
        df = self._calculate_funding_signals(data)
        df = self._calculate_oi_signals(df)
        
        # 시그널 통합
        df['signal'] = 'HOLD'
        
        # 매수 시그널:
        # 1) 극단적 음수 펀딩비
        # 2) OI 급증 + 가격 횡보
        long_condition = (
            (df['funding_extreme'] == 1) |  # 극단적 음수 펀딩비
            (df['oi_surge'] == 1)  # OI 급증 + 횡보
        )
        df.loc[long_condition, 'signal'] = 'BUY'
        
        # 매도 시그널:
        # 1) 극단적 양수 펀딩비
        short_condition = (df['funding_extreme'] == -1)
        df.loc[short_condition, 'signal'] = 'SELL'
        
        # 시그널 점수 계산 (-100 ~ +100)
        df['score'] = 0
        
        # 펀딩비 기반 점수
        df['score'] += -df['fundingRate'] * 5000  # 펀딩비 -0.01 = +50점
        
        # OI 기반 점수
        df.loc[df['oi_surge'] == 1, 'score'] += 30  # OI 급증 시 +30점
        
        # 점수 범위 제한
        df['score'] = df['score'].clip(-100, 100)
        
        signal_counts = df['signal'].value_counts()
        self.logger.info(
            f"시그널 생성 완료: BUY {signal_counts.get('BUY', 0)}개, "
            f"SELL {signal_counts.get('SELL', 0)}개"
        )
        
        return df