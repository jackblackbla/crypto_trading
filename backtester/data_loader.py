import pandas as pd
import numpy as np
from typing import Optional, Dict, Union
from datetime import datetime, timedelta
import logging

class DataLoader:
    def __init__(self):
        self.data = None
        self._setup_logging()
    
    def _setup_logging(self):
        self.logger = logging.getLogger('backtester.data_loader')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def load_ohlcv(self, 
                   filepath: str,
                   start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None) -> pd.DataFrame:
        """OHLCV 데이터 로드 및 전처리"""
        try:
            # CSV 파일 로드
            df = pd.read_csv(filepath)
            
            # 타임스탬프 변환
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # 기간 필터링
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
                
            # 데이터 정합성 체크
            self._validate_ohlcv(df)
            
            # 컬럼 타입 변환
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_columns] = df[numeric_columns].astype(float)
            
            self.data = df
            self.logger.info(f"OHLCV 데이터 로드 완료: {len(df)} 행")
            
            return df
            
        except Exception as e:
            self.logger.error(f"데이터 로드 실패: {e}")
            raise
            
    def load_funding_rate(self, 
                         filepath: str,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> pd.DataFrame:
        """펀딩비율 데이터 로드"""
        try:
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
                
            df['fundingRate'] = df['fundingRate'].astype(float)
            
            self.logger.info(f"펀딩비율 데이터 로드 완료: {len(df)} 행")
            return df
            
        except Exception as e:
            self.logger.error(f"펀딩비율 데이터 로드 실패: {e}")
            raise
            
    def merge_data(self,
                   ohlcv: pd.DataFrame,
                   funding_rate: Optional[pd.DataFrame] = None,
                   fillna: bool = True) -> pd.DataFrame:
        """OHLCV와 펀딩비율 데이터 병합"""
        try:
            result = ohlcv.copy()
            
            if funding_rate is not None:
                # 펀딩비율을 OHLCV와 같은 시간 간격으로 리샘플링
                funding_resampled = funding_rate.resample('5T').ffill()  # 5분 기준
                result = pd.merge(
                    result,
                    funding_resampled,
                    left_index=True,
                    right_index=True,
                    how='left'
                )
                
                if fillna:
                    result['fundingRate'].fillna(method='ffill', inplace=True)
                    
            self.logger.info(f"데이터 병합 완료: {len(result)} 행")
            return result
            
        except Exception as e:
            self.logger.error(f"데이터 병합 실패: {e}")
            raise
    
    def _validate_ohlcv(self, df: pd.DataFrame) -> None:
        """OHLCV 데이터 유효성 검증"""
        try:
            # 필수 컬럼 체크
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"필수 컬럼 누락: {missing_columns}")
            
            # 결측치 체크
            null_counts = df[required_columns].isnull().sum()
            if null_counts.any():
                self.logger.warning(f"결측치 발견:\n{null_counts[null_counts > 0]}")
            
            # 시간 순서 체크
            if not df.index.is_monotonic_increasing:
                raise ValueError("타임스탬프가 오름차순이 아님")
            
            # 중복 인덱스 체크
            if df.index.duplicated().any():
                duplicates = df.index[df.index.duplicated()].unique()
                raise ValueError(f"중복된 타임스탬프 발견: {duplicates}")
            
            # OHLC 값 검증
            invalid_candles = df[
                (df['high'] < df['low']) |  # 고가가 저가보다 낮은 경우
                (df['open'] > df['high']) |  # 시가가 고가보다 높은 경우
                (df['open'] < df['low']) |   # 시가가 저가보다 낮은 경우
                (df['close'] > df['high']) | # 종가가 고가보다 높은 경우
                (df['close'] < df['low'])    # 종가가 저가보다 낮은 경우
            ]
            
            if not invalid_candles.empty:
                self.logger.warning(
                    f"비정상 캔들 {len(invalid_candles)}개 발견:\n"
                    f"{invalid_candles.head()}"
                )
            
        except Exception as e:
            self.logger.error(f"데이터 검증 실패: {e}")
            raise