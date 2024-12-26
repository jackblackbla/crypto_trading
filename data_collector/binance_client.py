import os
import time
import requests
from datetime import datetime, timedelta
import pandas as pd
from typing import Optional, Dict, Any, Union
from dotenv import load_dotenv

load_dotenv()

class BinanceClient:
    def __init__(self):
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        self.base_url = "https://fapi.binance.com"
        self.max_retries = 3
        
    def _handle_response(self, response: requests.Response, max_retries: int = 3) -> Dict:
        """API 응답 처리 및 에러 핸들링"""
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:  # Too Many Requests
                if max_retries > 0:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    print(f"Rate limit exceeded. Retrying after {retry_after} seconds...")
                    time.sleep(retry_after)
                    return self._handle_response(response, max_retries - 1)
                else:
                    raise Exception("최대 재시도 횟수를 초과했습니다.")
            else:
                raise Exception(f"HTTP 에러: {e}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"요청 실패: {e}")
            
    def save_to_csv(self, 
                    df: pd.DataFrame, 
                    symbol: str, 
                    interval: str,
                    data_type: str = 'klines') -> None:
        """데이터를 CSV로 저장"""
        if df.empty:
            print("[Warning] DataFrame이 비어있습니다. 저장을 건너뜁니다.")
            return
            
        # data 디렉토리가 없으면 생성
        os.makedirs('data', exist_ok=True)
            
        start_dt = df['timestamp'].min()
        end_dt = df['timestamp'].max()
        
        if isinstance(start_dt, (int, float)):
            start_str = datetime.utcfromtimestamp(start_dt/1000).strftime("%Y%m%d")
            end_str = datetime.utcfromtimestamp(end_dt/1000).strftime("%Y%m%d")
        else:
            start_str = start_dt.strftime("%Y%m%d")
            end_str = end_dt.strftime("%Y%m%d")
            
        filename = f"data/{symbol}_{data_type}"
        if interval:
            filename += f"_{interval}"
        filename += f"_{start_str}_{end_str}.csv"
        
        df.to_csv(filename, index=False)
        print(f"[Info] 데이터 저장 완료: {filename}")
        
    def fetch_klines(self, 
                    symbol: str = "BTCUSDT",
                    interval: str = "5m",
                    start: Optional[int] = None,
                    end: Optional[int] = None,
                    limit: int = 1000,
                    save_csv: bool = True) -> pd.DataFrame:
        """OHLCV 데이터를 가져옵니다."""
        endpoint = f"{self.base_url}/fapi/v1/klines"
        
        params: Dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        if start:
            params["startTime"] = start
        if end:
            params["endTime"] = end
            
        try:
            response = requests.get(endpoint, params=params)
            data = self._handle_response(response)
            
            # 응답 데이터를 데이터프레임으로 변환
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades_count',
                'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
            ])
            
            # 데이터 타입 변환
            numeric_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'quote_volume', 'taker_buy_volume', 'taker_buy_quote_volume'
            ]
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # 불필요한 컬럼 제거
            df = df.drop(['close_time', 'ignore'], axis=1)
            
            # 데이터 검증
            if not self.validate_data(df):
                raise Exception("데이터 검증 실패")
                
            # CSV 저장 (옵션)
            if save_csv:
                self.save_to_csv(df, symbol, interval, 'klines')
                
            return df
            
        except Exception as e:
            print(f"[Error] 데이터 수집 실패: {e}")
            return pd.DataFrame()
            
    def validate_data(self, df: pd.DataFrame) -> bool:
        """데이터 유효성 검사"""
        if df.empty:
            print("[Error] 데이터프레임이 비어있습니다.")
            return False
            
        try:
            assert df.index.is_monotonic_increasing, "시간이 오름차순이 아닙니다"
            assert not df.isnull().any().any(), "누락된 데이터가 있습니다"
            assert (df['high'] >= df['low']).all(), "high가 low보다 작은 데이터가 있습니다"
            assert (df['volume'] >= 0).all(), "거래량이 음수인 데이터가 있습니다"
            return True
        except AssertionError as e:
            print(f"[Error] 데이터 검증 실패: {e}")
            return False