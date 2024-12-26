import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union
from dotenv import load_dotenv

load_dotenv()

class BinanceFundingOIClient:
    def __init__(self):
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        self.base_url = "https://fapi.binance.com"
        self.max_retries = 3
        
    def _handle_response(self, response: requests.Response, retry_count: int = 0) -> Dict:
        """API 응답 처리 및 에러 핸들링"""
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429 and retry_count < self.max_retries:
                retry_after = int(response.headers.get('Retry-After', 60))
                print(f"[Rate Limit] {retry_after}초 후 재시도... (시도 {retry_count + 1}/{self.max_retries})")
                time.sleep(retry_after)
                return self._handle_response(response, retry_count + 1)
            else:
                raise Exception(f"HTTP 에러: {e}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"요청 실패: {e}")
            
    def validate_funding_data(self, df: pd.DataFrame) -> bool:
        """펀딩비율 데이터 검증"""
        try:
            # 기본 검증
            assert not df.empty, "데이터프레임이 비어있음"
            assert not df.isnull().values.any(), "결측치(NaN) 존재"
            
            # 펀딩비율 범위 검증 (-1% ~ +1%)
            assert df['fundingRate'].between(-0.01, 0.01).all(), "비정상적인 펀딩비율 존재"
            
            # 타임스탬프 간격 검증 (8시간 ± 30분)
            time_diff = df['timestamp'].diff().dropna()
            assert time_diff.between(pd.Timedelta(hours=7.5), pd.Timedelta(hours=8.5)).all(), \
                   "펀딩비율 시간 간격 오류"
            
            return True
        except AssertionError as e:
            print(f"[Error] 펀딩비율 데이터 검증 실패: {e}")
            return False
            
    def validate_oi_data(self, df: pd.DataFrame, expected_interval: str) -> bool:
        """OI 데이터 검증"""
        try:
            # 기본 검증
            assert not df.empty, "데이터프레임이 비어있음"
            assert not df.isnull().values.any(), "결측치(NaN) 존재"
            
            # OI 값 검증
            assert (df['sumOpenInterest'] >= 0).all(), "음수 OI 존재"
            assert (df['sumOpenInterestValue'] >= 0).all(), "음수 OI 가치 존재"
            
            # 시간 간격 검증
            interval_minutes = int(expected_interval.replace('m', ''))
            time_diff = df['timestamp'].diff().dropna()
            min_interval = pd.Timedelta(minutes=interval_minutes * 0.95)  # 5% 오차 허용
            max_interval = pd.Timedelta(minutes=interval_minutes * 1.05)
            assert time_diff.between(min_interval, max_interval).all(), \
                   f"OI 시간 간격 오류 (예상: {interval_minutes}분)"
            
            return True
        except AssertionError as e:
            print(f"[Error] OI 데이터 검증 실패: {e}")
            return False
    
    def save_to_csv(self, df: pd.DataFrame, symbol: str, data_type: str) -> None:
        """데이터를 CSV로 저장."""
        if df.empty:
            print(f"[Warning] {symbol} {data_type} DataFrame이 비어있습니다. 저장을 건너뜁니다.")
            return

        # data 디렉토리가 없으면 생성
        os.makedirs('data', exist_ok=True)

        start_date = df['timestamp'].min().strftime("%Y%m%d")
        end_date = df['timestamp'].max().strftime("%Y%m%d")

        filename = f"data/{symbol}_{data_type}_{start_date}_{end_date}.csv"
        df.to_csv(filename, index=False)
        print(f"[Info] {symbol} {data_type} 데이터 저장 완료: {filename}")
    
    def fetch_funding_rate(self,
                         symbol: str,
                         start_time: Optional[int] = None,
                         end_time: Optional[int] = None,
                         limit: int = 1000,
                         save_csv: bool = True) -> pd.DataFrame:
        """펀딩비율 데이터를 가져옵니다."""
        endpoint = f"{self.base_url}/fapi/v1/fundingRate"
        
        params = {
            "symbol": symbol,
            "limit": limit
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
            
        try:
            response = requests.get(endpoint, params=params)
            data = self._handle_response(response)
            
            if not data:
                print("[Warning] 데이터가 없습니다.")
                return pd.DataFrame()
            
            # DataFrame 변환 및 전처리
            df = pd.DataFrame(data)
            df['fundingRate'] = df['fundingRate'].astype(float)
            df['timestamp'] = pd.to_datetime(df['fundingTime'], unit='ms')
            df = df.drop('fundingTime', axis=1)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # 데이터 검증
            if not self.validate_funding_data(df):
                print("[Warning] 데이터 검증 실패. 저장을 건너뜁니다.")
                return pd.DataFrame()
                
            # CSV 저장
            if save_csv:
                self.save_to_csv(df, symbol, 'funding_rate')
                
            return df
            
        except Exception as e:
            print(f"[Error] 펀딩비율 데이터 수집 실패: {e}")
            return pd.DataFrame()
            
    def fetch_open_interest_hist(self,
                               symbol: str = "BTCUSDT",
                               period: str = "5m",
                               start_time: Optional[int] = None,
                               end_time: Optional[int] = None,
                               limit: int = 500,
                               save_csv: bool = True) -> pd.DataFrame:
        """미결제약정(OI) 데이터를 가져옵니다 (기간 분할)."""
        endpoint = f"{self.base_url}/futures/data/openInterestHist"
        all_data = []
        
        if start_time and end_time:
            current_start_ms = start_time
            chunk_size_ms = 3 * 24 * 60 * 60 * 1000  # 3일
            
            while current_start_ms < end_time:
                current_end_ms = min(current_start_ms + chunk_size_ms, end_time)
                
                params = {
                    "symbol": symbol,
                    "period": period,
                    "limit": limit,
                    "startTime": current_start_ms,
                    "endTime": current_end_ms
                }
                
                print(f"OI 요청 파라미터: {params}")
                try:
                    response = requests.get(endpoint, params=params)
                    response.raise_for_status()
                    data = response.json()
                    if data:
                        all_data.extend(data)
                except requests.exceptions.HTTPError as e:
                    print(f"[Error] OI 데이터 수집 실패 ({datetime.fromtimestamp(current_start_ms / 1000)} ~ {datetime.fromtimestamp(current_end_ms / 1000)}): {e}")
                    #return pd.DataFrame() # 임시 주석 처리
                
                current_start_ms = current_end_ms
                time.sleep(1)  # API 부하 방지
        else:
            params = {
                "symbol": symbol,
                "period": period,
                "limit": limit
            }
            try:
                response = requests.get(endpoint, params=params)
                response.raise_for_status()
                data = response.json()
                if data:
                    all_data.extend(data)
            except requests.exceptions.HTTPError as e:
                print(f"[Error] OI 데이터 수집 실패: {e}")
                return pd.DataFrame()
        
        if not all_data:
            print("[Warning] OI 데이터가 없습니다.")
            return pd.DataFrame()
        
        # DataFrame 변환 및 전처리
        df = pd.DataFrame(all_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['sumOpenInterest'] = df['sumOpenInterest'].astype(float)
        df['sumOpenInterestValue'] = df['sumOpenInterestValue'].astype(float)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 데이터 검증
        if not self.validate_oi_data(df, period):
            print("[Warning] OI 데이터 검증 실패. 저장을 건너뜁니다.")
            return pd.DataFrame()
        
        # CSV 저장
        if save_csv:
            self.save_to_csv(df, symbol, f'oi_{period}')
        
        return df
            
    def collect_historical_data(self,
                                symbols: list,
                                start_date: datetime = None,
                                end_date: datetime = None) -> pd.DataFrame:
        """특정 기간의 펀딩비율과 OI 데이터를 수집합니다."""
        funding_dataframes = []
        if not symbols:
            print("[Warning] 수집할 심볼이 없습니다.")
            return pd.DataFrame()

        today = datetime.now().date()
        if start_date and (today - start_date.date()).days > 30:
            print(f"[Info] {start_date.strftime('%Y-%m-%d')}는 현재 날짜로부터 30일 이전이므로 OI 데이터 수집을 건너뜁니다.")
            return pd.DataFrame()

        if not start_date:
            start_date = today - timedelta(days=30)
        if not end_date:
            end_date = today
            
        start_ms = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)
            
        for symbol in symbols:
            print(f"\n수집 기간: {start_date} ~ {end_date}, 심볼: {symbol}")

            # 펀딩비율 수집
            df_funding = self.fetch_funding_rate(
                symbol=symbol,
                start_time=start_ms,
                end_time=end_ms
            )
            if not df_funding.empty:
                self.save_to_csv(df_funding, symbol, 'funding_rate')
                print(f"{symbol} 펀딩비율 데이터 저장 완료: {len(df_funding)} 행")
                funding_dataframes.append(df_funding)
            else:
                print(f"{symbol} 펀딩비율 데이터 fetching 실패")
            
            # OI 수집
            df_oi = self.fetch_open_interest_hist(
                symbol=symbol,
                period="5m",
                start_time=start_ms,
                end_time=end_ms
            )
            print(f"{symbol} OI 데이터: {len(df_oi)}행")
            
        time.sleep(1)  # API 부하 방지
        
        if funding_dataframes:
            return pd.concat(funding_dataframes, ignore_index=True)
        else:
            return pd.DataFrame()