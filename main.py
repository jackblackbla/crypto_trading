import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from data_collector.binance_client import BinanceClient
from data_collector.funding_oi_client import BinanceFundingOIClient

load_dotenv()

def fetch_long_term_data(symbol, interval, start_date_str, end_date_str):
    binance_client = BinanceClient()
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # Adjust start_date and end_date to be earlier to fetch more data
    start_date -= pd.Timedelta(days=365)
    end_date += pd.Timedelta(days=365)
    
    start_ms = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)
    
    df = binance_client.fetch_klines(symbol, interval, start_ms, end_ms)
    return df

def main():
    # 작업 1.2: 장기 펀딩비 데이터 수집
    funding_start_date_str = "2020-12-01"
    funding_end_date_str = "2021-04-29"
    
    start_funding_date = datetime.strptime(funding_start_date_str, "%Y-%m-%d")
    end_funding_date = datetime.strptime(funding_end_date_str, "%Y-%m-%d")
    
    funding_symbols = ["BTCUSDT", "BNBUSDT", "SOLUSDT", "DOGEUSDT", "ADAUSDT", "DOTUSDT"]
    
    print("Fetching funding rate data for multiple symbols...")
    funding_client = BinanceFundingOIClient()
    
    df_funding = funding_client.collect_historical_data(
        symbols=funding_symbols,
        start_date=start_funding_date,
        end_date=end_funding_date
    )

    # 작업 1.3: 데이터 전처리 (펀딩비 데이터)
    if df_funding is not None and not df_funding.empty:
        print("\nPreprocessing BTCUSDT funding rate data...")
        # 결측치 처리
        print("Handling missing values...")
        df_funding.fillna(method='ffill', inplace=True)
        print("Missing values handled.")
        
        # 데이터 타입 변환
        print("Converting data types...")
        df_funding['timestamp'] = pd.to_datetime(df_funding['fundingTime'], unit='ms')
        df_funding['fundingRate'] = df_funding['fundingRate'].astype(float)
        print("Data types converted.")
        
        # 이상치 확인 (describe() 사용)
        print("\nDescriptive statistics for funding rate data:")
        print(df_funding.describe())
        
        print("\nBTCUSDT 펀딩비 데이터 전처리 완료")
    
    # OHLCV 데이터 수집 및 병합 (기존 코드 유지, 기간만 변경)
    symbols = ["BTCUSDT", "BNBUSDT", "SOLUSDT", "DOGEUSDT", "ADAUSDT", "DOTUSDT"]
    intervals = ["5m", "4h", "1d"]
    start_date_str = "2020-12-01"
    end_date_str = "2021-04-29"
    
    all_merged_data = {}
    
    for symbol in symbols:
        # 해당 symbol에 대한 펀딩비 데이터가 있는지 확인
        if df_funding is not None and not df_funding.empty:
            for interval in intervals:
                print(f"\nFetching {symbol} {interval} data...")
                df_ohlcv = fetch_long_term_data(symbol, interval, start_date_str, end_date_str)
                if not df_ohlcv.empty:
                    df_ohlcv['timestamp'] = pd.to_datetime(df_ohlcv['timestamp'])
                    
                    df_merged = pd.merge(df_ohlcv, df_funding, on='timestamp', how='left')
                    
                    all_merged_data[(symbol, interval)] = df_merged
                    
                    print(f"{symbol} {interval} OHLCV 데이터와 펀딩비 데이터 병합 완료")
                else:
                    print(f"{symbol} {interval} 데이터 fetching 실패")
        else:
            print(f"\n{symbol} 에 대한 펀딩비 데이터가 없습니다.")
    
    # 병합된 데이터 저장
    for (symbol, interval), df_merged in all_merged_data.items():
        file_name = f"data/{symbol}_{interval}_merged_{start_date_str.replace('-', '')[:6]}_{end_date_str.replace('-', '')[:6]}.csv"
        df_merged.to_csv(file_name, index=False)
        print(f"{symbol} {interval} 병합된 데이터 저장 완료: {file_name}")

if __name__ == "__main__":
    main()