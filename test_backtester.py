import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backtester.data_loader import DataLoader
from backtester.indicators import add_indicators

def create_sample_data(rows: int = 300):
    """테스트용 샘플 데이터 생성"""
    print("\n=== 샘플 데이터 생성 ===")
    
    # 현재 시점부터 과거로 5분 간격 데이터 생성
    end_time = datetime.now()
    dates = [end_time - timedelta(minutes=5*i) for i in range(rows)]
    dates.reverse()
    
    # 랜덤 가격 생성 (실제 비트코인 가격대 반영)
    base_price = 98000
    np.random.seed(42)  # 재현성을 위한 시드 설정
    
    prices = np.random.normal(0, 100, rows).cumsum() + base_price
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices + np.random.normal(0, 10, rows),
        'high': prices + np.abs(np.random.normal(0, 20, rows)),
        'low': prices - np.abs(np.random.normal(0, 20, rows)),
        'close': prices + np.random.normal(0, 10, rows),
        'volume': np.abs(np.random.normal(100, 30, rows)),
        'closed': True
    })
    
    # timestamp를 인덱스로 설정
    df.set_index('timestamp', inplace=True)
    
    return df

def test_indicators():
    """지표 계산 테스트"""
    print("\n=== 지표 계산 테스트 ===")
    
    # 샘플 데이터 생성
    df = create_sample_data(300)  # 300개의 데이터 포인트
    
    print("\n원본 데이터 샘플:")
    print(df.head())
    print("\n데이터 정보:")
    print(df.info())
    
    # 지표 설정
    indicator_config = {
        'ema': [50, 200],  # 50일, 200일 EMA
        'bollinger': {'period': 20, 'num_std': 2.0},
        'macd': {'fast': 12, 'slow': 26, 'signal': 9},
        'rsi': {'period': 14}
    }
    
    # 지표 계산
    df_with_indicators = add_indicators(df, indicator_config)
    
    print("\n지표 컬럼:")
    indicators = [col for col in df_with_indicators.columns if col not in df.columns]
    print(indicators)
    
    # 지표 유효성 검사
    print("\n지표별 유효한 데이터 포인트 수:")
    for indicator in indicators:
        valid_count = df_with_indicators[indicator].notna().sum()
        total_count = len(df_with_indicators)
        print(f"{indicator}: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
    
    print("\n최근 지표값 샘플:")
    print(df_with_indicators[indicators].tail())
    
    # 결과 데이터 파일로 저장
    output_file = 'data/test_indicators.csv'
    df_with_indicators.to_csv(output_file)
    print(f"\n테스트 데이터를 {output_file}에 저장했습니다.")

if __name__ == "__main__":
    test_indicators()