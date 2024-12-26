import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategies.swing_strategy import SwingStrategy
from strategies.scalping_strategy import ScalpingStrategy

def test_strategies():
    """전략 테스트"""
    print("\n=== 전략 테스트 시작 ===")
    
    # 테스트 데이터 생성 (300개 데이터 포인트)
    dates = [datetime.now() - timedelta(minutes=5*i) for i in range(300)]
    dates.reverse()
    
    # 실제와 비슷한 가격 변동성 생성
    np.random.seed(42)
    base_price = 98000
    prices = np.random.normal(0, 100, 300).cumsum() + base_price
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices + np.random.normal(0, 10, 300),
        'high': prices + np.abs(np.random.normal(0, 20, 300)),
        'low': prices - np.abs(np.random.normal(0, 20, 300)),
        'close': prices + np.random.normal(0, 10, 300),
        'volume': np.abs(np.random.normal(100, 30, 300))
    })
    df.set_index('timestamp', inplace=True)
    
    # 스윙 전략 테스트
    print("\n1. 스윙 전략 테스트")
    swing_params = {
        'fast_ema': 50,
        'slow_ema': 200,
        'rsi_period': 14,
        'rsi_threshold': 30
    }
    swing = SwingStrategy(swing_params)
    swing_results = swing.generate_signals(df)
    
    print("\n스윙 전략 시그널 요약:")
    print(swing_results['signal'].value_counts())
    print("\n스윙 전략 점수 분포:")
    print(swing_results['score'].describe())
    
    # 단타 전략 테스트
    print("\n2. 단타 전략 테스트")
    scalp_params = {
        'bb_period': 20,
        'bb_std': 2.0,
        'volume_ma_period': 20,
        'volume_mult': 2.0
    }
    scalp = ScalpingStrategy(scalp_params)
    scalp_results = scalp.generate_signals(df)
    
    print("\n단타 전략 시그널 요약:")
    print(scalp_results['signal'].value_counts())
    print("\n단타 전략 점수 분포:")
    print(scalp_results['score'].describe())
    
    # 결과 저장
    output_file = 'data/strategy_test_results.csv'
    
    # 두 전략의 시그널과 점수를 병합
    combined_results = pd.DataFrame({
        'swing_signal': swing_results['signal'],
        'swing_score': swing_results['score'],
        'scalp_signal': scalp_results['signal'],
        'scalp_score': scalp_results['score']
    })
    
    combined_results.to_csv(output_file)
    print(f"\n테스트 결과를 {output_file}에 저장했습니다.")
    
    return swing_results, scalp_results

if __name__ == "__main__":
    test_strategies()