import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategies.swing_strategy import SwingStrategy
from strategies.scalping_strategy import ScalpingStrategy
from strategies.market_sentiment import MarketSentimentStrategy
from strategies.ml_strategy import MLStrategy
from strategies.combined_strategy import calculate_combined_signals, calculate_funding_rate_change, calculate_oi_change
from backtester.engine import BacktestEngine

def load_test_data(days: int = 30) -> pd.DataFrame:
    """테스트 데이터 생성"""
    periods = days * 288  # 5분봉 * 24시간 * days
    dates = [datetime.now() - timedelta(minutes=5*i) for i in range(periods)]
    dates.reverse()
    
    np.random.seed(42)
    base_price = 98000
    prices = np.random.normal(0, 100, periods).cumsum() + base_price
    
    # OHLCV 데이터
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices + np.abs(np.random.normal(0, 20, periods)),
        'low': prices - np.abs(np.random.normal(0, 20, periods)),
        'close': prices,
        'volume': np.abs(np.random.normal(100, 30, periods)),
        'fundingRate': np.random.normal(0, 0.001, periods),
        'sumOpenInterest': np.abs(np.random.normal(100000, 1000, periods).cumsum())
    })
    
    # 펀딩비율 생성 (8시간마다 업데이트, -0.01 ~ 0.01 범위)
    funding_rates = []
    for i in range(periods):
        if i % 96 == 0:  # 8시간마다
            rate = np.random.normal(0, 0.003)  # 평균 0, 표준편차 0.003
        funding_rates.append(rate)
    df['fundingRate'] = funding_rates
    
    # OI 데이터 생성
    base_oi = 100000
    oi = np.random.normal(0, 1000, periods).cumsum() + base_oi
    df['sumOpenInterest'] = np.abs(oi)
    
    df.set_index('timestamp', inplace=True)
    return df

def test_all_strategies():
    """모든 전략 통합 테스트"""
    print("\n=== 전략 통합 테스트 시작 ===")
    
    # 테스트 데이터 로드
    df = load_test_data(days=30)  # 30일치 데이터
    
    # 1. 스윙 전략
    print("\n1. 스윙 전략 테스트")
    swing_params = {
        'fast_ema': 50,
        'slow_ema': 200,
        'rsi_period': 14,
        'rsi_threshold': 30
    }
    swing = SwingStrategy(swing_params)
    swing_results = swing.generate_signals(df)
    
    # 2. 단타 전략
    print("\n2. 단타 전략 테스트")
    scalp_params = {
        'bb_period': 20,
        'bb_std': 2.0,
        'volume_ma_period': 20,
        'volume_mult': 2.0,
        'atr_period': 14
    }
    scalp = ScalpingStrategy(scalp_params)
    scalp_results = scalp.generate_signals(df.copy())
    
    # 5. 결합 전략
    print("\n5. 결합 전략 테스트")
    combined_signals = calculate_combined_signals(df.copy(), swing.generate_signals, scalp.generate_signals)
    
    # 3. 시장 심리 전략
    print("\n3. 시장 심리 전략 테스트")
    sentiment_params = {
        'funding_extreme': 0.01,
        'funding_ma_period': 24,
        'oi_ma_period': 12,
        'oi_surge_threshold': 1.5
    }
    sentiment = MarketSentimentStrategy(sentiment_params)
    sentiment_results = sentiment.generate_signals(df)
    
    # 4. ML 전략
    print("\n4. ML 전략 테스트")
    ml_params = {
        'lookback': 20,
        'predict_horizon': 6,
        'return_threshold': 0.02
    }
    ml = MLStrategy(ml_params)
    ml_results = ml.generate_signals(df)
    
    # 펀딩비 및 OI 변화율 테스트
    print("\n6. 펀딩비 및 OI 변화율 테스트")
    funding_rate_change = calculate_funding_rate_change(df)
    oi_change = calculate_oi_change(df)
    print("\n펀딩비 변화율:")
    print(funding_rate_change.head())
    print("\nOI 변화율:")
    print(oi_change.head())
    
    # 결과 통합
    results = pd.DataFrame({
        'swing_signal': swing_results['signal'],
        'swing_score': swing_results['score'],
        'scalp_signal': scalp_results['signal'],
        'scalp_score': scalp_results['score'],
        'sentiment_signal': sentiment_results['signal'],
        'sentiment_score': sentiment_results['score'],
        'ml_signal': ml_results['signal'],
        'ml_score': ml_results['score'],
        'combined_signal': combined_signals['signal'],
        'combined_score': combined_signals['score']
    })
    
    # 결과 분석
    print("\n=== 전략별 시그널 분포 ===")
    for col in ['swing_signal', 'scalp_signal', 'sentiment_signal', 'ml_signal', 'combined_signal']:
        print(f"\n{col}:")
        print(results[col].value_counts())
    
    print("\n=== 전략별 점수 통계 ===")
    score_cols = ['swing_score', 'scalp_score', 'sentiment_score', 'ml_score', 'combined_score']
    print(results[score_cols].describe())
    
    # 결과 저장
    output_file = 'data/all_strategy_results.csv'
    results.to_csv(output_file)
    print(f"\n테스트 결과를 {output_file}에 저장했습니다.")
    
    return results

if __name__ == "__main__":
    test_all_strategies()