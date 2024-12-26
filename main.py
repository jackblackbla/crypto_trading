# main.py
import pandas as pd
from strategies.ml_strategy import MLStrategy
from backtester.engine import BacktestEngine

def main():
    # 1) 데이터 로딩
    df = pd.read_csv("data/your_ohlcv_data.csv", parse_dates=["timestamp"], index_col="timestamp")
    
    # 2) 전략 인스턴스 생성
    ml_params = {
        "lookback": 20,
        "predict_horizon": 12,
        # 기타 하이퍼파라미터...
    }
    ml_strat = MLStrategy(params=ml_params)

    # 3) 백테스트 엔진 초기화
    engine = BacktestEngine(strategy=ml_strat, initial_balance=10000, fee_rate=0.0005, slippage=0.0005)

    # 4) 백테스트 실행
    results_df = engine.run(df)

    # 5) 결과 요약
    engine.summary()

    # 6) 결과 저장/확인
    print(results_df.tail(10))
    results_df.to_csv("data/backtest_results.csv")

if __name__ == "__main__":
    main()