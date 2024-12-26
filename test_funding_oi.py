from datetime import datetime, timedelta
from data_collector.funding_oi_client import BinanceFundingOIClient

def test_recent_data():
    """최근 데이터 수집 테스트"""
    client = BinanceFundingOIClient()
    
    print("=== 최근 24시간 데이터 수집 테스트 ===")
    
    # 펀딩비율 테스트
    print("\n1. BTCUSDT 펀딩비율 데이터")
    df_funding = client.fetch_funding_rate("BTCUSDT")
    if not df_funding.empty:
        print("✓ 펀딩비율 데이터 수집 성공!")
        print("\n샘플 데이터:")
        print(df_funding.head(3))
        print("\n기초 통계:")
        print(df_funding.describe())
    
    # OI 테스트
    print("\n2. BTCUSDT OI 데이터 (5분봉)")
    df_oi = client.fetch_open_interest_hist("BTCUSDT", period="5m")
    if not df_oi.empty:
        print("✓ OI 데이터 수집 성공!")
        print("\n샘플 데이터:")
        print(df_oi.head(3))
        print("\n기초 통계:")
        print(df_oi.describe())

def test_historical_data():
    """과거 데이터 수집 테스트"""
    client = BinanceFundingOIClient()
    
    print("\n=== 과거 7일 데이터 수집 테스트 ===")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    client.collect_historical_data(
        symbol="BTCUSDT",
        start_date=start_date,
        end_date=end_date,
        interval_days=1  # 1일씩 나눠서 수집
    )

if __name__ == "__main__":
    print("=== Binance Futures 펀딩비율 & OI 데이터 수집 테스트 ===\n")
    
    # 최근 데이터 테스트
    test_recent_data()
    
    # 과거 데이터 테스트
    test_historical_data()