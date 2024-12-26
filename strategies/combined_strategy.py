import pandas as pd
import pandas_ta as ta

def calculate_funding_rate_change(df: pd.DataFrame, period=1):
    """펀딩비 변화율 계산"""
    return df['fundingRate'].diff(period)

def calculate_oi_change(df: pd.DataFrame, period=1):
    """OI 변화율 계산"""
    return df['sumOpenInterest'].diff(period)

def calculate_combined_signals(df: pd.DataFrame, swing_signal_func, scalp_signal_func):
    """스윙/단타 시그널과 펀딩비/OI 시그널을 결합"""
    swing_signals = swing_signal_func(df.copy())
    scalp_signals = scalp_signal_func(df.copy())
    
    # 펀딩비 및 OI 변화율 계산
    df['funding_rate_change'] = calculate_funding_rate_change(df)
    df['oi_change'] = calculate_oi_change(df)
    
    signals = []
    scores = []
    for i in range(len(df)):
        swing_signal = swing_signals['signal'].iloc[i]
        scalp_signal = scalp_signals['signal'].iloc[i]
        funding_rate_change = df['funding_rate_change'].iloc[i]
        oi_change = df['oi_change'].iloc[i]
        
        final_signal = 'HOLD'
        final_score = 0

        # 매수 시그널 결합
        if swing_signal == 'BUY' and scalp_signal == 'BUY':
            if funding_rate_change > 0 and oi_change > 0:
                final_signal = 'BUY'
                final_score = 100  # 높은 확신도
            elif funding_rate_change < 0 and oi_change < 0:
                final_signal = 'BUY'
                final_score = 50   # 낮은 확신도
            else:
                final_signal = 'BUY'
                final_score = 70
        # 매도 시그널 결합
        elif swing_signal == 'SELL' or scalp_signal == 'SELL':
            final_signal = 'SELL'
            final_score = -80
        
        signals.append(final_signal)
        scores.append(final_score)
    
    return pd.DataFrame({'signal': signals, 'score': scores}, index=df.index)