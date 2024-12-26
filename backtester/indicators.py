import pandas as pd
import numpy as np
from typing import Dict, Tuple, Union

def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """단순이동평균(Simple Moving Average) 계산"""
    return data.rolling(window=period).mean()

def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """지수이동평균(Exponential Moving Average) 계산"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_macd(data: pd.Series,
                  fast_period: int = 12,
                  slow_period: int = 26,
                  signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD(Moving Average Convergence Divergence) 계산"""
    fast_ema = calculate_ema(data, fast_period)
    slow_ema = calculate_ema(data, slow_period)
    
    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line, signal_period)
    macd_histogram = macd_line - signal_line
    
    return macd_line, signal_line, macd_histogram

def calculate_bollinger_bands(data: pd.Series,
                            period: int = 20,
                            num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """볼린저 밴드 계산"""
    middle_band = calculate_sma(data, period)
    std = data.rolling(window=period).std()
    
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    return upper_band, middle_band, lower_band

def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """RSI(Relative Strength Index) 계산"""
    delta = data.diff()
    
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def add_indicators(df: pd.DataFrame, config: Dict[str, Union[int, float]]) -> pd.DataFrame:
    """여러 지표를 한 번에 계산하여 DataFrame에 추가"""
    result = df.copy()
    
    # EMA
    if 'ema' in config:
        for period in config['ema']:
            result[f'ema_{period}'] = calculate_ema(df['close'], period)
    
    # Bollinger Bands
    if 'bollinger' in config:
        period = config['bollinger'].get('period', 20)
        num_std = config['bollinger'].get('num_std', 2.0)
        upper, middle, lower = calculate_bollinger_bands(
            df['close'], period, num_std
        )
        result[f'bb_upper_{period}'] = upper
        result[f'bb_middle_{period}'] = middle
        result[f'bb_lower_{period}'] = lower
    
    # MACD
    if 'macd' in config:
        fast = config['macd'].get('fast', 12)
        slow = config['macd'].get('slow', 26)
        signal = config['macd'].get('signal', 9)
        macd, signal, hist = calculate_macd(
            df['close'], fast, slow, signal
        )
        result['macd'] = macd
        result['macd_signal'] = signal
        result['macd_hist'] = hist
    
    # RSI
    if 'rsi' in config:
        period = config['rsi'].get('period', 14)
        result[f'rsi_{period}'] = calculate_rsi(df['close'], period)
    
    return result