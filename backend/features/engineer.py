import pandas as pd
import numpy as np
import ta

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a comprehensive set of features without lookahead bias.
    Input DataFrame must have OHLCV columns.
    """
    # Copy to avoid modifying original
    data = df.copy()
    
    # 1. Price Returns
    data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
    data['simple_return'] = data['Close'].pct_change()
    
    # 2. Moving Averages
    for window in [10, 20, 50, 200]:
        data[f'sma_{window}'] = data['Close'].rolling(window=window).mean()
        data[f'ema_{window}'] = ta.trend.ema_indicator(data['Close'], window=window)
        # Distance from MA
        data[f'dist_sma_{window}'] = data['Close'] / data[f'sma_{window}'] - 1
        
    # 3. Momentum & Oscillators
    data['rsi_14'] = ta.momentum.rsi(data['Close'], window=14)
    data['macd'] = ta.trend.macd_diff(data['Close'])
    
    # 4. Volatility
    data['bb_high'] = ta.volatility.bollinger_hband(data['Close'])
    data['bb_low'] = ta.volatility.bollinger_lband(data['Close'])
    data['bb_width'] = (data['bb_high'] - data['bb_low']) / data['Close']
    data['atr_14'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=14)
    data['roll_vol_20'] = data['log_return'].rolling(window=20).std()
    
    # 5. Volume Features
    data['vol_delta'] = data['Volume'].pct_change()
    data['vwap'] = ta.volume.volume_weighted_average_price(data['High'], data['Low'], data['Close'], data['Volume'])
    
    # 6. Candlestick shape
    data['body_size'] = abs(data['Close'] - data['Open'])
    data['upper_wick'] = data['High'] - data[['Open', 'Close']].max(axis=1)
    data['lower_wick'] = data[['Open', 'Close']].min(axis=1) - data['Low']
    data['candle_range'] = data['High'] - data['Low']
    # Avoid div by zero
    data['upper_wick_ratio'] = data['upper_wick'] / (data['candle_range'] + 1e-8)
    data['lower_wick_ratio'] = data['lower_wick'] / (data['candle_range'] + 1e-8)
    
    # Drop rows with NaN due to rolling windows
    data = data.dropna()
    return data
