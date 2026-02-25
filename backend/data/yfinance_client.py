import yfinance as yf
import pandas as pd
from typing import Optional

def fetch_ohlcv(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV historical data using yfinance.
    Returns DataFrame with columns: ['Open', 'High', 'Low', 'Close', 'Volume']
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        if df.empty:
            return None
            
        # Standardize column names
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        # yfinance index is timezone-aware DatetimeIndex
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None
