import pandas as pd
import numpy as np

def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes targets for Machine Learning.
    Target evaluation strictly avoids lookahead bias by shifting returns backwards
    (i.e., today's target is tomorrow's return).
    """
    data = df.copy()
    
    # Future Returns
    data['next_candle_return'] = data['Close'].shift(-1) / data['Close'] - 1
    
    # Primary Target: Binary
    # Bull = 1 if next candle > 0, Bear = 0
    data['target_bull'] = (data['next_candle_return'] > 0).astype(int)
    
    # Cannot train on the last row because next_candle_return is NaN
    data = data.dropna()
    
    return data
