"""
labeling.py — Volatility-adjusted 3-class labels.

label = +1  if future_return >  k * rolling_volatility   (strong upward)
label = -1  if future_return < -k * rolling_volatility   (strong downward)
label =  0  otherwise                                    (no-trade zone)

This prevents the model from learning noise on small moves.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def add_labels(
    df: pd.DataFrame,
    lookahead: int = 1,
    vol_window: int = 20,
    k: float = 0.5,
    return_col: str = "close",
    label_col: str = "label",
) -> pd.DataFrame:
    """
    Add volatility-adjusted labels to a candle DataFrame.

    Args:
        df:          DataFrame with at least a 'close' column.
        lookahead:   Number of bars ahead to measure return.
        vol_window:  Rolling window for volatility estimation.
        k:           Threshold multiplier (higher k = fewer but cleaner signals).
        return_col:  Column to compute returns from.
        label_col:   Output column name.

    Returns:
        df with label_col, future_return, rolling_vol columns added.
    """
    df = df.copy()

    # Log return over lookahead bars
    df["future_return"] = (
        df[return_col].shift(-lookahead) / df[return_col] - 1
    )

    # Rolling volatility (std of log returns)
    log_ret = np.log(df[return_col] / df[return_col].shift(1))
    df["rolling_vol"] = log_ret.rolling(vol_window).std()

    # Threshold
    threshold = k * df["rolling_vol"]
    df[label_col] = 0
    df.loc[df["future_return"] >  threshold, label_col] =  1
    df.loc[df["future_return"] < -threshold, label_col] = -1

    # Drop rows where we can't compute label or vol
    df.dropna(subset=["future_return", "rolling_vol"], inplace=True)

    counts = df[label_col].value_counts()
    total = len(df)
    print(f"[label] k={k}, lookahead={lookahead}, vol_window={vol_window}")
    print(f"  +1 (Bull):  {counts.get(1,0):5d}  ({100*counts.get(1,0)/total:.1f}%)")
    print(f"   0 (No-trade): {counts.get(0,0):5d}  ({100*counts.get(0,0)/total:.1f}%)")
    print(f"  -1 (Bear):  {counts.get(-1,0):5d}  ({100*counts.get(-1,0)/total:.1f}%)")
    return df


def binary_labels(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """
    Convert 3-class labels to binary by dropping 0-label rows.
    Used for models that only predict bull vs bear (ignores no-trade).
    """
    return df[df[label_col] != 0].copy()
