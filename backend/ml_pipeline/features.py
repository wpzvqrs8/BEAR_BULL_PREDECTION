"""
features.py — Full feature engineering pipeline.

Features included:
  Price-based:    EMAs, MACD, Bollinger Bands, RSI, ATR
  Volume-based:   Volume delta, OBV, VWAP deviation
  Volatility:     ATR percentile, volatility expansion ratio, rolling skewness
  Pattern:        Candle body/wick ratios
  Regime signals: ADX, DI+/DI- (consumed by regime.py)
  Correlation pruning: drops features with |corr| > 0.85
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────
def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    g = d.where(d > 0, 0.0).ewm(alpha=1/n, adjust=False).mean()
    l = (-d).where(d < 0, 0.0).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + g / (l + 1e-9))

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()

def _adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14):
    """Returns (ADX, DI+, DI-) as a DataFrame."""
    up   = high.diff()
    down = -low.diff()
    dm_p = up.where((up > down) & (up > 0), 0.0)
    dm_m = down.where((down > up) & (down > 0), 0.0)
    atr14 = _atr(high, low, close, n) + 1e-9
    di_p  = 100 * dm_p.ewm(span=n, adjust=False).mean() / atr14
    di_m  = 100 * dm_m.ewm(span=n, adjust=False).mean() / atr14
    dx    = 100 * (di_p - di_m).abs() / (di_p + di_m + 1e-9)
    adx   = dx.ewm(span=n, adjust=False).mean()
    return adx, di_p, di_m


# ── Main feature builder ──────────────────────────────────────────────────────
def build_features(df: pd.DataFrame,
                   drop_correlated: bool = True,
                   corr_threshold: float = 0.85) -> pd.DataFrame:
    """
    Build feature matrix from OHLCV DataFrame.

    Args:
        df: DataFrame with columns open, high, low, close, volume, ts.
        drop_correlated: remove features with |pearson| > corr_threshold.
        corr_threshold: max allowed pairwise correlation.

    Returns:
        df enriched with feature columns (original OHLCV columns preserved).
    """
    df = df.copy().reset_index(drop=True)
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    # ── EMAs ──────────────────────────────────────────────────────────────────
    df["ema9"]  = _ema(c, 9)
    df["ema21"] = _ema(c, 21)
    df["ema50"] = _ema(c, 50)
    df["ema200"]= _ema(c, 200)

    # EMA cross signals (normalised)
    df["ema9_21"]  = (df["ema9"]  - df["ema21"])  / (df["ema21"]  + 1e-9)
    df["ema21_50"] = (df["ema21"] - df["ema50"])  / (df["ema50"]  + 1e-9)
    df["ema50_200"]= (df["ema50"] - df["ema200"]) / (df["ema200"] + 1e-9)

    # ── MACD ──────────────────────────────────────────────────────────────────
    macd_line  = _ema(c, 12) - _ema(c, 26)
    signal     = _ema(macd_line, 9)
    df["macd_hist"] = (macd_line - signal) / (c + 1e-9)

    # ── RSI ───────────────────────────────────────────────────────────────────
    df["rsi14"] = _rsi(c, 14) / 100      # normalise 0–1
    df["rsi7"]  = _rsi(c, 7)  / 100

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    mid20   = c.rolling(20).mean()
    std20   = c.rolling(20).std()
    df["bb_pos"] = (c - mid20) / (2 * std20 + 1e-9)  # -1 to +1
    df["bb_wid"] = 2 * std20 / (mid20 + 1e-9)

    # ── ATR ───────────────────────────────────────────────────────────────────
    atr14 = _atr(h, l, c, 14)
    df["atr14_norm"] = atr14 / (c + 1e-9)

    # ATR percentile over rolling 100 bars
    df["atr_pct"] = (
        atr14.rolling(100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=True)
    )

    # ── Volatility expansion ratio ─────────────────────────────────────────────
    vol_short = c.rolling(5).std()
    vol_long  = c.rolling(20).std()
    df["vol_expand"] = vol_short / (vol_long + 1e-9)

    # ── Rolling skewness (returns) ────────────────────────────────────────────
    log_ret = np.log(c / c.shift(1))
    df["skew20"] = log_ret.rolling(20).skew()
    df["kurt20"] = log_ret.rolling(20).kurt()

    # ── Volume delta ──────────────────────────────────────────────────────────
    df["vol_delta"]    = v.diff()                           # raw volume change
    df["vol_delta_n"]  = df["vol_delta"] / (v.shift(1) + 1)   # normalised
    df["vol_ratio"]    = v / (v.rolling(20).mean() + 1)    # vs 20-bar average

    # ── OBV (normalised trend) ────────────────────────────────────────────────
    direction = np.sign(c.diff())
    obv = (direction * v).cumsum()
    df["obv_norm"] = (obv - obv.rolling(20).mean()) / (obv.rolling(20).std() + 1e-9)

    # ── VWAP deviation (intrabar proxy) ───────────────────────────────────────
    typical = (h + l + c) / 3
    vwap20  = (typical * v).rolling(20).sum() / (v.rolling(20).sum() + 1e-9)
    df["vwap_dev"] = (c - vwap20) / (vwap20 + 1e-9)

    # ── Candle shape features ─────────────────────────────────────────────────
    candle_rng = (h - l).clip(lower=1e-9)
    df["body_ratio"] = (c - o).abs() / candle_rng          # body / range
    df["upper_wick"] = (h - pd.concat([o,c],axis=1).max(axis=1)) / candle_rng
    df["lower_wick"] = (pd.concat([o,c],axis=1).min(axis=1) - l) / candle_rng
    df["is_bull_bar"] = (c > o).astype(float)

    # ── Momentum ──────────────────────────────────────────────────────────────
    for n in [3, 5, 10, 20]:
        df[f"mom{n}"] = c.pct_change(n)

    # ── ADX / DI ──────────────────────────────────────────────────────────────
    adx, di_p, di_m = _adx(h, l, c, 14)
    df["adx14"]  = adx / 100
    df["di_diff"]= (di_p - di_m) / 100   # positive = uptrend

    # ── ETH-specific: ETH/BTC ratio ───────────────────────────────────────────
    if "eth_btc_ratio" in df.columns:
        df["eth_btc_mom5"] = df["eth_btc_ratio"].pct_change(5)

    # ── Drop raw EMAs (highly correlated with close) ──────────────────────────
    df.drop(columns=["ema9","ema21","ema50","ema200"], inplace=True, errors="ignore")

    # ── Correlation pruning ────────────────────────────────────────────────────
    if drop_correlated:
        feature_cols = _get_feature_cols(df)
        df, dropped = _drop_correlated(df, feature_cols, corr_threshold)
        if dropped:
            print(f"[features] Dropped {len(dropped)} correlated features: {dropped}")

    return df


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return feature column names (excludes OHLCV, ts, label columns)."""
    exclude = {"ts","open","high","low","close","volume","close_time",
               "btc_close","eth_btc_ratio","future_return","rolling_vol",
               "label","label_binary"}
    return [c for c in df.columns if c not in exclude]


def _drop_correlated(df: pd.DataFrame, cols: list[str],
                     threshold: float) -> tuple[pd.DataFrame, list[str]]:
    """Drop features with pairwise |correlation| > threshold (keeps first)."""
    corr = df[cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
    df.drop(columns=to_drop, inplace=True, errors="ignore")
    return df, to_drop
