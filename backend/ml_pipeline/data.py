"""
data.py — Unified data loading for BTC (local CSVs) and ETH (Binance API).
"""
from __future__ import annotations
import os, glob, requests
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

# ── BTC: local 1-minute CSV files ─────────────────────────────────────────────
CSV_DIR = Path(__file__).parent.parent.parent / "datas"

_CSV_COLS = ["open_time", "open", "high", "low", "close", "volume",
             "close_time", "qav", "num_trades", "tbbav", "tbqav", "ignore"]

def load_btc_csv(max_rows: Optional[int] = None,
                 resample: Optional[str] = None) -> pd.DataFrame:
    """Load BTC 1-min CSVs → sorted DataFrame with OHLCV columns.

    Args:
        max_rows: cap rows (None = whole dataset).
        resample: e.g. '1h', '4h', '1d' — resamples after loading.
    """
    files = sorted(glob.glob(str(CSV_DIR / "BTCUSDT-1m-*.csv")))
    if not files:
        raise FileNotFoundError(f"No BTC CSVs found in {CSV_DIR}")

    dfs = []
    for f in files:
        df = pd.read_csv(f, header=None, names=_CSV_COLS, usecols=[0,1,2,3,4,5])
        dfs.append(df)
    raw = pd.concat(dfs, ignore_index=True)
    raw.rename(columns={"open_time": "ts"}, inplace=True)
    raw["ts"] = pd.to_datetime(raw["ts"], unit="ms", utc=True)
    raw = raw.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    for c in ["open", "high", "low", "close", "volume"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw.dropna(subset=["open","high","low","close","volume"], inplace=True)

    if max_rows is not None:
        raw = raw.tail(max_rows)

    if resample:
        raw = raw.set_index("ts").resample(resample).agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum"
        }).dropna().reset_index()

    print(f"[data] BTC loaded: {len(raw):,} rows | "
          f"{raw['ts'].min().date()} → {raw['ts'].max().date()}")
    return raw


# ── ETH: Binance public REST API ──────────────────────────────────────────────
BINANCE_BASE = "https://api.binance.com"

def _binance_klines(symbol: str, interval: str, limit: int = 1000,
                    start_ms: Optional[int] = None) -> list:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_ms:
        params["startTime"] = start_ms
    r = requests.get(f"{BINANCE_BASE}/api/v3/klines", params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def load_eth_binance(interval: str = "1d", pages: int = 4) -> pd.DataFrame:
    """Fetch up to `pages * 1000` ETH/USDT candles from Binance."""
    raw = []
    start = None
    for p in range(pages):
        rows = _binance_klines("ETHUSDT", interval, 1000, start)
        if not rows:
            break
        raw.extend(rows)
        start = int(rows[-1][0]) + 1
        print(f"[data] ETH page {p+1}: {len(raw)} candles")
        if len(rows) < 1000:
            break

    df = pd.DataFrame(raw, columns=[
        "ts","open","high","low","close","volume",
        "close_ts","qav","num_trades","tbbav","tbqav","ignore"
    ])[["ts","open","high","low","close","volume"]]
    df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms", utc=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c])
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    # Also fetch BTC for ratio feature
    btc_rows = []
    start = None
    for p in range(pages):
        rows = _binance_klines("BTCUSDT", interval, 1000, start)
        if not rows:
            break
        btc_rows.extend(rows)
        start = int(rows[-1][0]) + 1
        if len(rows) < 1000:
            break
    btc_df = pd.DataFrame(btc_rows, columns=[
        "ts","open","high","low","close","volume",
        "close_ts","qav","num_trades","tbbav","tbqav","ignore"
    ])[["ts","close"]].rename(columns={"close": "btc_close"})
    btc_df["ts"] = pd.to_datetime(btc_df["ts"].astype(int), unit="ms", utc=True)
    btc_df["btc_close"] = pd.to_numeric(btc_df["btc_close"])
    btc_df = btc_df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    df = df.merge(btc_df, on="ts", how="left")
    df["eth_btc_ratio"] = df["close"] / df["btc_close"].replace(0, np.nan)
    print(f"[data] ETH loaded: {len(df):,} rows | "
          f"{df['ts'].min().date()} → {df['ts'].max().date()}")
    return df
