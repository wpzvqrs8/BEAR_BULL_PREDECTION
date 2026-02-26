"""
regime.py — Market regime detection.

Classifies each bar into:
  0 = Ranging / low-trend / low-volatility
  1 = Trending up
 -1 = Trending down

Uses:
  - ADX > 25 → trending (direction from DI+/DI-)
  - ATR percentile for volatility clustering
  - ROC for momentum confirmation
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def detect_regime(df: pd.DataFrame,
                  adx_threshold: float = 25.0,
                  use_clustering: bool = True,
                  n_clusters: int = 3) -> pd.DataFrame:
    """
    Add 'regime' column to df.

    Regimes:
        1  = Strong uptrend      (ADX > threshold AND DI+ > DI-)
       -1  = Strong downtrend    (ADX > threshold AND DI- > DI+)
        0  = Ranging / sideways  (ADX < threshold)

    When use_clustering=True, K-Means on [ADX, vol_expand, atr_pct] is used
    to label each cluster as trending or ranging, providing a data-driven split.

    Args:
        df:             DataFrame with adx14, di_diff, vol_expand, atr_pct columns.
        adx_threshold:  ADX level separating trending from ranging.
        use_clustering: Whether to additionally apply KMeans regime labeling.
        n_clusters:     Number of regime clusters for KMeans.

    Returns:
        df with 'regime' column.
    """
    df = df.copy()

    if "adx14" not in df.columns:
        raise ValueError("df must have 'adx14' and 'di_diff' — run features.build_features() first")

    adx  = df["adx14"] * 100      # scale back to 0–100
    di_d = df["di_diff"] * 100    # DI+ - DI-

    # Rule-based base regime
    df["regime"] = 0
    df.loc[(adx > adx_threshold) & (di_d > 0), "regime"]  =  1
    df.loc[(adx > adx_threshold) & (di_d < 0), "regime"]  = -1

    if use_clustering and all(c in df.columns for c in ["vol_expand", "atr_pct"]):
        _add_volatility_cluster(df, n_clusters)

    counts = df["regime"].value_counts()
    print(f"[regime] Trending up:   {counts.get(1,0):6d} "
          f"({100*counts.get(1,0)/len(df):.1f}%)")
    print(f"[regime] Ranging:       {counts.get(0,0):6d} "
          f"({100*counts.get(0,0)/len(df):.1f}%)")
    print(f"[regime] Trending down: {counts.get(-1,0):6d} "
          f"({100*counts.get(-1,0)/len(df):.1f}%)")
    return df


def _add_volatility_cluster(df: pd.DataFrame, n_clusters: int = 3) -> None:
    """
    Add 'vol_regime' column using KMeans on volatility features.
    High-vol cluster: vol_regime = 1
    Low-vol cluster:  vol_regime = 0
    This is stored separately and used for model selection, not overwriting regime.
    """
    feat_cols = [c for c in ["adx14","vol_expand","atr_pct","skew20","bb_wid"]
                 if c in df.columns]
    sub = df[feat_cols].dropna()
    if len(sub) < n_clusters * 10:
        return

    scaler = StandardScaler()
    X = scaler.fit_transform(sub)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    km.fit(X)
    labels = pd.Series(km.labels_, index=sub.index)

    # Map cluster → high/low vol using mean ADX
    cluster_adx = {c: sub.loc[labels == c, feat_cols[0]].mean() for c in range(n_clusters)}
    high_vol_cluster = max(cluster_adx, key=cluster_adx.get)
    df.loc[labels.index, "vol_regime"] = (labels == high_vol_cluster).astype(int)
    df["vol_regime"].fillna(0, inplace=True)


def split_by_regime(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Split df into sub-DataFrames by regime label."""
    if "regime" not in df.columns:
        raise ValueError("Run detect_regime() first.")
    return {r: df[df["regime"] == r].copy() for r in df["regime"].unique()}
