"""
backtest.py — Walk-forward validation + full performance metrics.

Walk-forward protocol:
  - Expanding window (train grows, test = next N bars)
  - Strict time-series order (no shuffle)
  - Trades only when confidence >= CONF_THRESHOLD

Metrics reported:
  - Accuracy, Precision, Recall, F1 (on tradeable signals only)
  - Sharpe ratio, Max drawdown, Total return
  - Win rate, Trade count, No-trade filter rate
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional


# ─── Walk-Forward Split ────────────────────────────────────────────────────────
def walk_forward_splits(n: int, min_train: int = 2000,
                        test_size: int = 500,
                        step: int = 250) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate (train_idx, test_idx) pairs for expanding-window walk-forward.

    Args:
        n:          Total number of samples.
        min_train:  Minimum training set size before first test.
        test_size:  Number of bars in each test window.
        step:       Step size between successive test windows.

    Returns:
        List of (train_indices, test_indices) tuples.
    """
    splits = []
    start = min_train
    while start + test_size <= n:
        train = np.arange(0, start)
        test  = np.arange(start, min(start + test_size, n))
        splits.append((train, test))
        start += step
    return splits


# ─── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    returns_on_trade: Optional[np.ndarray] = None,
                    label: str = "") -> dict:
    """
    Compute full set of classification + financial metrics.

    Args:
        y_true:            True labels (-1, +1). 0 rows should be excluded.
        y_pred:            Predicted labels (-1, 0, +1). 0 = no-trade.
        returns_on_trade:  Actual future returns aligned with y_true rows.
        label:             String prefix for printing.
    """
    # Filter to tradeable signals only
    trade_mask = y_pred != 0
    n_trade = trade_mask.sum()
    n_total = len(y_true)
    filter_rate = 1 - n_trade / max(n_total, 1)

    if n_trade == 0:
        print(f"[{label}] NO TRADES — confidence threshold too high or no signals.")
        return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0,
                "n_trades": 0, "filter_rate": filter_rate,
                "sharpe": 0.0, "max_drawdown": 0.0, "total_return": 0.0}

    yt = y_true[trade_mask]
    yp = y_pred[trade_mask]

    # Map -1/+1 → 0/1 for sklearn
    yt_bin = (yt == 1).astype(int)
    yp_bin = (yp == 1).astype(int)

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    acc  = accuracy_score(yt_bin, yp_bin)
    f1   = f1_score(yt_bin, yp_bin, zero_division=0)
    prec = precision_score(yt_bin, yp_bin, zero_division=0)
    rec  = recall_score(yt_bin, yp_bin, zero_division=0)

    # Financial metrics
    sharpe, max_dd, total_ret = 0.0, 0.0, 0.0
    if returns_on_trade is not None:
        r = returns_on_trade[trade_mask]
        trade_returns = np.where(yp == 1, r, -r)   # long if bull, short if bear
        sharpe    = _sharpe(trade_returns)
        max_dd    = _max_drawdown(trade_returns)
        total_ret = float(np.sum(trade_returns))

    print(f"\n{'─'*50}")
    print(f"  [{label}] Walk-Forward Results")
    print(f"{'─'*50}")
    print(f"  Trades:       {n_trade}/{n_total} ({100*n_trade/n_total:.1f}% after filter)")
    print(f"  Confidence filter removed: {100*filter_rate:.1f}% of signals")
    print(f"  Accuracy:     {100*acc:.2f}%")
    print(f"  F1 Score:     {f1:.4f}")
    print(f"  Precision:    {prec:.4f}")
    print(f"  Recall:       {rec:.4f}")
    if returns_on_trade is not None:
        print(f"  Sharpe:       {sharpe:.3f}")
        print(f"  Max Drawdown: {100*max_dd:.2f}%")
        print(f"  Total Return: {100*total_ret:.2f}%")
    print(f"{'─'*50}\n")

    return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec,
            "n_trades": n_trade, "filter_rate": filter_rate,
            "sharpe": sharpe, "max_drawdown": max_dd, "total_return": total_ret}


def _sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / (returns.std() + 1e-9))


def _max_drawdown(returns: np.ndarray) -> float:
    equity = np.cumprod(1 + np.clip(returns, -0.5, 0.5))
    roll_max = np.maximum.accumulate(equity)
    dd = (equity - roll_max) / (roll_max + 1e-9)
    return float(np.min(dd))


# ─── Full Walk-Forward Evaluation ─────────────────────────────────────────────
def walk_forward_evaluate(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "label",
    suffix: str = "model",
    regime_ensembles: Optional[dict] = None,
    min_train: int = 2000,
    test_size: int = 500,
    step: int = 250,
    conf_threshold: float = 0.60,
) -> dict:
    """
    Run full walk-forward evaluation across the dataset.

    Args:
        df:               DataFrame with features + labels + regime + future_return.
        feature_cols:     Feature column names.
        label_col:        Column with -1/0/+1 labels.
        suffix:           Prefix for saved models.
        regime_ensembles: Pre-fitted {regime: RegimeEnsemble}. If None, trains fresh.
        min_train:        Minimum bars for first training window.
        test_size:        Test window size.
        step:             Step size between windows.
        conf_threshold:   Probability threshold for trading.

    Returns:
        dict of aggregated metrics across all folds.
    """
    from .model import RegimeEnsemble
    from .features import _get_feature_cols

    df = df.dropna(subset=feature_cols + [label_col]).reset_index(drop=True)
    # Only binary labels for walk-forward (exclude 0-class)
    df = df[df[label_col] != 0].reset_index(drop=True)
    n = len(df)
    splits = walk_forward_splits(n, min_train, test_size, step)
    print(f"[backtest] Walk-forward: {len(splits)} folds, "
          f"min_train={min_train}, test_size={test_size}")

    all_y_true, all_y_pred, all_returns = [], [], []

    for fold, (tr_idx, te_idx) in enumerate(splits):
        tr_df = df.iloc[tr_idx]
        te_df = df.iloc[te_idx]

        # Train per-regime ensembles
        regimes = tr_df["regime"].unique() if "regime" in tr_df.columns else [0]
        fold_ensembles: dict[int, RegimeEnsemble] = {}
        for reg in regimes:
            sub = tr_df[tr_df["regime"] == reg] if "regime" in tr_df.columns else tr_df
            if len(sub) < 50:
                continue
            ens = RegimeEnsemble(regime_label=int(reg))
            ens.fit(sub[feature_cols], sub[label_col])
            fold_ensembles[int(reg)] = ens

        # Predict on test set — dispatch per regime
        fold_preds = pd.Series(0, index=te_df.index)
        for reg, ens in fold_ensembles.items():
            if not ens.is_fitted:
                continue
            mask = (te_df["regime"] == reg) if "regime" in te_df.columns else pd.Series(True, index=te_df.index)
            sub = te_df[mask]
            if len(sub) == 0:
                continue
            result = ens.predict_with_confidence(sub[feature_cols], conf_threshold)
            fold_preds.loc[sub.index] = result["pred"].values

        all_y_true.extend(te_df[label_col].values)
        all_y_pred.extend(fold_preds.reindex(te_df.index).values)
        if "future_return" in te_df.columns:
            all_returns.extend(te_df["future_return"].values)

        fold_acc = accuracy_score(
            (pd.Series(all_y_true[-len(te_idx):]) == 1).astype(int),
            (pd.Series(all_y_pred[-len(te_idx):]) == 1).astype(int)
        ) if (pd.Series(all_y_pred[-len(te_idx):]) != 0).any() else 0.0
        print(f"  Fold {fold+1:2d}/{len(splits)}: "
              f"test={len(te_idx)}, acc={100*fold_acc:.1f}%, "
              f"regime_models={list(fold_ensembles.keys())}")

    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    returns = np.array(all_returns) if all_returns else None

    return compute_metrics(y_true, y_pred, returns, label=suffix.upper())


from sklearn.metrics import accuracy_score
