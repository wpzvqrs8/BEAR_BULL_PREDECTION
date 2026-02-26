"""
train_btc.py — BTC training entry point.

Run: python -m ml_pipeline.train_btc

Pipeline:
  1. Load BTC 1-min CSVs, resample to 1h
  2. Build features (corr-pruned)
  3. Add volatility-adjusted labels
  4. Detect regimes (ADX + volatility clustering)
  5. Walk-forward evaluate (prints metrics per fold)
  6. Final fit on full dataset, save per-regime ensemble models
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_pipeline.data      import load_btc_csv
from ml_pipeline.features  import build_features, _get_feature_cols
from ml_pipeline.labeling  import add_labels, binary_labels
from ml_pipeline.regime    import detect_regime
from ml_pipeline.model     import RegimeEnsemble, compute_shap_importance, MODELS_DIR
from ml_pipeline.backtest  import walk_forward_evaluate

# ── Config ────────────────────────────────────────────────────────────────────
RESAMPLE     = "1h"       # 1h candles (better signal/noise than 1m for ML)
LOOKAHEAD    = 1          # predict next 1h candle direction
VOL_WINDOW   = 20         # rolling bars for volatility
K            = 0.5        # threshold multiplier (lower → more labels, more noise)
CONF_THRESH  = 0.60       # minimum predicted probability to trade
MIN_TRAIN    = 2000       # minimum bars for walk-forward first split
TEST_SIZE    = 500        # bars per walk-forward test window
STEP         = 250        # step size between folds
PREFIX       = "btc_v2"  # model save prefix

def main():
    print("=" * 60)
    print("BTC ML Pipeline v2 — Walk-Forward Training")
    print("=" * 60)

    # 1. Data
    df = load_btc_csv(resample=RESAMPLE)

    # 2. Features
    df = build_features(df, drop_correlated=True, corr_threshold=0.85)
    feature_cols = _get_feature_cols(df)
    print(f"\n[train] Features after correlation pruning: {len(feature_cols)}")

    # 3. Labels
    df = add_labels(df, lookahead=LOOKAHEAD, vol_window=VOL_WINDOW, k=K)

    # 4. Regime
    df = detect_regime(df)

    # 5. Walk-forward evaluation
    print("\n[train] Running walk-forward evaluation...")
    metrics = walk_forward_evaluate(
        df, feature_cols,
        suffix="BTC",
        min_train=MIN_TRAIN,
        test_size=TEST_SIZE,
        step=STEP,
        conf_threshold=CONF_THRESH,
    )
    print(f"\n[train] Walk-forward accuracy: {100*metrics['accuracy']:.2f}%")
    print(f"[train] Walk-forward Sharpe:   {metrics['sharpe']:.3f}")
    print(f"[train] Max Drawdown:          {100*metrics['max_drawdown']:.2f}%")

    # 6. Final fit on full dataset, save models
    print("\n[train] Final fit on full dataset...")
    df_binary = binary_labels(df)    # exclude no-trade zone rows
    regimes = df_binary["regime"].unique()
    for reg in regimes:
        sub = df_binary[df_binary["regime"] == reg]
        if len(sub) < 100:
            print(f"[train] Regime {reg}: too few samples ({len(sub)}), skipping.")
            continue
        ens = RegimeEnsemble(regime_label=int(reg))
        ens.fit(sub[feature_cols], sub["label"])
        if ens.is_fitted:
            # SHAP importance
            imp = compute_shap_importance(ens.lgb_model, sub[feature_cols])
            print(f"\n[train] Top features (regime={reg}):")
            print(imp.to_string(index=False))
            ens.save(PREFIX)

    print(f"\n[train] Done. Models saved to {MODELS_DIR}/")
    print("To use in ws.py, set MODEL_PREFIX='btc_v2' in the pipeline config.")

if __name__ == "__main__":
    main()
