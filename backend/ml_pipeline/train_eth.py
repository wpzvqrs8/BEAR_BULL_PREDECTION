"""
train_eth.py — ETH training entry point.

Run: python -m ml_pipeline.train_eth

Same pipeline as BTC but uses Binance API (daily candles) and ETH/BTC ratio feature.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_pipeline.data      import load_eth_binance
from ml_pipeline.features  import build_features, _get_feature_cols
from ml_pipeline.labeling  import add_labels, binary_labels
from ml_pipeline.regime    import detect_regime
from ml_pipeline.model     import RegimeEnsemble, compute_shap_importance, MODELS_DIR
from ml_pipeline.backtest  import walk_forward_evaluate

# ── Config ────────────────────────────────────────────────────────────────────
INTERVAL     = "1d"
LOOKAHEAD    = 1
VOL_WINDOW   = 14         # smaller window for daily data (fewer bars)
K            = 0.4        # slightly lower k (ETH more volatile)
CONF_THRESH  = 0.60
MIN_TRAIN    = 500        # ETH daily: fewer bars total than BTC 1h
TEST_SIZE    = 100
STEP         = 50
PREFIX       = "eth_v2"

def main():
    print("=" * 60)
    print("ETH ML Pipeline v2 — Walk-Forward Training")
    print("=" * 60)

    df = load_eth_binance(interval=INTERVAL, pages=4)
    df = build_features(df, drop_correlated=True, corr_threshold=0.85)
    feature_cols = _get_feature_cols(df)
    print(f"\n[train] Features after correlation pruning: {len(feature_cols)}")

    df = add_labels(df, lookahead=LOOKAHEAD, vol_window=VOL_WINDOW, k=K)
    df = detect_regime(df)

    print("\n[train] Running walk-forward evaluation...")
    metrics = walk_forward_evaluate(
        df, feature_cols,
        suffix="ETH",
        min_train=MIN_TRAIN,
        test_size=TEST_SIZE,
        step=STEP,
        conf_threshold=CONF_THRESH,
    )
    print(f"\n[train] Walk-forward accuracy: {100*metrics['accuracy']:.2f}%")
    print(f"[train] Walk-forward Sharpe:   {metrics['sharpe']:.3f}")
    print(f"[train] Max Drawdown:          {100*metrics['max_drawdown']:.2f}%")

    print("\n[train] Final fit on full dataset...")
    df_binary = binary_labels(df)
    for reg in df_binary["regime"].unique():
        sub = df_binary[df_binary["regime"] == reg]
        if len(sub) < 50:
            continue
        ens = RegimeEnsemble(regime_label=int(reg))
        ens.fit(sub[feature_cols], sub["label"])
        if ens.is_fitted:
            imp = compute_shap_importance(ens.lgb_model, sub[feature_cols])
            print(f"\n[train] Top features (regime={reg}):")
            print(imp.to_string(index=False))
            ens.save(PREFIX)

    print(f"\n[train] Done. Models saved to {MODELS_DIR}/")

if __name__ == "__main__":
    main()
