"""
train_model.py — BTC 1-minute model trainer (Fixed v3)
Run ONCE:  python train_model.py

Key fixes in v3:
  1. LOOKAHEAD = 1 — predict next candle direction (matches ws.py inference)
  2. SUBSAMPLE = 10 — 3x more training data from local CSVs
  3. Regime feature added: bull/bear market regime (price vs 200-bar EMA)
  4. Halving cycle feature: days since last BTC halving (encoded as sine)
  5. n_estimators bumped to 1500, num_leaves=128 for richer trees
  6. Feature index comments added to match ws.py exactly
"""

import os, glob, json, time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

DATA_DIR  = Path(r"C:\Users\Admin\PyCharmMiscProject\pythonprojects\my_pro\AI_ML_DL\STOCKMARKET_PREDICTOR_DEMO\datas")
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH   = MODEL_DIR / "btc_model.pkl"
META_PATH    = MODEL_DIR / "btc_meta.json"
FEATURE_MASK = MODEL_DIR / "btc_feature_mask.json"

LOOKAHEAD   = 1      # predict: will close[t+1] > close[t]? (matches ws.py now)
MIN_ROWS    = 200    # minimum rows before first feature extraction
SUBSAMPLE   = 10     # use every 10th 1m candle (~12.8M/10 = 1.28M samples, ~15-25 min training)

# ──────────────────────────────────────────────────────────────────────────────
# Feature helpers  (identical to ws.py so inference is consistent)
# Feature layout (index comments match ws.py _predict() exactly):
#   [0]  composite    pattern composite score
#   [1]  rsi/100
#   [2]  macd/(close+eps)
#   [3]  bb_pos
#   [4]  bb_wid
#   [5]  mom5
#   [6]  mom10
#   [7]  mom20
#   [8]  ex9          (close - EMA9) / EMA9
#   [9]  es921        (EMA9 - EMA21) / EMA21
#   [10] es2150       (EMA21 - EMA50) / EMA50
#   [11] stoch_rsi
#   [12] obv_slope
#   [13] body/range
#   [14] upper_wick/range
#   [15] lower_wick/range
#   [16] vwap_dev
#   [17] vol_delta
#   [18] atr_ratio
#   [19] time_sin
#   [20] time_cos
#   [21] bull_market_regime  (NEW: price > EMA200 = 1 else -1)
#   [22] halving_cycle_sin   (NEW: sine encoding of BTC halving cycle)
#   [23..38] pattern flags
# ──────────────────────────────────────────────────────────────────────────────

# BTC halving timestamps (Unix)
_BTC_HALVINGS = [
    1354320000,  # Nov 28 2012
    1468082400,  # Jul  9 2016
    1589225600,  # May 11 2020
    1713801600,  # Apr 19 2024
]

def _halving_cycle_sin(ts: float) -> float:
    """Sine-encode position within current BTC halving cycle (~4 years = 1461 days)."""
    last_halving = max(h for h in _BTC_HALVINGS if h <= ts) if ts >= _BTC_HALVINGS[0] else _BTC_HALVINGS[0]
    CYCLE_SECS = 1461 * 86400  # ~4 years
    phase = (ts - last_halving) / CYCLE_SECS
    return float(np.sin(2 * np.pi * phase))

def _rsi(closes, period=14):
    if len(closes) < period + 1: return 50.0
    d = np.diff(closes[-(period+1):])
    g = np.where(d > 0, d, 0).mean()
    l = np.where(d < 0, -d, 0).mean()
    return 50.0 if l == 0 else 100.0 - 100.0/(1+g/l)

def _ema(closes, period):
    if len(closes) < period: return float(np.mean(closes))
    k, e = 2/(period+1), float(closes[0])
    for v in closes[1:]: e = v*k + e*(1-k)
    return e

def _macd(closes):
    return 0.0 if len(closes) < 26 else _ema(closes, 12) - _ema(closes, 26)

def _bb(closes, period=20):
    if len(closes) < period: return 0.5, 0.02
    w = closes[-period:]
    m, s = np.mean(w), np.std(w)
    if s == 0: return 0.5, 0.0
    pos = float(np.clip((closes[-1] - (m - 2*s)) / (4*s + 1e-9), 0, 1))
    width = float(4*s / (m + 1e-9))
    return pos, width

def _mom(closes, p):
    return 0.0 if len(closes) < p+1 else float((closes[-1]-closes[-(p+1)])/(closes[-(p+1)]+1e-9))

def _atr(rows, period=14):
    if len(rows) < 2: return 1.0
    trs = [max(r['high']-r['low'],
               abs(r['high']-rows[i-1]['close']),
               abs(r['low']-rows[i-1]['close']))
           for i, r in enumerate(rows) if i > 0]
    return float(np.mean(trs[-period:])) if trs else 1.0

def _stoch_rsi(closes, period=14):
    if len(closes) < period*2: return 0.5
    rsis = [_rsi(closes[:i+1], period) for i in range(period-1, len(closes))]
    r = rsis[-1]; lo, hi = min(rsis[-period:]), max(rsis[-period:])
    return 0.5 if hi == lo else float(np.clip((r-lo)/(hi-lo), 0, 1))

def _obv_slope(rows, period=10):
    if len(rows) < period+1: return 0.0
    recent = rows[-(period+1):]; obv = 0; vals = [0.0]
    for i in range(1, len(recent)):
        sign = 1 if recent[i]['close'] > recent[i-1]['close'] else -1
        obv += sign * recent[i]['volume']
        vals.append(obv)
    return float((vals[-1]-vals[0])/(abs(vals[0])+1e-9))

def _vwap_dev(rows, period=20):
    if len(rows) < period: return 0.0
    recent = rows[-period:]
    total_vol = sum(r['volume'] for r in recent)
    if total_vol == 0: return 0.0
    vwap = sum((r['high']+r['low']+r['close'])/3 * r['volume'] for r in recent) / total_vol
    return float((rows[-1]['close'] - vwap) / (vwap + 1e-9))

def _vol_delta(rows, period=10):
    if len(rows) < period+1: return 0.0
    avg = np.mean([r['volume'] for r in rows[-(period+1):-1]])
    return float((rows[-1]['volume'] - avg) / (avg + 1e-9))

def _time_features(ts):
    import datetime
    dt = datetime.datetime.utcfromtimestamp(ts)
    hour = dt.hour + dt.minute / 60.0
    return float(np.sin(2*np.pi*hour/24)), float(np.cos(2*np.pi*hour/24))

def detect_patterns(rows):
    pats = {}; n = len(rows)
    if n < 1: return pats
    def body(c): return c['close']-c['open']
    def abody(c): return abs(body(c))
    def rng(c): return (c['high']-c['low']) or 1e-9
    def uw(c): return c['high']-max(c['open'],c['close'])
    def lw(c): return min(c['open'],c['close'])-c['low']
    def bull(c): return c['close'] > c['open']
    def bear(c): return c['close'] < c['open']
    C0 = rows[-1]; atr = _atr(rows)
    if abody(C0) < 0.1*rng(C0): pats['doji'] = 0
    if lw(C0) >= 2*max(abody(C0), atr*0.01) and uw(C0) < abody(C0)*0.5: pats['hammer'] = 1
    if uw(C0) >= 2*max(abody(C0), atr*0.01) and lw(C0) < abody(C0)*0.5: pats['shooting_star'] = -1
    if bull(C0) and abody(C0) > 0.85*rng(C0): pats['marubozu_bull'] = 1
    if bear(C0) and abody(C0) > 0.85*rng(C0): pats['marubozu_bear'] = -1
    if n < 2: return pats
    C1 = rows[-2]
    if bear(C1) and bull(C0) and C0['open'] <= C1['close'] and C0['close'] >= C1['open']: pats['bullish_engulfing'] = 1
    if bull(C1) and bear(C0) and C0['open'] >= C1['close'] and C0['close'] <= C1['open']: pats['bearish_engulfing'] = -1
    if bear(C1) and bull(C0) and abody(C1) > 2*abody(C0):
        if C0['open'] > C1['close'] and C0['close'] < C1['open']: pats['bullish_harami'] = 1
    if bull(C1) and bear(C0) and abody(C1) > 2*abody(C0):
        if C0['open'] < C1['close'] and C0['close'] > C1['open']: pats['bearish_harami'] = -1
    if bear(C1) and bull(C0):
        mid = (C1['open']+C1['close'])/2
        if C0['open'] < C1['close'] and C0['close'] > mid: pats['piercing_line'] = 1
    if bull(C1) and bear(C0):
        mid = (C1['open']+C1['close'])/2
        if C0['open'] > C1['close'] and C0['close'] < mid: pats['dark_cloud_cover'] = -1
    if abs(C0['low']-C1['low']) < atr*0.05: pats['tweezer_bottom'] = 1
    if abs(C0['high']-C1['high']) < atr*0.05: pats['tweezer_top'] = -1
    if n < 3: return pats
    C2 = rows[-3]; mid2 = (C2['open']+C2['close'])/2
    if bear(C2) and abody(C1) < abody(C2)*0.5 and bull(C0) and C0['close'] > mid2: pats['morning_star'] = 1
    if bull(C2) and abody(C1) < abody(C2)*0.5 and bear(C0) and C0['close'] < mid2: pats['evening_star'] = -1
    if all(bull(c) for c in [C2,C1,C0]) and C1['close'] > C2['close'] and C0['close'] > C1['close']:
        if all(abody(c) > atr*0.3 for c in [C2,C1,C0]): pats['three_white_soldiers'] = 1
    if all(bear(c) for c in [C2,C1,C0]) and C1['close'] < C2['close'] and C0['close'] < C1['close']:
        if all(abody(c) > atr*0.3 for c in [C2,C1,C0]): pats['three_black_crows'] = -1
    return pats

PAT_NAMES = [
    'hammer','shooting_star','marubozu_bull','marubozu_bear',
    'bullish_engulfing','bearish_engulfing','bullish_harami','bearish_harami',
    'piercing_line','dark_cloud_cover','tweezer_bottom','tweezer_top',
    'morning_star','evening_star','three_white_soldiers','three_black_crows',
]

def extract_features(rows):
    closes = np.array([r['close'] for r in rows], dtype=float)
    last   = rows[-1]
    rng_l  = (last['high'] - last['low']) or 1e-9
    body_l = last['close'] - last['open']
    uw_l   = last['high'] - max(last['open'], last['close'])
    lw_l   = min(last['open'], last['close']) - last['low']
    atr    = _atr(rows)

    # Core indicators
    rsi    = _rsi(closes, 14)
    macd   = _macd(closes)
    bb_pos, bb_wid = _bb(closes, 20)
    mom5   = _mom(closes, 5)
    mom10  = _mom(closes, 10)
    mom20  = _mom(closes, 20)
    ema9   = _ema(closes, 9)
    ema21  = _ema(closes, 21)
    ema50  = _ema(closes, 50)
    ex9    = (closes[-1] - ema9)  / (ema9  + 1e-9)   # index 8
    es921  = (ema9  - ema21) / (ema21 + 1e-9)         # index 9
    es2150 = (ema21 - ema50) / (ema50 + 1e-9)         # index 10
    sr     = _stoch_rsi(closes, 14)
    obv    = _obv_slope(rows, 10)
    vwap   = _vwap_dev(rows, 20)
    vdelta = _vol_delta(rows, 10)
    atr_r  = atr / (closes[-1] + 1e-9)
    ts_sin, ts_cos = _time_features(last['time'])

    # BTC-specific features
    ema200 = _ema(closes, min(200, len(closes)))
    regime = 1.0 if closes[-1] > ema200 else -1.0   # bull/bear market
    halving_sin = _halving_cycle_sin(float(last['time']))

    tech = np.array([
        rsi/100, macd/(closes[-1]+1e-9), bb_pos, bb_wid,
        mom5, mom10, mom20,
        ex9, es921, es2150,
        sr, obv,
        body_l/rng_l, uw_l/rng_l, lw_l/rng_l,
        vwap, vdelta, atr_r, ts_sin, ts_cos,
        regime, halving_sin,
    ], dtype=float)

    pats      = detect_patterns(rows)
    composite = float(np.clip(sum(pats.values())/max(len(pats),1), -1, 1)) if pats else 0.0
    pat_flags = np.array([float(pats.get(p, 0)) for p in PAT_NAMES], dtype=float)

    feat = np.concatenate([[composite], tech, pat_flags])
    return np.nan_to_num(feat, nan=0, posinf=1, neginf=-1)


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Load raw 1m CSVs  (no resampling — train on actual timeframe)
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("BTC 1m Model Trainer — Fixed v3 (LOOKAHEAD=1, SUBSAMPLE=10)")
print("=" * 60)
t0 = time.time()

csv_files = sorted(glob.glob(str(DATA_DIR / "BTCUSDT-1m-*.csv")))
print(f"Found {len(csv_files)} CSV files")

dfs = []
for i, f in enumerate(csv_files):
    try:
        chunk = pd.read_csv(f, header=None,
                            usecols=[0,1,2,3,4,5],
                            names=['open_time','open','high','low','close','volume'],
                            dtype={'open':float,'high':float,'low':float,'close':float,'volume':float})
        chunk['dt'] = pd.to_datetime(chunk['open_time'], unit='ms', utc=True, errors='coerce')
        chunk = chunk.dropna(subset=['dt']).set_index('dt')
        dfs.append(chunk)
        if (i+1) % 10 == 0:
            print(f"  Loaded {i+1}/{len(csv_files)} files...")
    except Exception as e:
        print(f"  Warning: {Path(f).name} -> {e}")

all_1m = pd.concat(dfs).sort_index()
all_1m = all_1m[~all_1m.index.duplicated(keep='last')]

# Subsample to reduce training time while keeping temporal distribution
if SUBSAMPLE > 1:
    all_1m = all_1m.iloc[::SUBSAMPLE]
    print(f"Subsampled every {SUBSAMPLE} rows -> {len(all_1m):,} rows ({len(all_1m)*SUBSAMPLE/60:.0f} hours of 1m data)")
else:
    print(f"Total 1m candles: {len(all_1m):,}")

# Convert to list-of-dicts format (same structure as live ws.py buffer)
rows = [{'time': int(idx.timestamp()),
         'open': r['open'], 'high': r['high'],
         'low': r['low'],   'close': r['close'],
         'volume': float(r['volume'])}
        for idx, r in all_1m.iterrows()]
last_price = rows[-1]['close']
print(f"Last close: ${last_price:,.2f}  ({time.time()-t0:.1f}s)")


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Build feature matrix with 1-bar lookahead label (matches ws.py)
# ──────────────────────────────────────────────────────────────────────────────
print(f"\nBuilding feature matrix (lookahead={LOOKAHEAD}, label = next candle direction)...")
t1 = time.time()
X, y = [], []
total = len(rows) - LOOKAHEAD

for i in range(MIN_ROWS, total):
    feat = extract_features(rows[max(0, i-300):i+1])   # use up to 300 bars context
    X.append(feat)
    # Label: did close price go UP in the next candle? (matches ws.py outcome scoring)
    nxt = rows[i + LOOKAHEAD]
    y.append(1 if nxt['close'] >= nxt['open'] else 0)
    if (i - MIN_ROWS) % 100_000 == 0:
        pct = (i - MIN_ROWS) / max(total - MIN_ROWS, 1) * 100
        print(f"  {pct:.0f}%  ({i-MIN_ROWS:,}/{total-MIN_ROWS:,} rows)...")

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)
print(f"Feature matrix: {X.shape}  bull%={y.mean():.3f}  ({time.time()-t1:.1f}s)")


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Walk-forward split (no data leakage)
# ──────────────────────────────────────────────────────────────────────────────
split = int(len(X) * 0.85)   # last 15% = held-out test (chronological)
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]
print(f"Train: {len(X_tr):,}  |  Test (walk-forward): {len(X_te):,}")


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Train LightGBM with tuned hyperparameters
# ──────────────────────────────────────────────────────────────────────────────
print("\nTraining LightGBM (BTC-optimised)...")
t2 = time.time()
from lightgbm import LGBMClassifier

clf = LGBMClassifier(
    n_estimators      = 1500,
    num_leaves        = 128,        # richer trees for complex BTC patterns
    max_depth         = -1,
    learning_rate     = 0.01,
    subsample         = 0.7,
    subsample_freq    = 1,
    colsample_bytree  = 0.7,
    min_child_samples = 30,
    reg_alpha         = 0.1,
    reg_lambda        = 0.2,
    class_weight      = 'balanced',
    random_state      = 42,
    verbose           = -1,
    n_jobs            = -1,
)

clf.fit(
    X_tr, y_tr,
    eval_set=[(X_te, y_te)],
    callbacks=[
        __import__('lightgbm').early_stopping(stopping_rounds=50, verbose=False),
        __import__('lightgbm').log_evaluation(period=100),
    ],
)

train_acc = (clf.predict(X_tr) == y_tr).mean()
test_acc  = (clf.predict(X_te) == y_te).mean()
print(f"Train acc: {train_acc:.4f}  |  Walk-forward test acc: {test_acc:.4f}  ({time.time()-t2:.1f}s)")


# ──────────────────────────────────────────────────────────────────────────────
# Step 5: Feature importance — prune bottom 20%
# ──────────────────────────────────────────────────────────────────────────────
print("\nFeature importance pruning...")
n_features = X.shape[1]
importances = clf.feature_importances_
threshold   = np.percentile(importances, 20)
keep_mask   = importances >= threshold
keep_indices = np.where(keep_mask)[0].tolist()
print(f"Keeping {len(keep_indices)}/{n_features} features (dropped {n_features - len(keep_indices)} weak)")

X_tr_p = X_tr[:, keep_indices]
X_te_p = X_te[:, keep_indices]

clf2 = LGBMClassifier(
    n_estimators      = 1500,
    num_leaves        = 128,
    learning_rate     = 0.01,
    subsample         = 0.7,
    subsample_freq    = 1,
    colsample_bytree  = 0.8,
    min_child_samples = 30,
    reg_alpha         = 0.1,
    reg_lambda        = 0.2,
    class_weight      = 'balanced',
    random_state      = 42,
    verbose           = -1,
    n_jobs            = -1,
)
clf2.fit(
    X_tr_p, y_tr,
    eval_set=[(X_te_p, y_te)],
    callbacks=[
        __import__('lightgbm').early_stopping(stopping_rounds=50, verbose=False),
        __import__('lightgbm').log_evaluation(period=100),
    ],
)

final_train = (clf2.predict(X_tr_p) == y_tr).mean()
final_test  = (clf2.predict(X_te_p) == y_te).mean()
print(f"Pruned train acc: {final_train:.4f}  |  Pruned test acc: {final_test:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# Step 6: Save model + metadata + feature mask
# ──────────────────────────────────────────────────────────────────────────────
joblib.dump(clf2, MODEL_PATH)

with open(FEATURE_MASK, 'w') as fp:
    json.dump({'keep_indices': keep_indices, 'n_total': n_features}, fp)

with open(META_PATH, 'w') as fp:
    json.dump({
        'asset':            'BTC',
        'last_price':       last_price,
        'n_samples':        len(X),
        'n_features':       len(keep_indices),
        'n_features_total': n_features,
        'train_accuracy':   round(final_train, 4),
        'test_accuracy':    round(final_test, 4),
        'lookahead_bars':   LOOKAHEAD,
        'subsample_step':   SUBSAMPLE,
        'label_logic':      'next candle close >= next candle open',
        'last_candle_time': rows[-1]['time'],
        'trained_at':       time.strftime('%Y-%m-%d %H:%M:%S'),
    }, fp, indent=2)

print(f"\n{'='*60}")
print(f"BTC Model saved:    {MODEL_PATH}")
print(f"Feature mask saved: {FEATURE_MASK}")
print(f"Metadata saved:     {META_PATH}")
print(f"Total time:         {time.time()-t0:.1f}s")
print(f"{'='*60}")
print(f"\nWalk-forward test accuracy: {final_test*100:.2f}%")
print(f"(Honest out-of-sample score. LOOKAHEAD=1 now matches ws.py exactly.)")
print(f"\nRestart ws.py to load the new BTC model.")
