"""
train_gold_model.py — GOLD (XAU/USD) dedicated model trainer (yfinance daily)
Run:  python train_gold_model.py

GOLD characteristics:
  - Mon–Fri market (COMEX GC=F), mean-reverting
  - Driven by USD strength (inverse), inflation, risk-off flows
  - Only ~60-90 daily candles available via yfinance free tier
  - Weekly periodicity (Mon/Fri effects at open/close of week)
  - RSI extremes are highly predictive (oversold/overbought revert fast)
  - Bollinger Band width signals volatility regimes

Model approach:
  - Hybrid: LightGBM (if enough data) + rule-based fallback
  - Features tuned for mean-reversion + weekly patterns
  - LOOKAHEAD = 1 (next trading day direction)
"""
import json, time
import numpy as np
import joblib
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "gold_model.pkl"
META_PATH  = MODEL_DIR / "gold_meta.json"

LOOKAHEAD = 1
MIN_ROWS  = 20

print("=" * 60)
print("GOLD Daily Model Trainer v1 (yfinance GC=F)")
print("=" * 60)
t0 = time.time()


def fetch_gold_historical(days: int = 250) -> list:
    """Fetch GOLD daily candles. Tries GC=F, GLD, XAUUSD=X with multiple periods."""
    import yfinance as yf

    tickers   = ["GC=F", "GLD", "XAUUSD=X"]
    periods   = ["2y", "1y", "6mo", "max"]

    for sym in tickers:
        for period in periods:
            try:
                hist = yf.Ticker(sym).history(period=period, interval="1d")
                if hist is None or hist.empty:
                    continue
                candles = []
                for ts, row in hist.iterrows():
                    c = float(row.get("Close", 0) or 0)
                    if c <= 0:
                        continue
                    candles.append({
                        "time":   int(ts.timestamp()),
                        "open":   round(float(row.get("Open",  c)), 2),
                        "high":   round(float(row.get("High",  c)), 2),
                        "low":    round(float(row.get("Low",   c)), 2),
                        "close":  round(c, 2),
                        "volume": round(float(row.get("Volume", 0) or 0), 2),
                    })
                if candles:
                    print(f"  [{sym}] {period}: {len(candles)} candles, last=${candles[-1]['close']}")
                    return candles
            except Exception as e:
                print(f"  [{sym}] {period}: {e}")
    return []


print("Fetching GOLD historical from yfinance (GC=F, max 2 years)...")
gold_rows = fetch_gold_historical(730)

if not gold_rows:
    print("ERROR: Could not fetch GOLD data. Ensure yfinance is installed: pip install yfinance")
    exit(1)

print(f"GOLD candles: {len(gold_rows)}")
last_price = gold_rows[-1]['close']
print(f"Last GOLD close: ${last_price:,.2f}  ({time.time()-t0:.1f}s)")


def _rsi(c, p=14):
    if len(c) < p+1: return 50.0
    d = np.diff(c[-(p+1):]); g = np.where(d>0,d,0).mean(); l = np.where(d<0,-d,0).mean()
    return 50.0 if l==0 else 100.0-100.0/(1+g/l)

def _ema(c, p):
    if len(c)<p: return float(np.mean(c))
    k, e = 2/(p+1), float(c[0])
    for v in c[1:]: e = v*k + e*(1-k)
    return e

def _bb(c, p=20):
    if len(c)<p: return 0.5, 0.02
    w = c[-p:]; m, s = np.mean(w), np.std(w)
    if s==0: return 0.5, 0.0
    return float(np.clip((c[-1]-(m-2*s))/(4*s+1e-9),0,1)), float(4*s/(m+1e-9))

def _mom(c, p):
    return 0.0 if len(c)<p+1 else float((c[-1]-c[-(p+1)])/(c[-(p+1)]+1e-9))

def _atr(rows, p=14):
    if len(rows)<2: return 1.0
    trs = [max(r['high']-r['low'], abs(r['high']-rows[i-1]['close']),
               abs(r['low']-rows[i-1]['close'])) for i,r in enumerate(rows) if i>0]
    return float(np.mean(trs[-p:])) if trs else 1.0

def _vol_delta(rows, p=10):
    if len(rows)<p+1: return 0.0
    avg = np.mean([r['volume'] for r in rows[-(p+1):-1]])
    return float((rows[-1]['volume']-avg)/(avg+1e-9))

def _day_of_week(ts):
    import datetime
    wd = datetime.datetime.utcfromtimestamp(ts).weekday()
    return float(np.sin(2*np.pi*wd/5)), float(np.cos(2*np.pi*wd/5))  # 5-day week

def _week_of_month(ts):
    import datetime
    d = datetime.datetime.utcfromtimestamp(ts)
    return float((d.day - 1) // 7) / 4.0   # 0..1 within month

def extract_gold_features(rows):
    """
    GOLD-specific feature set. Mean-reversion focused.
    Layout:
      [0]  rsi/100
      [1]  rsi_extreme  (|rsi-50|/50, high = reversal expected)
      [2]  bb_pos
      [3]  bb_wid
      [4]  mom3  (3-day)
      [5]  mom5  (5-day)
      [6]  mom10 (10-day)
      [7]  ex9   (close vs EMA9)
      [8]  es921  (EMA9 vs EMA21)
      [9]  atr_ratio
      [10] vol_delta
      [11] body/range
      [12] upper_wick/range
      [13] lower_wick/range
      [14] day_sin
      [15] day_cos
      [16] week_of_month
      [17] bull_regime  (close > EMA20)
      [18] distance_from_52w_high  (mean-reversion signal)
      [19] distance_from_52w_low
    """
    c = np.array([r['close'] for r in rows], dtype=float)
    last = rows[-1]
    rng_l = (last['high']-last['low']) or 1e-9
    body_l = last['close']-last['open']
    uw_l = last['high']-max(last['open'],last['close'])
    lw_l = min(last['open'],last['close'])-last['low']
    atr = _atr(rows)
    ema9 = _ema(c,9); ema21 = _ema(c,21); ema20 = _ema(c,20)
    bb_pos, bb_wid = _bb(c,20)
    rsi_val = _rsi(c)
    rsi_extreme = abs(rsi_val-50)/50.0
    ex9    = (c[-1]-ema9)/(ema9+1e-9)
    es921  = (ema9-ema21)/(ema21+1e-9)
    day_sin, day_cos = _day_of_week(last['time'])
    wom = _week_of_month(last['time'])
    regime = 1.0 if c[-1]>ema20 else -1.0
    # 52-week anchor (mean-reversion: far from high = buy signal, far from low = sell)
    lookback = min(252, len(c))
    hi52 = float(np.max(c[-lookback:]))
    lo52 = float(np.min(c[-lookback:]))
    dist_hi = (hi52 - c[-1]) / (hi52 + 1e-9)   # > 0 means below 52w high = buy bias
    dist_lo = (c[-1] - lo52) / (c[-1] + 1e-9)  # > 0 means above 52w low = sell bias

    feat = np.array([
        rsi_val/100, rsi_extreme, bb_pos, bb_wid,
        _mom(c,3), _mom(c,5), _mom(c,10),
        ex9, es921, atr/(c[-1]+1e-9),
        _vol_delta(rows), body_l/rng_l, uw_l/rng_l, lw_l/rng_l,
        day_sin, day_cos, wom, regime,
        dist_hi, dist_lo,
    ], dtype=float)
    return np.nan_to_num(feat, nan=0, posinf=1, neginf=-1)


print(f"\nBuilding GOLD feature matrix (LOOKAHEAD={LOOKAHEAD})...")
t1 = time.time()
X, y = [], []
for i in range(MIN_ROWS, len(gold_rows)-LOOKAHEAD):
    X.append(extract_gold_features(gold_rows[:i+1]))
    nxt = gold_rows[i+LOOKAHEAD]
    y.append(1 if nxt['close'] >= nxt['open'] else 0)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)
print(f"Feature matrix: {X.shape}  bull%={y.mean():.3f}  ({time.time()-t1:.1f}s)")

if len(X) < 20 or len(set(y)) < 2:
    print("WARNING: Very few samples. Saving rule-based fallback metadata only.")
    with open(META_PATH,'w') as fp:
        json.dump({'asset':'GOLD','last_price':last_price,'n_samples':len(X),
                   'n_features':20,'train_accuracy':0.0,'test_accuracy':0.0,
                   'lookahead_bars':LOOKAHEAD,'timeframe':'daily',
                   'model_type':'rule_only',
                   'trained_at':time.strftime('%Y-%m-%d %H:%M:%S')}, fp, indent=2)
    print("Saved metadata (rule-only mode). Restart ws.py.")
    exit(0)

split = int(len(X)*0.80)  # 80/20 split (small dataset)
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]
print(f"Train: {len(X_tr)}  |  Test: {len(X_te)}")

from lightgbm import LGBMClassifier
print("\nTraining LightGBM (GOLD-optimised, mean-reversion)...")
t2 = time.time()
clf = LGBMClassifier(
    n_estimators=300,
    num_leaves=16,        # very small to avoid overfitting ~60 samples
    max_depth=4,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_samples=3,
    reg_alpha=0.2,
    reg_lambda=0.3,       # stronger regularisation for tiny dataset
    class_weight='balanced',
    random_state=42,
    verbose=-1,
    n_jobs=-1,
)
clf.fit(X_tr, y_tr)
tr_acc = (clf.predict(X_tr)==y_tr).mean()
te_acc = (clf.predict(X_te)==y_te).mean()
print(f"Train acc: {tr_acc:.4f}  |  Walk-forward test acc: {te_acc:.4f}  ({time.time()-t2:.1f}s)")

joblib.dump(clf, MODEL_PATH)
with open(META_PATH,'w') as fp:
    json.dump({
        'asset':'GOLD','last_price':last_price,
        'n_samples':len(X),'n_features':X.shape[1],
        'train_accuracy':round(tr_acc,4),'test_accuracy':round(te_acc,4),
        'lookahead_bars':LOOKAHEAD,'timeframe':'daily',
        'model_type':'lightgbm_mean_reversion',
        'label_logic':'next candle close >= next candle open',
        'trained_at':time.strftime('%Y-%m-%d %H:%M:%S'),
    }, fp, indent=2)

print(f"\n{'='*60}")
print(f"GOLD Model: {MODEL_PATH}")
print(f"Walk-forward test accuracy: {te_acc*100:.2f}%")
print(f"Total time: {time.time()-t0:.1f}s\nRestart ws.py to load.")
