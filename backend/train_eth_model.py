"""
train_eth_model.py — ETH/USDT dedicated model trainer (daily Binance candles)
Run:  python train_eth_model.py

ETH characteristics:
  - 24/7 trading, follows BTC + DeFi events
  - Daily ETHUSDT from Binance (~1000 candles = ~2.7 years)
  - More mean-reverting at extremes than BTC
  - ETH/BTC relative performance is a strong signal
  - Day-of-week periodicity (ETH has Monday/Friday effects)
"""
import json, time
import numpy as np
import requests
import joblib
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "eth_model.pkl"
META_PATH  = MODEL_DIR / "eth_meta.json"

LOOKAHEAD = 1
MIN_ROWS  = 30

print("=" * 60)
print("ETH Daily Model Trainer v1")
print("=" * 60)
t0 = time.time()


def fetch_binance_daily(symbol: str) -> list:
    url = "https://api.binance.com/api/v3/klines"
    all_candles = []
    end_time_ms = None
    for page in range(5):
        params = {"symbol": symbol, "interval": "1d", "limit": 1000}
        if end_time_ms:
            params["endTime"] = end_time_ms
        try:
            resp = requests.get(url, params=params, timeout=20)
            if not resp.ok:
                break
            rows = resp.json()
            if not rows:
                break
            batch = [{
                "time":   int(r[0]) // 1000,
                "open":   float(r[1]),
                "high":   float(r[2]),
                "low":    float(r[3]),
                "close":  float(r[4]),
                "volume": float(r[5]),
            } for r in rows]
            all_candles = batch + all_candles
            end_time_ms = int(rows[0][0]) - 1
            if len(batch) < 1000:
                break
            print(f"  [{symbol}] page {page+1}: {len(all_candles)} candles")
        except Exception as e:
            print(f"  [{symbol}] error: {e}"); break
    return all_candles


print("Fetching ETHUSDT daily...")
eth_rows = fetch_binance_daily("ETHUSDT")
print("Fetching BTCUSDT daily (for ETH/BTC ratio feature)...")
btc_rows = fetch_binance_daily("BTCUSDT")

if not eth_rows:
    print("ERROR: Could not fetch ETH data."); exit(1)

print(f"ETH: {len(eth_rows)} candles  |  BTC: {len(btc_rows)} candles")
last_price = eth_rows[-1]['close']
btc_close_by_time = {r['time']: r['close'] for r in btc_rows}


def _rsi(c, p=14):
    if len(c) < p+1: return 50.0
    d = np.diff(c[-(p+1):]); g = np.where(d>0,d,0).mean(); l = np.where(d<0,-d,0).mean()
    return 50.0 if l == 0 else 100.0 - 100.0/(1+g/l)

def _ema(c, p):
    if len(c) < p: return float(np.mean(c))
    k, e = 2/(p+1), float(c[0])
    for v in c[1:]: e = v*k + e*(1-k)
    return e

def _bb(c, p=20):
    if len(c) < p: return 0.5, 0.02
    w = c[-p:]; m, s = np.mean(w), np.std(w)
    if s == 0: return 0.5, 0.0
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

def _stoch_rsi(c, p=14):
    if len(c)<p*2: return 0.5
    rsis = [_rsi(c[:i+1],p) for i in range(p-1,len(c))]
    r = rsis[-1]; lo,hi = min(rsis[-p:]),max(rsis[-p:])
    return 0.5 if hi==lo else float(np.clip((r-lo)/(hi-lo),0,1))

def _obv_slope(rows, p=10):
    if len(rows)<p+1: return 0.0
    recent = rows[-(p+1):]; obv=0; vals=[0.0]
    for i in range(1,len(recent)):
        obv += recent[i]['volume'] if recent[i]['close']>recent[i-1]['close'] else -recent[i]['volume']
        vals.append(obv)
    return float((vals[-1]-vals[0])/(abs(vals[0])+1e-9))

def _day_features(ts):
    import datetime
    wd = datetime.datetime.utcfromtimestamp(ts).weekday()
    return float(np.sin(2*np.pi*wd/7)), float(np.cos(2*np.pi*wd/7))

def extract_eth_features(rows, btc_map):
    c = np.array([r['close'] for r in rows], dtype=float)
    last = rows[-1]
    rng_l = (last['high']-last['low']) or 1e-9
    body_l = last['close']-last['open']
    uw_l = last['high']-max(last['open'],last['close'])
    lw_l = min(last['open'],last['close'])-last['low']
    atr = _atr(rows)
    ema9 = _ema(c,9); ema21 = _ema(c,21); ema50 = _ema(c,50)
    ex9    = (c[-1]-ema9)  / (ema9+1e-9)
    es921  = (ema9-ema21)  / (ema21+1e-9)
    es2150 = (ema21-ema50) / (ema50+1e-9)
    bb_pos, bb_wid = _bb(c,20)
    day_sin, day_cos = _day_features(last['time'])
    # ETH/BTC relative performance
    btc_now = btc_map.get(last['time'], 0.0)
    if len(rows)>=6 and btc_now>0:
        prev_ts = rows[-6]['time']
        btc_prev = btc_map.get(prev_ts, btc_now)
        eth_btc_rel = _mom(c,5) - (btc_now-btc_prev)/(btc_prev+1e-9)
    else:
        eth_btc_rel = 0.0
    regime = 1.0 if c[-1]>ema50 else -1.0
    feat = np.array([
        _rsi(c)/100,  (_ema(c,12)-_ema(c,26))/(c[-1]+1e-9),
        bb_pos, bb_wid,
        _mom(c,5), _mom(c,10), _mom(c,20),
        ex9, es921, es2150,
        _stoch_rsi(c), _obv_slope(rows), _vol_delta(rows),
        atr/(c[-1]+1e-9),
        body_l/rng_l, uw_l/rng_l, lw_l/rng_l,
        eth_btc_rel, day_sin, day_cos, regime,
    ], dtype=float)
    return np.nan_to_num(feat, nan=0, posinf=1, neginf=-1)


print(f"\nBuilding ETH feature matrix (LOOKAHEAD={LOOKAHEAD})...")
t1 = time.time()
X, y = [], []
for i in range(MIN_ROWS, len(eth_rows)-LOOKAHEAD):
    X.append(extract_eth_features(eth_rows[:i+1], btc_close_by_time))
    nxt = eth_rows[i+LOOKAHEAD]
    y.append(1 if nxt['close'] >= nxt['open'] else 0)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)
print(f"Feature matrix: {X.shape}  bull%={y.mean():.3f}  ({time.time()-t1:.1f}s)")
if len(set(y)) < 2:
    print("ERROR: Only one class."); exit(1)

split = int(len(X)*0.85)
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]
print(f"Train: {len(X_tr)}  |  Test: {len(X_te)}")

from lightgbm import LGBMClassifier
print("\nTraining LightGBM (ETH-optimised)...")
t2 = time.time()
clf = LGBMClassifier(
    n_estimators=500, num_leaves=32, max_depth=6,
    learning_rate=0.02, subsample=0.8, colsample_bytree=0.8,
    min_child_samples=5, reg_alpha=0.1, reg_lambda=0.1,
    class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1,
)
clf.fit(X_tr, y_tr)
tr_acc = (clf.predict(X_tr)==y_tr).mean()
te_acc = (clf.predict(X_te)==y_te).mean()
print(f"Train acc: {tr_acc:.4f}  |  Walk-forward test acc: {te_acc:.4f}  ({time.time()-t2:.1f}s)")

joblib.dump(clf, MODEL_PATH)
with open(META_PATH,'w') as fp:
    json.dump({
        'asset':'ETH','last_price':last_price,
        'n_samples':len(X),'n_features':X.shape[1],
        'train_accuracy':round(tr_acc,4),'test_accuracy':round(te_acc,4),
        'lookahead_bars':LOOKAHEAD,'timeframe':'daily',
        'label_logic':'next candle close >= next candle open',
        'trained_at':time.strftime('%Y-%m-%d %H:%M:%S'),
    }, fp, indent=2)

print(f"\n{'='*60}")
print(f"ETH Model: {MODEL_PATH}")
print(f"Walk-forward test accuracy: {te_acc*100:.2f}%")
print(f"Total time: {time.time()-t0:.1f}s\nRestart ws.py to load.")
