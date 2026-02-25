import asyncio
import json
import time
import random
import datetime
import numpy as np
import requests
from collections import deque
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Any, List, Deque, Optional
try:
    import pytz
    _HAS_PYTZ = True
except ImportError:
    _HAS_PYTZ = False

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

manager = ConnectionManager()

BUFFER_SIZE = 3000
candle_buffers: Dict[str, Deque[Dict]] = {}
last_known_prices: Dict[str, float] = {}

# ── Per-asset model registry ─────────────────────────────────────────────────
# Each asset has its own dedicated LightGBM model with asset-tuned hyperparams.
_models:       Dict[str, Any]           = {"BTC-USD": None, "ETH-USD": None, "GOLD": None}
_feature_masks: Dict[str, Optional[List[int]]] = {"BTC-USD": None, "ETH-USD": None, "GOLD": None}
_model_trained = False   # single flag: True once all startup loading is done

# ── Per-symbol prediction smoothing ──────────────────────────────────────────
# EMA-smooth consecutive predictions to reduce flip-flopping near 50% confidence
_prev_bull: Dict[str, float] = {"BTC-USD": 0.5, "ETH-USD": 0.5, "GOLD": 0.5}

# ── News sentiment cache ──────────────────────────────────────────────────────
_news_cache: Dict[str, Dict] = {}   # {symbol: {score, expires_at}}
_NEWS_TTL = 300   # refresh news every 5 minutes


# ═══════════════════════════════════════════════════════════════
# 1. FETCH REAL HISTORICAL DATA FROM BINANCE (free, no API key)
# ═══════════════════════════════════════════════════════════════
_BINANCE_SYMBOL_MAP = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
    "BNB-USD": "BNBUSDT",
}

def _fetch_binance_klines(symbol: str, interval: str = "1d", limit: int = 1000) -> List[Dict]:
    """
    Fetch up to `limit` candles from Binance public REST API.
    Returns list of OHLCV dicts sorted oldest → newest.
    No API key required.
    """
    binance_sym = _BINANCE_SYMBOL_MAP.get(symbol, "BTCUSDT")
    url = "https://api.binance.com/api/v3/klines"
    try:
        resp = requests.get(url, params={
            "symbol": binance_sym,
            "interval": interval,
            "limit": limit
        }, timeout=15)
        if not resp.ok:
            return []
        rows = resp.json()
        candles = []
        for r in rows:
            open_time_s = int(r[0]) // 1000
            candles.append({
                "time": open_time_s,
                "open":   float(r[1]),
                "high":   float(r[2]),
                "low":    float(r[3]),
                "close":  float(r[4]),
                "volume": float(r[5]),
            })
        return candles
    except Exception as e:
        print(f"[Binance] Fetch error: {e}")
        return []


def fetch_all_history(symbol: str) -> List[Dict]:
    """
    Fetch as much history as possible via multiple Binance API calls.
    Binance returns max 1000 candles per request, so we page backwards.
    """
    print(f"[Binance] Fetching historical daily data for {symbol}...")
    all_candles: List[Dict] = []
    binance_sym = _BINANCE_SYMBOL_MAP.get(symbol, "BTCUSDT")
    url = "https://api.binance.com/api/v3/klines"
    end_time_ms = None  # start from now, page backwards

    for page in range(10):  # up to 10 pages × 1000 candles = 10,000 days (way more than BTC lifespan)
        params = {"symbol": binance_sym, "interval": "1d", "limit": 1000}
        if end_time_ms:
            params["endTime"] = end_time_ms

        try:
            resp = requests.get(url, params=params, timeout=15)
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

            all_candles = batch + all_candles  # prepend (older data first)
            end_time_ms = int(rows[0][0]) - 1   # go back before earliest candle in this batch

            if len(batch) < 1000:
                break  # reached the beginning of available data

        except Exception as e:
            print(f"[Binance] Page {page} error: {e}")
            break

    print(f"[Binance] Fetched {len(all_candles)} total daily candles for {symbol}")
    return all_candles


# ═══════════════════════════════════════════════════════════════
# 2. TECHNICAL INDICATORS (pure numpy)
# ═══════════════════════════════════════════════════════════════
def _rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1: return 50.0
    d = np.diff(closes[-(period + 1):])
    g = np.where(d > 0, d, 0.0).mean()
    l = np.where(d < 0, -d, 0.0).mean()
    return 50.0 if l == 0 else 100.0 - 100.0 / (1.0 + g / l)

def _ema(closes: np.ndarray, period: int) -> float:
    if len(closes) < period: return float(np.mean(closes))
    k, e = 2.0 / (period + 1), closes[0]
    for v in closes[1:]: e = v * k + e * (1 - k)
    return e

def _macd(closes: np.ndarray) -> float:
    return 0.0 if len(closes) < 26 else _ema(closes, 12) - _ema(closes, 26)

def _bb(closes: np.ndarray, period: int = 20):
    """Returns (position 0-1, normalised bandwidth)."""
    if len(closes) < period: return 0.5, 0.02
    w = closes[-period:]
    m, s = np.mean(w), np.std(w)
    if s == 0: return 0.5, 0.0
    pos   = float(np.clip((closes[-1] - (m - 2*s)) / (4*s + 1e-9), 0.0, 1.0))
    width = float(4*s / (m + 1e-9))
    return pos, width

def _mom(closes: np.ndarray, p: int) -> float:
    return 0.0 if len(closes) < p + 1 else float((closes[-1] - closes[-(p+1)]) / (closes[-(p+1)] + 1e-9))

def _atr(buf: List[Dict], period: int = 14) -> float:
    if len(buf) < 2: return 1.0
    trs = [max(c["high"] - c["low"],
               abs(c["high"] - buf[i-1]["close"]),
               abs(c["low"]  - buf[i-1]["close"]))
           for i, c in enumerate(buf) if i > 0]
    return float(np.mean(trs[-period:])) if trs else 1.0

def _stoch_rsi(closes: np.ndarray, period: int = 14) -> float:
    """Stochastic RSI — RSI normalised within its own min/max window."""
    if len(closes) < period * 2: return 0.5
    rsi_vals = [_rsi(closes[:i+1], period) for i in range(period - 1, len(closes))]
    if not rsi_vals: return 0.5
    r = rsi_vals[-1]
    lo, hi = min(rsi_vals[-period:]), max(rsi_vals[-period:])
    return 0.5 if hi == lo else float(np.clip((r - lo) / (hi - lo), 0.0, 1.0))

def _obv_slope(buf: List[Dict], period: int = 10) -> float:
    """On-Balance Volume normalised trend."""
    if len(buf) < period + 1: return 0.0
    recent = buf[-(period + 1):]
    obv = 0.0
    vals = [0.0]
    for i in range(1, len(recent)):
        obv += recent[i]["volume"] if recent[i]["close"] > recent[i-1]["close"] else -recent[i]["volume"]
        vals.append(obv)
    return float((vals[-1] - vals[0]) / (abs(vals[0]) + 1e-9))


def _vwap_dev(buf: List[Dict], period: int = 20) -> float:
    """Price deviation from VWAP (volume-weighted average price)."""
    if len(buf) < period: return 0.0
    recent = buf[-period:]
    total_vol = sum(r["volume"] for r in recent)
    if total_vol == 0: return 0.0
    vwap = sum((r["high"]+r["low"]+r["close"])/3 * r["volume"] for r in recent) / total_vol
    return float((buf[-1]["close"] - vwap) / (vwap + 1e-9))


def _vol_delta(buf: List[Dict], period: int = 10) -> float:
    """Recent volume vs rolling average volume."""
    if len(buf) < period + 1: return 0.0
    avg = float(np.mean([r["volume"] for r in buf[-(period+1):-1]]))
    return float((buf[-1]["volume"] - avg) / (avg + 1e-9))


def _time_features(ts: int):
    """Sine/cosine encoding of hour-of-day (UTC). BTC has strong intraday patterns."""
    import datetime
    dt = datetime.datetime.utcfromtimestamp(ts)
    hour = dt.hour + dt.minute / 60.0
    return float(np.sin(2*np.pi*hour/24)), float(np.cos(2*np.pi*hour/24))


# ═══════════════════════════════════════════════════════════════
# 3. CANDLESTICK PATTERN DETECTOR (pure Python, 16+ patterns)
# ═══════════════════════════════════════════════════════════════
def detect_patterns(buf: List[Dict]) -> Dict[str, int]:
    patterns: Dict[str, int] = {}
    n = len(buf)
    if n < 1: return patterns

    def body(c):        return c["close"] - c["open"]
    def abody(c):       return abs(body(c))
    def rng(c):         return (c["high"] - c["low"]) or 1e-9
    def upper_wick(c):  return c["high"] - max(c["open"], c["close"])
    def lower_wick(c):  return min(c["open"], c["close"]) - c["low"]
    def is_bull(c):     return c["close"] > c["open"]
    def is_bear(c):     return c["close"] < c["open"]

    C0 = buf[-1]
    atr = _atr(buf)

    # ── Single candle ──────────────────────────────────────────
    if abody(C0) < 0.10 * rng(C0):
        patterns["doji"] = 0

    if lower_wick(C0) >= 2.0 * max(abody(C0), atr * 0.01) and upper_wick(C0) < abody(C0) * 0.5:
        patterns["hammer"] = 1

    if upper_wick(C0) >= 2.0 * max(abody(C0), atr * 0.01) and lower_wick(C0) < abody(C0) * 0.5:
        patterns["shooting_star"] = -1

    if is_bull(C0) and abody(C0) > 0.85 * rng(C0):
        patterns["marubozu_bull"] = 1

    if is_bear(C0) and abody(C0) > 0.85 * rng(C0):
        patterns["marubozu_bear"] = -1

    if 0.10 * rng(C0) < abody(C0) < 0.40 * rng(C0) and min(upper_wick(C0), lower_wick(C0)) > abody(C0) * 0.5:
        patterns["spinning_top"] = 0

    if n < 2: return patterns
    C1 = buf[-2]

    # ── Two-candle ─────────────────────────────────────────────
    if is_bear(C1) and is_bull(C0) and C0["open"] <= C1["close"] and C0["close"] >= C1["open"]:
        patterns["bullish_engulfing"] = 1

    if is_bull(C1) and is_bear(C0) and C0["open"] >= C1["close"] and C0["close"] <= C1["open"]:
        patterns["bearish_engulfing"] = -1

    if is_bear(C1) and is_bull(C0) and abody(C1) > 2*abody(C0):
        if C0["open"] > C1["close"] and C0["close"] < C1["open"]:
            patterns["bullish_harami"] = 1

    if is_bull(C1) and is_bear(C0) and abody(C1) > 2*abody(C0):
        if C0["open"] < C1["close"] and C0["close"] > C1["open"]:
            patterns["bearish_harami"] = -1

    if is_bear(C1) and is_bull(C0):
        mid = (C1["open"] + C1["close"]) / 2
        if C0["open"] < C1["close"] and C0["close"] > mid:
            patterns["piercing_line"] = 1

    if is_bull(C1) and is_bear(C0):
        mid = (C1["open"] + C1["close"]) / 2
        if C0["open"] > C1["close"] and C0["close"] < mid:
            patterns["dark_cloud_cover"] = -1

    if abs(C0["low"] - C1["low"]) < atr * 0.05:
        patterns["tweezer_bottom"] = 1

    if abs(C0["high"] - C1["high"]) < atr * 0.05:
        patterns["tweezer_top"] = -1

    if n < 3: return patterns
    C2 = buf[-3]

    # ── Three-candle ───────────────────────────────────────────
    mid_c2 = (C2["open"] + C2["close"]) / 2
    if is_bear(C2) and abody(C1) < abody(C2)*0.5 and is_bull(C0) and C0["close"] > mid_c2:
        patterns["morning_star"] = 1

    if is_bull(C2) and abody(C1) < abody(C2)*0.5 and is_bear(C0) and C0["close"] < mid_c2:
        patterns["evening_star"] = -1

    if all(is_bull(c) for c in [C2, C1, C0]) and C1["close"] > C2["close"] and C0["close"] > C1["close"]:
        if all(abody(c) > atr*0.3 for c in [C2, C1, C0]):
            patterns["three_white_soldiers"] = 1

    if all(is_bear(c) for c in [C2, C1, C0]) and C1["close"] < C2["close"] and C0["close"] < C1["close"]:
        if all(abody(c) > atr*0.3 for c in [C2, C1, C0]):
            patterns["three_black_crows"] = -1

    if is_bear(C2) and is_bull(C1) and is_bull(C0):
        if C1["open"] > C2["close"] and C1["close"] < C2["open"] and C0["close"] > C2["open"]:
            patterns["three_inside_up"] = 1

    if is_bull(C2) and is_bear(C1) and is_bear(C0):
        if C1["open"] < C2["close"] and C1["close"] > C2["open"] and C0["close"] < C2["open"]:
            patterns["three_inside_down"] = -1

    return patterns


def pattern_score(patterns: Dict[str, int]) -> float:
    if not patterns: return 0.0
    total = sum(v for v in patterns.values())
    return float(np.clip(total / max(len(patterns), 1), -1.0, 1.0))


# ═══════════════════════════════════════════════════════════════
# 4. FEATURE EXTRACTION  (20 tech + 1 composite + 16 pattern flags = 37 total)
# ═══════════════════════════════════════════════════════════════
PATTERN_NAMES = [
    "hammer", "shooting_star", "marubozu_bull", "marubozu_bear",
    "bullish_engulfing", "bearish_engulfing",
    "bullish_harami", "bearish_harami",
    "piercing_line", "dark_cloud_cover",
    "tweezer_bottom", "tweezer_top",
    "morning_star", "evening_star",
    "three_white_soldiers", "three_black_crows",
]

def extract_features(buf: List[Dict]) -> np.ndarray:
    closes = np.array([c["close"] for c in buf], dtype=float)
    last   = buf[-1]
    rng_l  = (last["high"] - last["low"]) or 1e-9
    body_l = last["close"] - last["open"]
    uw     = last["high"] - max(last["open"], last["close"])
    lw     = min(last["open"], last["close"]) - last["low"]
    atr    = _atr(buf)

    rsi    = _rsi(closes, 14)
    macd   = _macd(closes)
    bb_pos, bb_wid = _bb(closes, 20)
    mom5   = _mom(closes, 5)
    mom10  = _mom(closes, 10)
    mom20  = _mom(closes, 20)
    ema9   = _ema(closes, 9)
    ema21  = _ema(closes, 21)
    ema50  = _ema(closes, 50)
    ex9    = (closes[-1] - ema9)  / (ema9  + 1e-9)
    es921  = (ema9  - ema21)      / (ema21 + 1e-9)
    es2150 = (ema21 - ema50)      / (ema50 + 1e-9)
    sr     = _stoch_rsi(closes, 14)
    obv    = _obv_slope(buf, 10)
    vwap   = _vwap_dev(buf, 20)
    vdelta = _vol_delta(buf, 10)
    atr_r  = atr / (closes[-1] + 1e-9)
    ts_sin, ts_cos = _time_features(last["time"])

    tech = np.array([
        rsi/100.0, macd/(closes[-1]+1e-9), bb_pos, bb_wid,
        mom5, mom10, mom20,
        ex9, es921, es2150,
        sr, obv,
        body_l/rng_l, uw/rng_l, lw/rng_l,
        vwap, vdelta, atr_r, ts_sin, ts_cos,
    ], dtype=float)

    patterns  = detect_patterns(buf)
    composite = pattern_score(patterns)
    pat_flags = np.array([float(patterns.get(p, 0)) for p in PATTERN_NAMES], dtype=float)

    feat = np.concatenate([[composite], tech, pat_flags])
    return np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)


# ═══════════════════════════════════════════════════════════════
# 5. MODEL TRAINING (on real historical data)
# ═══════════════════════════════════════════════════════════════
def _train_model(buf: List[Dict], label: str = "buffer"):
    global _model
    try:
        from lightgbm import LGBMClassifier
        min_rows = 60
        if len(buf) < min_rows + 1:
            print(f"[Model] Not enough data ({len(buf)} rows) to train")
            return

        X, y = [], []
        for i in range(min_rows, len(buf) - 1):
            X.append(extract_features(buf[:i + 1]))
            nxt = buf[i + 1]
            y.append(1 if nxt["close"] >= nxt["open"] else 0)

        if len(set(y)) < 2:
            print("[Model] Only one class present, skipping training")
            return

        X_arr, y_arr = np.array(X), np.array(y)
        clf = LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
            reg_alpha=0.1, reg_lambda=0.1,
            random_state=42, verbose=-1,
        )
        clf.fit(X_arr, y_arr)
        _model = clf

        # Quick accuracy on training set as sanity check
        preds = clf.predict(X_arr)
        acc = (preds == y_arr).mean()
        print(f"[Model] Trained on {len(X)} {label} samples ({len(buf)} candles), "
              f"train-acc={acc:.3f}, features={X_arr.shape[1]}, bull%={y_arr.mean():.2f}")

    except Exception as e:
        print(f"[Model] Training error: {e}")
        import traceback; traceback.print_exc()


# ═══════════════════════════════════════════════════════════════
# 6. NEWS SENTIMENT (free, no API key, cached 5 min)
# ═══════════════════════════════════════════════════════════════
_BULLISH_WORDS = [
    'rally', 'surge', 'soar', 'breakout', 'ath', 'all-time high', 'adoption',
    'bullish', 'buy', 'gain', 'rise', 'pump', 'inflow', 'etf approved',
    'positive', 'upgrade', 'partnership', 'institutional', 'record',
]
_BEARISH_WORDS = [
    'crash', 'dump', 'plunge', 'ban', 'hack', 'exploit', 'sell', 'bearish',
    'regulation', 'shutdown', 'fine', 'lawsuit', 'outflow', 'crisis',
    'negative', 'downgrade', 'fear', 'fraud', 'collapse', 'liquidation',
]

def _fetch_news_sentiment(symbol: str) -> float:
    """
    Fetch free RSS headlines and score sentiment. Returns float in [-1, 1].
    Cached for _NEWS_TTL seconds. No API key required.
    """
    now = time.time()
    cached = _news_cache.get(symbol)
    if cached and cached.get('expires_at', 0) > now:
        return cached['score']

    # Choose RSS feed based on symbol
    if symbol == 'GOLD':
        feeds = [
            'https://feeds.finance.yahoo.com/rss/2.0/headline?s=GC%3DF&region=US&lang=en-US',
        ]
        extra_bearish = ['rate hike', 'dollar strong', 'fed hawkish']
        extra_bullish = ['inflation', 'safe haven', 'uncertainty', 'dollar weak']
    else:
        coin = 'bitcoin' if symbol == 'BTC-USD' else 'ethereum'
        feeds = [
            f'https://cryptopanic.com/news/rss/?currencies={coin[:3].upper()}',
            f'https://feeds.finviz.com/rss.ashx?t={coin[:3].lower()}usd',
        ]
        extra_bearish = []; extra_bullish = []

    bull_words = _BULLISH_WORDS + extra_bullish
    bear_words = _BEARISH_WORDS + extra_bearish
    score = 0.0
    count = 0
    try:
        for feed_url in feeds:
            r = requests.get(feed_url, timeout=5,
                             headers={'User-Agent': 'Mozilla/5.0'})
            if not r.ok:
                continue
            text = r.text.lower()
            bull_hits = sum(text.count(w) for w in bull_words)
            bear_hits = sum(text.count(w) for w in bear_words)
            total = bull_hits + bear_hits
            if total > 0:
                score += (bull_hits - bear_hits) / total
                count += 1
    except Exception:
        pass

    final_score = float(np.clip(score / max(count, 1), -1, 1)) if count > 0 else 0.0
    _news_cache[symbol] = {'score': final_score, 'expires_at': now + _NEWS_TTL}
    print(f"[News] {symbol} sentiment={final_score:+.3f} (from {count} feed(s))")
    return final_score


# ═══════════════════════════════════════════════════════════════
# 7. PREDICTION ENGINE (per-asset model + feature mask + smoothing + news)
# ═══════════════════════════════════════════════════════════════

def _predict(buf: List[Dict], symbol: str = "BTC-USD") -> Dict:
    """
    Unified prediction engine — dispatches to the correct per-asset model.
    Feature indices (matches train_model.py exactly):
      [0] composite, [1] rsi/100, [2] macd, [3] bb_pos, [4] bb_wid,
      [5] mom5, [6] mom10, [7] mom20, [8] ex9, [9] es921, [10] es2150,
      [11] stoch_rsi, [12] obv_slope, ...
    """
    features = extract_features(buf)
    model = _models.get(symbol)
    mask  = _feature_masks.get(symbol)
    feat_for_ml = features[mask] if mask is not None else features

    # ── Rule-based signal (fixed indices) ──────────────────────
    composite = features[0]   # pattern composite
    rsi_norm  = features[1]   # rsi/100
    mom10     = features[6]   # 10-period momentum  [6 is correct]
    ex9       = features[8]   # price vs EMA9       [8 is correct]
    es921     = features[9]   # EMA9 vs EMA21       [9 is correct]

    rule = 0.0
    if rsi_norm < 0.30:  rule += 0.22   # oversold → bullish
    elif rsi_norm > 0.70: rule -= 0.22  # overbought → bearish
    rule += mom10     * 4.5
    rule += ex9       * 3.0
    rule += es921     * 2.0
    rule += composite * 0.40

    # GOLD: stronger mean-reversion rule
    if symbol == 'GOLD':
        rule *= 1.3  # amplify RSI/BB signals (mean-reverting market)

    rule_bull = float(np.clip(0.5 + rule, 0.15, 0.85))

    # ── ML model ───────────────────────────────────────────────
    if model is not None:
        try:
            ml_prob = float(model.predict_proba(feat_for_ml.reshape(1, -1))[0][1])
            if symbol == 'ETH-USD':
                bull = 0.70 * ml_prob + 0.30 * rule_bull  # ETH: trust ML more (good model)
            elif symbol == 'BTC-USD':
                bull = 0.65 * ml_prob + 0.35 * rule_bull  # BTC: balanced
            else:  # GOLD
                bull = 0.55 * ml_prob + 0.45 * rule_bull  # GOLD: trust rules more (tiny data)
        except Exception:
            bull = rule_bull
    else:
        bull = rule_bull

    # ── News sentiment nudge (±3% max) ──────────────────────────
    try:
        news_score = _fetch_news_sentiment(symbol)
        bull = float(np.clip(bull + news_score * 0.03, 0.10, 0.90))
    except Exception:
        pass

    # ── EMA smoothing (reduces flip-flopping near 50%) ──────────
    prev = _prev_bull.get(symbol, 0.5)
    bull_smoothed = 0.65 * bull + 0.35 * prev
    _prev_bull[symbol] = bull_smoothed
    bull = float(np.clip(bull_smoothed, 0.10, 0.90))

    return {
        "bull":       round(bull, 4),
        "bear":       round(1.0 - bull, 4),
        "confidence": round(0.50 + abs(bull - 0.50), 2),
    }


# ═══════════════════════════════════════════════════════════════
# 8. STARTUP: load all per-asset models + seed buffers
# ═══════════════════════════════════════════════════════════════
from pathlib import Path
import joblib
import json as _json

_MODELS_DIR = Path(__file__).parent.parent / "models"

# Per-asset model file definitions
_ASSET_MODEL_FILES = {
    "BTC-USD": {
        "model":  _MODELS_DIR / "btc_model.pkl",
        "meta":   _MODELS_DIR / "btc_meta.json",
        "mask":   _MODELS_DIR / "btc_feature_mask.json",
    },
    "ETH-USD": {
        "model":  _MODELS_DIR / "eth_model.pkl",
        "meta":   _MODELS_DIR / "eth_meta.json",
        "mask":   None,
    },
    "GOLD": {
        "model":  _MODELS_DIR / "gold_model.pkl",
        "meta":   _MODELS_DIR / "gold_meta.json",
        "mask":   None,
    },
}

_historical_real_data: List[Dict] = []

def _load_asset_model(asset: str) -> bool:
    """Load a single asset's model + mask from disk. Returns True on success."""
    cfg = _ASSET_MODEL_FILES.get(asset, {})
    model_path = cfg.get("model")
    meta_path  = cfg.get("meta")
    mask_path  = cfg.get("mask")

    if not (model_path and model_path.exists() and meta_path and meta_path.exists()):
        print(f"[Model/{asset}] No saved model found at {model_path}. Run the trainer first.")
        return False

    try:
        model = joblib.load(model_path)
        with open(meta_path) as f:
            meta = _json.load(f)

        _models[asset] = model
        if meta.get("model_type") == "rule_only":
            _models[asset] = None  # GOLD rule-only fallback
            print(f"[Model/{asset}] Rule-only mode (too few training samples)")
        else:
            _models[asset] = model
            print(f"[Model/{asset}] Loaded: acc={meta.get('train_accuracy')}, "
                  f"test_acc={meta.get('test_accuracy','N/A')}, "
                  f"n={meta.get('n_samples')}, tf={meta.get('timeframe','N/A')}")

        # Load feature mask if applicable
        if mask_path and mask_path.exists():
            with open(mask_path) as f:
                m = _json.load(f)
            _feature_masks[asset] = m.get("keep_indices")
            print(f"[Model/{asset}] Feature mask: keeping {len(_feature_masks[asset])} features")
        return True

    except Exception as e:
        print(f"[Model/{asset}] Load error: {e}")
        return False


def _startup_train(symbol: str = "BTC-USD"):
    """
    Load all per-asset models from disk at startup, then seed chart buffers.
    Falls back to rule-based predictions if any model is missing.
    """
    global _model_trained, _historical_real_data

    if _model_trained:
        return

    print("[Startup] Loading per-asset models...")
    print("[Startup] BTC model (from local 1m CSVs)...")
    btc_ok = _load_asset_model("BTC-USD")
    print("[Startup] ETH model (from Binance daily)...")
    _load_asset_model("ETH-USD")
    print("[Startup] GOLD model (from yfinance daily)...")
    _load_asset_model("GOLD")

    # Seed BTC chart with 300 Binance daily candles
    print("[Startup] Seeding BTC chart from Binance (300 daily candles)...")
    seed = _fetch_binance_klines(symbol, interval="1d", limit=300)
    if seed:
        last_known_prices[symbol] = seed[-1]["close"]
        candle_buffers[symbol] = deque(seed, maxlen=BUFFER_SIZE)
        print(f"[Startup] BTC seeded: {len(seed)} candles, last=${seed[-1]['close']:,.2f}")
    else:
        candle_buffers[symbol] = deque(maxlen=BUFFER_SIZE)
        if not btc_ok:
            last_known_prices[symbol] = 85000.0
        # Fallback: Binance on-the-fly training if no saved model
        if _models["BTC-USD"] is None:
            print("[Startup] No BTC model found. Fetching Binance history for on-the-fly training...")
            real_data = fetch_all_history(symbol)
            if real_data:
                _historical_real_data = real_data
                last_known_prices[symbol] = real_data[-1]["close"]
                candle_buffers[symbol] = deque(real_data[-BUFFER_SIZE:], maxlen=BUFFER_SIZE)
                _train_model(real_data, label="Binance daily")
                _models["BTC-USD"] = None  # _train_model sets legacy _model; wrap it
            else:
                print("[Startup] ⚠️  No BTC data. Using rule-based predictions only.")
                last_known_prices[symbol] = 85000.0


    _model_trained = True


# ═══════════════════════════════════════════════════════════════
# 8-A. ETH-USD: Seed buffer from Binance at startup (no model needed)
# ═══════════════════════════════════════════════════════════════
_eth_seeded = False

def _startup_seed_eth():
    """Seeds candle_buffers['ETH-USD'] with 60 daily Binance candles."""
    global _eth_seeded
    if _eth_seeded:
        return
    print("[ETH] Seeding chart from Binance (300 daily candles)...")
    seed = _fetch_binance_klines("ETH-USD", interval="1d", limit=300)
    if seed:
        last_known_prices["ETH-USD"] = seed[-1]["close"]
        candle_buffers["ETH-USD"] = deque(seed, maxlen=BUFFER_SIZE)
        print(f"[ETH] Seeded {len(seed)} candles, last=${seed[-1]['close']:,.2f}")
    else:
        candle_buffers["ETH-USD"] = deque(maxlen=BUFFER_SIZE)
        last_known_prices["ETH-USD"] = 3000.0
        print("[ETH] Binance unreachable — buffer empty, price fallback $3000")
    _eth_seeded = True


# ═══════════════════════════════════════════════════════════════
# 8-B. GOLD: yfinance helpers + COMEX market-hours check
# ═══════════════════════════════════════════════════════════════
_gold_seeded = False

def _is_gold_market_open() -> bool:
    """
    COMEX Gold Futures (GC=F) trading hours (approximate UTC):
      Sunday 23:00 UTC → Friday 22:00 UTC
      with a daily 60-min break: 22:00–23:00 UTC each weekday
    Returns False on weekends and during the daily break.
    """
    if _HAS_PYTZ:
        now = datetime.datetime.now(pytz.UTC)
    else:
        now = datetime.datetime.utcnow()
    wd  = now.weekday()   # 0=Mon … 6=Sun
    h, m = now.hour, now.minute
    utc_mins = h * 60 + m

    # Saturday — always closed
    if wd == 5:
        return False
    # Sunday — open only after 23:00 UTC
    if wd == 6 and utc_mins < 23 * 60:
        return False
    # Friday — closed after 22:00 UTC
    if wd == 4 and utc_mins >= 22 * 60:
        return False
    # Daily break Mon–Thu 22:00–23:00 UTC
    if wd < 4 and 22 * 60 <= utc_mins < 23 * 60:
        return False
    return True


def _fetch_gold_historical(limit: int = 60) -> List[Dict]:
    """Fetch last `limit` daily candles for XAU/USD via yfinance GC=F."""
    try:
        import yfinance as yf
        ticker = yf.Ticker("GC=F")
        hist = ticker.history(period="90d", interval="1d")
        candles = []
        for ts, row in hist.iterrows():
            candles.append({
                "time":   int(ts.timestamp()),
                "open":   round(float(row["Open"]),  2),
                "high":   round(float(row["High"]),  2),
                "low":    round(float(row["Low"]),   2),
                "close":  round(float(row["Close"]), 2),
                "volume": round(float(row["Volume"]), 2),
            })
        candles = [c for c in candles if c["close"] > 0]
        return candles[-limit:]
    except Exception as e:
        print(f"[GOLD] Historical fetch error: {e}")
        return []


def _get_gold_live_price() -> Optional[float]:
    """Fetch current gold spot price via yfinance (fastest free source)."""
    try:
        import yfinance as yf
        fi = yf.Ticker("GC=F").fast_info
        p = fi.get("last_price") or fi.get("regularMarketPrice")
        return float(p) if p else None
    except Exception:
        return None


def _get_gold_forming_kline() -> Optional[Dict]:
    """Return a pseudo-forming kline for GOLD using the latest yfinance tick."""
    price = _get_gold_live_price()
    if price is None:
        return None
    last = last_known_prices.get("GOLD", price)
    t = int(time.time()) - (int(time.time()) % 60)   # floor to current minute
    return {
        "time":   t,
        "open":   round(last, 2),
        "high":   round(max(last, price), 2),
        "low":    round(min(last, price), 2),
        "close":  round(price, 2),
        "volume": 0.0,
    }


def _get_gold_closed_kline() -> Optional[Dict]:
    """Return a closed kline for GOLD (uses current tick as close)."""
    price = _get_gold_live_price()
    if price is None:
        return None
    prev = last_known_prices.get("GOLD", price)
    t = int(time.time()) - 60
    return {
        "time":   t,
        "open":   round(prev, 2),
        "high":   round(max(prev, price), 2),
        "low":    round(min(prev, price), 2),
        "close":  round(price, 2),
        "volume": 0.0,
    }


def _startup_seed_gold():
    """Seeds candle_buffers['GOLD'] with historical yfinance daily candles."""
    global _gold_seeded
    if _gold_seeded:
        return
    print("[GOLD] Seeding chart from yfinance GC=F (60 daily candles)...")
    seed = _fetch_gold_historical(60)
    if seed:
        last_known_prices["GOLD"] = seed[-1]["close"]
        candle_buffers["GOLD"] = deque(seed, maxlen=BUFFER_SIZE)
        print(f"[GOLD] Seeded {len(seed)} candles, last=${seed[-1]['close']:,.2f}")
    else:
        candle_buffers["GOLD"] = deque(maxlen=BUFFER_SIZE)
        last_known_prices["GOLD"] = 2300.0
        print("[GOLD] yfinance unavailable — buffer empty, price fallback $2300")
    _gold_seeded = True


# ═══════════════════════════════════════════════════════════════
# 8-C. Universal dispatch wrappers (BTC/ETH → Binance, GOLD → yfinance)
#      These call the untouched existing BTC helpers for BTC/ETH.
# ═══════════════════════════════════════════════════════════════
def _any_live_price(symbol: str) -> Optional[float]:
    if symbol == "GOLD":
        return _get_gold_live_price()
    return _get_live_price(symbol)   # existing Binance helper (BTC & ETH)


def _any_forming_kline(symbol: str) -> Optional[Dict]:
    if symbol == "GOLD":
        return _get_gold_forming_kline()
    return _get_forming_kline(symbol)   # existing Binance helper


def _any_make_live_candle(symbol: str, candle_id: int,
                          open_price: float, interval: str) -> Dict[str, Any]:
    if symbol == "GOLD":
        kline = _get_gold_closed_kline()
        if kline:
            op, cl = kline["open"], kline["close"]
            last_known_prices[symbol] = cl
            candle = {"time": kline["time"],
                      "open": op, "high": kline["high"],
                      "low": kline["low"], "close": cl, "volume": 0.0}
            if symbol in candle_buffers:
                candle_buffers[symbol].append(candle)
            return {"type": "candle", "candle_id": candle_id, "symbol": symbol,
                    **candle, "price": round(cl, 2),
                    "actual_direction": "bull" if cl >= op else "bear"}
        # Fallback
        cl = last_known_prices.get(symbol, 2300.0)
        return {"type": "candle", "candle_id": candle_id, "symbol": symbol,
                "time": int(time.time()), "open": cl, "high": cl, "low": cl,
                "close": cl, "volume": 0.0, "price": round(cl, 2),
                "actual_direction": "bull"}
    return _make_live_candle(symbol, candle_id, open_price, interval)   # existing BTC/ETH



# ─────────────────────────────────────────────────────────────
# Timeframe → seconds + Binance interval mapping
# ─────────────────────────────────────────────────────────────
TF_SECONDS: Dict[str, int] = {
    "1m": 60,   "5m": 300,  "15m": 900,
    "30m": 1800, "1h": 3600, "1d": 86400,
}
TF_BINANCE: Dict[str, str] = {
    "1m": "1m", "5m": "5m", "15m": "15m",
    "30m": "30m", "1h": "1h", "1d": "1d",
}


def _get_live_price(symbol: str) -> Optional[float]:
    """Fetch current spot price from Binance (fast, single ticker call)."""
    binance_sym = _BINANCE_SYMBOL_MAP.get(symbol, "BTCUSDT")
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbol": binance_sym},
            timeout=5
        )
        if r.ok:
            return float(r.json()["price"])
    except Exception:
        pass
    return None


def _get_closed_kline(symbol: str, interval: str) -> Optional[Dict]:
    """
    Fetch the most recently CLOSED kline from Binance.
    Binance returns the last 2 klines; the first one is the closed one.
    """
    binance_sym = _BINANCE_SYMBOL_MAP.get(symbol, "BTCUSDT")
    binance_interval = TF_BINANCE.get(interval, "1m")
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": binance_sym, "interval": binance_interval, "limit": 2},
            timeout=8
        )
        if r.ok:
            rows = r.json()
            if len(rows) >= 1:
                row = rows[0]  # rows[0] = previous (closed), rows[1] = current (open)
                return {
                    "time":   int(row[0]) // 1000,
                    "open":   float(row[1]),
                    "high":   float(row[2]),
                    "low":    float(row[3]),
                    "close":  float(row[4]),
                    "volume": float(row[5]),
                }
    except Exception:
        pass
    return None


def _make_live_candle(symbol: str, candle_id: int,
                      open_price: float, interval: str) -> Dict[str, Any]:
    """
    Build the candle message for the just-completed interval.
    Tries Binance first, falls back to synthetic walk.
    """
    kline = _get_closed_kline(symbol, interval)
    if kline:
        op   = kline["open"]
        cl   = kline["close"]
        hi   = kline["high"]
        lo   = kline["low"]
        vol  = kline["volume"]
        t    = kline["time"]
    else:
        # Synthetic fallback
        base = last_known_prices.get(symbol, 63000.0)
        vol  = base * 0.0015
        op   = open_price
        cl   = base + random.gauss(0, base * 0.001)
        hi   = max(op, cl) + abs(random.gauss(0, base * 0.0003))
        lo   = min(op, cl) - abs(random.gauss(0, base * 0.0003))
        vol  = random.randint(100, 5000)
        t    = int(time.time())

    last_known_prices[symbol] = cl
    candle = {"time": t, "open": round(op,2), "high": round(hi,2),
              "low": round(lo,2), "close": round(cl,2), "volume": round(vol,2)}
    if symbol in candle_buffers:
        candle_buffers[symbol].append(candle)

    return {
        "type": "candle", "candle_id": candle_id, "symbol": symbol,
        **candle, "price": round(cl, 2),
        "actual_direction": "bull" if cl >= op else "bear",
    }


def _get_last_60_for_chart(symbol: str) -> List[Dict]:
    """Return the last 60 real historical candles for initial chart display."""
    buf = list(candle_buffers.get(symbol, []))
    return buf[-60:] if buf else []

def _get_forming_kline(symbol: str) -> Optional[Dict]:
    """Fetch the CURRENTLY FORMING (not yet closed) 1m kline — limit=1 returns the open bar."""
    binance_sym = _BINANCE_SYMBOL_MAP.get(symbol, "BTCUSDT")
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": binance_sym, "interval": "1m", "limit": 1},
            timeout=5
        )
        if r.ok:
            row = r.json()[0]
            return {
                "time":   int(row[0]) // 1000,
                "open":   float(row[1]),
                "high":   float(row[2]),
                "low":    float(row[3]),
                "close":  float(row[4]),
                "volume": float(row[5]),
            }
    except Exception:
        pass
    return None


def _build_reason(buf: List[Dict], features: np.ndarray, bull: float, predicted: str) -> str:
    """Build a short plain-English reason for the prediction."""
    rsi_norm  = features[1]
    mom10     = features[6]   # correct index 6
    es921     = features[9]   # EMA 9-21 cross at index 9
    composite = features[0]   # pattern composite
    rsi       = rsi_norm * 100

    parts = []
    if rsi > 68:   parts.append(f"RSI {rsi:.0f} overbought")
    elif rsi < 32: parts.append(f"RSI {rsi:.0f} oversold")
    else:          parts.append(f"RSI {rsi:.0f} neutral")

    if mom10 > 0.008:  parts.append("strong upward momentum")
    elif mom10 < -0.008: parts.append("strong downward momentum")

    if es921 > 0.003:  parts.append("EMA9 > EMA21 bullish cross")
    elif es921 < -0.003: parts.append("EMA9 < EMA21 bearish cross")

    patterns = detect_patterns(buf)
    bull_pats = [k.replace('_', ' ') for k, v in patterns.items() if v > 0]
    bear_pats = [k.replace('_', ' ') for k, v in patterns.items() if v < 0]
    if bull_pats: parts.append(f"pattern: {bull_pats[0]}")
    if bear_pats: parts.append(f"pattern: {bear_pats[0]}")

    action = "predicted UP" if predicted == "bull" else "predicted DOWN"
    prob   = int(max(bull, 1 - bull) * 100)
    return f"{action} ({prob}% conf) — {' | '.join(parts)}"


def make_prediction(symbol: str, candle_id: int, msg_type: str = "prediction") -> Dict[str, Any]:
    buf = list(candle_buffers.get(symbol, []))
    if len(buf) < 20:
        bull, bear, conf = 0.50, 0.50, 0.50
        detected_patterns: list = []
        reason = "Not enough data for analysis yet"
    else:
        features = extract_features(buf)
        r = _predict(buf, symbol)   # pass symbol for per-asset model dispatch
        bull, bear, conf = r["bull"], r["bear"], r["confidence"]
        detected_patterns = [
            {"name": k.replace("_", " ").title(), "signal": v}
            for k, v in detect_patterns(buf).items()
            if v != 0
        ]
        predicted = "bull" if bull >= 0.50 else "bear"
        reason = _build_reason(buf, features, bull, predicted)

    has_model = _models.get(symbol) is not None
    news_cached = _news_cache.get(symbol, {}).get("score", None)
    model_label = f"{'LightGBM' if has_model else 'Rule'}+Patterns"
    if news_cached is not None:
        model_label += "+News"

    return {
        "type": msg_type, "candle_id": candle_id, "symbol": symbol,
        "bull_probability": bull, "bear_probability": bear,
        "predicted_direction": "bull" if bull >= 0.50 else "bear",
        "confidence_score": conf,
        "current_price": round(last_known_prices.get(symbol, 0), 2),
        "model": model_label,
        "detected_patterns": detected_patterns,
        "reason": reason,
    }


# ═══════════════════════════════════════════════════════════════
# 9. WEBSOCKET ENDPOINT
# ═══════════════════════════════════════════════════════════════
@router.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    symbol, interval, candle_id = "BTC-USD", "1m", 0

    async def _recv_nonblocking(timeout: float = 0.05) -> Optional[dict]:
        """Try to read a client message without blocking. Returns None if nothing."""
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=timeout)
            return json.loads(raw)
        except (asyncio.TimeoutError, Exception):
            return None

    try:
        # Ensure BTC model is ready (loads from disk in lifespan, but guard anyway)
        if not _model_trained:
            await asyncio.get_event_loop().run_in_executor(None, _startup_train, "BTC-USD")

        # Seed ETH and GOLD buffers on first connection (non-blocking)
        if not _eth_seeded:
            await asyncio.get_event_loop().run_in_executor(None, _startup_seed_eth)
        if not _gold_seeded:
            await asyncio.get_event_loop().run_in_executor(None, _startup_seed_gold)

        # ── Send history immediately ─────────────────────────────────
        history_candles = _get_last_60_for_chart(symbol)
        await websocket.send_text(json.dumps({
            "type": "history", "symbol": symbol,
            "interval": "1m",  # chart always shows 1m candles
            "data": history_candles
        }))

        while True:
            # ── GOLD: check market hours first ───────────────────────
            if symbol == "GOLD" and not _is_gold_market_open():
                await websocket.send_text(json.dumps({
                    "type": "market_closed",
                    "symbol": "GOLD",
                    "message": "COMEX Gold market is closed. Opens Sunday 23:00 UTC."
                }))
                await asyncio.sleep(30)
                # Check for symbol switch while sleeping
                msg = await _recv_nonblocking(0.1)
                if msg and "symbol" in msg:
                    symbol = msg["symbol"]
                    history_candles = _get_last_60_for_chart(symbol)
                    await websocket.send_text(json.dumps({
                        "type": "history", "symbol": symbol,
                        "interval": "1m", "data": history_candles
                    }))
                continue

            # ── Check for symbol/interval change ────────────────────────
            msg = await _recv_nonblocking()
            if msg:
                if "symbol" in msg and msg["symbol"] != symbol:
                    symbol = msg["symbol"]
                    history_candles = _get_last_60_for_chart(symbol)
                    await websocket.send_text(json.dumps({
                        "type": "history", "symbol": symbol,
                        "interval": "1m", "data": history_candles
                    }))
                if "interval" in msg:
                    interval = msg["interval"]  # prediction timeframe only

            candle_id += 1

            # ── Phase 1: Get live price then predict (symbol-aware) ────
            live_p = await asyncio.get_event_loop().run_in_executor(
                None, _any_live_price, symbol
            )
            if live_p:
                last_known_prices[symbol] = live_p
            _default_price = {"BTC-USD": 85000.0, "ETH-USD": 3000.0, "GOLD": 2300.0}
            open_price = last_known_prices.get(symbol, _default_price.get(symbol, 1000.0))

            wait_secs  = TF_SECONDS.get(interval, 60)  # prediction resolution window
            chart_step = 60   # push a new 1-min candle to chart every 60s
            tick_step  = 5    # send live-price tick every 5s

            pred_msg = make_prediction(symbol, candle_id)
            pred_msg["current_price"] = round(open_price, 2)
            pred_msg["resolve_in_seconds"] = wait_secs
            await websocket.send_text(json.dumps(pred_msg))

            # ── Phase 2: Wait full TF — every 5s: update live chart bar + re-predict ──
            elapsed_total = 0
            last_pred = pred_msg   # track most recent prediction for final scoring

            while elapsed_total < wait_secs:
                step = min(5, wait_secs - elapsed_total)
                await asyncio.sleep(step)
                elapsed_total += step

                # Check for client control messages
                msg = await _recv_nonblocking(0.01)
                if msg:
                    if "symbol" in msg and msg["symbol"] != symbol:
                        symbol = msg["symbol"]
                        await websocket.send_text(json.dumps({
                            "type": "history", "symbol": symbol,
                            "interval": "1m",
                            "data": _get_last_60_for_chart(symbol)
                        }))
                        break
                    if "interval" in msg:
                        interval = msg["interval"]

                remaining = max(0, wait_secs - elapsed_total)

                # 1) Fetch live price and forming kline (symbol-aware dispatch)
                tick_p, forming = await asyncio.gather(
                    asyncio.get_event_loop().run_in_executor(None, _any_live_price, symbol),
                    asyncio.get_event_loop().run_in_executor(None, _any_forming_kline, symbol),
                )
                if tick_p:
                    last_known_prices[symbol] = tick_p

                live_price_now = last_known_prices.get(symbol, open_price)

                # 2) Update candle buffer with forming kline BEFORE predicting
                #    so make_prediction uses the freshest live market data
                if forming and symbol in candle_buffers:
                    buf = candle_buffers[symbol]
                    if buf and buf[-1]["time"] == forming["time"]:
                        buf[-1] = forming          # same minute: update in-place
                    else:
                        buf.append(forming)        # new minute: append

                # 3) Send live chart update (forming candle — updates current bar)
                if forming:
                    await websocket.send_text(json.dumps({
                        "type": "chart_candle",
                        "symbol": symbol,
                        "forming": True,
                        **forming,
                        "price": round(forming["close"], 2),
                    }))

                # 4) Re-run prediction with the freshest buffer data
                updated_pred = make_prediction(symbol, candle_id, msg_type="prediction_update")
                updated_pred["current_price"] = round(live_price_now, 2)
                updated_pred["remaining_seconds"] = remaining
                updated_pred["total_seconds"] = wait_secs
                last_pred = updated_pred
                await websocket.send_text(json.dumps(updated_pred))

            # ── Phase 3: Resolve — fetch real closed candle (symbol-aware) ──
            candle_msg = await asyncio.get_event_loop().run_in_executor(
                None, _any_make_live_candle, symbol, candle_id, open_price, interval
            )
            # Attach the last prediction's reason so frontend can show miss explanation
            candle_msg["reason"] = last_pred.get("reason", "")
            candle_msg["final_predicted"] = last_pred.get("predicted_direction", "")
            await websocket.send_text(json.dumps(candle_msg))

    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)

