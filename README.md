# 🐻🐂 Bear & Bull Predictor

> AI-powered crypto & gold market predictor with real-time WebSocket streaming, per-asset ML models, and live paper trading.

[![Vercel](https://img.shields.io/badge/Frontend-Vercel-black?logo=vercel)](https://vercel.com)
[![Railway](https://img.shields.io/badge/Backend-Railway-purple?logo=railway)](https://railway.app)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Vercel (Frontend — Next.js 14)                 │
│  - Real-time chart    - Prediction log    - Trading Desk    │
│  - Asset selector     - Accuracy stats    - Auto-trading    │
└─────────────────────┬───────────────────────────────────────┘
                      │ WebSocket (wss://)
┌─────────────────────▼───────────────────────────────────────┐
│            Railway / Render (Backend — FastAPI)             │
│  - LightGBM per-asset models (BTC / ETH / GOLD)            │
│  - Binance WebSocket for real-time prices                   │
│  - yfinance for GOLD market data                            │
│  - News sentiment (RSS, no API key)                         │
└─────────────────────────────────────────────────────────────┘
```

### Why Separate Backend?
WebSocket connections + long-running ML models cannot run on Vercel serverless functions. The backend runs persistently on Railway (free tier available) with full WebSocket support.

---

## 🚀 Quick Deploy

### Step 1 — Deploy Backend (Railway)

1. Go to [railway.app](https://railway.app) → **New Project → Deploy from GitHub**
2. Select this repo → set **Root Directory** to `backend`
3. Railway auto-detects Python and installs `requirements.txt`
4. Set these environment variables in Railway:
   ```
   ALLOWED_ORIGIN=https://your-vercel-app.vercel.app
   ```
5. Copy your Railway public URL (e.g., `https://xyz.up.railway.app`)

### Step 2 — Deploy Frontend (Vercel)

1. Go to [vercel.com](https://vercel.com) → **New Project → Import from GitHub**
2. Select this repo → set **Root Directory** to `frontend`
3. Set these environment variables in Vercel:
   ```
   NEXT_PUBLIC_BACKEND_URL=https://xyz.up.railway.app
   NEXT_PUBLIC_WS_URL=wss://xyz.up.railway.app
   ```
4. Click Deploy ✅

---

## 🛠️ Local Development

### Prerequisites
- Python 3.11+
- Node.js 18+

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate         # Windows
# source venv/bin/activate    # Linux/Mac
pip install -r requirements.txt

# (Optional) Train per-asset models
python train_model.py         # BTC  (~20 min, uses local CSVs)
python train_eth_model.py     # ETH  (~2 min, fetches from Binance)
python train_gold_model.py    # GOLD (~1 min, fetches from yfinance)

# Start server
python main.py                # → http://localhost:8000
```

### Frontend
```bash
cd frontend
npm install

# Create local env file
cp .env.example .env.local
# Edit .env.local — local dev defaults point to localhost:8000

npm run dev                   # → http://localhost:3000
```

---

## 🧠 Per-Asset Models

| Asset | Data | Candles | Walk-Forward Acc |
|---|---|---|---|
| **BTC** | Local 1m CSVs (2017–2026) | ~4M | Retrain: run `train_model.py` |
| **ETH** | Binance daily API | 3115 | **53.78%** |
| **GOLD** | yfinance GC=F daily | ~500 | Rule-based fallback |

All models fall back to rule-based predictions if `.pkl` files are missing (safe for first deploy).

---

## 📊 Features

- **Real-time WebSocket** streaming for BTC, ETH, GOLD
- **LightGBM ML models** tuned per-asset (regime, halving cycle, ETH/BTC ratio)
- **Prediction smoothing** — reduces flip-flopping near 50% confidence
- **News sentiment** — free RSS feeds, no API key required
- **Paper trading** — 10-unit demo desk (BTC/ETH/GOLD units, USD P&L)
  - Manual trades with optional stop-loss and auto-exit timer
  - Auto-trading with configurable confidence threshold and SL%
- **GOLD market hours** detection (COMEX Mon–Fri)

---

## 📁 Project Structure

```
BEAR_BULL_PREDECTION/
├── frontend/                   # Next.js 14 app (deploys to Vercel)
│   ├── src/app/
│   │   ├── page.tsx           # Main trader page
│   │   └── components/
│   │       └── TradingPanel.tsx
│   ├── .env.example
│   └── vercel.json            # (optional, also at root)
│
├── backend/                    # FastAPI app (deploys to Railway)
│   ├── api/
│   │   ├── ws.py              # WebSocket endpoint + prediction engine
│   │   └── rest.py            # REST endpoints
│   ├── models/                # Trained .pkl files (gitignored)
│   ├── main.py
│   ├── requirements.txt
│   ├── railway.toml
│   ├── train_model.py         # BTC trainer
│   ├── train_eth_model.py     # ETH trainer
│   └── train_gold_model.py    # GOLD trainer
│
├── vercel.json                 # Root Vercel config
└── .gitignore
```

---

## ⚠️ Notes

- **CSV data files** (`datas/`) are excluded from git (102 files, ~700MB). BTC model will auto-fall back to Binance live data on Railway if no CSVs.
- **Model `.pkl` files** are excluded from git (too large). The backend gracefully falls back to rule-based predictions without them.
- Update `vercel.json` → `rewrites.destination` with your actual Railway URL.
