from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
import time
from typing import Dict, Any, List

router = APIRouter()

class TrainRequest(BaseModel):
    symbol: str
    model_type: str
    start_date: str = "2020-01-01"
    end_date: str = "2023-01-01"

class PredictRequest(BaseModel):
    symbol: str

APP_STATE = {
    "is_training": False,
    "last_trained_model": None,
    "backtest_results": None
}

def mock_training_task(symbol: str, model_type: str):
    APP_STATE["is_training"] = True
    time.sleep(5)
    APP_STATE["is_training"] = False
    APP_STATE["last_trained_model"] = f"{model_type}_{symbol}"
    APP_STATE["backtest_results"] = {
        "sharpe_ratio": 1.5,
        "max_drawdown_pct": -15.2,
        "win_rate": 0.58,
        "total_return_pct": 45.3
    }

@router.post("/train")
async def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    if APP_STATE["is_training"]:
        raise HTTPException(status_code=400, detail="Training already in progress")
    background_tasks.add_task(mock_training_task, req.symbol, req.model_type)
    return {"message": "Training started in background", "symbol": req.symbol}

@router.get("/status")
async def get_status():
    return APP_STATE

@router.post("/predict")
async def make_prediction(req: PredictRequest):
    return {
        "symbol": req.symbol,
        "bull_probability": 0.65,
        "bear_probability": 0.35,
        "confidence_score": 0.8,
        "model_used": APP_STATE["last_trained_model"] or "baseline_lgbm",
        "timestamp": time.time()
    }

# ── Historical candle endpoint (used by frontend on page load) ────────────────
@router.get("/history/{symbol}")
async def get_history(symbol: str, limit: int = 365) -> Dict[str, Any]:
    """
    Return stored daily candles for a symbol from Supabase.
    Frontend calls this on page load to pre-populate the chart with past data.
    """
    try:
        from api.db import fetch_candles
        rows = fetch_candles(symbol.upper(), limit=min(limit, 500))
        return {"symbol": symbol, "type": "history", "data": rows, "count": len(rows)}
    except Exception as e:
        # DB unavailable — return empty (websocket will fill in live data)
        print(f"[REST] history/{symbol} error: {e}")
        return {"symbol": symbol, "type": "history", "data": [], "count": 0}
