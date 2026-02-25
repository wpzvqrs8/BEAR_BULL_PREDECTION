from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
import time
from typing import Dict, Any

router = APIRouter()

class TrainRequest(BaseModel):
    symbol: str
    model_type: str
    start_date: str = "2020-01-01"
    end_date: str = "2023-01-01"

class PredictRequest(BaseModel):
    symbol: str

# Mock global state for simulation
APP_STATE = {
    "is_training": False,
    "last_trained_model": None,
    "backtest_results": None
}

def mock_training_task(symbol: str, model_type: str):
    APP_STATE["is_training"] = True
    time.sleep(5)  # Simulate training
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
    # Output format specifically requested
    return {
      "symbol": req.symbol,
      "bull_probability": 0.65,
      "bear_probability": 0.35,
      "confidence_score": 0.8,
      "model_used": APP_STATE["last_trained_model"] or "baseline_xgboost",
      "timestamp": time.time()
    }
