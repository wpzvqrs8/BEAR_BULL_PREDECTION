from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import asyncio

from api.rest import router as rest_router
from api.ws import router as ws_router, _startup_train

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Fetch real Binance data and train the LightGBM model at server startup."""
    print("[Startup] Fetching BTC-USD historical data from Binance and training model...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _startup_train, "BTC-USD")
    print("[Startup] Model ready.")
    yield
    # (cleanup on shutdown if needed)


app = FastAPI(
    title="Stock Market Predictor API",
    description="Institutional-grade system for stock market prediction & backtesting",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rest_router, prefix="/api")
app.include_router(ws_router)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Stock Market Predictor API is running."}
