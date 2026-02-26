from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import asyncio, os

from api.rest import router as rest_router
from api.ws import router as ws_router, _startup_train

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Init DB tables
    try:
        from api.db import init_db
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, init_db)
    except Exception as e:
        print(f"[Startup] DB init skipped (no DATABASE_URL?): {e}")

    # 2. Load ML models + seed candle buffers
    print("[Startup] Loading models and seeding buffers...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _startup_train, "BTC-USD")
    print("[Startup] Ready.")

    # 3. Start background DB sync task (saves candles every 30 min)
    sync_task = asyncio.create_task(_db_sync_loop())

    yield

    sync_task.cancel()
    try:
        await sync_task
    except asyncio.CancelledError:
        pass


async def _db_sync_loop():
    """Save completed daily candles to Supabase every 30 minutes."""
    INTERVAL = 30 * 60   # seconds
    await asyncio.sleep(60)  # wait for buffers to fill first
    while True:
        try:
            from api.db import upsert_candles, prune_old_candles
            from api.ws import candle_buffers
            for symbol, buf in candle_buffers.items():
                rows = list(buf)
                if rows:
                    n = upsert_candles(symbol, rows)
                    print(f"[DB sync] {symbol}: {n} rows upserted")
                    prune_old_candles(symbol)
        except Exception as e:
            print(f"[DB sync] Error: {e}")
        await asyncio.sleep(INTERVAL)


app = FastAPI(
    title="Stock Market Predictor API",
    description="Institutional-grade system for stock market prediction & backtesting",
    version="2.1.0",
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
    return {"status": "ok", "message": "Stock Market Predictor API v2.1 is running."}
