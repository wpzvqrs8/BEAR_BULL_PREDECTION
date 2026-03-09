import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

from api.db import init_db, upsert_candles, prune_old_candles
from api.ws import _fetch_binance_klines, _fetch_gold_historical, _ASSET_MODEL_FILES
from api.ws import _fetch_binance_klines

def seed_database():
    print("Initializing Database Schema...")
    init_db()
    
    symbols = {
        "BTC-USD": lambda: _fetch_binance_klines("BTCUSDT", "1d", 300),
        "ETH-USD": lambda: _fetch_binance_klines("ETHUSDT", "1d", 300),
        "GOLD": lambda: _fetch_gold_historical(300),
        "USD-PHP": lambda: _fetch_binance_klines("USDTPHP", "1d", 300)
    }
    
    for symbol, fetch_func in symbols.items():
        print(f"\nFetching data for {symbol}...")
        try:
            rows = fetch_func()
            if rows:
                n = upsert_candles(symbol, rows)
                print(f"[SUCCESS] Upserted {n} rows for {symbol} into Supabase.")
                prune_old_candles(symbol)
            else:
                print(f"[FAIL] Failed to fetch data for {symbol}.")
        except Exception as e:
            print(f"[ERROR] Error seeding {symbol}: {e}")

if __name__ == "__main__":
    seed_database()
