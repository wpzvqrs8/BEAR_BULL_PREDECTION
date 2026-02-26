"""
db.py — Supabase PostgreSQL persistence for OHLCV candles.

Storage strategy (minimal footprint):
  - One table: `candles` (symbol, ts, open, high, low, close, volume)
  - Daily candles only  → 3 assets × 365 rows ≈ ~65 KB/year
  - Auto-prune: keeps only last KEEP_ROWS rows per symbol
  - UPSERT on (symbol, ts) to avoid duplicates
"""
from __future__ import annotations
import os, time
from typing import Optional
import psycopg2
from psycopg2.extras import execute_values

# ── Connection ────────────────────────────────────────────────────────────────
# Store DATABASE_URL in backend/.env — never commit the real URL to git
_DATABASE_URL: str = os.getenv("DATABASE_URL", "")
_conn: Optional[psycopg2.extensions.connection] = None

KEEP_ROWS = 500   # rows to keep per symbol (500 daily ≈ 1.4 years, ~60KB total)


def get_conn() -> psycopg2.extensions.connection:
    """Return (or lazily create) a persistent DB connection."""
    global _conn
    if _conn is None or _conn.closed:
        if not _DATABASE_URL:
            raise RuntimeError("DATABASE_URL env var is not set.")
        _conn = psycopg2.connect(_DATABASE_URL, connect_timeout=10)
        _conn.autocommit = False
    return _conn


# ── Schema init ───────────────────────────────────────────────────────────────
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS candles (
    symbol  VARCHAR(12)   NOT NULL,
    ts      BIGINT        NOT NULL,   -- Unix epoch seconds
    open    NUMERIC(18,4) NOT NULL,
    high    NUMERIC(18,4) NOT NULL,
    low     NUMERIC(18,4) NOT NULL,
    close   NUMERIC(18,4) NOT NULL,
    volume  NUMERIC(24,2) NOT NULL DEFAULT 0,
    PRIMARY KEY (symbol, ts)
);
CREATE INDEX IF NOT EXISTS idx_candles_symbol_ts ON candles (symbol, ts DESC);
"""

def init_db() -> None:
    """Create tables if they don't exist. Called once at startup."""
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
        conn.commit()
        print("[db] Tables ready ✓")
    except Exception as e:
        print(f"[db] init_db error: {e}")


# ── UPSERT candles ────────────────────────────────────────────────────────────
def upsert_candles(symbol: str, rows: list[dict]) -> int:
    """
    Insert or update candle rows.

    Args:
        symbol: e.g. 'BTC-USD', 'ETH-USD', 'GOLD'
        rows:   list of dicts with keys: time/ts, open, high, low, close, volume

    Returns:
        Number of rows upserted.
    """
    if not rows:
        return 0
    try:
        conn = get_conn()
        values = []
        for r in rows:
            ts = int(r.get("time") or r.get("ts") or 0)
            if ts <= 0:
                continue
            values.append((
                symbol, ts,
                float(r.get("open",  r.get("close", 0))),
                float(r.get("high",  r.get("close", 0))),
                float(r.get("low",   r.get("close", 0))),
                float(r.get("close", 0)),
                float(r.get("volume", 0)),
            ))
        if not values:
            return 0
        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO candles (symbol, ts, open, high, low, close, volume)
                VALUES %s
                ON CONFLICT (symbol, ts) DO UPDATE SET
                    open   = EXCLUDED.open,
                    high   = EXCLUDED.high,
                    low    = EXCLUDED.low,
                    close  = EXCLUDED.close,
                    volume = EXCLUDED.volume
            """, values)
        conn.commit()
        return len(values)
    except Exception as e:
        print(f"[db] upsert_candles error ({symbol}): {e}")
        try:
            get_conn().rollback()
        except Exception:
            pass
        return 0


# ── Fetch candles ─────────────────────────────────────────────────────────────
def fetch_candles(symbol: str, limit: int = KEEP_ROWS) -> list[dict]:
    """
    Return the most recent `limit` candles for a symbol, sorted ascending by ts.
    """
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ts, open, high, low, close, volume
                FROM candles
                WHERE symbol = %s
                ORDER BY ts DESC
                LIMIT %s
            """, (symbol, limit))
            rows = cur.fetchall()
        return [
            {"time": r[0], "open": float(r[1]), "high": float(r[2]),
             "low": float(r[3]), "close": float(r[4]), "volume": float(r[5])}
            for r in reversed(rows)    # ascending order for chart
        ]
    except Exception as e:
        print(f"[db] fetch_candles error ({symbol}): {e}")
        return []


# ── Auto-prune old rows ────────────────────────────────────────────────────────
def prune_old_candles(symbol: str, keep: int = KEEP_ROWS) -> None:
    """Delete rows older than the most recent `keep` rows for a symbol."""
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM candles
                WHERE symbol = %s AND ts < (
                    SELECT ts FROM candles
                    WHERE symbol = %s
                    ORDER BY ts DESC
                    OFFSET %s LIMIT 1
                )
            """, (symbol, symbol, keep))
        conn.commit()
    except Exception as e:
        print(f"[db] prune error ({symbol}): {e}")
        try:
            get_conn().rollback()
        except Exception:
            pass
