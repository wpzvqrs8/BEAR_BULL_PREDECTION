import time
import asyncio
import pandas as pd
from typing import AsyncGenerator

class LiveSimulator:
    def __init__(self, historical_df: pd.DataFrame, playback_speed: float = 1.0):
        """
        historical_df: Processed DataFrame containing features and targets
        playback_speed: 1.0 = normal speed (e.g. 1 sec per candle)
        """
        self.df = historical_df.sort_index()
        self.playback_speed = playback_speed
        self.paused = False
        self.is_running = False

    def toggle_pause(self):
        self.paused = not self.paused
        
    def stop(self):
        self.is_running = False

    async def stream_candles(self) -> AsyncGenerator[dict, None]:
        self.is_running = True
        
        for idx, row in self.df.iterrows():
            if not self.is_running:
                break
                
            while self.paused:
                await asyncio.sleep(0.5)
                
            # Yield dictionary with candle info and features
            payload = {
                "timestamp": str(idx),
                "open": row.get("Open"),
                "high": row.get("High"),
                "low": row.get("Low"),
                "close": row.get("Close"),
                "volume": row.get("Volume")
            }
            
            yield payload
            
            # Simulate real-time delay adjusted by playback speed
            await asyncio.sleep(1.0 / self.playback_speed)
