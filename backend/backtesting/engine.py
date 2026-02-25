import pandas as pd
import numpy as np

class BacktestEngine:
    def __init__(self, initial_capital: float = 100000.0, transaction_cost: float = 0.001, slippage: float = 0.0005):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        
    def run(self, df: pd.DataFrame, predictions: pd.Series, probabilities: pd.Series) -> dict:
        """
        Chronological simulation evaluating row by row.
        df must contain 'Close' prices and return data.
        """
        equity = self.initial_capital
        equity_curve = []
        trades = []
        
        # Ensure aligned indices
        aligned_df = df.loc[predictions.index].copy()
        
        position = 0 # 1 for Long, 0 for Flat (we won't short for simplicity, or 1/-1 if we short)
        
        # Simulating candle by candle evolution
        for idx, row in aligned_df.iterrows():
            pred = predictions.loc[idx]
            prob = probabilities.loc[idx]
            close_price = row['Close']
            
            # Determine target position: 1 if Bullish (pred==1), 0 if Bearish
            target_position = 1 if pred == 1 else 0
            
            # Execute Trade
            if target_position != position:
                # Calculate cost on capital traded
                cost_ratio = self.transaction_cost + self.slippage
                cost = equity * cost_ratio
                equity -= cost
                
                trade = {
                    'timestamp': idx,
                    'type': 'BUY' if target_position == 1 else 'SELL',
                    'price': close_price,
                    'cost': cost,
                    'probability': prob
                }
                trades.append(trade)
                position = target_position
                
            # Accrue return if in position (using next_candle_return pre-calculated for truth,
            # or calculating from shift. Since 'next_candle_return' shifts -1, 
            # multiplying current position by next_candle_return gives the realized PnL of this candle
            if position == 1 and 'next_candle_return' in row:
                equity *= (1 + row['next_candle_return'])
                
            equity_curve.append({
                'timestamp': idx,
                'equity': equity
            })
            
        equity_df = pd.DataFrame(equity_curve).set_index('timestamp')
        
        # Calculate Metrics
        returns = equity_df['equity'].pct_change().dropna()
        total_return = (equity - self.initial_capital) / self.initial_capital
        
        if len(returns) > 0 and returns.std() != 0:
            sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())
        else:
            sharpe_ratio = 0.0
            
        # Drawdown
        cummax = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - cummax) / cummax
        max_drawdown = drawdown.min()
        
        # Win Rate calculation
        profitable_trades = sum(1 for i in range(1, len(trades)) if trades[i]['price'] > trades[i-1]['price'] and trades[i-1]['type'] == 'BUY')
        total_round_trips = len(trades) // 2
        win_rate = profitable_trades / total_round_trips if total_round_trips > 0 else 0.0

        return {
            "initial_capital": self.initial_capital,
            "final_equity": equity,
            "total_return_pct": total_return * 100,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown_pct": max_drawdown * 100,
            "win_rate": win_rate,
            "equity_curve": equity_curve,
            "drawdown_curve": drawdown.tolist(),
            "trades": trades
        }
