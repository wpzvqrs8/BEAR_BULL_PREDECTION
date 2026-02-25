# This directory stores trained ML model files (.pkl, .json).
# These files are NOT committed to git (too large for GitHub).
#
# To populate this directory:
#   1. Run:  python backend/train_model.py      (BTC — ~20 min)
#   2. Run:  python backend/train_eth_model.py  (ETH — ~2 min)
#   3. Run:  python backend/train_gold_model.py (GOLD — requires yfinance)
#
# On Railway/Render deployment:
#   The backend will auto-fall back to rule-based predictions if no .pkl files
#   are present and will fetch live data from Binance/yfinance at startup.
