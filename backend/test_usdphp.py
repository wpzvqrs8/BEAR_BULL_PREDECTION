import yfinance as yf
fi = yf.Ticker("PHP=X").fast_info
print("lastPrice:", fi.get("lastPrice", "Not found via get"))
print("Attr lastPrice:", getattr(fi, "lastPrice", "Not found via attr"))
print("price:", fi.get("price"))
