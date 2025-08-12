Run it

1) Put your minute CSV at data/minute_feed.csv  (rename MNQ_09-12_24_MARKET_HOURS_FILLED.csv)
2) Put reversed_moby_strategy.py in the project root
3) pip install -r requirements.txt
4) streamlit run ui/app.py

What you get
- Strategy run with your parameters
- Pre-context dropdown in steps of 5
- Vectors + labels per trade
- KDTree nearest neighbors for any trade

Next
- TradingView-like trade list with pattern banner
- Group creation from top K with envelope and whitelist
- Group backtest + exports