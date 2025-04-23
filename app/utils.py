import pandas as pd
import yfinance as yf

# Fetch all NSE stock symbols
def get_nse_stock_list():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    df = pd.read_csv(url)
    return df['SYMBOL'].dropna().unique().tolist()

# Fetch historical stock data for Indian and Global stocks
def fetch_stock_data(symbol, is_indian=True, period="6mo"):
    if is_indian:
        ticker = yf.Ticker(f"{symbol}.NS")
    else:
        ticker = yf.Ticker(symbol)
    return ticker.history(period=period)
