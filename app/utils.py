import pandas as pd
import yfinance as yf

# Fetch all NSE stock symbols
def get_nse_stock_list():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    df = pd.read_csv(url)
    return df['SYMBOL'].dropna().unique().tolist()

# Fetch historical stock data
def fetch_stock_data(symbol, period="6mo"):
    ticker = yf.Ticker(f"{symbol}.NS")
    return ticker.history(period=period)
