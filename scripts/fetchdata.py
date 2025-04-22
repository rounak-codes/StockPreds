import yfinance as yf
import os
import pandas as pd
import time

def save_stock_data(symbol, path, period="1y"):
    try:
        data = yf.Ticker(symbol).history(period=period)
        if not data.empty:
            data.reset_index(inplace=True)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data.to_csv(path, index=False)
            print(f"✅ Saved {symbol} to {path}")
        else:
            print(f"❌ Empty data for {symbol}")
    except Exception as e:
        print(f"❌ Error fetching {symbol}: {e}")

# Nifty 50 (India)
NIFTY_50_SYMBOLS = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "LT", "HINDUNILVR", "SBIN", "AXISBANK", "KOTAKBANK",
    "ITC", "BHARTIARTL", "BAJFINANCE", "ASIANPAINT", "HCLTECH", "WIPRO", "MARUTI", "ULTRACEMCO", "TITAN", "NESTLEIND",
    "SUNPHARMA", "TECHM", "BAJAJFINSV", "POWERGRID", "NTPC", "ADANIENT", "ONGC", "JSWSTEEL", "CIPLA", "INDUSINDBK",
    "TATASTEEL", "HDFCLIFE", "COALINDIA", "BPCL", "HINDALCO", "DIVISLAB", "UPL", "BRITANNIA", "GRASIM", "EICHERMOT",
    "ADANIPORTS", "APOLLOHOSP", "BAJAJ-AUTO", "SBILIFE", "HEROMOTOCO", "M&M", "DRREDDY", "SHREECEM", "ICICIPRULI", "TATAMOTORS"
]

# Top 50 Global Stocks (mostly US-based, ticker symbols)
GLOBAL_TOP_50 = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "BRK-B", "UNH", "JNJ",
    "V", "WMT", "JPM", "MA", "PG", "XOM", "HD", "CVX", "LLY", "ABBV",
    "PEP", "KO", "BAC", "MRK", "AVGO", "ADBE", "COST", "CSCO", "TMO", "DIS",
    "DHR", "NKE", "PFE", "MCD", "ABT", "ACN", "CRM", "INTC", "WFC", "LIN",
    "TXN", "AMD", "NEE", "PM", "BMY", "UNP", "RTX", "LOW", "IBM", "QCOM"
]

if __name__ == "__main__":
    delay_seconds = 4  # Delay to avoid rate limiting
    period = "2y"

    # Global Stocks
    for stock in GLOBAL_TOP_50:
        save_stock_data(stock, f"data/global/{stock}.csv", period=period)
        time.sleep(delay_seconds)

    # Indian Stocks
    for stock in NIFTY_50_SYMBOLS:
        save_stock_data(f"{stock}.NS", f"data/india/{stock}.NS.csv", period=period)
        time.sleep(delay_seconds)
