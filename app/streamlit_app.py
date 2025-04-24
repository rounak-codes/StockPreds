import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

import streamlit as st
import pandas as pd
from utils import get_nse_stock_list, fetch_stock_data
from predict import predict_from_dataframe

# Where your .pkl models live
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts/models'))

def get_available_stocks(stock_list, region):
    suffix = ".ns.pkl" if region == "india" else ".pkl"
    prefix = f"{region}_model_"
    files = [f for f in os.listdir(MODELS_DIR) if f.startswith(prefix) and f.endswith(suffix)]
    syms = [f[len(prefix):-len(suffix)] for f in files]
    return [s for s in stock_list if s.lower() in syms]

# Global candidates
GLOBAL_TOP_50 = [
    "AAPL","MSFT","GOOGL","AMZN","META","TSLA","NVDA","BRK-B","UNH","JNJ",
    "V","WMT","JPM","MA","PG","XOM","HD","CVX","LLY","ABBV",
    "PEP","KO","BAC","MRK","AVGO","ADBE","COST","CSCO","TMO","DIS",
    "DHR","NKE","PFE","MCD","ABT","ACN","CRM","INTC","WFC","LIN",
    "TXN","AMD","NEE","PM","BMY","UNP","RTX","LOW","IBM","QCOM"
]

st.title("🌍 Stock Trend Predictor")
tab1, tab2 = st.tabs(["🇮🇳 Indian Stocks", "🌎 Global Stocks"])

for region, tab, is_indian in [
    ("india", tab1, True),
    ("global", tab2, False)
]:
    with tab:
        st.subheader(f"{'Indian' if is_indian else 'Global'} Stock Trend Predictor")
        base_list = get_nse_stock_list() if is_indian else GLOBAL_TOP_50
        avail = get_available_stocks(base_list, region)
        symbol = st.selectbox("Choose a Stock", avail, key=f"sel_{region}")

        if not symbol:
            st.info("No model available.")
            continue

        # Historical fetch
        data = fetch_stock_data(symbol, is_indian=is_indian)
        if data.empty:
            st.error("No data fetched.")
            continue

        df = data.reset_index()
        model_files = [f for f in os.listdir(MODELS_DIR)
                       if f.startswith(f"{region}_model_{symbol.lower()}")]
        if not model_files:
            st.warning("Model file missing.")
            continue
        model_path = os.path.join(MODELS_DIR, model_files[0])

        # Future duration selector
        dur = st.selectbox("Prediction Duration", ["None", "6 Months", "1 Year", "2 Years"],
                           key=f"dur_{region}")
        months_map = {"6 Months": 6, "1 Year": 12, "2 Years": 24}
        future_months = months_map.get(dur, 0)

        # Run predictions
        in_sample, future = predict_from_dataframe(df.copy(), model_path, future_months)

        # Show in-sample
        in_sample['Date'] = pd.to_datetime(in_sample['Date'])
        in_sample = in_sample.set_index('Date')
        st.markdown("**Historical vs In-Sample Prediction**")
        st.line_chart(in_sample[['Close', 'Predicted_Close']])
        st.dataframe(in_sample[['Close', 'Predicted_Close']].tail().reset_index())

        # Show future if selected
        if future_months:
            future['Date'] = pd.to_datetime(future['Date'])
            future = future.set_index('Date')
            st.markdown(f"**Out-of-Sample Forecast ({dur})**")
            st.line_chart(future[['Predicted_Close']])
            st.dataframe(future.reset_index())
