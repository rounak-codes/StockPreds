import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

import streamlit as st
from utils import get_nse_stock_list, fetch_stock_data
from predict import predict_from_dataframe
import pandas as pd

# Define correct model directory path from streamlit_app's perspective
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts/models'))

def get_available_stocks(stock_list, region):
    if region == "india":
        suffix = ".ns.pkl"
    elif region == "global":
        suffix = ".pkl"
    else:
        return []

    prefix = f"{region}_model_"
    model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith(prefix) and f.endswith(suffix)]

    # Extract symbol names from filenames
    model_symbols = [f[len(prefix):-len(suffix)] for f in model_files]

    # Match only those stocks that have trained models
    return [s for s in stock_list if s.lower() in model_symbols]


# Global Stock List (from fetchdata.py)
GLOBAL_TOP_50 = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "BRK-B", "UNH", "JNJ",
    "V", "WMT", "JPM", "MA", "PG", "XOM", "HD", "CVX", "LLY", "ABBV",
    "PEP", "KO", "BAC", "MRK", "AVGO", "ADBE", "COST", "CSCO", "TMO", "DIS",
    "DHR", "NKE", "PFE", "MCD", "ABT", "ACN", "CRM", "INTC", "WFC", "LIN",
    "TXN", "AMD", "NEE", "PM", "BMY", "UNP", "RTX", "LOW", "IBM", "QCOM"
]

# UI
st.title("🌍 Stock Trend Predictor")

# Tabs for Indian and Global Stocks
tab1, tab2 = st.tabs(["🇮🇳 Indian Stocks", "🌎 Global Stocks"])

# ========== INDIAN STOCKS TAB ==========
with tab1:
    st.subheader("📈 Indian Stock Trend Predictor")
    stock_list = get_available_stocks(get_nse_stock_list(), "india")
    selected_stock = st.selectbox("Choose an NSE Stock", stock_list, key="india")

    if selected_stock:
        st.subheader(f"Stock: {selected_stock}")
        data = fetch_stock_data(selected_stock, is_indian=True)

        if not data.empty:
            df = data.reset_index()
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
            model_filename = next((f for f in os.listdir(MODELS_DIR)
                      if f.startswith(f"india_model_{selected_stock.lower()}") and f.endswith(".ns.pkl")), None)

            if model_filename:
                model_path = os.path.join(MODELS_DIR, model_filename)
            else:
                st.warning(f"No trained model found for {selected_stock}. Please train it first.")
                st.stop()


            if os.path.exists(model_path):
                try:
                    prediction_df = predict_from_dataframe(df.copy(), model_path)
                    df['Predicted_Close'] = prediction_df['Predicted_Close']
                    st.line_chart(df[['Close', 'Predicted_Close']])
                    st.dataframe(df[['Date', 'Close', 'Predicted_Close']].tail())
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
            else:
                st.warning(f"No trained model found for {selected_stock}. Please train it first.")
        else:
            st.error("No data found for this stock.")

# ========== GLOBAL STOCKS TAB ==========
with tab2:
    st.subheader("📊 Global Stock Trend Predictor")
    global_stock_list = get_available_stocks(GLOBAL_TOP_50, "global")
    selected_global_stock = st.selectbox("Choose a Global Stock", global_stock_list, key="global")

    if selected_global_stock:
        st.subheader(f"Stock: {selected_global_stock}")
        data = fetch_stock_data(selected_global_stock, is_indian=False)

        if not data.empty:
            df = data.reset_index()
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
            model_path = os.path.join(MODELS_DIR, f"global_model_{selected_global_stock.lower()}.pkl")

            if os.path.exists(model_path):
                try:
                    prediction_df = predict_from_dataframe(df.copy(), model_path)
                    df['Predicted_Close'] = prediction_df['Predicted_Close']
                    st.line_chart(df[['Close', 'Predicted_Close']])
                    st.dataframe(df[['Date', 'Close', 'Predicted_Close']].tail())
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
            else:
                st.warning(f"No trained model found for {selected_global_stock}. Please train it first.")
        else:
            st.error("No data found for this stock.")
