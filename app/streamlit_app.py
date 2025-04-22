import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

import streamlit as st
from utils import get_nse_stock_list, fetch_stock_data
from predict import predict_from_dataframe
import pandas as pd

# UI
st.title("🌍 Stock Trend Predictor")

# Tabs for Indian and Global Stocks
tab1, tab2 = st.tabs(["🇮🇳 Indian Stocks", "🌎 Global Stocks"])

# ========== INDIAN STOCKS TAB ==========
with tab1:
    st.subheader("📈 Indian Stock Trend Predictor")
    stock_list = get_nse_stock_list()
    selected_stock = st.selectbox("Choose an NSE Stock", stock_list, key="india")

    if selected_stock:
        st.subheader(f"Stock: {selected_stock}")
        data = fetch_stock_data(selected_stock, is_indian=True)

        if not data.empty:
            df = data.reset_index()
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
            
            # Update model path to follow the new naming convention
            model_path = f"models/india_model_{selected_stock.lower()}.pkl"

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
    # Expanded list of global stocks
    global_stock_list = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX", "BABA", "IBM"]
    selected_global_stock = st.selectbox("Choose a Global Stock", global_stock_list, key="global")

    if selected_global_stock:
        st.subheader(f"Stock: {selected_global_stock}")
        data = fetch_stock_data(selected_global_stock, is_indian=False)

        if not data.empty:
            df = data.reset_index()
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
            
            # Update model path to follow the new naming convention
            model_path = f"models/global_model_{selected_global_stock.lower()}.pkl"

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
