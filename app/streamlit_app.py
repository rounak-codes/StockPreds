import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

import streamlit as st
from utils import get_nse_stock_list, fetch_stock_data
from predict import predict_from_dataframe
import pandas as pd
import os

st.title("📈 Indian Stock Trend Predictor")
st.write("Select a stock to view its historical trend and prediction:")

stock_list = get_nse_stock_list()
selected_stock = st.selectbox("Choose a Stock", stock_list)

if selected_stock:
    st.subheader(f"Stock: {selected_stock}")
    data = fetch_stock_data(selected_stock)

    if not data.empty:
        # Reset index to get Date column
        df = data.reset_index()

        # Add Date column in string format (if not already)
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

        # Run prediction using your model
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
            st.warning(f"No trained model found for {selected_stock}. Please train it and place it at `{model_path}`")

    else:
        st.error("No data found for this stock.")
