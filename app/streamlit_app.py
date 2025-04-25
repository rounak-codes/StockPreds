import os
import sys
import pandas as pd
import streamlit as st
import altair as alt

from prophet.plot import plot_components_plotly

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))
from lstm_predict import predict_with_lstm
from prophet_predict import predict_with_prophet
from utils import get_nse_stock_list, fetch_stock_data
from rf_predict import predict_from_dataframe

# Model directory
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts/models'))

def get_available_stocks(stock_list, region):
    suffix = ".ns.pkl" if region == "india" else ".pkl"
    prefix = f"{region}_model_"
    files = [f for f in os.listdir(MODELS_DIR) if f.startswith(prefix) and f.endswith(suffix)]
    syms = [f[len(prefix):-len(suffix)] for f in files]
    return [s for s in stock_list if s.lower() in syms]

def ensemble_predictions(df1, df2):
    # Convert both dataframes' Date columns to timezone-naive
    df1 = df1.copy()
    df2 = df2.copy()
    
    # Convert to timezone-naive if timezone info exists
    if hasattr(df1['Date'].dtype, 'tz') and df1['Date'].dtype.tz is not None:
        df1['Date'] = df1['Date'].dt.tz_localize(None)
    
    if hasattr(df2['Date'].dtype, 'tz') and df2['Date'].dtype.tz is not None:
        df2['Date'] = df2['Date'].dt.tz_localize(None)
    
    # Now merge with timezone-naive dates
    merged = pd.merge(df1, df2, on="Date", suffixes=('_lstm', '_prophet'))
    
    # Check if 'Close' exists in either of the input dataframes
    return_cols = ['Date', 'Predicted_Close']
    
    # Only include 'Close' in output if it exists in at least one of the dataframes
    if 'Close' in df1.columns or 'Close' in df2.columns:
        # Add 'Close' from whichever dataframe has it
        if 'Close' in df1.columns:
            merged['Close'] = df1['Close']
        elif 'Close' in df2.columns:
            merged['Close'] = df2['Close']
        return_cols.insert(1, 'Close')
    
    merged['Predicted_Close'] = (merged['Predicted_Close_lstm'] + merged['Predicted_Close_prophet']) / 2
    
    # Return only selected columns
    return merged[return_cols]

# Global candidates
GLOBAL_TOP_50 = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "BRK-B", "UNH", "JNJ",
    "V", "WMT", "JPM", "MA", "PG", "XOM", "HD", "CVX", "LLY", "ABBV",
    "PEP", "KO", "BAC", "MRK", "AVGO", "ADBE", "COST", "CSCO", "TMO", "DIS",
    "DHR", "NKE", "PFE", "MCD", "ABT", "ACN", "CRM", "INTC", "WFC", "LIN",
    "TXN", "AMD", "NEE", "PM", "BMY", "UNP", "RTX", "LOW", "IBM", "QCOM"
]

st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("🌍 Stock Trend Predictor")
tab1, tab2 = st.tabs(["🇮🇳 Indian Stocks", "🌎 Global Stocks"])

for region, tab, is_indian in [("india", tab1, True), ("global", tab2, False)]:
    with tab:
        st.subheader(f"{'Indian' if is_indian else 'Global'} Stock Trend Predictor")
        base_list = get_nse_stock_list() if is_indian else GLOBAL_TOP_50
        avail = get_available_stocks(base_list, region)
        # three columns: stock selector, duration selector, model selector
        col_stock, col_dur, col_model = st.columns([3, 2, 5])

        with col_stock:
            symbol = st.selectbox("Choose a Stock", avail, key=f"sel_{region}")

        with col_dur:
            dur = st.selectbox(
                "Prediction Duration",
                ["6 Months", "1 Year", "2 Years"],
                key=f"dur_{region}_{symbol}_unique"  # Make the key unique by including symbol
            )

        with col_model:
            model_type = st.radio(
                "Model Type",
                ["Traditional", "LSTM", "Prophet", "Prophet + LSTM"],
                horizontal=True,
                key=f"model_{region}_unique"
            )

        # now you have symbol, dur, and model_type defined
        months_map = {"6 Months": 6, "1 Year": 12, "2 Years": 24}
        future_months = months_map.get(dur, 0)

        if not symbol:
            st.info("No model available.")
            continue

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
        future_months = months_map.get(dur, 0)

        in_sample, future = pd.DataFrame(), pd.DataFrame()

        if model_type == "Traditional":
            in_sample, future = predict_from_dataframe(df.copy(), model_path, future_months)

        elif model_type == "LSTM":
            in_sample, future = predict_with_lstm(df.copy(), symbol, region, future_months)

        elif model_type == "Prophet":
            in_sample, future, _, _ = predict_with_prophet(df.copy(), future_months)

        elif model_type == "Prophet + LSTM":
            ins1, fut1 = predict_with_lstm(df.copy(), symbol, region, future_months)[:2]
            ins2, fut2, _, _ = predict_with_prophet(df.copy(), future_months)
            in_sample = ensemble_predictions(ins1, ins2)
            future = ensemble_predictions(fut1, fut2)

        if not in_sample.empty:
            # Make sure in_sample has 'Close' and 'Predicted_Close' columns
            required_cols = ['Close', 'Predicted_Close']
            if not all(col in in_sample.columns for col in required_cols):
                st.error(f"Missing columns in data: {set(required_cols) - set(in_sample.columns)}. Available columns: {in_sample.columns}")
                continue
                
            in_sample['Date'] = pd.to_datetime(in_sample['Date'])
            in_sample = in_sample.set_index('Date')

            left_col, right_col = st.columns([1, 1])

            with left_col:
                st.markdown("**Historical vs In-Sample Prediction**")
                chart = alt.Chart(in_sample.reset_index()).transform_fold(
                    ['Close', 'Predicted_Close']
                ).mark_line().encode(
                    x='Date:T',
                    y='value:Q',
                    color='key:N',
                    tooltip=['Date:T', 'key:N', 'value:Q']
                ).properties(width=675, height=400).interactive()
                st.altair_chart(chart, use_container_width=False)
                
                # Display dataframe safely checking for columns first
                display_cols = [col for col in ['Close', 'Predicted_Close'] if col in in_sample.columns]
                if display_cols:
                    st.dataframe(in_sample[display_cols].tail().reset_index())
                else:
                    st.error("Required columns not found in the data")

            with right_col:
                if future_months and not future.empty:
                    future['Date'] = pd.to_datetime(future['Date'])
                    future = future.set_index('Date')
                    st.markdown(f"**Forecast ({dur})**")
                    future_chart = alt.Chart(future.reset_index()).mark_line().encode(
                        x='Date:T',
                        y='Predicted_Close:Q',
                        tooltip=['Date:T', 'Predicted_Close:Q']  # Fixed tooltip field name
                    ).properties(width='container', height=400).interactive()
                    st.altair_chart(future_chart, use_container_width=True)
                    st.dataframe(future.reset_index())