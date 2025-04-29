import os
import sys
import pandas as pd
import streamlit as st
import altair as alt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Stock Predictor", layout="wide")

from auth_utils import init_firebase, register_user, verify_user, firebase_google_login

init_firebase()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.email = None

def login_ui():
    st.title("🔐 Login/Register")
    tab1, tab2, tab3 = st.tabs(["Login", "Register", "Google Login"])
    
    with tab1:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", key="login_button"):
            if verify_user(email, password):
                st.session_state.authenticated = True
                st.session_state.email = email
                st.success("Logged in successfully")
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        email = st.text_input("Register Email", key="register_email")
        username = st.text_input("Username", key="register_username")
        password = st.text_input("Password", type="password", key="register_password")
        if st.button("Register", key="register_button"):
            if register_user(email, username, password):
                st.success("Registered successfully")
            else:
                st.error("Registration failed")

    with tab3:
        token = st.text_area("Paste Firebase ID Token (from client-side Google login)", key="google_token")
        if st.button("Login with Google", key="google_login_button"):
            email = firebase_google_login(token)
            if email:
                st.session_state.authenticated = True
                st.session_state.email = email
                st.success(f"Logged in as {email}")
                st.rerun()
            else:
                st.error("Google login failed")

if not st.session_state.authenticated:
    login_ui()
    st.stop()
else:
    st.sidebar.success(f"Logged in as: {st.session_state.email}")
    if st.sidebar.button("Logout", key="logout_button"):
        st.session_state.authenticated = False
        st.session_state.email = None
        st.experimental_rerun()

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
    df1 = df1.copy()
    df2 = df2.copy()
    
    # Remove timezone information if present
    for df in [df1, df2]:
        if 'Date' in df.columns and hasattr(df['Date'].dtype, 'tz') and df['Date'].dtype.tz is not None:
            df['Date'] = df['Date'].dt.tz_localize(None)
    
    # Merge predictions on 'Date' with suffixes
    merged = pd.merge(df1, df2, on="Date", suffixes=('_lstm', '_prophet'))
    
    # Calculate ensemble prediction
    merged['Predicted_Close'] = (merged['Predicted_Close_lstm'] + merged['Predicted_Close_prophet']) / 2
    
    # Attempt to include 'Close' if present in either dataframe
    if 'Close' in df1.columns:
        merged['Close'] = df1['Close']
    elif 'Close' in df2.columns:
        merged['Close'] = df2['Close']
    
    # Dynamically select columns based on availability
    columns = ['Date', 'Predicted_Close']
    if 'Close' in merged.columns:
        columns.insert(1, 'Close')  # Insert 'Close' after 'Date' if present
    
    return merged[columns]

# Function to display dataframe with pagination
def display_dataframe_with_pagination(df, page_size=50, key_prefix=""):
    n_pages = (len(df) + page_size - 1) // page_size
    if n_pages > 1:
        page_num = st.number_input(f'Page ({1}-{n_pages})', min_value=1, max_value=n_pages, value=1, key=f"{key_prefix}_page")
        start_idx = (page_num - 1) * page_size
        end_idx = min(start_idx + page_size, len(df))
        
        # Display row numbers correctly based on the pagination
        st.write(f"Showing rows {start_idx + 1} to {end_idx} of {len(df)}")
        
        # Adjusting dataframe for the correct pagination range
        st.dataframe(df.iloc[start_idx:end_idx].reset_index(drop=True), use_container_width=True)
        
    else:
        st.dataframe(df.reset_index(drop=True), use_container_width=True)

# Function to compute metrics for model evaluation
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    # Avoid division by zero in MAPE calculation
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-10, y_true))) * 100
    return mae, mse, rmse, mape

# Global candidates
GLOBAL_TOP_50 = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "BRK-B", "UNH", "JNJ",
    "V", "WMT", "JPM", "MA", "PG", "XOM", "HD", "CVX", "LLY", "ABBV",
    "PEP", "KO", "BAC", "MRK", "AVGO", "ADBE", "COST", "CSCO", "TMO", "DIS",
    "DHR", "NKE", "PFE", "MCD", "ABT", "ACN", "CRM", "INTC", "WFC", "LIN",
    "TXN", "AMD", "NEE", "PM", "BMY", "UNP", "RTX", "LOW", "IBM", "QCOM"
]

st.title("🌍 Stock Trend Predictor")

# Add Model Info Section with expander
with st.sidebar.expander("📘 Model Info", expanded=False):
    st.markdown("""
    ### Model Information
    
    - **RainForest**: A tree-based ensemble model similar to Random Forest that's great for structured, tabular data. Excellent for capturing non-linear relationships and quick estimates.
    
    - **LSTM (Long Short-Term Memory)**: A specialized deep learning model designed for time series data. Ideal for capturing long-term dependencies and sequential patterns in stock prices.
    
    - **Prophet**: Developed by Facebook (Meta), this model excels at decomposing time series into trend, seasonality, and holiday components. Best for identifying seasonal patterns and trends.
    
    - **Prophet + LSTM**: An ensemble approach that combines the strengths of both models - Prophet's ability to capture seasonality with LSTM's capacity to learn complex patterns.
    """)

# Stock search and filter - adding to sidebar
st.sidebar.markdown("### 🔍 Stock Search")
search_region = st.sidebar.radio("Region", ["India", "Global"], horizontal=True)
search_term = st.sidebar.text_input("Search Stock", "")

tab1, tab2 = st.tabs(["🇮🇳 Indian Stocks", "🌎 Global Stocks"])

for region, tab, is_indian in [("india", tab1, True), ("global", tab2, False)]:
    with tab:
        st.subheader(f"{'Indian' if is_indian else 'Global'} Stock Trend Predictor")
        base_list = get_nse_stock_list() if is_indian else GLOBAL_TOP_50
        avail = get_available_stocks(base_list, region)
        
        # Apply filter if search term is provided and region matches
        if search_term and search_region.lower() == region:
            filtered_stocks = [s for s in avail if search_term.lower() in s.lower()]
        else:
            filtered_stocks = avail
        
        col_stock, col_dur, col_model = st.columns([3, 2, 5])

        with col_stock:
            symbol = st.selectbox("Choose a Stock", filtered_stocks, key=f"sel_{region}")

        with col_dur:
            dur = st.selectbox(
                "Prediction Duration",
                ["6 Months", "1 Year", "2 Years"],
                key=f"dur_{region}_{symbol}_unique"
            )

        with col_model:
            selected_models = st.multiselect(
                "Select Models",
                ["RainForest", "LSTM", "Prophet", "Prophet + LSTM"],
                default=["RainForest"],
                key=f"model_select_{region}"
            )

        color_legend = {
            "Actual": "#3060c6",
            "RainForest": "#ffbe1b",
            "LSTM": "#009E73",
            "Prophet": "#ff1e1e",
            "Prophet + LSTM": "#B10DC9"
        }

        legend_text = "<div style='display: flex; gap: 30px; align-items: center;'>"
        for model, color in color_legend.items():
            legend_text += f"<div style='display: flex; align-items: center; gap: 5px; margin-bottom:30px; margin-top:10px'>"
            legend_text += f"<span style='color:{color}; font-weight:bold;'>────</span>"
            legend_text += f"<span style='font-weight:bold;'>{model}</span>"
            legend_text += "</div>"
        legend_text += "</div>"

        st.markdown("### 📊 Color Legend for Models")
        st.markdown(legend_text, unsafe_allow_html=True)

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

        if not selected_models:
            st.info("Please select at least one model.")
            continue
            
        # Key Statistics Box - Show latest price and some stats
        st.markdown("### 📈 Key Statistics")
        latest_price = df['Close'].iloc[-1]
        avg_price = df['Close'].mean()
        min_price = df['Close'].min()
        max_price = df['Close'].max()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Latest Close", f"${latest_price:.2f}")
        col2.metric("Avg Price (1yr)", f"${avg_price:.2f}")
        col3.metric("52-Week Low", f"${min_price:.2f}")
        col4.metric("52-Week High", f"${max_price:.2f}")

        predictions_in_sample = {}
        predictions_future = {}

        for model_type in selected_models:
            in_sample, future = pd.DataFrame(), pd.DataFrame()

            if model_type == "RainForest":
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
                # Ensure Date column is properly formatted for combining data later
                if 'Date' in in_sample.columns:
                    in_sample['Date'] = pd.to_datetime(in_sample['Date'])
                predictions_in_sample[model_type] = in_sample.copy()
            
            if not future.empty:
                # Ensure Date column is properly formatted for combining data later
                if 'Date' in future.columns:
                    future['Date'] = pd.to_datetime(future['Date'])
                predictions_future[model_type] = future.copy()

        left_col, right_col = st.columns(2)

        color_map = {
            "RainForest": "#FF851B",
            "LSTM": "#009E73",
            "Prophet": "#ff1e1e",
            "Prophet + LSTM": "#B10DC9"
        }
        
        with left_col:
            st.markdown("**Historical vs In-Sample Predictions**")
            base_chart = alt.Chart(df).mark_line(color='#3060c6').encode(
                x='Date:T',
                y='Close:Q',
                tooltip=['Date:T', 'Close:Q']
            ).properties(width=800, height=450).interactive()

            prediction_charts = []
            for model_name, pred_df in predictions_in_sample.items():
                pred_chart = alt.Chart(pred_df).mark_line(color=color_map.get(model_name, "gray")).encode(
                    x='Date:T',
                    y='Predicted_Close:Q',
                    tooltip=['Date:T', alt.Tooltip('Predicted_Close', title=model_name)]
                )
                prediction_charts.append(pred_chart)

            combined_chart = base_chart + alt.layer(*prediction_charts)
            st.altair_chart(combined_chart, use_container_width=True)
            
            # Model Comparison Metrics Table
            st.subheader("📊 Model Comparison Metrics")
            
            metrics = {
                "Model": [],
                "MAE": [],
                "MSE": [],
                "RMSE": [],
                "MAPE (%)": []
            }
            
            for model_name, pred_df in predictions_in_sample.items():
                if 'Close' in pred_df.columns and 'Predicted_Close' in pred_df.columns:
                    # Drop rows with NA values
                    valid_df = pred_df.dropna(subset=['Close', 'Predicted_Close'])
                    if not valid_df.empty:
                        y_true = valid_df['Close'].values
                        y_pred = valid_df['Predicted_Close'].values
                        mae, mse, rmse, mape = compute_metrics(y_true, y_pred)
                        
                        metrics["Model"].append(model_name)
                        metrics["MAE"].append(f"{mae:.2f}")
                        metrics["MSE"].append(f"{mse:.2f}")
                        metrics["RMSE"].append(f"{rmse:.2f}")
                        metrics["MAPE (%)"].append(f"{mape:.2f}")
            
            if metrics["Model"]:
                metrics_df = pd.DataFrame(metrics)
                st.dataframe(metrics_df, use_container_width=True)
            else:
                st.info("No valid metrics data available for comparison.")
                
            # Add complete tables below the charts with pagination
            st.subheader("In-Sample Prediction Data (Complete)")
            for model_name, pred_df in predictions_in_sample.items():
                if not pred_df.empty:
                    expander = st.expander(f"**{model_name} Model** - {len(pred_df)} rows", expanded=True)
                    with expander:
                        display_cols = [col for col in ['Date', 'Close', 'Predicted_Close'] if col in pred_df.columns]
                        display_dataframe_with_pagination(
                            pred_df[display_cols], 
                            page_size=50, 
                            key_prefix=f"in_sample_{region}_{symbol}_{model_name}"
                        )
                        
                        # Add download button for each model's data
                        csv = pred_df[display_cols].to_csv(index=False).encode()
                        st.download_button(
                            f"⬇️ Download {model_name} In-Sample Data", 
                            csv, 
                            file_name=f"{symbol}_{model_name}_in_sample_predictions.csv", 
                            mime="text/csv",
                            key=f"download_insample_{region}_{symbol}_{model_name}"
                        )

        with right_col:
            if future_months and predictions_future:
                st.markdown(f"**Forecast ({dur})**")

                # Combine all future prediction values to calculate dynamic y-axis limits
                all_predicted_values = pd.concat([df['Predicted_Close'] for df in predictions_future.values() if 'Predicted_Close' in df.columns])
                if not all_predicted_values.empty:
                    y_min = all_predicted_values.min() * 0.98  # 2% margin below
                    y_max = all_predicted_values.max() * 1.02  # 2% margin above

                    future_charts = []

                    for model_name, pred_df in predictions_future.items():
                        future_chart = alt.Chart(pred_df).mark_line(color=color_map.get(model_name, "gray")).encode(
                            x='Date:T',
                            y=alt.Y('Predicted_Close:Q', scale=alt.Scale(domain=[y_min, y_max])),
                            tooltip=['Date:T', alt.Tooltip('Predicted_Close', title=model_name)]
                        ).interactive()
                        future_charts.append(future_chart)

                    combined_future_chart = alt.layer(*future_charts).properties(width=800, height=450)
                    st.altair_chart(combined_future_chart, use_container_width=True)

                # Key future predictions
                st.subheader("📌 Future Price Predictions")
                forecast_cols = st.columns(len(predictions_future))
                
                for i, (model_name, pred_df) in enumerate(predictions_future.items()):
                    if not pred_df.empty and 'Predicted_Close' in pred_df.columns:
                        latest_pred = pred_df['Predicted_Close'].iloc[-1]
                        first_pred = pred_df['Predicted_Close'].iloc[0]
                        change = ((latest_pred - first_pred) / first_pred) * 100
                        
                        with forecast_cols[i]:
                            st.metric(
                                f"{model_name}", 
                                f"${latest_pred:.2f}",
                                f"{change:.2f}% over {dur}"
                            )
                
                # Add complete tables below the future charts with pagination
                st.subheader(f"Forecast Data (Complete)")
                for model_name, pred_df in predictions_future.items():
                    if not pred_df.empty:
                        expander = st.expander(f"**{model_name} Model** - {len(pred_df)} rows", expanded=True)
                        with expander:
                            display_cols = [col for col in ['Date', 'Predicted_Close'] if col in pred_df.columns]
                            display_dataframe_with_pagination(
                                pred_df[display_cols], 
                                page_size=50, 
                                key_prefix=f"future_{region}_{symbol}_{model_name}"
                            )
                            
                            # Add download button for each model's future data
                            csv = pred_df[display_cols].to_csv(index=False).encode()
                            st.download_button(
                                f"⬇️ Download {model_name} Forecast Data", 
                                csv, 
                                file_name=f"{symbol}_{model_name}_forecast_predictions.csv", 
                                mime="text/csv",
                                key=f"download_forecast_{region}_{symbol}_{model_name}"
                            )