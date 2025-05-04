import os
import sys
import pandas as pd
import streamlit as st
import altair as alt
import numpy as np
import json
from datetime import datetime, timedelta, timezone, date

from components import MODELS_DIR, compute_metrics, display_dataframe_with_pagination, ensemble_predictions, get_available_stocks, homepage_ui, profile_tab, recently_viewed_section # Import timezone and date

# Adjust path to include scripts directory
try:
    scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts'))
    if scripts_path not in sys.path:
        sys.path.append(scripts_path)
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from lstm_predict import predict_with_lstm
    from prophet_predict import predict_with_prophet
    from utils import get_nse_stock_list, fetch_stock_data
    from rf_predict import predict_from_dataframe
    from auth_utils import (
        get_user_preferences, add_to_watchlist, get_watchlist,
        add_prediction_history, get_prediction_history, delete_prediction_history_item,
        get_oldest_prediction_id_for_stock) # Import the new helper function
except ImportError as e:
    st.error(f"Failed to import necessary modules. Please ensure '../scripts' directory is accessible and contains required files (lstm_predict.py, prophet_predict.py, utils.py, rf_predict.py, auth_utils.py). Error: {e}")
    # You might want to add placeholder functions or stop execution if imports fail critically
    st.stop()


st.set_page_config(page_title="Stock Predictor", layout="wide")

# --- Initialize Session State ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.email = None
    st.session_state.username = None
    st.session_state.user_id = None
    # Initialize search terms per tab to avoid conflicts
    st.session_state.search_india = ""
    st.session_state.search_global = ""
    st.session_state.recent_stocks = [] # Initialize here

prefs = {} 
if st.session_state.authenticated and st.session_state.user_id is not None:
    prefs = get_user_preferences(st.session_state.user_id)
default_region_pref = prefs.get('preferred_region', 'India') if prefs else 'India' #
tab_titles = ["📊 Prediction Models","🇮🇳 Indian Stocks", "🌎 Global Stocks",  "📌 Recently Viewed", "📋 My Profile"] #
default_tab_index = 0 if default_region_pref == 'India' else 1 #

# Global candidates
GLOBAL_TOP_50 = [ #
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "BRK-B", "UNH", "JNJ", #
    "V", "WMT", "JPM", "MA", "PG", "XOM", "HD", "CVX", "LLY", "ABBV", #
    "PEP", "KO", "BAC", "MRK", "AVGO", "ADBE", "COST", "CSCO", "TMO", "DIS", #
    "DHR", "NKE", "PFE", "MCD", "ABT", "ACN", "CRM", "INTC", "WFC", "LIN", #
    "TXN", "AMD", "NEE", "PM", "BMY", "UNP", "RTX", "LOW", "IBM", "QCOM" #
] #

query_params = st.query_params
url_symbol = None
url_region = None

if "symbol" in query_params and "region" in query_params:
    url_symbol = query_params["symbol"]
    url_region = query_params["region"].lower()

    # Redirect to appropriate tab or trigger content rendering
    if url_region == "models":
        active_tab_index = 0
    elif url_region == "india":
        active_tab_index = 1
    elif url_region == "global":
        active_tab_index = 2
    elif url_region == "profile":
        active_tab_index = 4
    elif url_region == "recent":
        active_tab_index = 3
    else:
        active_tab_index = default_tab_index

    # Optionally clear query_params after handling to prevent unnecessary reruns

if not st.session_state.authenticated:
    homepage_ui()
    if st.session_state.get("trigger_logout_rerun", False):
        st.session_state.pop("trigger_logout_rerun")
        st.rerun()

    st.stop()

else:
    # Main Content Area
    st.title("🌍 Stock Trend Predictor") #

    # Check if URL parameters specify a stock, override default tab if needed
    active_tab_index = default_tab_index #
    if url_symbol and url_region: #
        if url_region == 'india': #
            active_tab_index = 1 #
        elif url_region == 'global': #
            active_tab_index = 2 #

    tab0, tab1, tab2, tab3, tab4= st.tabs(tab_titles) #

    # --- Model Info Tab ---
    with tab0: #
        st.title("📊 Prediction Models")
        st.markdown(
            """
            ### Overview
            This tab provides insights into the machine learning models used in this application, including their performance metrics, comparisons, advantages, disadvantages, and typical use cases.

            ### Models Available:
            - **Random Forest (Random Forest)**: Ensemble-based model for robust general predictions.
            - **LSTM (Long Short-Term Memory)**: Neural network designed for sequence prediction and temporal dependencies.
            - **Prophet**: Additive model for time series with seasonality, developed by Facebook.
            - **Prophet + LSTM (Ensemble)**: Combines Prophet and LSTM for enhanced forecasting accuracy.

            ### Performance Metrics:
            - **Mean Absolute Error (MAE):** Measures average magnitude of errors.
            - **Mean Squared Error (MSE):** Penalizes larger errors more than smaller ones.
            - **Root Mean Squared Error (RMSE):** Square root of MSE, interpretable as standard deviation of residuals.
            - **Mean Absolute Percentage Error (MAPE):** Expresses prediction accuracy as a percentage.

            ### Model Comparisons:
            | Model              | Advantages                          | Disadvantages                       |
            |--------------------|-------------------------------------|-------------------------------------|
            | Random Forest         | Handles outliers well, interpretable feature importance | Limited temporal capabilities |
            | LSTM               | Captures temporal dependencies, powerful for sequences | Requires large datasets, computationally intensive |
            | Prophet            | Simple, interpretable, handles seasonality well | Limited to time-based patterns |
            | Prophet + LSTM     | Combines strengths of both models for improved accuracy | High computational cost |

            ### Use Cases:
            - **Random Forest:** General tabular data predictions.
            - **LSTM:** Time series analysis and predictions.
            - **Prophet:** Forecasting with seasonal trends.
            - **Prophet + LSTM:** Complex datasets requiring both trend and temporal analysis.
            """
        )

    # --- Profile Tab ---
    with tab4: #
        profile_tab(st.session_state.user_id, st.session_state.email) #

    with tab3: #
        recently_viewed_section()

    # --- Stock Tabs Logic ---
    # Consolidate common elements for stock tabs
    color_legend = { #
        "Actual": "#3060c6", #
        "Random Forest": "#ffbe1b", # Was #FF851B before, adjusted to match sidebar version #
        "LSTM": "#009E73", #
        "Prophet": "#ff1e1e", #
        "Prophet + LSTM": "#B10DC9" #
    } #
    # Prepare legend HTML once
    legend_html = "<div style='display: flex; flex-wrap: wrap; gap: 20px; align-items: center; margin-bottom: 20px; margin-top: 10px;'>" #
    for model, color in color_legend.items(): #
        legend_html += f"<div style='display: flex; align-items: center; gap: 5px;'>" #
        legend_html += f"<span style='color:{color}; font-weight:bold; font-size: 1.5em;'>───</span>" # Thicker line #
        legend_html += f"<span style='font-weight:bold;'>{model}</span>" #
        legend_html += "</div>" #
    legend_html += "</div>" #


    # Process each region tab (India, Global)
    tab_data = {} #

    for region_name, tab_widget, is_indian, tab_index in [("india", tab1, True, 0), ("global", tab2, False, 1)]: #
        with tab_widget: #
            st.subheader(f"{'Indian' if is_indian else 'Global'} Stock Analysis") #
            tab_key_prefix = f"{region_name}_tab" #
            default_select_option = "-- Select a Stock --" # Define default option text #

            # --- Search within Tab ---
            search_col, search_btn_col, clear_btn_col = st.columns([5, 1, 1], gap="small", vertical_alignment="bottom") #
            with search_col: #
                 # Use tab-specific search state keys
                current_search_term = st.text_input(
                    f"Search Stock Symbol ({region_name.capitalize()})",
                    key=f"{tab_key_prefix}_search",
                    value=st.session_state.get(f"search_{region_name}", ""), # Use region-specific key
                    label_visibility="collapsed",
                    placeholder=f"Search Stock Symbol ({region_name.capitalize()})"
                ) #
            with search_btn_col: #
                 if st.button("Search", key=f"{tab_key_prefix}_search_btn", use_container_width=True): #
                     st.session_state[f"search_{region_name}"] = current_search_term # Update region-specific key
                     st.rerun() #
            with clear_btn_col: #
                if st.button("Clear", key=f"{tab_key_prefix}_clear_btn", use_container_width=True): #
                    if st.session_state.get(f"search_{region_name}"): # Only rerun if clearing something #
                         st.session_state[f"search_{region_name}"] = "" # Clear region-specific key
                         st.rerun() #


            # --- Stock Selection ---
            base_stock_list = get_nse_stock_list() if is_indian else GLOBAL_TOP_50 # #
            available_stocks = get_available_stocks(base_stock_list, region_name) #

            # Apply filtering based on the tab-specific search term
            tab_search_term = st.session_state.get(f"search_{region_name}", "").lower() # Use region-specific key #
            if tab_search_term: #
                # Filter based on symbol containing the search term
                filtered_stocks = [s for s in available_stocks if tab_search_term in s.lower()] #
                if not filtered_stocks: #
                    st.warning(f"No stocks found matching '{tab_search_term}' in this region.") #
                    # Keep available_stocks unfiltered if search yields nothing? Or show empty?
                    # Showing empty might be better feedback.
            else: #
                filtered_stocks = available_stocks #

            # Add the default "Select" option to the list for the dropdown
            display_stock_options = [default_select_option] + sorted(filtered_stocks) # Sort alphabetically #

            # Check if models are available *before* adding default option
            if not available_stocks: #
                 st.info(f"No models currently available for the {region_name} region.") #
                 continue # Skip rest of the tab processing if no models #

            # Determine selected stock: URL param > User selection > Default "Select"
            default_selection_index = 0 # Default to "-- Select a Stock --" #
            # Check if URL params match this tab's region and symbol exists
            if url_symbol and url_region == region_name: #
                 # Check against the options list *including* the default placeholder
                 normalized_url_symbol = url_symbol.upper() # Normalize for comparison #
                 try: #
                     # Find the index in display_stock_options (case-insensitive match)
                     match_index = next((i for i, option in enumerate(display_stock_options) if option.upper() == normalized_url_symbol), -1) #
                     if match_index != -1: #
                         default_selection_index = match_index #
                     else: #
                         # Symbol from URL not found in the available list for this region
                         st.warning(f"URL symbol '{url_symbol}' not found or available in {region_name}. Showing default selection.") #
                         # Optionally clear the query param if it's invalid?
                         # st.query_params.clear() # Or manage state to ignore after check
                 except Exception: # Catch potential errors during index finding #
                      st.error("Error processing URL symbol.") #
                      default_selection_index = 0 #


            selected_symbol = st.selectbox( #
                "Choose a Stock to Analyze", #
                display_stock_options, # Use the sorted list with the default option #
                index=default_selection_index, # Set the determined index #
                key=f"{tab_key_prefix}_stock_select" #
            ) #

            # --- MAIN LOGIC BLOCK: Only proceed if a valid stock is selected ---
            if selected_symbol != default_select_option: #

                user_watchlist = get_watchlist(st.session_state.user_id) # #
                is_in_watchlist = any(item['symbol'] == selected_symbol and item['region'] == region_name for item in user_watchlist) #

                if is_in_watchlist: #
                    st.success(f"⭐ '{selected_symbol}' is in your Watchlist (view in Profile tab)") #
                else: #
                    if st.button(f"➕ Add '{selected_symbol}' to Watchlist", key=f"add_watchlist_{region_name}_{selected_symbol}"): #
                        if add_to_watchlist(st.session_state.user_id, selected_symbol, region_name): # Function handles errors/duplicates #
                            st.success(f"Added {selected_symbol} to watchlist!") #
                            st.rerun() #
  

                    st.markdown("---") # Separator #

                # Update recently viewed list when a new stock is selected/viewed
                current_view = (selected_symbol, region_name) #
                # Add to front if not already the first item
                if "recent_stocks" not in st.session_state:
                    st.session_state.recent_stocks = [] # Initialize if somehow missing
                if not st.session_state.recent_stocks or st.session_state.recent_stocks[0] != current_view: #
                    # Remove if exists elsewhere in list
                    st.session_state.recent_stocks = [item for item in st.session_state.recent_stocks if item != current_view] #
                    # Add to front
                    st.session_state.recent_stocks.insert(0, current_view) #
                    # Limit size
                    max_recent = 5 #
                    st.session_state.recent_stocks = st.session_state.recent_stocks[:max_recent] #

                # --- Model & Duration Selection ---
                st.markdown("---") #
                col_model, col_duration = st.columns(2) #

                with col_model: #
                    prefs = get_user_preferences(st.session_state.user_id) #
                    # Set default to empty list for no default selection
                    all_models_list = ["Random Forest", "LSTM", "Prophet", "Prophet + LSTM"] #

                    selected_models = st.multiselect( #
                        "Select Models to Compare", #
                        all_models_list, #
                        default=[], # Changed default to empty list
                        key=f"{tab_key_prefix}_model_select" #
                    ) #

                with col_duration: #
                     preferred_duration = prefs.get('prediction_duration', '1 Year') if prefs else '1 Year' #
                     duration_options = ["6 Months", "1 Year", "2 Years"] #
                     if preferred_duration not in duration_options: preferred_duration = "1 Year" #

                     selected_duration = st.select_slider( #
                         "Select Prediction Duration", #
                         options=duration_options, #
                         value=preferred_duration, #
                         key=f"{tab_key_prefix}_duration_slider" #
                     ) #
                     months_map = {"6 Months": 6, "1 Year": 12, "2 Years": 24} #
                     future_months = months_map.get(selected_duration, 12) #

                st.markdown("---") #

                # --- Run Prediction Button ---
                run_prediction_button = st.button(
                    "Run Prediction",
                    key=f"{tab_key_prefix}_run_prediction_button",
                    use_container_width=True,
                    disabled=not selected_models # Disable if no models are selected
                )
                st.markdown("---")

                # --- Conditional block for prediction logic ---
                if run_prediction_button: # Only run prediction when button is clicked

                    # --- Check Prediction History & Limit ---
                    # This logic now runs *before* generating a new prediction
                    user_id = st.session_state.user_id #
                    # Define limit for *total* history entries for a specific stock
                    prediction_history_limit = 2 # Max number of entries per stock/region allowed #
                    # Get *all* history entries for this specific stock/region
                    stock_specific_history = [ #
                        h for h in get_prediction_history(user_id) # Fetch all history once #
                        if h['symbol'] == selected_symbol and h['region'] == region_name #
                    ] #

                    # If the limit is reached or exceeded
                    if len(stock_specific_history) >= prediction_history_limit: #
                         # REMOVED EXPLICIT WARNING - This info message is now removed
                         # st.info(...)

                        # Find the ID of the oldest prediction for this stock/region using the helper function
                        oldest_id = get_oldest_prediction_id_for_stock(user_id, selected_symbol, region_name) #
                        if oldest_id: #
                            # Attempt to delete the oldest entry
                            if delete_prediction_history_item(user_id, oldest_id): #
                                pass 
                            else: 
                                st.warning("Could not remove the oldest prediction history entry. Please check logs or contact support.") #
                        else: #
                        
                             st.warning("Could not identify the oldest prediction entry to replace.") #

                    # --- Data Fetching and Processing (Always proceed now, replacement handled above) ---
                    if not selected_models: #
                        st.warning("Please select at least one model to generate predictions.") #
                        st.stop() #

                    # Fetch data (consider caching)
                    @st.cache_data(ttl=3600) # Cache data for 1 hour #
                    def cached_fetch_stock_data(symbol, is_indian_flag): #
                        return fetch_stock_data(symbol.upper(), is_indian=is_indian_flag) # #

                    try: #
                        data = cached_fetch_stock_data(selected_symbol, is_indian) #
                        if data is None or data.empty: #
                            st.error(f"Could not fetch data for {selected_symbol}. It might be delisted or an issue with the data source.") #
                            st.stop() #
                        df = data.reset_index() #
                        # Ensure Date column is datetime and timezone-naive for consistency
                        if 'Date' in df.columns: #
                            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None) #
                        else: #
                            st.error("Fetched data is missing the 'Date' column.") #
                            st.stop() #

                    except Exception as e: #
                        st.error(f"An error occurred fetching data for {selected_symbol}: {e}") #
                        st.stop() #


                    # --- Key Statistics ---
                    st.markdown("### 📈 Key Statistics (Based on Historical Data)") #
                    if not df.empty and 'Close' in df.columns and pd.api.types.is_numeric_dtype(df['Close']): #
                        latest_price = df['Close'].iloc[-1] #
                        one_year_ago = df['Date'].max() - pd.Timedelta(days=365) #
                        df_1yr = df[df['Date'] >= one_year_ago] #
                        if not df_1yr.empty: #
                            avg_price = df_1yr['Close'].mean() #
                            min_price = df_1yr['Close'].min() #
                            max_price = df_1yr['Close'].max() #
                        else: # Fallback #
                            avg_price = df['Close'].mean() #
                            min_price = df['Close'].min() #
                            max_price = df['Close'].max() #

                        stat_cols = st.columns(4) #
                        stat_cols[0].metric("Latest Close", f"{latest_price:.2f}") #
                        stat_cols[1].metric("Avg Price (Last Yr)", f"{avg_price:.2f}") #
                        stat_cols[2].metric("52-Week Low", f"{min_price:.2f}") #
                        stat_cols[3].metric("52-Week High", f"{max_price:.2f}") #
                    else: #
                        st.info("Statistics cannot be calculated.") #
                    st.markdown("---") #

                    # --- Predictions ---
                    @st.cache_data(ttl=3600) # Cache predictions #
                    def generate_predictions(symbol, region, historical_df, models, months): #
                        predictions_in_sample = {} #
                        predictions_future = {} #
                        metrics_data = {} #
                        symbol_lower = symbol.lower() # Use lower case for model filenames #

                        # Find the model path
                        model_file_name = f"{region}_model_{symbol_lower}.ns.pkl" if region == 'india' else f"{region}_model_{symbol_lower}.pkl" #
                        model_path = os.path.join(MODELS_DIR, model_file_name) #
                        model_exists = os.path.exists(model_path) #


                        for model_type in models: #
                            in_sample, future = pd.DataFrame(), pd.DataFrame() #
                            mae, mse, rmse, mape = np.nan, np.nan, np.nan, np.nan #

                            try: #
                                if model_type == "Random Forest": #
                                    if model_exists: #
                                        in_sample, future = predict_from_dataframe(historical_df.copy(), model_path, months) # #
                                    else: #
                                        st.warning(f"Random Forest model file not found for {symbol}. Skipping.") #
                                        continue #

                                elif model_type == "LSTM": #
                                    # Pass uppercase symbol if required by underlying function
                                    in_sample, future = predict_with_lstm(historical_df.copy(), symbol.upper(), region, months)[:2] # #

                                elif model_type == "Prophet": #
                                    in_sample, future, _, _ = predict_with_prophet(historical_df.copy(), months) # #

                                elif model_type == "Prophet + LSTM": #
                                    ins1, fut1, ins2, fut2 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame() #
                                    try: #
                                        # Pass uppercase symbol if required by underlying function
                                        ins1, fut1 = predict_with_lstm(historical_df.copy(), symbol.upper(), region, months)[:2] # #
                                    except Exception as e_lstm: #
                                        st.warning(f"LSTM failed for ensemble: {e_lstm}. Skipping Prophet+LSTM.") #
                                        continue #
                                    try: #
                                        ins2, fut2, _, _ = predict_with_prophet(historical_df.copy(), months) # #
                                    except Exception as e_prophet: #
                                        st.warning(f"Prophet failed for ensemble: {e_prophet}. Skipping Prophet+LSTM.") #
                                        continue #

                                    if not ins1.empty and not ins2.empty: #
                                        in_sample = ensemble_predictions(ins1, ins2) #
                                    if not fut1.empty and not fut2.empty: #
                                        future = ensemble_predictions(fut1, fut2) #

                                # Store results and calculate metrics
                                if not in_sample.empty and 'Predicted_Close' in in_sample.columns: #
                                    if 'Date' in in_sample.columns: #
                                        in_sample['Date'] = pd.to_datetime(in_sample['Date']).dt.tz_localize(None) #
                                    predictions_in_sample[model_type] = in_sample.copy() #

                                    if 'Close' in in_sample.columns: #
                                        valid_df = in_sample.dropna(subset=['Close', 'Predicted_Close']) #
                                        if not valid_df.empty: #
                                            mae, mse, rmse, mape = compute_metrics(valid_df['Close'].values, valid_df['Predicted_Close'].values) #

                                if not future.empty and 'Predicted_Close' in future.columns: #
                                    if 'Date' in future.columns: #
                                        future['Date'] = pd.to_datetime(future['Date']).dt.tz_localize(None) #
                                    predictions_future[model_type] = future.copy() #

                                metrics_data[model_type] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE (%)': mape} #

                            except FileNotFoundError: #
                                st.error(f"Model file required for {model_type} for {symbol} not found. Skipping.") #
                            except Exception as e: #
                                # Log the full traceback for debugging
                                import traceback #
                                st.error(f"Error generating prediction for {model_type} ({symbol}): {e}\n```\n{traceback.format_exc()}\n```") #


                        return predictions_in_sample, predictions_future, metrics_data #

                    # --- Generate and Display Predictions & Metrics ---
                    predictions_in_sample, predictions_future, metrics_data = generate_predictions( #
                        selected_symbol, region_name, df, selected_models, future_months #
                    ) #

                    # --- Model Comparison Metrics Table ---
                    st.subheader("📊 Model Comparison Metrics (In-Sample)") #
                    metrics_list = [] #
                    for model_name, mets in metrics_data.items(): #
                        if not all(np.isnan(m) for m in mets.values()): # Check if any metric was calculated #
                            metrics_list.append({ #
                                "Model": model_name, #
                                "MAE": f"{mets['MAE']:.2f}" if not np.isnan(mets['MAE']) else "N/A", #
                                "MSE": f"{mets['MSE']:.2f}" if not np.isnan(mets['MSE']) else "N/A", #
                                "RMSE": f"{mets['RMSE']:.2f}" if not np.isnan(mets['RMSE']) else "N/A", #
                                "MAPE (%)": f"{mets['MAPE (%)']:.2f}" if not np.isnan(mets['MAPE (%)']) else "N/A" #
                            }) #

                    if metrics_list: #
                        metrics_df = pd.DataFrame(metrics_list) #
                        st.dataframe(metrics_df.set_index('Model'), use_container_width=True) # Set index for better display #
                    else: #
                        st.info("No metrics available for the selected models or data.") #
                    st.markdown("---") #

                    # --- Key Future Predictions Summary ---
                    if future_months and predictions_future: #
                        st.subheader(f"📌 Key Future Predictions ({selected_duration})") #
                        valid_forecasts = {k: v for k, v in predictions_future.items() if not v.empty and 'Predicted_Close' in v.columns} #
                        if valid_forecasts: #
                            forecast_cols = st.columns(len(valid_forecasts)) #
                            col_idx = 0 #
                            for model_name, pred_df in valid_forecasts.items(): #
                                latest_pred = pred_df['Predicted_Close'].iloc[-1] #
                                first_pred = pred_df['Predicted_Close'].iloc[0] #
                                change = ((latest_pred - first_pred) / first_pred * 100) if first_pred else 0 #
                                with forecast_cols[col_idx]: #
                                    st.metric( #
                                        f"{model_name} (End Price)", #
                                        f"{latest_pred:.2f}", #
                                        f"{change:.1f}% vs Start" # Shorten delta label #
                                    ) #
                                col_idx += 1 #
                        else: #
                             st.info("No valid future predictions generated.") #
                        st.markdown("---") #


                    # --- Save Prediction to History --- (Do this *after* generating predictions)
                    if selected_symbol and selected_models and predictions_future: #
                        # Only save if at least one model produced a future prediction
                         if any(not df.empty for df in predictions_future.values()): #
                            current_utc_time = datetime.now(timezone.utc) #
                            forecast_data_to_save = { #
                                "symbol": selected_symbol, #
                                "region": region_name, #
                                # Store prediction date as ISO 8601 string (UTC)
                                "prediction_date": current_utc_time.isoformat(), #
                                "models_used": selected_models, #
                                "duration_months": future_months, #
                                "data": {} #
                            } #
                            for model_name, pred_df in predictions_future.items(): #
                                if not pred_df.empty: #
                                    serializable_df = pred_df.copy() #
                                    if 'Date' in serializable_df.columns: #
                                        serializable_df['Date'] = serializable_df['Date'].dt.strftime('%Y-%m-%d') #
                                    forecast_data_to_save["data"][model_name] = serializable_df.to_dict(orient="records") #

                            try: #
                                # Check if data part is too large (simple check based on string length)
                                # Adjust the limit as needed based on your database column size
                                MAX_JSON_LENGTH = 2000000 # Example limit (adjust based on DB TEXT/JSON size)
                                # Serialize only the data part first to check its size
                                data_json_string = json.dumps(forecast_data_to_save["data"])
                                if len(data_json_string) > MAX_JSON_LENGTH:
                                    st.warning(f"Forecast data for {selected_symbol} is too large to save in history. Only metadata will be stored.")
                                    forecast_data_to_save["data"] = {"error": "data_too_large"} # Replace large data

                                forecast_json_string = json.dumps(forecast_data_to_save)


                                # Add prediction history (handles internal errors)
                                add_prediction_history(
                                    st.session_state.user_id,
                                    selected_symbol,
                                    region_name,
                                    selected_models, # Pass list directly
                                    forecast_json_string # Pass the potentially modified JSON string
                                )
                            except Exception as e: # Catch errors during JSON serialization or saving
                                 st.error(f"Could not save prediction to history for {selected_symbol}: {e}")


                    # --- Legend ---
                    st.markdown("### 📊 Color Legend for Models") #
                    st.markdown(legend_html, unsafe_allow_html=True) #
                    st.markdown("---") #

                    # --- Plotting Area ---
                    plot_area_col1, plot_area_col2 = st.columns(2) #

                    color_map = { #
                        "Actual": "#3060c6", #
                        "Random Forest": "#ffbe1b", #
                        "LSTM": "#009E73", #
                        "Prophet": "#ff1e1e", #
                        "Prophet + LSTM": "#B10DC9" #
                    } #

                    # In-Sample Plot
                    with plot_area_col1: #
                        st.markdown("**Historical vs In-Sample Predictions**") #
                        if not df.empty: #
                            base_chart = alt.Chart(df).mark_line(color=color_map['Actual']).encode( #
                                x=alt.X('Date:T', title='Date'), #
                                y=alt.Y('Close:Q', title='Price', scale=alt.Scale(zero=False)), # Disable zero-based scale #
                                tooltip=['Date:T', 'Open:Q', 'High:Q', 'Low:Q', 'Close:Q', 'Volume:Q'] #
                            ) #

                            prediction_layers = [] #
                            for model_name, pred_df in predictions_in_sample.items(): #
                                if not pred_df.empty and 'Predicted_Close' in pred_df.columns: #
                                    plot_df_in_sample = pred_df.copy() #
                                    if 'Date' in plot_df_in_sample.columns: #
                                         plot_df_in_sample['Date'] = pd.to_datetime(plot_df_in_sample['Date']).dt.tz_localize(None) #

                                    layer = alt.Chart(plot_df_in_sample).mark_line(color=color_map.get(model_name, "grey"), strokeDash=[5,5]).encode( #
                                        x='Date:T', #
                                        y='Predicted_Close:Q', #
                                        tooltip=[alt.Tooltip('Date:T'), alt.Tooltip('Predicted_Close:Q', title=f'{model_name} Pred.')] #
                                    ) #
                                    prediction_layers.append(layer) #

                            if prediction_layers: #
                                combined_chart = alt.layer(base_chart, *prediction_layers).interactive() #
                                st.altair_chart(combined_chart, use_container_width=True) #
                            else: #
                                st.altair_chart(base_chart.interactive(), use_container_width=True) #
                        else: #
                            st.info("No historical data to plot.") #

                    # Future Forecast Plot
                    with plot_area_col2: #
                        st.markdown(f"**Forecast ({selected_duration})**") #
                        if future_months and predictions_future: #
                            future_layers = [] #
                            all_future_values = [] #

                            for model_name, pred_df in predictions_future.items(): #
                                if not pred_df.empty and 'Predicted_Close' in pred_df.columns: #
                                    all_future_values.extend(pred_df['Predicted_Close'].dropna().tolist()) # Ensure no NaNs affect min/max #
                                    plot_df_future = pred_df.copy() #
                                    if 'Date' in plot_df_future.columns: #
                                        plot_df_future['Date'] = pd.to_datetime(plot_df_future['Date']).dt.tz_localize(None) #

                                    layer = alt.Chart(plot_df_future).mark_line(color=color_map.get(model_name, "grey")).encode( #
                                        x=alt.X('Date:T', title=f'Future Date ({selected_duration})'), #
                                        y=alt.Y('Predicted_Close:Q', title='Predicted Price'), #
                                        tooltip=[alt.Tooltip('Date:T'), alt.Tooltip('Predicted_Close:Q', title=f'{model_name} Forecast')] #
                                    ) #
                                    future_layers.append(layer) #

                            if future_layers: #
                                y_min = min(all_future_values) * 0.98 if all_future_values else 0 #
                                y_max = max(all_future_values) * 1.02 if all_future_values else 100 #

                                combined_future_chart = alt.layer(*future_layers).encode( #
                                    y=alt.Y(scale=alt.Scale(domain=[y_min, y_max], zero=False)) # Disable zero-based scale here too #
                                ).interactive() #
                                st.altair_chart(combined_future_chart, use_container_width=True) #
                            else: #
                                st.info("No future forecast data available to plot.") #
                        else: #
                            st.info("Future forecast plot requires future predictions.") #


                    # --- Data Tables and Downloads ---
                    st.markdown("---") #
                    st.subheader("Detailed Prediction Data & Downloads") #

                    tabs_data = st.tabs(["In-Sample Data", "Forecast Data"]) #

                    with tabs_data[0]: #
                        st.markdown("**In-Sample Prediction Data**") #
                        if predictions_in_sample: #
                            for model_name, pred_df in predictions_in_sample.items(): #
                                if not pred_df.empty: #
                                    with st.expander(f"**{model_name} Model** (In-Sample) - {len(pred_df)} rows", expanded=False): #
                                        display_cols = [col for col in ['Date', 'Close', 'Predicted_Close'] if col in pred_df.columns] #
                                        display_df_in_sample = pred_df[display_cols].copy() #
                                        if 'Date' in display_df_in_sample.columns: #
                                            display_df_in_sample['Date'] = display_df_in_sample['Date'].dt.strftime('%Y-%m-%d') #

                                        display_dataframe_with_pagination( #
                                            display_df_in_sample, page_size=10, #
                                            key_prefix=f"in_sample_{tab_key_prefix}_{selected_symbol}_{model_name}" #
                                        ) #
                                        csv_in_sample = display_df_in_sample.to_csv(index=False).encode('utf-8') #
                                        st.download_button( #
                                            f"⬇️ Download {model_name} In-Sample Data", csv_in_sample, #
                                            file_name=f"{selected_symbol}_{model_name}_in_sample_predictions.csv", mime="text/csv", #
                                            key=f"download_insample_{tab_key_prefix}_{selected_symbol}_{model_name}" #
                                        ) #
                        else: #
                            st.info("No in-sample prediction data generated.") #


                    with tabs_data[1]: #
                        st.markdown("**Forecast Data**") #
                        if predictions_future: #
                            for model_name, pred_df in predictions_future.items(): #
                                if not pred_df.empty: #
                                    with st.expander(f"**{model_name} Model** (Forecast: {selected_duration}) - {len(pred_df)} rows", expanded=False): #
                                        display_cols = [col for col in ['Date', 'Predicted_Close'] if col in pred_df.columns] #
                                        display_df_future = pred_df[display_cols].copy() #
                                        if 'Date' in display_df_future.columns: #
                                             display_df_future['Date'] = display_df_future['Date'].dt.strftime('%Y-%m-%d') #

                                        display_dataframe_with_pagination( #
                                            display_df_future, page_size=10, #
                                            key_prefix=f"future_{tab_key_prefix}_{selected_symbol}_{model_name}" #
                                        )
                                        csv_future = display_df_future.to_csv(index=False).encode('utf-8') #
                                        st.download_button( #
                                            f"⬇️ Download {model_name} Forecast Data", csv_future, #
                                            file_name=f"{selected_symbol}_{model_name}_forecast_{selected_duration}.csv", mime="text/csv", #
                                            key=f"download_forecast_{tab_key_prefix}_{selected_symbol}_{model_name}" #
                                        )
                        else:
                            st.info("No forecast prediction data generated.") #

                    # Store results for this tab if needed later outside the loop (optional)
                    tab_data[region_name] = { #
                        'selected_symbol': selected_symbol, #'selected_models': selected_models, 'selected_duration': selected_duration, #
                        #'predictions_in_sample': predictions_in_sample, 'predictions_future': predictions_future,'metrics': metrics_data #
                    }
                # End of the conditional block for the "Run Prediction" button
            else:
                st.info("Please select a stock from the dropdown above to generate analysis and predictions.") #