from datetime import date, datetime, timezone
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st

from auth_utils import clear_all_prediction_history, delete_prediction_history_item, get_prediction_history, get_user_email, get_user_id_by_username, get_user_preferences, get_user_profile, get_watchlist, register_user, remove_from_watchlist, set_user_preferences, update_username, verify_user


# --- Define the NEW Homepage UI Function ---
def homepage_ui():
    st.title("Welcome to the 🌍 Stock Trend Predictor!")
    st.markdown("""
        This application leverages advanced machine learning models to help you analyze stock trends
        and make informed decisions. Explore historical data, generate future predictions, and
        manage your personalized stock watchlist.
    """)

    # --- ADD IMAGES HERE ---
    # Assuming 1.png and 2.png are in the same directory as the script or a known path
    # If they are in a subdirectory (e.g., 'images'), use 'images/1.png'
    img_col1, img_col2 = st.columns(2)
    try:
        with img_col1:
            # Add width or use_column_width=True if needed for sizing
            st.image("../images/1.png")
        with img_col2:
            st.image("../images/2.png")
    except FileNotFoundError:
        st.warning("Could not find image files (1.png, 2.png). Please ensure they are in the correct directory.")
    except Exception as e:
        st.error(f"Error loading images: {e}")

    st.divider() # Divider after images

    col1, col2 = st.columns([2, 1]) # Adjust ratio as needed

    with col1:
        st.subheader("🚀 Key Features:")
        st.markdown("""
        * **Multi-Model Predictions:** Compare forecasts using RainForest, LSTM, Prophet, and an ensemble model.
        * **Indian & Global Stocks:** Analyze stocks from both NSE (India) and major global markets.
        * **Historical Data Analysis:** View interactive charts and key statistics based on up to 10 years of data.
        * **Personalized Experience:**
            * Save default preferences for regions, models, and prediction duration.
            * Maintain a watchlist of your favorite stocks.
            * Review your past prediction history.
        * **Data Export:** Download prediction data in CSV format.
        """)
        st.markdown("---")
    model_expander = st.expander("📘 Learn About Our Prediction Models", expanded=False)  # Collapsed by default
    with model_expander:
        # --- MAKE THIS SECTION TWO COLUMNS ---
        model_col1, model_col2 = st.columns(2)

        with model_col1:
            st.subheader("📘 Model Information")
            st.markdown("""
                Here's a breakdown of the models we use:
            """)

            # --- RainForest ---
            st.markdown("### 🌳 RainForest (Random Forest)", unsafe_allow_html=True)
            st.markdown("""
                * **Description:** Ensemble of decision trees. Good for general prediction.
                * **Use Cases:** Tabular data, feature importance.
                * **Advantages:** Robust to outliers, accurate.
                * **Disadvantages:** "Black box," less time-sensitive.
            """)

            # --- LSTM ---
            st.markdown("### 🧠 LSTM (Long Short-Term Memory)", unsafe_allow_html=True)
            st.markdown("""
                * **Description:** Recurrent Neural Network for sequences.
                * **Use Cases:** Time series, language.
                * **Advantages:** Captures temporal patterns.
                * **Disadvantages:** Needs lots of data, complex.
            """)

        with model_col2:
            # --- Prophet ---
            st.markdown("### 🔮 Prophet", unsafe_allow_html=True)
            st.markdown("""
                * **Description:** Additive model for time series with seasonality.
                * **Use Cases:** Business forecasting.
                * **Advantages:** Handles missing data, interpretable.
                * **Disadvantages:** Primarily time-driven.
            """)

            # --- Ensemble ---
            st.markdown("### 🤝 Prophet + LSTM (Ensemble)", unsafe_allow_html=True)
            st.markdown("""
                * **Description:** Combines Prophet and LSTM.
                * **Use Cases:** Complex time series.
                * **Advantages:** Potentially more accurate.
                * **Disadvantages:** Complex, higher cost.
            """)
            st.markdown("---")
            st.info("To start analyzing stocks, please log in or register below.")


    with col2:
        st.subheader("🔐 Access Your Account")
        # --- Integrate Login/Register Tabs ---
        tab1, tab2 = st.tabs(["Login", "Register"])

        with tab1:
            # Using 'home_login' prefix for keys to avoid conflicts if login_ui still exists
            username = st.text_input("Username", key="home_login_username")
            password = st.text_input("Password", type="password", key="home_login_password")
            if st.button("Login", key="home_login_button", use_container_width=True): # Use full width
                if verify_user(username, password): # Function from auth_utils.py
                    st.session_state.authenticated = True
                    user_email = get_user_email(username) # Function from auth_utils.py
                    st.session_state.email = user_email
                    st.session_state.username = username
                    # Fetch and store user_id upon successful login
                    st.session_state.user_id = get_user_id_by_username(username) # Use get_user_id_by_username
                    st.success("Logged in successfully!")
                    # Use st.rerun() to refresh the app state after successful login
                    st.rerun()
                else:
                    st.error("Invalid username or password")

        with tab2:
            # Using 'home_register' prefix for keys
            email = st.text_input("Register Email", key="home_register_email")
            username = st.text_input("Username", key="home_register_username") # Corrected key
            password = st.text_input("Password", type="password", key="home_register_password")
            if st.button("Register", key="home_register_button", use_container_width=True): # Use full width
                if register_user(email, username, password): # Function from auth_utils.py
                    st.success("Registered successfully! Please login.")
                else:
                    pass

# Model directory
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts/models')) #
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts/models'))

def get_available_stocks(stock_list, region):
    """Checks the models directory for available pre-trained models for a region."""
    suffix = ".ns.pkl" if region == "india" else ".pkl"
    prefix = f"{region}_model_"
    if not os.path.isdir(MODELS_DIR):
        st.error(f"Models directory not found: {MODELS_DIR}")
        return []
    try:
        files = [f for f in os.listdir(MODELS_DIR) if f.startswith(prefix) and f.endswith(suffix)]
        # Extract symbol, assuming format like 'india_model_reliance.ns.pkl' or 'global_model_aapl.pkl'
        # Ensure case consistency, converting extracted symbols to uppercase for comparison
        syms = [f[len(prefix):-len(suffix)].upper() for f in files]
    except FileNotFoundError:
        st.error(f"Error accessing models directory: {MODELS_DIR}")
        return []

    stock_list_map_upper = {s.upper(): s for s in stock_list}
    available_original_case = [stock_list_map_upper[s_upper] for s_upper in syms if s_upper in stock_list_map_upper]

    return available_original_case


def ensemble_predictions(df1, df2): #
    df1 = df1.copy() #
    df2 = df2.copy() #

    # Remove timezone information if present
    for df in [df1, df2]: #
        if 'Date' in df.columns and hasattr(df['Date'].dtype, 'tz') and df['Date'].dtype.tz is not None: #
            df['Date'] = df['Date'].dt.tz_localize(None) #

    # Merge predictions on 'Date' with suffixes
    merged = pd.merge(df1, df2, on="Date", suffixes=('_lstm', '_prophet'), how='inner') # Use inner merge #

    # Check if merge resulted in empty dataframe
    if merged.empty: #
        st.warning("Ensemble failed: No matching dates between LSTM and Prophet predictions.") #
        return pd.DataFrame(columns=['Date', 'Close', 'Predicted_Close']) # Return empty frame #

    # Calculate ensemble prediction
    merged['Predicted_Close'] = (merged['Predicted_Close_lstm'] + merged['Predicted_Close_prophet']) / 2 #

    # Attempt to include 'Close' if present in either dataframe (prefer df1)
    close_col = None #
    if 'Close_lstm' in merged.columns: #
        close_col = 'Close_lstm' #
    elif 'Close_prophet' in merged.columns: #
        close_col = 'Close_prophet' #
    if close_col: #
        merged['Close'] = merged[close_col] #


    # Dynamically select columns based on availability
    columns = ['Date', 'Predicted_Close'] #
    if 'Close' in merged.columns: #
        columns.insert(1, 'Close')  # Insert 'Close' after 'Date' if present #

    return merged[columns] #

# Function to display dataframe with pagination
def display_dataframe_with_pagination(df, page_size=50, key_prefix=""): #
    if df is None or df.empty: #
        st.info("No data to display.") #
        return #

    total_rows = len(df) #
    n_pages = (total_rows + page_size - 1) // page_size #
    if n_pages > 1: #
        # Ensure unique key for pagination input
        page_key = f"{key_prefix}_page_{total_rows}" # Add total_rows to make key unique if data changes #
        page_num = st.number_input(f'Page ({1}-{n_pages})', min_value=1, max_value=n_pages, value=1, key=page_key) #
        start_idx = (page_num - 1) * page_size #
        end_idx = min(start_idx + page_size, total_rows) #

        # Display row numbers correctly based on the pagination
        st.write(f"Showing rows {start_idx + 1} to {end_idx} of {total_rows}") #

        # Adjusting dataframe for the correct pagination range
        st.dataframe(df.iloc[start_idx:end_idx].reset_index(drop=True), use_container_width=True) #

    else: #
        st.dataframe(df.reset_index(drop=True), use_container_width=True) #

# Function to compute metrics for model evaluation
def compute_metrics(y_true, y_pred): #
    if len(y_true) == 0 or len(y_pred) == 0: #
        return np.nan, np.nan, np.nan, np.nan #
    mae = mean_absolute_error(y_true, y_pred) #
    mse = mean_squared_error(y_true, y_pred) #
    rmse = np.sqrt(mse) #
    # Avoid division by zero and handle potential NaNs/Infs in MAPE calculation
    mask = y_true != 0 #
    if np.sum(mask) == 0: # All true values are zero #
        mape = 0.0 if np.allclose(y_true, y_pred) else np.inf #
    else: #
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 #
    return mae, mse, rmse, mape #

# Function to log out user
def logout_user():
    st.session_state.authenticated = False
    st.session_state.email = None
    st.session_state.pop('username', None)
    st.session_state.pop('user_id', None)
    st.session_state.pop('recent_stocks', None)
    st.session_state.pop('search_india', None)
    st.session_state.pop('search_global', None)
    st.session_state["trigger_logout_rerun"] = True 


def profile_tab(user_id, email): #
    st.title("My Profile") #

    # Logout Button - Moved here for visibility within the profile tab
    st.markdown("### 👤 User Account") #
    # Use st.session_state.username for display if available
    display_name = st.session_state.get('username', email) # Fallback to email #
    st.success(f"Logged in as: **{display_name}**") #
    st.button("Logout", key="logout_button_profile", #
              on_click=lambda: logout_user()) #
    st.divider() #

    # Create tabs for different profile sections
    profile_tabs = st.tabs([ #
        "Profile Information", #
        "Watchlist", #
        "Prediction History", 
    ]) 

    # Profile Information Tab
    with profile_tabs[0]: #
        profile_info_section(user_id) #

    # Watchlist Tab
    with profile_tabs[1]: #
        watchlist_section(user_id) #

    # Prediction History Tab
    with profile_tabs[2]: #
        prediction_history_section(user_id) #


def profile_info_section(user_id): #
    st.header("Profile Information") #

    # Get user profile data
    profile = get_user_profile(user_id) # #
    if not profile: #
        st.error("Could not retrieve profile information") #
        return #

    # Display current profile info
    st.write(f"**Email:** {profile['email']}") #

    # Username edit section
    current_username = profile['username'] #
    col1, col2 = st.columns([3, 1], vertical_alignment="bottom") # Align button #
    with col1: #
        new_username = st.text_input("Username", value=current_username, key="profile_username_input") #
    with col2: #
        #st.write("")  # Spacing removed
        if st.button("Update Username", key="update_username_profile", use_container_width=True): # Use container width #
            if new_username and new_username != current_username: #
                if update_username(user_id, new_username): # Errors handled inside function #
                    st.success("Username updated successfully!") #
                    st.session_state.username = new_username # Update session state #
                    st.rerun() #
            elif not new_username: #
                 st.warning("Username cannot be empty.") #
            else: #
                 st.info("Username is the same.") #

    # Account info
    if 'created_at' in profile and profile['created_at']: #
        # Ensure created_at is datetime object
        created_at_dt = profile['created_at'] #
        if isinstance(created_at_dt, datetime): #
            # Assuming stored as UTC, display in a standard format
            if created_at_dt.tzinfo is None: #
                 created_at_dt = created_at_dt.replace(tzinfo=timezone.utc) # Assume UTC if naive #
            st.write(f"**Account Created:** {created_at_dt.strftime('%d %B, %Y')}") #
        else: #
             st.write(f"**Account Created:** {created_at_dt}") # Fallback #


def user_preferences_section(user_id): #
    st.header("User Preferences") #

    # Get current preferences
    prefs = get_user_preferences(user_id) # #

    # Region preference
    region_options = ["India", "Global"] #
    selected_region = st.selectbox( #
        "Preferred Default Region Tab", #
        region_options, #
        index=region_options.index(prefs.get('preferred_region', 'India')) if prefs.get('preferred_region') in region_options else 0, #
        key="pref_region" #
    ) #

    # Model preferences
    all_models = ["RainForest", "LSTM", "Prophet", "Prophet + LSTM"] #
    # Ensure default_models is a list of strings from the DB
    preferred_models_str = prefs.get('preferred_models', '') #
    default_models = preferred_models_str.split(',') if preferred_models_str else ["RainForest"] #
    default_models = [m for m in default_models if m in all_models]  # Clean up any invalid models #

    selected_models = st.multiselect( #
        "Default Models for Prediction", #
        all_models, #
        default=default_models, #
        key="pref_models" #
    ) #

    # Prediction duration preference
    duration_options = ["6 Months", "1 Year", "2 Years"] #
    selected_duration = st.select_slider( #
        "Default Prediction Duration", #
        options=duration_options, #
        value=prefs.get('prediction_duration', '1 Year') if prefs.get('prediction_duration') in duration_options else "1 Year", #
        key="pref_duration" #
    ) #

    # Dark mode toggle - Note: Streamlit doesn't directly support programmatic theme switching easily.
    # This might control custom CSS or just be stored for future use.
    dark_mode = st.toggle("Dark Mode Preference (Future Feature)", value=bool(prefs.get('dark_mode', False)), key="pref_dark_mode") #

    # Save button
    if st.button("Save Preferences", key="save_prefs_button"): #
        if set_user_preferences(user_id, selected_region, selected_models, selected_duration, dark_mode): # Pass selected_models list #
            st.success("Preferences saved successfully!") #
            # Optionally update session state if preferences affect immediate UI
        else: #
            st.error("Failed to save preferences.") #


def watchlist_section(user_id): #
    st.header("My Watchlist") #

    # Get watchlist items
    watchlist = get_watchlist(user_id) # #

    if not watchlist: #
        st.info("Your watchlist is empty. Add stocks using the '➕ Add to Watchlist' button on the stock prediction pages.") #
        return #

    # Display watchlist items
    st.write("Click 'View' to jump to the stock's prediction page.") #
    # Add search/filter for watchlist? (Optional enhancement)

    for item in watchlist: #
        # Use item ID for unique keys
        item_id = item['id'] #
        cols = st.columns([4, 1, 1]) # Adjust ratios as needed #

        with cols[0]: #
            st.write(f"**{item['symbol']}** ({item['region'].capitalize()})") # Capitalize region for display #
            # Display added date?
            added_at_dt = item.get('added_at') #
            if isinstance(added_at_dt, datetime): #
                 # Format date nicely
                 st.caption(f"Added: {added_at_dt.strftime('%d %b %Y')}") #

        with cols[1]: #
             if st.button("View", key=f"view_watchlist_{item_id}", use_container_width=True): #
                 st.query_params["symbol"] = item['symbol'] #
                 st.query_params["region"] = item['region'] #
                 st.success(f"Set to view {item['symbol']}. Go to the '{item['region'].capitalize()} Stocks' tab.") #
                 st.rerun() # Rerun to make query_params take effect #

        with cols[2]: #
            if st.button("Remove", key=f"remove_watchlist_{item_id}", use_container_width=True): #
                if remove_from_watchlist(user_id, item_id): # Function handles errors #
                    st.success(f"Removed {item['symbol']} from watchlist") #
                    st.rerun() #


# --- MODIFIED prediction_history_section ---
# --- MODIFIED prediction_history_section ---
def prediction_history_section(user_id): #
    st.header("Prediction History") #
    st.write("Past predictions you have generated:") #

    # --- Date Filter ---
    st.markdown("---") #
    st.subheader("Filter History by Date") #
    filter_cols = st.columns(2) #
    with filter_cols[0]: #
        start_date = st.date_input("Start Date", value=None, key="hist_filter_start", max_value=date.today()) #
    with filter_cols[1]: #
        # Ensure end date cannot be before start date if start date is selected
        min_end_date = start_date if start_date else None #
        end_date = st.date_input("End Date", value=None, key="hist_filter_end", min_value=min_end_date, max_value=date.today()) #

    st.markdown("---") #

    # Get prediction history using the date filter
    # Pass start_date and end_date directly (they will be date objects or None)
    history = get_prediction_history(user_id, start_date=start_date, end_date=end_date) # Use modified function #

    if not history: #
        if start_date or end_date: #
             st.info("No predictions found within the selected date range.") #
        else: #
             st.info("You haven't made any predictions yet.") #
        return #

    # --- Clear All Button (for the filtered view or all history?) ---
    # Decide if "Clear All" should respect the date filter or always clear everything.
    # Current implementation clears ALL history regardless of filter.
    if st.button("⚠️ Clear All Prediction History", key="clear_all_history_confirm"): #
        st.session_state['confirm_delete_all_history'] = True #

    # Confirmation logic for clearing all history
    if st.session_state.get('confirm_delete_all_history', False): #
        st.warning("**Are you sure you want to delete ALL prediction history? This action cannot be undone.**") #
        col_confirm, col_cancel, _ = st.columns([1, 1, 4]) #
        with col_confirm: #
            if st.button("Yes, Delete All", key="confirm_delete_all_yes", type="primary"): #
                if clear_all_prediction_history(user_id): # Function handles errors #
                    st.success("All prediction history cleared.") #
                    st.session_state['confirm_delete_all_history'] = False # Reset confirmation state #
                    st.rerun() #
                # Error message handled by clear_all_prediction_history
        with col_cancel: #
             if st.button("Cancel", key="confirm_delete_all_no"): #
                 st.session_state['confirm_delete_all_history'] = False # Reset confirmation state #
                 st.rerun() #
    st.markdown("---") #

    # Display filtered prediction history
    st.write(f"Displaying {len(history)} historical prediction(s):") # Show count of filtered items #

    for item in history: #
        # Ensure prediction_date is datetime object for formatting
        pred_date = item.get('prediction_date') #
        pred_date_str = "N/A" #
        if isinstance(pred_date, datetime): #
             # Ensure datetime is timezone-aware (assuming UTC storage if naive)
            if pred_date.tzinfo is None: #
                pred_date = pred_date.replace(tzinfo=timezone.utc) #
            # Convert to local timezone for display (optional)
            try: #
                local_tz = datetime.now().astimezone().tzinfo #
                pred_date_local = pred_date.astimezone(local_tz) #
                pred_date_str = pred_date_local.strftime('%d %b %Y, %H:%M %Z') # Include local timezone #
            except Exception: # Fallback if local timezone fails #
                pred_date_str = pred_date.strftime('%d %b %Y, %H:%M UTC') # Show as UTC #

        elif pred_date: #
            pred_date_str = str(pred_date) # Fallback to string if not datetime #

        expander_title = f"{item.get('symbol', 'N/A')} ({item.get('region', 'N/A').capitalize()}) - {pred_date_str}" #

        with st.expander(expander_title): #
            col_details, col_actions = st.columns([4, 1]) # Adjust column ratio as needed #

            with col_details: #
                st.write(f"**Symbol:** {item.get('symbol', 'N/A')} ({item.get('region', 'N/A').capitalize()})") #
                st.write(f"**Models Used:** {item.get('models_used', 'N/A')}") #
                st.write(f"**Prediction Generated On:** {pred_date_str}") #

                # Display forecast data if available
                forecast_json = item.get('forecast_json') #
                forecast_data_models = {} # Initialize dictionary to hold parsed data #
                if forecast_json: #
                    try: #
                        # Parse the outer JSON string, then access the nested data
                        forecast_data_outer = json.loads(forecast_json) #
                        # Access the 'data' dictionary which contains model predictions
                        forecast_data_models = forecast_data_outer.get('data', {}) #

                        if isinstance(forecast_data_models, dict) and forecast_data_models: #
                            st.markdown("**Forecasted Data (Preview):**") #
                            # Show preview for one model if available
                            first_model = next(iter(forecast_data_models), None) #
                            if first_model and isinstance(forecast_data_models[first_model], list) and forecast_data_models[first_model]: #
                                 st.write(f"*{first_model} Model Forecast (first 5 rows):*") #
                                 # Attempt to convert dict list back to DataFrame for display
                                 try: #
                                     preview_df = pd.DataFrame(forecast_data_models[first_model][:5]) # Limit preview size #
                                     if 'Date' in preview_df.columns: #
                                         # Ensure Date conversion is robust
                                         try: #
                                             preview_df['Date'] = pd.to_datetime(preview_df['Date']) #
                                             preview_df['Date'] = preview_df['Date'].dt.strftime('%Y-%m-%d') # Format for display #
                                         except Exception as date_err: #
                                             st.warning(f"Could not parse dates in stored forecast preview: {date_err}") #
                                     st.dataframe(preview_df, use_container_width=True) #
                                 except Exception as pd_err: #
                                     st.error(f"Error creating DataFrame from stored forecast: {pd_err}") #
                                     st.json(forecast_data_models[first_model][:5]) # Show raw data instead #

                            else: #
                                st.info("No detailed forecast data preview available.") #
                        else: #
                            st.info("No detailed forecast data found in this record.") #
                    except json.JSONDecodeError: #
                        st.error("Error decoding forecast data for this entry.") #
                    except Exception as e: #
                        st.error(f"Error displaying forecast data: {e}") #
                else: #
                    st.info("No forecast data recorded for this entry.") #

            with col_actions: #
                 st.write("") # Spacer #

                 # --- Add Download Buttons for Forecast Data (replacing Revisit Stock) ---
                 if forecast_data_models: # Only show download options if forecast data exists #
                     st.markdown("**Download Forecast**") #
                     # Iterate through each model's data available in the history entry
                     for model_name, model_data_list in forecast_data_models.items(): #
                         if isinstance(model_data_list, list) and model_data_list: # Check if data is a non-empty list #
                             try: #
                                 # Convert list of dicts to DataFrame
                                 forecast_df = pd.DataFrame(model_data_list) #
                                 # Ensure Date column is formatted as string for CSV
                                 if 'Date' in forecast_df.columns: #
                                     forecast_df['Date'] = pd.to_datetime(forecast_df['Date']).dt.strftime('%Y-%m-%d') #
                                 # Convert DataFrame to CSV string
                                 csv_data = forecast_df.to_csv(index=False).encode('utf-8') #

                                 # Create a unique file name
                                 symbol_clean = item.get('symbol', 'stock').replace('.', '_') # Replace dots for filename safety #
                                 region_clean = item.get('region', 'none') #
                                 # Format prediction date for filename
                                 pred_date_filename = "" #
                                 if isinstance(item.get('prediction_date'), datetime): #
                                     # Use UTC datetime for consistent filenames
                                     utc_date = item.get('prediction_date').replace(tzinfo=timezone.utc) if item.get('prediction_date').tzinfo is None else item.get('prediction_date').astimezone(timezone.utc) #
                                     pred_date_filename = utc_date.strftime('%Y%m%d_%H%M') # YYYYMMDD_HHMM format #

                                 file_name = f"{symbol_clean}_{region_clean}_{model_name.replace(' + ', '_')}_forecast_{pred_date_filename}.csv" #

                                 st.download_button( #
                                     f"⬇️ {model_name} CSV", # Button label #
                                     csv_data, # Data to download #
                                     file_name=file_name, # Suggested file name #
                                     mime="text/csv", # MIME type #
                                     key=f"download_hist_forecast_{item['id']}_{model_name.replace(' + ', '_')}" # Unique key #
                                 ) #
                             except Exception as download_err: #
                                 st.warning(f"Could not generate download for {model_name} data: {download_err}") #
                         else: #
                             st.info(f"No downloadable forecast data for {model_name}.") #

                 else: # If forecast_data_models is empty or not a dict #
                     st.info("No forecast data available for download.") #

                 st.write("") # Spacer #

                 # --- Keep Delete Button for individual item ---
                 if st.button("🗑️ Delete Entry", key=f"delete_pred_hist_{item['id']}", use_container_width=True): #
                     if delete_prediction_history_item(user_id, item['id']): # Error handling inside function #
                         st.success(f"Deleted history entry for {item.get('symbol', 'N/A')}.") #
                         st.rerun() # Rerun to refresh the history list #


def recently_viewed_section(): #
    """Displays recently viewed stocks, moved from sidebar.""" #
    st.header("📌 Recently Viewed Stocks") #
    if "recent_stocks" not in st.session_state or not st.session_state.recent_stocks: #
        st.info("You haven't viewed any stocks in this session yet.") #
        return #

    st.write("Stocks you have looked at recently in this session:") #
    # Display horizontally if few items, wrap if many
    num_recent = len(st.session_state.recent_stocks) #
    cols_per_row = 5 # Adjust as needed #
    num_rows = (num_recent + cols_per_row - 1) // cols_per_row #

    item_index = 0 #
    for _ in range(num_rows): #
         cols = st.columns(min(cols_per_row, num_recent - item_index)) #
         for j in range(len(cols)): #
             if item_index < num_recent: #
                 recent_item = st.session_state.recent_stocks[item_index] #
                 with cols[j]: #
                    # Expects recent_stocks stores tuples: (symbol, region)
                    if isinstance(recent_item, tuple) and len(recent_item) == 2: #
                         symbol, region = recent_item #
                         button_label = f"{symbol} ({region.capitalize()})" #
                         # Ensure unique key across sessions/users if needed, here just per session view
                         key = f"recent_{symbol}_{region}_{item_index}" #
                    else: # Fallback if format is unexpected #
                         symbol = str(recent_item) #
                         region = "india" # Default assumption #
                         button_label = f"{symbol} (?)" #
                         key = f"recent_{symbol}_{item_index}" # Ensure unique key #

                    if st.button(button_label, key=key, use_container_width=True): #
                        st.query_params["symbol"] = symbol #
                        st.query_params["region"] = region #
                        st.success(f"Set to view {symbol}. Go to the '{region.capitalize()} Stocks' tab.") #
                        st.rerun() #
                 item_index += 1 #


def application_settings_section(): #
    """Displays application settings, moved from sidebar.""" #
    st.header("⚙️ Application Settings") #
    st.checkbox("Auto-refresh Data (Not Implemented)", key="auto_refresh", disabled=True) #
    refresh_interval = st.select_slider( #
        "Refresh Interval (Not Implemented)", #
        options=["5 min", "15 min", "30 min", "1 hour"], #
        disabled=True
        #disabled=not st.session_state.get("auto_refresh", False) # Original logic if implemented
    ) #
    st.caption("Auto-refresh functionality is currently not implemented.") #

# --- MODIFIED model_info_section ---
def model_info_section():
    """Displays detailed information about the prediction models."""
    st.header("📘 Model Information")
    st.markdown("""
    This application utilizes several machine learning models for stock prediction. Each model has its strengths and is suited for different aspects of time series analysis:

    #### RainForest (Based on Random Forest)
    * **Description:** A tree-based ensemble model that builds multiple decision trees during training and outputs the average prediction (regression) of the individual trees. It's effective for tabular data.
    * **Use Cases:** General prediction tasks on structured data, identifying feature importance, establishing a strong baseline model. Good for capturing non-linear relationships in stock features if provided (though this implementation might primarily use price history).
    * **Advantages:**
        * Robust to outliers and non-linear data.
        * Generally provides high accuracy.
        * Reduces overfitting compared to a single decision tree.
        * Relatively fast to train compared to complex deep learning models.
    * **Disadvantages:**
        * Can be a "black box," making it harder to interpret the reasoning behind specific predictions compared to simpler models.
        * May not explicitly model time dependencies or seasonality as well as time-series specific models.
        * Can require careful tuning of hyperparameters (like the number of trees).

    #### LSTM (Long Short-Term Memory)
    * **Description:** A type of Recurrent Neural Network (RNN) specifically designed to learn long-range dependencies in sequential data, making it suitable for time series forecasting.
    * **Use Cases:** Time series forecasting (stock prices, weather), natural language processing, speech recognition, sequence generation.
    * **Advantages:**
        * Excellent at capturing complex temporal patterns and long-term dependencies in data.
        * Can model highly non-linear relationships.
        * Stateful nature allows it to "remember" past information over extended periods.
    * **Disadvantages:**
        * Requires significant amounts of data for effective training.
        * Computationally intensive and slower to train than simpler models.
        * Sensitive to hyperparameter choices (network architecture, learning rate, etc.).
        * Can be complex to implement and tune correctly.

    #### Prophet
    * **Description:** Developed by Facebook (Meta), Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.
    * **Use Cases:** Business forecasting (e.g., sales, capacity planning), time series with strong seasonal patterns, forecasting with missing data or outliers, providing interpretable forecasts.
    * **Advantages:**
        * Works well with time series that have strong seasonal effects and several seasons of historical data.
        * Robust to missing data, shifts in the trend, and large outliers.
        * Provides interpretable components (trend, seasonality, holidays).
        * Generally fast and easy to use, with sensible default parameters.
    * **Disadvantages:**
        * Primarily models time as the main driver; may not capture complex interactions with other external regressors as effectively as some other models.
        * Might be less accurate for very short-term, high-frequency predictions compared to models like LSTM in some cases.
        * Assumes trends are piecewise linear or logistic, which might not always hold.

    #### Prophet + LSTM (Ensemble)
    * **Description:** An ensemble approach that aims to combine the strengths of Prophet and LSTM. Typically, Prophet models the main trend and seasonality, and LSTM models the residuals (the part Prophet couldn't explain) or they are averaged. (*Note: This implementation averages the predictions.*)
    * **Use Cases:** Complex time series where both clear seasonal/trend components and subtle, non-linear patterns exist. Aiming for potentially higher accuracy by leveraging both models.
    * **Advantages:**
        * Potential to capture both interpretable components (from Prophet) and complex sequential patterns (from LSTM).
        * May lead to improved forecast accuracy over using either model alone in certain situations.
        * Increased complexity in implementation, training, and tuning.
    * **Disadvantages:**
        * Higher computational cost than using a single model.
        * Interpretation can become more difficult as it combines two different model types.
        * Performance gains are not guaranteed and depend heavily on the specific dataset.

    ---
    **To analyze and predict stocks using these models, log in or register.**
    """, unsafe_allow_html=True) # unsafe_allow_html might be needed if using HTML tags within markdown
