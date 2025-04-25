import pandas as pd
from prophet import Prophet
from prophet.plot import plot_components_plotly

def get_prophet_components(model, forecast):
    fig = plot_components_plotly(model, forecast)
    return fig

def predict_with_prophet(df, future_months=6):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
   
    # Keep a copy of the original Close column
    # Convert Date to timezone-naive datetime
    original_dates_and_close = df[['Date', 'Close']].copy()
    original_dates_and_close['Date'] = original_dates_and_close['Date'].dt.tz_localize(None)
   
    # Rename for Prophet
    prophet_df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
   
    # Remove timezone information if any
    prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
    
    # Ensure the 'y' column is numeric
    prophet_df['y'] = pd.to_numeric(prophet_df['y'], errors='coerce')
    
    # Remove rows where 'y' is NaN
    prophet_df = prophet_df.dropna(subset=['y'])
    
    model = Prophet(daily_seasonality=True)
    model.fit(prophet_df)
    
    future_days = future_months * 21  # trading days
    future_df = model.make_future_dataframe(periods=future_days, freq='B')
    forecast = model.predict(future_df)
    
    # Result with ds and yhat for Prophet
    result = forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Predicted_Close'})
   
    # Merge with original Close values for in-sample data
    # Both dataframes now have timezone-naive datetimes
    in_sample = pd.merge(result, original_dates_and_close, on='Date', how='left')
   
    # Splitting in-sample and future predictions
    in_sample = in_sample[in_sample['Date'].isin(original_dates_and_close['Date'])].copy()
    future = result[~result['Date'].isin(original_dates_and_close['Date'])].copy()
    
    return in_sample, future, model, forecast