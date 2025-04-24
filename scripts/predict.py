import pandas as pd
import joblib
from preprocess import add_features
from datetime import timedelta

def predict_future(df_features, model, days):
    """
    Generate future predictions for the next `days` business days,
    using the last known feature values.
    """
    # Ensure Date is datetime
    df_features['Date'] = pd.to_datetime(df_features['Date'])

    # Create business-day future dates
    last_date = df_features['Date'].max()
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=days, freq='B')

    # Use last known features (MA10, MA50, Volatility)
    if not all(x in df_features.columns for x in ['MA10', 'MA50', 'Volatility']):
        raise ValueError("Required feature columns missing for future prediction.")
        
    last_feats = df_features[['MA10', 'MA50', 'Volatility']].iloc[-1]
    future_feats = pd.DataFrame([last_feats.values] * len(future_dates),
                                 columns=['MA10', 'MA50', 'Volatility'],
                                 index=future_dates)

    preds = model.predict(future_feats)
    future_df = future_feats.copy()
    future_df['Predicted_Close'] = preds
    future_df = future_df[['Predicted_Close']].reset_index().rename(columns={'index': 'Date'})
    return future_df

def predict_from_dataframe(df, model_path, future_months: int = 0):
    """
    Returns two DataFrames:
      - in_sample: historical + model predictions
      - future: out-of-sample predictions for future_months
    """
    # Ensure Date column is datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Compute features
    df_feat = add_features(df.copy())

    # Check if features are available after dropna
    if df_feat.empty or not all(col in df_feat.columns for col in ['MA10', 'MA50', 'Volatility']):
        raise ValueError("Insufficient data after adding features for prediction.")

    # Load model
    model = joblib.load(model_path)

    # In-sample prediction
    X = df_feat[['MA10', 'MA50', 'Volatility']]
    df_feat['Predicted_Close'] = model.predict(X)
    in_sample = df_feat[['Date', 'Close', 'Predicted_Close']].copy()

    # Out-of-sample forecast
    future = pd.DataFrame()
    if future_months and future_months > 0:
        days = int(future_months * 21)  # approx 21 trading days per month
        future = predict_future(df_feat, model, days)

    return in_sample, future

# Optional CLI test
if __name__ == "__main__":
    df = pd.read_csv("data/global/AAPL.csv")
    hist, fut = predict_from_dataframe(df, "models/global_model_aapl.pkl", future_months=6)
    print(hist.tail())
    print(fut.head())
