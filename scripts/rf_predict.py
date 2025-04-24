import pandas as pd
import joblib
import numpy as np
from pandas.tseries.offsets import BDay
from preprocess import add_features

def predict_future_rf(df_features, model, days):
    df_features = df_features.copy()
    df_features['Date'] = pd.to_datetime(df_features['Date'])
    df_features.set_index('Date', inplace=True)

    # Ensure at least 2 rows for calculating Log_Return
    history = df_features[['Close']].copy()
    if len(history) < 2:
        raise ValueError("Not enough data for prediction (need at least 2 rows)")

    history = history.iloc[-2:]
    current_date = history.index.max()

    predictions = []

    for _ in range(days):
        current_date += BDay(1)
        history.loc[current_date] = history['Close'].iloc[-1]  # placeholder

        feats = add_features(history.reset_index()).set_index('Date')

        # Skip if features couldn't be computed properly
        if feats.empty or feats.iloc[-1].isnull().any():
            continue

        try:
            latest_feats = feats.iloc[-1][['MA10', 'MA50', 'Volatility', 'Log_Return']].values.reshape(1, -1)
            predicted = model.predict(latest_feats)[0]
        except Exception as e:
            print(f"Skipping prediction on {current_date} due to: {e}")
            continue

        predicted += np.random.normal(loc=0, scale=1)
        predictions.append({'Date': current_date, 'Predicted_Close': predicted})
        history.at[current_date, 'Close'] = predicted

    return pd.DataFrame(predictions)


def predict_from_dataframe(df, model_path, future_months=0):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df_feat = add_features(df.copy())

    model = joblib.load(model_path)
    X = df_feat[['MA10', 'MA50', 'Volatility','Log_Return']].values
    df_feat['Predicted_Close'] = model.predict(X)

    in_sample = df_feat[['Date', 'Close', 'Predicted_Close']].copy()

    future = pd.DataFrame()
    if future_months > 0:
        days = int(future_months * 21)
        future = predict_future_rf(df_feat, model, days)

    return in_sample, future

if __name__ == "__main__":
    hist, fut = predict_from_dataframe(
        pd.read_csv("data/global/AAPL.csv"),
        "models/global_model_aapl.pkl",
        future_months=6
    )
    print(hist.tail())
    print(fut.head())
