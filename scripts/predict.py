# predict.py
import pandas as pd
import joblib
from preprocess import add_features

def predict_from_dataframe(df, model_path):
    df = add_features(df)
    X = df[['MA10', 'MA50', 'Volatility']]
    model = joblib.load(model_path)
    df['Predicted_Close'] = model.predict(X)
    return df[['Close', 'Predicted_Close']]

# Optional CLI test
if __name__ == "__main__":
    df = pd.read_csv("data/global/AAPL.csv")
    result = predict_from_dataframe(df, "models/global_model_aapl.pkl")
    print(result.tail())
