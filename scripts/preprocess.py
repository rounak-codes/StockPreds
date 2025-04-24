import pandas as pd
import numpy as np

def add_features(df):
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['Volatility'] = df['Close'].rolling(window=10).std()
    df['Log_Return'] = df['Close'].pct_change().apply(lambda x: np.log(1+x))
    df.dropna(inplace=True)
    return df
