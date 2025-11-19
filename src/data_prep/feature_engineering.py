import pandas as pd
import numpy as np

def add_time_feature(df):
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    return df
def add_statistical_features(df):
    df['rolling_mean'] = df.groupby('customer_id')['consumption'].transform(
        lambda x: x.rolling(window=24, min_periods=1).mean()
    )
    df['rolling_std'] = df.groupby('customer_id')['consumption'].transform(
        lambda x: x.rolling(window=24, min_periods=1).std()
    )
    return df
def engineer_features(df):
    df = add_time_feature(df)
    df = add_statistical_features(df)
    df = df.fillna(0)
    return df