import pandas as pd

def clean_data(df):
    df = df.dropna()
    df = df[df['consumption']>=0]
    df = df.drop_duplicates(subset=['customer_id','timestamp'])
    return df.reset_index(drop=True)