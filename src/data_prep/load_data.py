import pandas as pd

def load_smart_meter_data(path):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.sort_values(by=['customer_id','timestamp'],inplace=True)
    return df