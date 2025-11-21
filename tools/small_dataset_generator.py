import pandas as pd
import numpy as np
import os

def generate_small_dataset(num_customers=20, num_days=30, seed=42):
    np.random.seed(seed)

    # Ensure folder exists
    os.makedirs("data/raw", exist_ok=True)

    # Create customer IDs
    cons_nos = [f"CUST_{i+1:04d}" for i in range(num_customers)]

    # Dummy flag (all 1s)
    flags = [1] * num_customers

    # Generate date columns
    start_date = pd.to_datetime("2014-01-01")
    dates = [(start_date + pd.Timedelta(days=i)).strftime("%Y/%m/%d") for i in range(num_days)]

    data = {
        "CONS_NO": cons_nos,
        "FLAG": flags
    }

    # Random consumption
    for d in dates:
        data[d] = np.round(np.random.uniform(0.1, 15.0, size=num_customers), 2)

    df = pd.DataFrame(data)

    # Save CSV
    path = "data/raw/data.csv"
    df.to_csv(path, index=False)

    print("data.csv created with shape:", df.shape)
    print("Saved at:", path)

    return df

if __name__ == "__main__":
    generate_small_dataset()
