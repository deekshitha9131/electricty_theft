import pandas as pd
import numpy as np
import os

INPUT = "data/processed/cleaned_long.csv"
OUTPUT = "data/processed/engineered_features.csv"


def add_time_features(df):
    df["YEAR"] = df["DATE"].dt.year
    df["MONTH"] = df["DATE"].dt.month
    df["DAY"] = df["DATE"].dt.day
    df["DAY_OF_WEEK"] = df["DATE"].dt.dayofweek
    return df


def add_rolling_features(df):
    df = df.sort_values(["CONS_NO", "DATE"])

    df["ROLL_MEAN_3"] = df.groupby("CONS_NO")["CONSUMPTION"].rolling(3).mean().reset_index(0, drop=True)
    df["ROLL_STD_3"] = df.groupby("CONS_NO")["CONSUMPTION"].rolling(3).std().reset_index(0, drop=True)
    df["ROLL_MEAN_7"] = df.groupby("CONS_NO")["CONSUMPTION"].rolling(7).mean().reset_index(0, drop=True)

    df["DIFF_1"] = df.groupby("CONS_NO")["CONSUMPTION"].diff(1)

    return df


def add_new_daily_features(df):
    df = df.sort_values(["CONS_NO", "DATE"])

    # 1. DAILY ENERGY
    df["DAILY_ENERGY"] = df["CONSUMPTION"]

    # 2. LOAD FACTOR (mean7 / max7)
    rolling_max7 = df.groupby("CONS_NO")["CONSUMPTION"].rolling(7).max().reset_index(0, drop=True)
    df["LOAD_FACTOR"] = df["ROLL_MEAN_7"] / rolling_max7

    # 3. MIN-MAX RATIO
    rolling_min7 = df.groupby("CONS_NO")["CONSUMPTION"].rolling(7).min().reset_index(0, drop=True)
    df["MIN_MAX_RATIO"] = rolling_min7 / rolling_max7

    # 4. DELTA CONSUMPTION improvements
    df["DIFF_3"] = df.groupby("CONS_NO")["CONSUMPTION"].diff(3)
    df["DIFF_7"] = df.groupby("CONS_NO")["CONSUMPTION"].diff(7)

    # 5. VOLATILITY INDEX (std / mean)
    df["VOLATILITY_INDEX"] = df["ROLL_STD_3"] / df["ROLL_MEAN_3"]

    # 6. PEAK-OFFPEAK RATIO (daily definition)
    df["OFFPEAK_3"] = df.groupby("CONS_NO")["ROLL_MEAN_3"].shift(3)
    df["PEAK_OFFPEAK_RATIO"] = df["ROLL_MEAN_3"] / df["OFFPEAK_3"]

    return df


def main():
    print("Loading cleaned dataset:", INPUT)
    df = pd.read_csv(INPUT, parse_dates=["DATE"])

    print("Adding time features...")
    df = add_time_features(df)

    print("Adding rolling features...")
    df = add_rolling_features(df)

    print("Adding additional Day-4 features (daily-safe)...")
    df = add_new_daily_features(df)

    print("Saving:", OUTPUT)
    df.to_csv(OUTPUT, index=False)
    print("Feature engineering completed successfully!")


if __name__ == "__main__":
    main()