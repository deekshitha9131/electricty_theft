import pandas as pd
import os

INPUT_FILE = "data/processed/cleaned_long.csv"
OUTPUT_FILE = "data/processed/featured.csv"


def add_time_features(df):
    print("Adding time-based features...")

    df["YEAR"] = df["DATE"].dt.year
    df["MONTH"] = df["DATE"].dt.month
    df["DAY"] = df["DATE"].dt.day
    df["DAY_OF_WEEK"] = df["DATE"].dt.dayofweek

    return df


def add_rolling_features(df):
    print("Adding rolling features...")

    df = df.sort_values(["CONS_NO", "DATE"])

    df["ROLL_MEAN_3"] = (
        df.groupby("CONS_NO")["CONSUMPTION"]
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["ROLL_STD_3"] = (
        df.groupby("CONS_NO")["CONSUMPTION"]
        .rolling(3, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
    )

    df["ROLL_MEAN_7"] = (
        df.groupby("CONS_NO")["CONSUMPTION"]
        .rolling(7, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return df


def add_difference_features(df):
    print("Adding consumption difference features...")

    df["DIFF_1"] = (
        df.groupby("CONS_NO")["CONSUMPTION"]
        .diff(1)
    )

    return df


def main():
    print("Loading cleaned dataset...")

    if not os.path.exists(INPUT_FILE):
        print("ERROR: cleaned_long.csv not found.")
        return

    df = pd.read_csv(INPUT_FILE, parse_dates=["DATE"])

    print("Applying feature engineering...")
    df = add_time_features(df)
    df = add_rolling_features(df)
    df = add_difference_features(df)

    print("Saving:", OUTPUT_FILE)
    df.to_csv(OUTPUT_FILE, index=False)
    print("Feature file created successfully.")


if __name__ == "__main__":
    main()