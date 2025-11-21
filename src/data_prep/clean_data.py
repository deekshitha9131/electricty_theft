import pandas as pd
import os

RAW_FILE = "data/raw/data.csv"
OUTPUT_FILE = "data/processed/cleaned_long.csv"

def load_and_convert_wide_to_long(path):
    print("Loading dataset...")
    df = pd.read_csv(path)

    print("Identifying date columns...")
    date_cols = [col for col in df.columns if "/" in col]

    print("Converting wide to long format...")
    df_long = df.melt(
        id_vars=["CONS_NO", "FLAG"],
        value_vars=date_cols,
        var_name="DATE",
        value_name="CONSUMPTION"
    )

    print("Fixing DATE column format...")
    df_long["DATE"] = pd.to_datetime(df_long["DATE"], format="%Y/%m/%d")

    print("Sorting values...")
    df_long = df_long.sort_values(["CONS_NO", "DATE"])

    return df_long


def main():
    os.makedirs("data/processed", exist_ok=True)

    df_long = load_and_convert_wide_to_long(RAW_FILE)

    print("Saving cleaned file to:", OUTPUT_FILE)
    df_long.to_csv(OUTPUT_FILE, index=False)

    print("cleaned_long.csv created successfully!")


if __name__ == "__main__":
    main()