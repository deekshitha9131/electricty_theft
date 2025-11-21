# tools/generate_synthetic_theft.py
import os
import random
import pandas as pd
import numpy as np
from datetime import timedelta

INPUT = "data/processed/cleaned_long.csv"           # long-format input
OUT_SYN = "data/external/synthetic_theft.csv"      # synthetic output
OUT_MERGED = "data/processed/cleaned_with_synthetic.csv"  # optional merged output

# Parameters: tune these as you like
NUM_CUSTOMERS_TO_USE = 8     # how many original customers to clone/create theft on
SYN_PER_CUSTOMER = 2         # how many synthetic variants per selected customer
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def choose_customers(df, n):
    uniq = df["CONS_NO"].unique().tolist()
    random.shuffle(uniq)
    return uniq[:n]


def make_flatline(series, level=0.2):
    """Replace consumption with a small constant for a long period."""
    return np.full_like(series, fill_value=level, dtype=float)


def make_sharp_dip(series, dip_fraction=0.1, dip_duration_frac=0.2):
    """Create a sudden drop for a contiguous window."""
    series = series.copy().astype(float)
    L = len(series)
    duration = max(1, int(L * dip_duration_frac))
    start = random.randint(0, max(0, L - duration))
    series[start:start+duration] = series[start:start+duration] * dip_fraction
    return series


def make_spike_burst(series, spike_count=5, spike_multiplier=6.0):
    """Insert a few large spikes in random places."""
    series = series.copy().astype(float)
    L = len(series)
    for _ in range(spike_count):
        i = random.randint(0, L-1)
        series[i] = max(series[i] * spike_multiplier, series.mean() * spike_multiplier / 2.0)
    return series


def make_periodic_cut(series, every_n=7):
    """Zero out every nth value (e.g., meter bypass on every weekend)."""
    series = series.copy().astype(float)
    for i in range(0, len(series), every_n):
        series[i] = 0.0
    return series


def make_meter_tamper(series, block_size=3):
    """Repeat the same small block of values (meter stuck to repeating pattern)."""
    series = series.copy().astype(float)
    L = len(series)
    if L < block_size:
        return series
    template = series[:block_size] * 0.2
    for i in range(0, L, block_size):
        series[i:i+block_size] = template[:min(block_size, L - i)]
    return series


ANOMALY_FUNCTIONS = [
    ("flatline", make_flatline),
    ("sharp_dip", make_sharp_dip),
    ("spike_burst", make_spike_burst),
    ("periodic_cut", make_periodic_cut),
    ("meter_tamper", make_meter_tamper),
]


def synthesize_for_customer(df_cust, cust_id, variant_idx):
    """
    df_cust: DataFrame of single customer (sorted by DATE) -> columns: CONS_NO, FLAG, DATE, CONSUMPTION
    returns DataFrame of synthetic customer series (same DATEs), with new CONS_NO and FLAG=1
    """
    dates = df_cust["DATE"].values
    values = df_cust["CONSUMPTION"].values.astype(float)

    # pick anomaly type randomly
    anomaly_name, func = random.choice(ANOMALY_FUNCTIONS)
    new_values = func(values)

    # build new customer id (original + suffix)
    new_cons_id = f"{cust_id}SYN{variant_idx}{anomaly_name}"

    synth_df = pd.DataFrame({
        "CONS_NO": [new_cons_id] * len(dates),
        "FLAG": [1] * len(dates),   # mark synthetic as theft
        "DATE": dates,
        "CONSUMPTION": new_values
    })
    synth_df["anomaly_type"] = anomaly_name
    synth_df["source_consumer"] = cust_id
    return synth_df


def main():
    os.makedirs(os.path.dirname(OUT_SYN), exist_ok=True)
    print("Loading cleaned long dataset:", INPUT)
    df = pd.read_csv(INPUT, parse_dates=["DATE"])

    # Quick sanity
    required_cols = {"CONS_NO", "FLAG", "DATE", "CONSUMPTION"}
    if not required_cols.issubset(set(df.columns)):
        raise RuntimeError(f"Input must contain {required_cols}. Got: {df.columns.tolist()}")

    # pick customers to clone
    chosen = choose_customers(df, NUM_CUSTOMERS_TO_USE)
    print("Selected customers for synthetic generation:", chosen)

    synthetic_rows = []
    for cust in chosen:
        df_c = df[df["CONS_NO"] == cust].sort_values("DATE")
        if df_c.empty:
            continue
        for v in range(1, SYN_PER_CUSTOMER + 1):
            synth = synthesize_for_customer(df_c, cust, v)
            synthetic_rows.append(synth)

    if not synthetic_rows:
        print("No synthetic rows generated. Exiting.")
        return

    synth_all = pd.concat(synthetic_rows, ignore_index=True)
    synth_all = synth_all[["CONS_NO", "FLAG", "DATE", "CONSUMPTION", "anomaly_type", "source_consumer"]]

    print("Saving synthetic dataset to:", OUT_SYN)
    synth_all.to_csv(OUT_SYN, index=False)

    # Optional: create merged dataset (append synthetic to cleaned)
    merged_path = OUT_MERGED
    print("Also saving merged file with synthetic rows appended to:", merged_path)
    merged = pd.concat([df, synth_all.drop(columns=["anomaly_type", "source_consumer"])], ignore_index=True)
    merged.to_csv(merged_path, index=False)

    print("Done. Synthetic rows:", len(synth_all), "Merged rows:", len(merged))

if __name__ == "__main__":
    main()