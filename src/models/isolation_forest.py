import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import os

def run_isolation_forest(input_path="data/processed/engineered_features.csv",
                         output_path="data/processed/anomaly_iforest.csv",
                         contamination=0.05):

    # -----------------------------
    # Validate input file
    # -----------------------------
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading features: {input_path}")
    df = pd.read_csv(input_path, parse_dates=["DATE"])

    # -----------------------------
    # Features used for modeling
    # -----------------------------
    feature_cols = [
        'CONSUMPTION',
        'ROLL_MEAN_3', 'ROLL_STD_3',
        'ROLL_MEAN_7',
        'DIFF_1', 'DIFF_3', 'DIFF_7',
        'DAILY_ENERGY', 'LOAD_FACTOR',
        'MIN_MAX_RATIO',
        'VOLATILITY_INDEX',
        'OFFPEAK_3', 'PEAK_OFFPEAK_RATIO'
    ]

    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column in featured.csv: {col}")

    X = df[feature_cols].fillna(0)

    # -----------------------------
    # Train Isolation Forest
    # -----------------------------
    print("Training Isolation Forest...")
    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=42
    )

    df["IFOREST_LABEL"] = model.fit_predict(X)
    df["IFOREST_ANOMALY"] = (df["IFOREST_LABEL"] == -1).astype(int)
    df["IFOREST_SCORE"] = model.score_samples(X)

    print("Anomalies detected:", df["IFOREST_ANOMALY"].sum())

    # -----------------------------
    # Save results
    # -----------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Isolation Forest results saved to: {output_path}")
    return df


if __name__ == "__main__":
    run_isolation_forest()