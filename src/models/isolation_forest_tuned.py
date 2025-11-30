"""
src/models/isolation_forest_tuned.py

Train and save a tuned Isolation Forest model using engineered features,
and write out a scored CSV and a pickled model.

Expected input:
    data/processed/engineered_features.csv

Outputs:
    models/isolation_forest_final.pkl
    data/processed/anomaly_iforest_tuned.csv
    models/final_isolation_forest_params.json  (written/overwritten)
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

# ---------- CONFIG ----------
ENGINEERED_FEATURES_PATH = "../../data/processed/engineered_features.csv"
OUTPUT_SCORED_CSV = "../../data/processed/anomaly_iforest_tuned.csv"
OUTPUT_MODEL_PKL = "../../models/isolation_forest_final.pkl"
PARAMS_JSON = "../../models/final_isolation_forest_params.json"

# default params (will be overridden if params JSON exists)
DEFAULT_PARAMS = {
    "n_estimators": 100,
    "contamination": 0.03,
    "max_samples": 256,
    "random_state": 42,
    "n_jobs": -1
}

FEATURE_COLS = [
    "CONSUMPTION",
    "ROLL_MEAN_3", "ROLL_STD_3", "ROLL_MEAN_7",
    "DIFF_1", "DIFF_3", "DIFF_7",
    "DAILY_ENERGY", "LOAD_FACTOR", "MIN_MAX_RATIO",
    "VOLATILITY_INDEX", "OFFPEAK_3", "PEAK_OFFPEAK_RATIO"
]


def load_params(path: str):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                p = json.load(f)
            print(f"[INFO] Loaded params from {path}: {p}")
            return p
        except Exception as e:
            print(f"[WARN] Could not read params file {path}: {e}")
    print("[INFO] Using DEFAULT_PARAMS")
    return DEFAULT_PARAMS.copy()


def save_params(path: str, params: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(params, f, indent=4)
    print(f"[INFO] Saved params to {path}")


def main():
    # verify input exists
    if not os.path.exists(ENGINEERED_FEATURES_PATH):
        raise FileNotFoundError(f"Engineered features file not found: {ENGINEERED_FEATURES_PATH}")

    # load data
    df = pd.read_csv(ENGINEERED_FEATURES_PATH, parse_dates=["DATE"])
    print(f"[INFO] Loaded engineered features: {df.shape}")

    # check required feature columns
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns in engineered_features.csv: {missing}")

    # drop rows with NaN in feature cols (or you can choose imputation)
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS)
    after = len(df)
    print(f"[INFO] Dropped {before - after} rows with NA in feature columns. Remaining: {after}")

    # load params (if present)
    params = load_params(PARAMS_JSON)

    # ensure numeric types for features
    X = df[FEATURE_COLS].astype(float).values

    # construct IsolationForest with params (filter out any unexpected keys)
    allowed_keys = {"n_estimators", "contamination", "max_samples", "random_state", "n_jobs"}
    if isinstance(params, dict):
        model_kwargs = {k: params[k] for k in params if k in allowed_keys}
    else:
        model_kwargs = {k: DEFAULT_PARAMS[k] for k in DEFAULT_PARAMS if k in allowed_keys}

    # fill any missing fallback keys
    for k in allowed_keys:
        if k not in model_kwargs:
            model_kwargs[k] = DEFAULT_PARAMS[k]

    print(f"[INFO] Final model kwargs: {model_kwargs}")

    # train model
    model = IsolationForest(**model_kwargs)
    model.fit(X)
    print("[INFO] Isolation Forest training complete.")

    # scoring & labels
    # decision_function: higher = more normal, lower = more anomalous
    scores = model.decision_function(X)
    labels = model.predict(X)  # -1 anomaly, 1 normal

    # attach results to dataframe copy
    out = df.copy()
    out["IFOREST_SCORE"] = scores
    out["IFOREST_LABEL"] = labels
    out["IFOREST_ANOMALY"] = (out["IFOREST_LABEL"] == -1).astype(int)

    # ensure output folders
    os.makedirs(os.path.dirname(OUTPUT_SCORED_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_MODEL_PKL), exist_ok=True)

    # save scored csv
    out.to_csv(OUTPUT_SCORED_CSV, index=False)
    print(f"[INFO] Saved scored results to: {OUTPUT_SCORED_CSV}")

    # save model (joblib)
    joblib.dump(model, OUTPUT_MODEL_PKL)
    print(f"[INFO] Saved trained model to: {OUTPUT_MODEL_PKL}")

    # persist params used (write back final config to JSON)
    save_params(PARAMS_JSON, model_kwargs)

    print("[DONE] isolation_forest_tuned pipeline finished.")


if __name__ == "__main__":
    main()