import numpy as np
import pandas as pd
import os

# ------------------------------------------
# CONFIG
# ------------------------------------------

SEQ_LEN = 48   # Your chosen sequence length

INPUT_PATH = "../../data/processed/engineered_features.csv"
OUTPUT_DIR = "../../data/processed/lstm/"

FEATURE_COLS = [
    "CONSUMPTION", 
    "ROLL_MEAN_3", "ROLL_STD_3", "ROLL_MEAN_7",
    "DIFF_1", "DIFF_3", "DIFF_7",
    "VOLATILITY_INDEX",
    "OFFPEAK_3", "PEAK_OFFPEAK_RATIO",
    "DAILY_ENERGY", "LOAD_FACTOR", "MIN_MAX_RATIO"
]

# ------------------------------------------
# SEQUENCE CREATOR
# ------------------------------------------

def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        sequences.append(seq)
    return np.array(sequences)


# ------------------------------------------
# MAIN PIPELINE
# ------------------------------------------

def main():
    print("[INFO] Loading engineered feature file...")
    df = pd.read_csv(INPUT_PATH)

    # Validate columns
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    print(f"[INFO] Loaded dataframe shape: {df.shape}")

    data = df[FEATURE_COLS].values.astype("float32")

    print("[INFO] Creating sliding windows...")
    sequences = create_sequences(data, SEQ_LEN)

    print(f"[INFO] Total sequences created: {sequences.shape}")

    # Train / validation split
    train_size = int(sequences.shape[0] * 0.8)
    X_train = sequences[:train_size]
    X_val = sequences[train_size:]

    print(f"[INFO] Train: {X_train.shape}, Val: {X_val.shape}")

    # Create directory if missing
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save arrays
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)

    # Store seq length text file
    with open(os.path.join(OUTPUT_DIR, "seq_length.txt"), "w") as f:
        f.write(str(SEQ_LEN))

    print("[DONE] LSTM preparation complete.")
    print(f"[SAVED] X_train.npy, X_val.npy → {OUTPUT_DIR}")


# ------------------------------------------
if __name__ == "__main__":
    main()
