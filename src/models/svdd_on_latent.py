"""
svdd_on_latent.py
- Loads trained LSTM autoencoder weights (same class as in lstm_autoencoder.py)
- Extracts latent vectors for train/val (and optionally test) windows
- Trains sklearn OneClassSVM on train latent features
- Saves model, latent features, scores, labels and a small config json
"""

import os
import json
import numpy as np
import joblib
from datetime import datetime
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------- PATHS (adjust if needed) ----------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LSTM_DIR = os.path.join(ROOT, "data", "processed", "lstm")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "lstm_autoencoder.pth")  # trained AE weights
OUT_DIR = os.path.join(ROOT, "models")
os.makedirs(OUT_DIR, exist_ok=True)

LATENT_OUT = os.path.join(LSTM_DIR, "latent_features.npy")
LATENT_TRAIN_OUT = os.path.join(LSTM_DIR, "latent_train.npy")
LATENT_VAL_OUT = os.path.join(LSTM_DIR, "latent_val.npy")
SCORES_OUT = os.path.join(LSTM_DIR, "svdd_scores.npy")
LABELS_OUT = os.path.join(LSTM_DIR, "svdd_labels.npy")
MODEL_OUT = os.path.join(OUT_DIR, "svdd_oneclass.pkl")
CONFIG_OUT = os.path.join(OUT_DIR, "svdd_config.json")

X_TRAIN_PATH = os.path.join(LSTM_DIR, "X_train.npy")
X_VAL_PATH = os.path.join(LSTM_DIR, "X_val.npy")
THRESH_PATH = os.path.join(LSTM_DIR, "recon_threshold.txt")  # optional use

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Model class (must match your training class) ----------------
# Minimal version: only encoder + enc_fc needed
class LSTMAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Use same config as training
        n_features = 13  # from data shape
        hidden_dim = 256
        latent_dim = 64
        num_layers = 2
        
        self.encoder = nn.LSTM(n_features, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.enc_fc = nn.Linear(hidden_dim, latent_dim)
        self.dropout_enc = nn.Dropout(0.2)
        # decoder not needed here

    def encode(self, x):
        # x: (batch, seq, feat)
        enc_out, _ = self.encoder(x)          # (batch, seq, hidden)
        last_h = enc_out[:, -1, :]            # (batch, hidden)
        last_h = self.dropout_enc(last_h)
        z = self.enc_fc(last_h)               # (batch, latent)
        return z

# ---------------- Helpers ----------------
def load_autoencoder_for_encoding(n_features, hidden_dim, latent_dim, num_layers, model_path):
    model = LSTMAutoencoder().to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    # state dict may contain keys for decoder too; we only need matched keys for encoder/enc_fc
    model.load_state_dict({k.replace('encoder.', 'encoder.').replace('enc_fc.', 'enc_fc.'): v
                           for k, v in state.items() if k.startswith(('encoder', 'enc_fc'))}, strict=False)
    model.eval()
    return model

def extract_latents(model, X):
    """
    X: numpy array shape (N, seq_len, features)
    returns: numpy array shape (N, latent_dim)
    """
    dataset = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=64, shuffle=False)
    latents = []
    with torch.no_grad():
        for batch in dataset:
            xb = batch[0].to(DEVICE)
            z = model.encode(xb)               # torch tensor (batch, latent)
            latents.append(z.cpu().numpy())
    return np.vstack(latents)

# ---------------- Main ----------------
def main():
    print("[INFO] Device:", DEVICE)

    # sanity files
    if not os.path.exists(X_TRAIN_PATH):
        raise FileNotFoundError(f"Missing X_train: {X_TRAIN_PATH}")
    if not os.path.exists(X_VAL_PATH):
        raise FileNotFoundError(f"Missing X_val: {X_VAL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing AE model weights: {MODEL_PATH}")

    X_train = np.load(X_TRAIN_PATH)
    X_val = np.load(X_VAL_PATH)

    n_features = X_train.shape[2]

    # Set these to match your training config
    # If you modified HIDDEN_DIM/LATENT_DIM/NUM_LAYERS in training, copy same values here.
    HIDDEN_DIM = 256
    LATENT_DIM = 64
    NUM_LAYERS = 2

    print("[INFO] Loading AE (encoder) to extract latents")
    ae = load_autoencoder_for_encoding(n_features, HIDDEN_DIM, LATENT_DIM, NUM_LAYERS, MODEL_PATH)

    print("[INFO] Extracting latent features for train set")
    lat_train = extract_latents(ae, X_train)
    print("  lat_train shape:", lat_train.shape)
    
    # Handle NaN values
    if np.any(np.isnan(lat_train)):
        print("[WARNING] NaN values in lat_train, replacing with zeros")
        lat_train = np.nan_to_num(lat_train, nan=0.0)
    
    np.save(LATENT_TRAIN_OUT, lat_train)

    print("[INFO] Extracting latent features for val set")
    lat_val = extract_latents(ae, X_val)
    print("  lat_val shape:", lat_val.shape)
    
    # Handle NaN values
    if np.any(np.isnan(lat_val)):
        print("[WARNING] NaN values in lat_val, replacing with zeros")
        lat_val = np.nan_to_num(lat_val, nan=0.0)
    
    np.save(LATENT_VAL_OUT, lat_val)

    # Optional: combine for later analysis
    np.save(LATENT_OUT, np.concatenate([lat_train, lat_val], axis=0))

    # ------------- Scale latents -------------
    scaler = StandardScaler()
    lat_train_s = scaler.fit_transform(lat_train)
    lat_val_s = scaler.transform(lat_val)

    # Save scaler and latents
    joblib.dump(scaler, os.path.join(OUT_DIR, "svdd_scaler.pkl"))

    # ------------- Train OneClassSVM (SVDD-like) -------------
    print("[INFO] Training OneClassSVM on latent train features")
    # parameters to try/tune
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.01)  # nu ~ expected contamination
    ocsvm.fit(lat_train_s)

    joblib.dump(ocsvm, MODEL_OUT)
    print("[INFO] OneClassSVM saved to:", MODEL_OUT)

    # ------------- Scoring -------------
    print("[INFO] Scoring train and val latents")
    scores_train = ocsvm.decision_function(lat_train_s)   # higher -> more normal, lower -> outlier
    scores_val = ocsvm.decision_function(lat_val_s)

    # convert to anomaly label (1 = anomaly)
    labels_train = (scores_train < 0).astype(int)
    labels_val = (scores_val < 0).astype(int)

    np.save(os.path.join(LSTM_DIR, "svdd_scores_train.npy"), scores_train)
    np.save(os.path.join(LSTM_DIR, "svdd_scores_val.npy"), scores_val)
    np.save(os.path.join(LSTM_DIR, "svdd_labels_train.npy"), labels_train)
    np.save(os.path.join(LSTM_DIR, "svdd_labels_val.npy"), labels_val)

    # also save combined scores for quick plotting
    scores_comb = np.concatenate([scores_train, scores_val], axis=0)
    labels_comb = np.concatenate([labels_train, labels_val], axis=0)
    np.save(SCORES_OUT, scores_comb)
    np.save(LABELS_OUT, labels_comb)

    # ------------- Config/metadata -------------
    cfg = {
        "method": "OneClassSVM_on_latent",
        "ocsvm": {"kernel": "rbf", "gamma": "scale", "nu": 0.01},
        "latent_dim": LATENT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "generated_on": datetime.utcnow().isoformat() + "Z",
        "notes": "Train on latent vectors from trained LSTM autoencoder."
    }
    with open(CONFIG_OUT, "w") as f:
        json.dump(cfg, f, indent=2)

    print("[DONE] SVDD-like pipeline complete.")
    print("Saved:", MODEL_OUT, SCORES_OUT, LABELS_OUT, CONFIG_OUT)

if __name__ == "__main__":
    main()
