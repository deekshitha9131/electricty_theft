# e:\electricity_theft\src\models\lstm_autoencoder.py
"""
Final PyTorch LSTM Autoencoder that uses:
  e:\electricity_theft\data\processed\lstm\X_train.npy
  e:\electricity_theft\data\processed\lstm\X_val.npy
  e:\electricity_theft\data\processed\lstm\seq_length.txt

Overwrite previous file with this exact content and run.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------- CONFIG ----------------
BATCH_SIZE = 32
EPOCHS = 20
LR = 5e-4
HIDDEN_DIM = 256
LATENT_DIM = 64
NUM_LAYERS = 2
SEED = 42
GRAD_CLIP = 1.0

# ---------------- PATHS ----------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LSTM_DIR = os.path.join(ROOT, "data", "processed", "lstm")
X_TRAIN_PATH = os.path.join(LSTM_DIR, "X_train.npy")
X_VAL_PATH   = os.path.join(LSTM_DIR, "X_val.npy")
SEQ_PATH     = os.path.join(LSTM_DIR, "seq_length.txt")

MODEL_OUT = os.path.join(os.path.dirname(__file__), "lstm_autoencoder.pth")
RECON_TRAIN_OUT = os.path.join(LSTM_DIR, "recon_train.npy")
RECON_VAL_OUT = os.path.join(LSTM_DIR, "recon_val.npy")
THRESH_OUT = os.path.join(LSTM_DIR, "recon_threshold.txt")

# ---------------- seed ----------------
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------- validate files ----------------
for p in (X_TRAIN_PATH, X_VAL_PATH, SEQ_PATH):
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required file missing: {p}")

# ---------------- load data ----------------
print("[INFO] Loading numpy windows from:", LSTM_DIR)
X_train = np.load(X_TRAIN_PATH)
X_val = np.load(X_VAL_PATH)
with open(SEQ_PATH, "r") as f:
    seq_len = int(f.read().strip())

print(f"[INFO] X_train: {X_train.shape}")
print(f"[INFO] X_val:   {X_val.shape}")
print(f"[INFO] seq_len: {seq_len}")

# Check for invalid values
if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
    print("[WARNING] Invalid values in X_train, replacing with zeros")
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
if np.any(np.isnan(X_val)) or np.any(np.isinf(X_val)):
    print("[WARNING] Invalid values in X_val, replacing with zeros")
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

# Simple normalization
X_train = X_train / (np.max(np.abs(X_train)) + 1e-8)
X_val = X_val / (np.max(np.abs(X_val)) + 1e-8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] Device:", device)

n_features = X_train.shape[2]

# ---------------- DataLoaders ----------------
train_tensor = torch.tensor(X_train, dtype=torch.float32)
val_tensor = torch.tensor(X_val, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(train_tensor), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(val_tensor), batch_size=BATCH_SIZE, shuffle=False)

# ---------------- Model defined AFTER data load (no args needed) ----------------
class LSTMAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # we use the outer n_features variable captured from global scope
        input_dim = n_features
        hidden_dim = HIDDEN_DIM
        latent_dim = LATENT_DIM
        num_layers = NUM_LAYERS

        # encoder
        self.encoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                                num_layers=num_layers, batch_first=True, dropout=0.2)
        self.enc_fc = nn.Linear(hidden_dim, latent_dim)
        self.dropout_enc = nn.Dropout(0.2)

        # decoder
        self.dec_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim,
                                num_layers=num_layers, batch_first=True, dropout=0.2)
        self.dropout_dec = nn.Dropout(0.2)
        self.out_fc = nn.Linear(hidden_dim, input_dim)
        
        # Initialize weights
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        # x: (batch, seq_len, features)
        enc_out, _ = self.encoder(x)           # enc_out: (batch, seq, hidden)
        last_h = enc_out[:, -1, :]             # (batch, hidden)
        last_h = self.dropout_enc(last_h)
        z = self.enc_fc(last_h)                # (batch, latent)
        dec_in = self.dec_fc(z).unsqueeze(1).repeat(1, x.size(1), 1)  # (batch, seq, hidden)
        dec_out, _ = self.decoder(dec_in)      # (batch, seq, hidden)
        dec_out = self.dropout_dec(dec_out)
        out = self.out_fc(dec_out)             # (batch, seq, features)
        return out

# instantiate model (no positional/keyword mismatch risk)
model = LSTMAutoencoder().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss(reduction="mean")

print("[INFO] Model instantiated. Parameter count:", sum(p.numel() for p in model.parameters()))

# Initialize loss tracking
train_losses = []
val_losses = []

# ---------------- Training loop ----------------
print("[INFO] Training start")
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    count = 0
    for batch in train_loader:
        Xb = batch[0].to(device)
        optimizer.zero_grad()
        out = model(Xb)
        loss = criterion(out, Xb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        train_loss += loss.item()
        count += 1
    train_loss = train_loss / max(1, count)

    # validation
    model.eval()
    val_loss = 0.0
    vcount = 0
    with torch.no_grad():
        for batch in val_loader:
            Xb = batch[0].to(device)
            l = criterion(model(Xb), Xb)
            val_loss += l.item()
            vcount += 1
    val_loss = val_loss / max(1, vcount)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"[Epoch {epoch}/{EPOCHS}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

print("[INFO] Training finished")

# ---------------- Save model ----------------
torch.save(model.state_dict(), MODEL_OUT)
print("[INFO] Model saved to:", MODEL_OUT)

# ---------------- Compute reconstruction errors ----------------
def compute_recon_errors(loader):
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in loader:
            Xb = batch[0].to(device)
            out = model(Xb)
            mse_seq = torch.mean((out - Xb) ** 2, dim=(1,2)).cpu().numpy()
            errors.extend(mse_seq.tolist())
    return np.array(errors)

recon_train = compute_recon_errors(train_loader)
recon_val = compute_recon_errors(val_loader)

np.save(RECON_TRAIN_OUT, recon_train)
np.save(RECON_VAL_OUT, recon_val)
print("[INFO] Recon errors saved:", RECON_TRAIN_OUT, RECON_VAL_OUT)

# threshold (95th percentile)
thresh = float(np.percentile(recon_train, 95))
with open(THRESH_OUT, "w") as f:
    f.write(str(thresh))
print("[INFO] Threshold saved to:", THRESH_OUT, "value:", thresh)

# ---------------- Plot training curve ----------------
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train loss")
plt.plot(val_losses, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("LSTM Autoencoder Training Curve")
plt.legend()
plt.grid()

plot_path = os.path.join(os.path.dirname(__file__), "training_curve.png")
plt.savefig(plot_path)
print("[INFO] Training curve saved to:", plot_path)

print("[DONE] LSTM Autoencoder complete.")

# ---------------- DAY 12: INFERENCE (Anomaly Detection) ----------------

def load_model_for_inference():
    model_inf = LSTMAutoencoder().to(device)
    model_inf.load_state_dict(torch.load(MODEL_OUT, map_location=device))
    model_inf.eval()
    return model_inf

def detect_anomaly(window, model_inf, threshold):
    """
    window: numpy array shaped (seq_len, n_features)
    """
    if len(window.shape) != 2:
        raise ValueError("Input window must be 2D: (seq_len, features)")

    # normalize same as training
    window = window / (np.max(np.abs(window)) + 1e-8)

    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq, feat)

    with torch.no_grad():
        out = model_inf(x)
        mse = torch.mean((out - x) ** 2).item()

    is_anomaly = mse > threshold
    return mse, is_anomaly

def load_threshold():
    with open(THRESH_OUT, "r") as f:
        return float(f.read().strip())

print("[INFO] Day 12 inference utilities ready.")


# ---------------- DAY 12: INFERENCE (Anomaly Detection) ----------------

def load_model_for_inference():
    model_inf = LSTMAutoencoder().to(device)
    model_inf.load_state_dict(torch.load(MODEL_OUT, map_location=device))
    model_inf.eval()
    return model_inf

def detect_anomaly(window, model_inf, threshold):
    """
    window: numpy array shaped (seq_len, n_features)
    """
    if len(window.shape) != 2:
        raise ValueError("Input window must be 2D: (seq_len, features)")

    # normalize same as training
    window = window / (np.max(np.abs(window)) + 1e-8)

    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq, feat)

    with torch.no_grad():
        out = model_inf(x)
        mse = torch.mean((out - x) ** 2).item()

    is_anomaly = mse > threshold
    return mse, is_anomaly

def load_threshold():
    with open(THRESH_OUT, "r") as f:
        return float(f.read().strip())

print("[INFO] Day 12 inference utilities ready.")
