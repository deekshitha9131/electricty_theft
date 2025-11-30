import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Config
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
HIDDEN_DIM = 128
LATENT_DIM = 64
NUM_LAYERS = 2
SEED = 42

# Paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LSTM_DIR = os.path.join(ROOT, "data", "processed", "lstm")
X_TRAIN_PATH = os.path.join(LSTM_DIR, "X_train.npy")
X_VAL_PATH = os.path.join(LSTM_DIR, "X_val.npy")

torch.manual_seed(SEED)
np.random.seed(SEED)

print("Loading data...")
X_train = np.load(X_TRAIN_PATH)
X_val = np.load(X_VAL_PATH)

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")

# Clean data
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

# Normalize
train_max = np.max(np.abs(X_train))
val_max = np.max(np.abs(X_val))
global_max = max(train_max, val_max)
X_train = X_train / (global_max + 1e-8)
X_val = X_val / (global_max + 1e-8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

n_features = X_train.shape[2]

# DataLoaders
train_tensor = torch.tensor(X_train, dtype=torch.float32)
val_tensor = torch.tensor(X_val, dtype=torch.float32)
train_loader = DataLoader(TensorDataset(train_tensor), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(val_tensor), batch_size=BATCH_SIZE, shuffle=False)

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.enc_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        
        # Decoder
        self.dec_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.out_fc = nn.Linear(hidden_dim, input_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def encode(self, x):
        enc_out, _ = self.encoder(x)
        last_h = enc_out[:, -1, :]
        z = self.enc_fc(last_h)
        return z
    
    def decode(self, z, seq_len):
        dec_in = self.dec_fc(z).unsqueeze(1).repeat(1, seq_len, 1)
        dec_out, _ = self.decoder(dec_in)
        out = self.out_fc(dec_out)
        return out
    
    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z, x.size(1))
        return out

# Create model
model = LSTMAutoencoder(n_features, HIDDEN_DIM, LATENT_DIM, NUM_LAYERS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
criterion = nn.MSELoss()

print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

# Training
print("Starting training...")
train_losses = []
val_losses = []

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        Xb = batch[0].to(device)
        optimizer.zero_grad()
        out = model(Xb)
        loss = criterion(out, Xb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            Xb = batch[0].to(device)
            out = model(Xb)
            loss = criterion(out, Xb)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch}/{EPOCHS} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

print("Training completed!")

# Save model
model_path = os.path.join(ROOT, "models", "trained_models", "lstm_autoencoder.pth")
os.makedirs(os.path.dirname(model_path), exist_ok=True)
torch.save(model.state_dict(), model_path)
print(f"Model saved to: {model_path}")

# Extract latent features
print("Extracting latent features...")
model.eval()
all_latents = []

with torch.no_grad():
    # Process training data
    for batch in train_loader:
        Xb = batch[0].to(device)
        z = model.encode(Xb)
        all_latents.append(z.cpu().numpy())
    
    # Process validation data
    for batch in val_loader:
        Xb = batch[0].to(device)
        z = model.encode(Xb)
        all_latents.append(z.cpu().numpy())

latent_features = np.vstack(all_latents)
print(f"Latent features shape: {latent_features.shape}")

# Validate latent features
print("Validating latent features...")
nan_count = np.isnan(latent_features).sum()
print(f"NaN values: {nan_count}")

variances = np.var(latent_features, axis=0)
print(f"Variance per dimension - Min: {variances.min():.6f}, Max: {variances.max():.6f}")

if variances.max() > 1e-6:
    print("SUCCESS: Latent space has proper variance!")
    
    # Save latent features
    latent_path = os.path.join(LSTM_DIR, "latent_features.npy")
    np.save(latent_path, latent_features)
    print(f"Latent features saved to: {latent_path}")
else:
    print("ERROR: Latent space still appears flat!")

print("Done!")