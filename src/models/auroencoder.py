import numpy as np 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector
from tensorflow.keras.optimizers import Adam

def create_lstm_autoencoder(input_dim, timesteps):
    inputs = Input(shape=(timesteps, input_dim))
    
    encoded = LSTM(64, activation='relu')(inputs)
    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
    
    model = Model(inputs,decoded)
    model.compile(optimizer=Adam(0.001), loss='mse')
    return model
def calculate_reconstruction_error(model, X):
    reconstructed = model.predict(X)
    errors = np.mean((X - reconstructed)**2,axis=(1,2))
    return errors