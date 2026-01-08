import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, GlobalMaxPool1D

# --- CONFIGURATION ---
MAX_VOCAB_SIZE = 20000 
MAX_LEN = 150
EMBEDDING_DIM = 64
BATCH_SIZE = 32
EPOCHS = 3 # Can be increased for better results, kept low for demonstration speed

PROCESSED_DATA_DIR = os.path.join('data', 'processed')
MODEL_DIR = 'models'

def build_model():
    """
    Defines the LSTM architecture.
    """
    model = Sequential([
        Embedding(MAX_VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
        Bidirectional(LSTM(64, return_sequences=True)),
        GlobalMaxPool1D(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def main():
    print("--- STARTING TRAINING ---")
    
    # 1. Load Processed Data
    print("Loading preprocessed data...")
    try:
        X_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'))
        y_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'))
        X_val = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_val.npy'))
        y_val = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_val.npy'))
    except FileNotFoundError:
        print("Error: Processed data not found. Please run src/preprocess.py first.")
        return

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")

    # 2. Build Model
    model = build_model()
    model.summary()

    # 3. Train Model
    print(f"Training for {EPOCHS} epochs...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val)
    )

    # 4. Save Model
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, 'toxicity_model.h5')
    model.save(save_path)
    
    print("--- TRAINING COMPLETE ---")
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
