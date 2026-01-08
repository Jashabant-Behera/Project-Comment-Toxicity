import os
import pandas as pd
import numpy as np
import pickle
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# --- CONFIGURATION (Decided in Phase 1) ---
MAX_VOCAB_SIZE = 20000
MAX_LEN = 150
RAW_DATA_PATH = os.path.join('data', 'raw', 'train.csv')
PROCESSED_DATA_DIR = os.path.join('data', 'processed')
MODEL_ARTIFACTS_DIR = 'models'

def clean_text(text):
    """
    Applies text cleaning operations:
    1. Lowercase
    2. Remove URLs
    3. Remove special chars
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    print("--- STARTING PREPROCESSING ---")
    
    # 1. Ensure directories exist
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_ARTIFACTS_DIR, exist_ok=True)
    
    # 2. Load Data
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw data not found at {RAW_DATA_PATH}")
        
    print("Loading raw data...")
    df = pd.read_csv(RAW_DATA_PATH)
    
    # 3. Clean Text
    print("Cleaning text data...")
    df['comment_text'] = df['comment_text'].fillna('')
    df['cleaned_text'] = df['comment_text'].apply(clean_text)
    
    # 4. Split Data (Train/Val)
    # We split BEFORE tokenization to fully simulate a fresh test set
    X = df['cleaned_text'].values
    y = df['toxic'].values
    
    print("Splitting data...")
    X_train_text, X_val_text, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Tokenization
    # CRITICAL: Fit ONLY on training data
    print("Fitting tokenizer...")
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train_text)
    
    # 6. Transform to Sequences
    print("Transforming text to sequences...")
    X_train_seq = tokenizer.texts_to_sequences(X_train_text)
    X_val_seq = tokenizer.texts_to_sequences(X_val_text)
    
    # 7. Padding
    print(f"Padding sequences to {MAX_LEN} length...")
    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    X_val_pad = pad_sequences(X_val_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # 8. Save Artifacts
    print("Saving processed data and tokenizer...")
    
    # Save Numpy arrays (ready for model training)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'), X_train_pad)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'X_val.npy'), X_val_pad)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_val.npy'), y_val)
    
    # Save Tokenizer (Required for inference app)
    with open(os.path.join(MODEL_ARTIFACTS_DIR, 'tokenizer.pickle'), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("--- PREPROCESSING COMPLETE ---")
    print(f"Artifacts saved to {PROCESSED_DATA_DIR} and {MODEL_ARTIFACTS_DIR}")

if __name__ == "__main__":
    main()
