import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# --- CONFIGURATION ---
# These must match the training configuration exactly
MAX_LEN = 150
MODEL_PATH = os.path.join('models', 'toxicity_model.h5')
TOKENIZER_PATH = os.path.join('models', 'tokenizer.pickle')

class ToxicityPredictor:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._load_artifacts()

    def _load_artifacts(self):
        """
        Loads the trained model and tokenizer from disk.
        """
        try:
            if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
                raise FileNotFoundError("Model or Tokenizer not found. Please run training pipeline first.")
                
            self.model = tf.keras.models.load_model(MODEL_PATH)
            
            with open(TOKENIZER_PATH, 'rb') as handle:
                self.tokenizer = pickle.load(handle)
                
            print("Artifacts loaded successfully.")
            
        except Exception as e:
            print(f"Error loading artifacts: {e}")
            self.model = None
            self.tokenizer = None

    def _clean_text(self, text):
        """
        Same cleaning logic as src/preprocess.py
        """
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def predict(self, text):
        """
        Predicts toxicity score for a single string.
        Returns: float (probability of toxicity)
        """
        if self.model is None or self.tokenizer is None:
            return 0.0
            
        cleaned_text = self._clean_text(text)
        seq = self.tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
        
        # Predict
        prob = self.model.predict(padded, verbose=0)[0][0]
        return float(prob)

    def predict_batch(self, texts):
        """
        Predicts toxicity scores for a list of strings.
        Returns: list of floats
        """
        if self.model is None or self.tokenizer is None:
            return []
            
        cleaned_texts = [self._clean_text(t) for t in texts]
        seqs = self.tokenizer.texts_to_sequences(cleaned_texts)
        padded = pad_sequences(seqs, maxlen=MAX_LEN, padding='post', truncating='post')
        
        probs = self.model.predict(padded, verbose=0)
        return probs.flatten().tolist()
