import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
PROCESSED_DATA_DIR = os.path.join('data', 'processed')
MODEL_PATH = os.path.join('models', 'toxicity_model.h5')
RESULTS_DIR = 'results'

def main():
    print("--- STARTING EVALUATION ---")
    
    # 1. Load Resources
    print("Loading test data and model...")
    try:
        # Note: In a real scenario, we would have a separate 'X_test.npy' created in preprocess.py 
        # from the 'test.csv'. For this portfolio project structure, we use the validation set 
        # as our proxy for 'unseen' data to evaluate the final model performance.
        X_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_val.npy'))
        y_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_val.npy'))
        
        if not os.path.exists(MODEL_PATH):
             raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
             
        model = tf.keras.models.load_model(MODEL_PATH)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 2. Make Predictions
    print("Generating predictions...")
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int)

    # 3. Calculate Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n--- RESULTS ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    # 4. Save Results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Save textual metrics
    with open(os.path.join(RESULTS_DIR, 'metrics.txt'), 'w') as f:
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred))
        
    # Generate and Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
    
    print(f"\nMetrics and plots saved to {RESULTS_DIR}/")

if __name__ == "__main__":
    main()
