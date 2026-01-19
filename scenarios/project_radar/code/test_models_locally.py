import pandas as pd
import tensorflow as tf
import pickle
import os
import numpy as np

# --- PATHS ---
ANN_PATH = "../model_output/currentAiSolution.h5"
OLS_PATH = "../model_output/currentOlsSolution.pkl"

def test_models():
    print("---  Local Model Verification Test ---")

    # 1. Load ANN (TensorFlow)
    if os.path.exists(ANN_PATH):
        print(" Found ANN Model.")
        ann_model = tf.keras.models.load_model(ANN_PATH)
    else:
        print(" ANN Model missing!")
        return

    # 2. Load OLS (Pickle)
    if os.path.exists(OLS_PATH):
        print(" Found OLS Model.")
        with open(OLS_PATH, 'rb') as f:
            ols_model = pickle.load(f)
    else:
        print(" OLS Model missing!")
        return

    # 3. Create a Dummy Room for Testing
    # Let's say we have a room that is 20 sqm.
    # We need to normalize it because the AI learned on normalized data (0-1).
    # Assuming max size in training was ~80 and min was ~8.
    raw_size = 20
    size_norm = (raw_size - 8) / (80 - 8) 
    
    print(f"\n--- Testing Prediction for a {raw_size}m² Room ---")
    
    # 4. Predict with ANN
    # Input shape must be (1,1) -> [[size_norm]]
    ann_input = np.array([[size_norm]])
    ann_pred = ann_model.predict(ann_input)
    print(f" ANN Prediction: {ann_pred[0][0]:.2f} €")

    # 5. Predict with OLS
    # OLS needs [const, size_norm] -> [1.0, size_norm]
    ols_input = [1.0, size_norm] 
    ols_pred = ols_model.predict(ols_input)
    print(f" OLS Prediction: {ols_pred[0]:.2f} €")

    print("\n Verification Complete: Both models are readable and predicting!")

if __name__ == "__main__":
    test_models()