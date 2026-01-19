import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os

# --- PATHS ---
TRAIN_PATH = "../data/training_data.csv"
TEST_PATH = "../data/test_data.csv"

# We save the model and plots to a new 'model_output' folder
OUTPUT_DIR = "../model_output"
MODEL_PATH = os.path.join(OUTPUT_DIR, "currentAiSolution.h5")

def train_model():
    print("---  Starting AI Model Training (TensorFlow) ---")
    
    # 1. Setup Output Directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 2. Load Data
    print("   Loading data...")
    try:
        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)
    except FileNotFoundError:
        print(" Error: Data files not found. Run 'cleaner.py' first.")
        return

    # 3. Prepare Inputs (X) and Targets (Y)
    # INPUT: We use 'size_norm' (normalized size 0-1) so the AI learns faster
    # TARGET: We want to predict 'rent' (real Euro value)
    X_train = train_df[['size_norm']]
    y_train = train_df['rent']
    
    X_test = test_df[['size_norm']]
    y_test = test_df['rent']

    print(f"   Training on {len(X_train)} rows, Testing on {len(X_test)} rows.")

    # 4. Build the ANN Architecture (Subgoal 4 Requirement)
    # We use a simple Feed-Forward Network
    model = keras.Sequential([
        layers.Input(shape=(1,)),       # Input Layer: 1 feature (Size)
        layers.Dense(64, activation='relu'), # Hidden Layer 1: 64 neurons
        layers.Dense(64, activation='relu'), # Hidden Layer 2: 64 neurons
        layers.Dense(1)                 # Output Layer: 1 value (Predicted Rent)
    ])

    # Compile the model
    # Optimizer: Adam (Standard)
    # Loss: Mean Squared Error (MSE) - minimizes the squared difference between real/predicted rent
    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(0.001),
                  metrics=['mae', 'mse'])

    # 5. Train the Model
    print("   Training in progress... (This takes a few seconds)")
    history = model.fit(
        X_train, y_train,
        epochs=100,             # How many times we loop through the data
        validation_split=0.2,   # Use 20% of training data for internal validation
        verbose=0               # Silent mode (change to 1 to see progress bar)
    )

    # 6. Save the Model
    model.save(MODEL_PATH)
    print(f" Model saved to: {MODEL_PATH}")

    # 7. Evaluate Performance
    loss, mae, mse = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test Set Mean Absolute Error: {mae:.2f} €")
    print("   (This means on average, the AI prediction is off by this amount)")

    # --- VISUALIZATION (Subgoal 4 Requirements) ---
    plot_results(history, model, X_test, y_test)

def plot_results(history, model, X_test, y_test):
    print("   Generating required visualizations...")

    # A. Learning Curves (Loss vs Epochs)
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('1. Learning Curve (Loss over Time)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "learning_curve.png"))
    plt.close()

    # Geting Predictions for Test Data
    predictions = model.predict(X_test).flatten()

    # B. Scatter Plot (Real vs Predicted)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--') # Perfect prediction line
    plt.title('2. Actual Rent vs Predicted Rent')
    plt.xlabel('Actual Rent (€)')
    plt.ylabel('Predicted Rent (€)')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "scatter_plot.png"))
    plt.close()

    # C. Diagnostic Plot (Residuals)
    # Residual = Real - Predicted. want these to be close to 0.
    residuals = y_test - predictions
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=25)
    plt.title('3. Diagnostic Plot (Error Distribution)')
    plt.xlabel('Prediction Error (€)')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "diagnostic_plot.png"))
    plt.close()

    print(f"Visualizations saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    train_model()