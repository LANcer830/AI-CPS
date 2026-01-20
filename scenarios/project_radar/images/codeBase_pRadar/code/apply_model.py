import pandas as pd
import tensorflow as tf
import joblib
import statsmodels.api as sm
import os

ACTIVATION_PATH = "/tmp/activationBase/activation_data.csv"
KB_PATH = "/tmp/knowledgeBase/"

data = pd.read_csv(ACTIVATION_PATH)

# Extract only the feature column used during training
X = data[['size_norm']]  # ANN expects this format
print(f"   Feature shape: {X.shape}")
print(f"   Sample values:\n{X.head()}")

# For OLS, we need to add a constant column (intercept)
X_with_const = sm.add_constant(X)

# Load models
try:
    ann = tf.keras.models.load_model(KB_PATH + "currentAiSolution.h5")
except Exception as e:
    exit(1)

try:
    ols = joblib.load(KB_PATH + "currentOlsSolution.pkl")
except Exception as e:
    exit(1)

# Run predictions
try:
    ann_preds = ann.predict(X, verbose=0)
except Exception as e:
    exit(1)

try:
    ols_preds = ols.predict(X_with_const)
except Exception as e:
    exit(1)

# Create a summary DataFrame
results = pd.DataFrame({
    'id': data['id'],
    'city': data['city'],
    'size_sqm': data['size'],
    'size_norm': data['size_norm'],
    'ANN_predicted_rent': ann_preds.flatten(),
    'OLS_predicted_rent': ols_preds
})

print(results.head(10).to_string(index=False))

print(f"--- Summary Statistics ---")
print(f"ANN Average Predicted Rent: {ann_preds.mean():.2f} €")
print(f"OLS Average Predicted Rent: {ols_preds.mean():.2f} €")
print(f"ANN Min/Max: {ann_preds.min():.2f} € / {ann_preds.max():.2f} €")
print(f"OLS Min/Max: {ols_preds.min():.2f} € / {ols_preds.max():.2f} €")