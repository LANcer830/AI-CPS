import os
import sys
import pandas as pd
import tensorflow as tf
import joblib
import statsmodels.api as sm

# -----------------------------
# Paths (DO NOT CHANGE)
# -----------------------------
ACTIVATION_PATH = "/tmp/activationBase/activation_data.csv"
KB_PATH = "/tmp/knowledgeBase/"

ANN_MODEL_PATH = os.path.join(KB_PATH, "currentAiSolution.h5")
OLS_MODEL_PATH = os.path.join(KB_PATH, "currentOlsSolution.pkl")

# -----------------------------
# Load activation data
# -----------------------------
print("Loading activation data...")
if not os.path.exists(ACTIVATION_PATH):
    raise FileNotFoundError(f"Activation data not found at {ACTIVATION_PATH}")

data = pd.read_csv(ACTIVATION_PATH)

required_columns = {"id", "city", "size", "size_norm"}
missing = required_columns - set(data.columns)
if missing:
    raise ValueError(f"Missing required columns in activation data: {missing}")

print("Activation data loaded successfully")
print(f"Number of samples: {len(data)}")
print("\nFeature preview:")
print(data[["size_norm"]].head().to_string(index=False))

# -----------------------------
# Prepare features
# -----------------------------
X = data[["size_norm"]]

print(f"\nFeature shape used for inference: {X.shape}")

# -----------------------------
# Load models
# -----------------------------
print("\nLoading ANN model...")
if not os.path.exists(ANN_MODEL_PATH):
    raise FileNotFoundError(f"ANN model not found at {ANN_MODEL_PATH}")

ann_model = tf.keras.models.load_model(ANN_MODEL_PATH)
print("ANN model loaded")

print("\nLoading OLS model...")
if not os.path.exists(OLS_MODEL_PATH):
    raise FileNotFoundError(f"OLS model not found at {OLS_MODEL_PATH}")

ols_model = joblib.load(OLS_MODEL_PATH)
print("OLS model loaded")

# -----------------------------
# Run predictions
# -----------------------------
print("\nRunning predictions...")

ann_preds = ann_model.predict(X).flatten()

# We force statsmodels to add the constant even for a single row 
# by using 'has_constant='add' or manually checking the columns.
X_ols = sm.add_constant(X, has_constant='add')

# If X_ols still only has 1 column (happens with single-row DataFrames), 
# we manually insert the 'const' column that the model expects.
if len(X_ols.columns) == 1:
    X_ols.insert(0, 'const', 1.0)

ols_preds = ols_model.predict(X_ols)

# -----------------------------
# Combine results
# -----------------------------
results = pd.DataFrame({
    "id": data["id"],
    "city": data["city"],
    "size_sqm": data["size"],
    "ANN_predicted_rent": ann_preds,
    "OLS_predicted_rent": ols_preds
})

# -----------------------------
# Print results clearly
# -----------------------------
print("\n==============================")
print(" FINAL RENT PREDICTIONS (€) ")
print("==============================")

for _, row in results.iterrows():
    print(
        f"Listing {row['id']} in {row['city']} "
        f"({row['size_sqm']} sqm): "
        f"ANN = {row['ANN_predicted_rent']:.2f} €, "
        f"OLS = {row['OLS_predicted_rent']:.2f} €"
    )

print("\n------------------------------")
print(f"ANN Average Predicted Rent: {results['ANN_predicted_rent'].mean():.2f} €")
print(f"OLS Average Predicted Rent: {results['OLS_predicted_rent'].mean():.2f} €")
print("------------------------------")

print("\nInference completed successfully.")

sys.exit(0)