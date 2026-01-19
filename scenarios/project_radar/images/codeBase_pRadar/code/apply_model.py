import pickle
import pandas as pd
import tensorflow as tf

ACTIVATION_PATH = "/tmp/activationBase/activation_data.csv"
KB_PATH = "/tmp/knowledgeBase/"

data = pd.read_csv(ACTIVATION_PATH)

# ANN prediction
ann = tf.keras.models.load_model(KB_PATH + "currentAiSolution.h5")
ann_pred = ann.predict(data)

# OLS prediction
with open(KB_PATH + "currentOlsSolution.pkl", "rb") as f:
    ols = pickle.load(f)
ols_pred = ols.predict(data)

print("ANN Prediction:", ann_pred)
print("OLS Prediction:", ols_pred)
