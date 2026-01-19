import pandas as pd
import numpy as np
import os

# --- PATHS ---
RAW_DATA_PATH = "../data/raw_scraped_data.csv"
JOINT_DATA_PATH = "../data/joint_data_collection.csv"
TRAIN_DATA_PATH = "../data/training_data.csv"
TEST_DATA_PATH = "../data/test_data.csv"
ACTIVATION_DATA_PATH = "../data/activation_data.csv"

def clean_data():
    print("---  Starting Data Cleaning (Kaggle Edition) ---")
    
    # 1. Load Data
    if not os.path.exists(RAW_DATA_PATH):
        print(f" Error: {RAW_DATA_PATH} not found.")
        print("   Run 'python kaggle_import.py' first!")
        return

    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Loaded {len(df)} raw entries.")

    # 2. Clean Data (Robust Method)
    # Even though Kaggle data is clean, we treat it as text first to be safe
    # This ensures it works whether you scraped it OR imported it.
    
    # Clean Rent: Remove '€', spaces, and convert to number
    df['rent'] = df['rent_raw'].astype(str).str.replace('€', '').str.replace(',', '.').str.strip()
    df['rent'] = pd.to_numeric(df['rent'], errors='coerce')

    # Clean Size: Remove 'm²', spaces, and convert to number
    df['size'] = df['size_raw'].astype(str).str.replace('m²', '').str.strip()
    df['size'] = pd.to_numeric(df['size'], errors='coerce')

    # Drop rows that failed to convert
    df = df.dropna(subset=['rent', 'size'])

    # 3. Filter Outliers (Student Housing Logic)
    # We keep only realistic student rooms
    original_count = len(df)
    df = df[(df['rent'] > 100) & (df['rent'] < 2000)]
    df = df[(df['size'] > 8) & (df['size'] < 80)]
    
    if len(df) < original_count:
        print(f"Removed {original_count - len(df)} outliers (too expensive/big/small).")

    # 4. Normalization (Requirement)
    # Scale size between 0 and 1 for the Neural Network
    if not df.empty:
        df['size_norm'] = (df['size'] - df['size'].min()) / (df['size'].max() - df['size'].min())
    else:
        print(" CRITICAL: No data left after filtering! Check your raw data.")
        return

    # Save the "Joint Data Collection"
    df.to_csv(JOINT_DATA_PATH, index=False)
    print(f"Saved joint data: {JOINT_DATA_PATH}")

    # 5. Split Data 80/20 (Requirement)
    train_df = df.sample(frac=0.8, random_state=2026) # 80% Training
    test_df = df.drop(train_df.index)                 # 20% Testing

    # Save Train/Test files
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)
    print(f" SUCCESS: Created Training Set ({len(train_df)} rows) and Test Set ({len(test_df)} rows).")

    # 6. Create Activation Data (Requirement)
    # Just take the first row of the test set
    if not test_df.empty:
        activation_row = test_df.head(1)
        activation_row.to_csv(ACTIVATION_DATA_PATH, index=False)
        print(f"Saved activation data: {ACTIVATION_DATA_PATH}")

if __name__ == "__main__":
    clean_data()