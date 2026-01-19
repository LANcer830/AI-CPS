import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
import pickle

# --- PATHS ---
TRAIN_PATH = "../data/training_data.csv"
TEST_PATH = "../data/test_data.csv"
OUTPUT_DIR = "../model_output"
# We save as .pkl (Python Pickle) because OLS models aren't standard neural networks
MODEL_PATH = os.path.join(OUTPUT_DIR, "currentOlsSolution.pkl")

def train_ols():
    print("---  Starting OLS Model Training (Statsmodels) ---")
    
    # 1. Load Data
    try:
        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)
    except FileNotFoundError:
        print(" Error: Data files not found.")
        return

    # 2. Prepare Data 
    # Statsmodels requires us to manually add a "Constant" column (for the Y-intercept)
    X_train = train_df['size_norm']
    y_train = train_df['rent']
    
    X_test = test_df['size_norm']
    y_test = test_df['rent']

    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)

    # 3. Train OLS Model
    model = sm.OLS(y_train, X_train_const).fit()
    print(model.summary()) # Prints the statistical report (R-squared, P-values, etc.)

    # 4. Save Model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f" OLS Model saved to: {MODEL_PATH}")

    # 5. Evaluate & Visualize (Subgoal 5 Requirement)
    predictions = model.predict(X_test_const)
    
    # Calculate Error (MAE)
    mae = abs(y_test - predictions).mean()
    print(f"   Test Set MAE: {mae:.2f} €")

    # A. Scatter Plot (Green for OLS)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.6, color='green')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title('OLS: Actual Rent vs Predicted Rent')
    plt.xlabel('Actual Rent (€)')
    plt.ylabel('Predicted Rent (€)')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "ols_scatter_plot.png"))
    plt.close()

    # B. Diagnostic Plot (Residuals)
    residuals = y_test - predictions
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=25, color='green')
    plt.title('OLS: Diagnostic Plot (Error Distribution)')
    plt.xlabel('Prediction Error (€)')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "ols_diagnostic_plot.png"))
    plt.close()

    print(f" Visualizations saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    train_ols()