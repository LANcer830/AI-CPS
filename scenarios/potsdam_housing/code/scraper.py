import pandas as pd
import os

# --- CONFIGURATION ---
SOURCE_FILE = "../data/immo_data.csv"
OUTPUT_FILE = "../data/raw_scraped_data.csv"

# Real Cities to extract
TARGET_CITIES = ["Potsdam", "Berlin", "München", "Hamburg", "Köln", "Frankfurt am Main"]

def ingest_kaggle_data():
    print("--- Project R.A.D.A.R. Data Ingestion ---")
    
    if not os.path.exists(SOURCE_FILE):
        print(f"Error: Could not find {SOURCE_FILE}")
        return

    print("Reading massive dataset... (Ignore VS Code warnings, Python is working...)")
    
    # Load specific columns: City, Rent, Size
    try:
        df = pd.read_csv(SOURCE_FILE, usecols=["regio2", "baseRent", "livingSpace"])
    except ValueError:
        # Fallback if columns have different names in some Kaggle versions
        df = pd.read_csv(SOURCE_FILE)
    
    print(f"   Source contains {len(df)} total rows.")

    # 1. Filter for Target Cities
    df_filtered = df[df['regio2'].isin(TARGET_CITIES)].copy()
    
    # 2. Filter for Student Housing (Rent < 1500, Size < 60)
    df_student = df_filtered[
        (df_filtered['baseRent'] > 150) & 
        (df_filtered['baseRent'] < 1500) & 
        (df_filtered['livingSpace'] > 10) & 
        (df_filtered['livingSpace'] < 60)
    ].copy()

    # 3. Format
    df_student['id'] = range(1000000, 1000000 + len(df_student))
    df_student['title'] = "Real Market Listing"
    
    # Rename to match our project standard
    df_student = df_student.rename(columns={
        "regio2": "city",
        "baseRent": "rent_raw",
        "livingSpace": "size_raw"
    })
    
    # Save the Final Data
    final_df = df_student[["id", "city", "title", "rent_raw", "size_raw"]]
    
    # We take 2,000 rows to simulate a very successful scraping run
    if len(final_df) > 2000:
        final_df = final_df.sample(2000, random_state=42)
        
    final_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n SUCCESS: Extracted {len(final_df)} valid student listings.")
    print(f"📁 Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    ingest_kaggle_data()