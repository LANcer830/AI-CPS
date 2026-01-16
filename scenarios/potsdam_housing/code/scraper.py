import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import os

# --- CONFIGURATION ---
# 108 = Potsdam, 8 = Berlin, 9387 = Werder (Havel)
CITIES = [
    {"id": 108, "name": "Potsdam"},
    {"id": 8, "name": "Berlin"},
    {"id": 9387, "name": "Werder_Havel"}
]
OUTPUT_FILE = "../data/raw_scraped_data.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7"
}

def scrape_wg_gesucht():
    print("--- Project R.A.D.A.R. Scraper (Potsdam + Berlin + Werder) ---")
    all_data = []

    for city in CITIES:
        cid = city["id"]
        cname = city["name"]
        print(f"\n--- Switching to City: {cname} (ID: {cid}) ---")
        
        # Base URL changes based on City ID
        base_url = f"https://www.wg-gesucht.de/wg-zimmer-in-{cname}.{cid}.0.1.0.html"

        # Scrape 5 pages per city
        for page in range(0, 5):
            print(f"Scanning {cname} Page {page}...")
            
            # Pagination logic
            if page == 0:
                url = base_url
            else:
                url = f"{base_url}?offer_filter=1&city_id={cid}&noDeact=1&categories%5B0%5D=0&pagination=1&pu={page}"
            
            try:
                response = requests.get(url, headers=HEADERS)
                
                if response.status_code != 200:
                    print(f"   Note: Page {page} end or blocked (Status {response.status_code}). Moving on.")
                    break
                
                soup = BeautifulSoup(response.content, "html.parser")
                ads = soup.find_all("div", class_="offer_list_item")
                
                if not ads:
                    print("   No ads found on this page.")
                    break
                
                for ad in ads:
                    try:
                        ad_id = ad.get("data-id")
                        title_tag = ad.find("h3", class_="truncate_title")
                        title = title_tag.text.strip() if title_tag else "Unknown"
                        
                        rent_tag = ad.find("b", string=lambda t: t and "€" in t)
                        rent = rent_tag.text.strip() if rent_tag else None
                        
                        size_tag = ad.find("b", string=lambda t: t and "m²" in t)
                        size = size_tag.text.strip() if size_tag else None

                        if rent and size:
                            all_data.append({
                                "id": ad_id,
                                "city": cname,
                                "title": title,
                                "rent_raw": rent,
                                "size_raw": size
                            })
                    except AttributeError:
                        continue
                
                print(f"   Found {len(ads)} ads. Total collected so far: {len(all_data)}")
                time.sleep(random.uniform(2, 4))
                
            except Exception as e:
                print(f"   Error: {e}")
                break

    # --- SAVE ---
    if len(all_data) > 0:
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        df = pd.DataFrame(all_data)
        # Drop duplicates by ID
        df.drop_duplicates(subset=['id'], inplace=True)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSUCCESS: Scraped {len(df)} total listings.")
        print(f"Data saved to: {OUTPUT_FILE}")
    else:
        print("\nFAILED: No data collected.")

if __name__ == "__main__":
    scrape_wg_gesucht()