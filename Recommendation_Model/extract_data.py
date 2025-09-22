import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import json

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}

# Categories mapped to occasions
categories = {
    "Wedding": {
        "Saree": "https://www.myntra.com/saree",
        "Jewellery": "https://www.myntra.com/jewellery",
        "Heels": "https://www.myntra.com/heels"
    },
    "Casual": {
        "Jeans": "https://www.myntra.com/jeans",
        "Tshirts": "https://www.myntra.com/tshirts",
        "Sneakers": "https://www.myntra.com/sneakers"
    },
    "Formal": {
        "Shirts": "https://www.myntra.com/formal-shirts",
        "Trousers": "https://www.myntra.com/formal-trousers",
        "Shoes": "https://www.myntra.com/formal-shoes"
    }
}

def scrape_myntra(url, category, occasion, limit=10):
    """Scrape Myntra product data from a category page"""
    try:
        r = requests.get(url, headers=headers)
        if r.status_code != 200:
            print(f"⚠️ Failed to fetch {url}, status {r.status_code}")
            return []

        soup = BeautifulSoup(r.text, "html.parser")

        # Find script with JSON data
        script = None
        for s in soup.find_all("script"):
            if s.string and "results" in s.string:
                script = s
                break

        if not script:
            print(f"⚠️ No JSON block found for {category} ({occasion})")
            return []

        # Extract JSON
        match = re.search(r'({.*"products".*})', script.string)
        if not match:
            print(f"⚠️ JSON regex failed for {category} ({occasion})")
            return []

        json_text = match.group(1)
        data = json.loads(json_text)

        # Handle structure variations
        if "searchData" in data:
            products = data["searchData"]["results"]["products"]
        elif "props" in data:  # NEXT.js style
            products = data["props"]["pageProps"]["initialData"]["searchData"]["results"]["products"]
        else:
            print(f"⚠️ Unknown JSON structure for {category} ({occasion})")
            return []

        # Build dataset
        dataset = []
        for p in products[:limit]:
            dataset.append({
                "id": p.get("productId"),
                "name": p.get("product"),
                "brand": p.get("brand"),
                "color": p.get("primaryColour"),
                "category": category,
                "occasion": occasion,
                "price": p.get("price"),
                "discount_price": p.get("discountedPrice"),
                "image_url": p.get("searchImage"),
                "warehouse": "Expressway Hub"  # mock value
            })
        return dataset

    except Exception as e:
        print(f"❌ Error scraping {category} ({occasion}):", e)
        return []


# Main script
all_data = []

for occasion, cats in categories.items():
    for cat, url in cats.items():
        print(f"⏳ Scraping {occasion} - {cat} ...")
        limit = 10 if cat != "Shoes" else 5  # Shoes only 5
        items = scrape_myntra(url, cat, occasion, limit=limit)
        all_data.extend(items)

# Save to CSV
df = pd.DataFrame(all_data)
df.to_csv("expressway_dataset.csv", index=False)

print("✅ Dataset created with", len(df), "items")
print(df.head())
