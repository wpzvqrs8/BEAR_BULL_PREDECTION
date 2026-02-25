import os
import requests
import zipfile
from datetime import datetime

# ==============================
# CONFIG
# ==============================

base_url = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/"
save_path = r"C:\Users\Admin\PyCharmMiscProject\pythonprojects\my_pro\AI_ML_DL\STOCKMARKET_PREDICTOR_DEMO\datas"

start_year = 2017
start_month = 8
end_year = 2026
end_month = 1

# ==============================
# CREATE FOLDER IF NOT EXISTS
# ==============================

os.makedirs(save_path, exist_ok=True)

# ==============================
# GENERATE ALL MONTHS
# ==============================

dates = []
current = datetime(start_year, start_month, 1)
end_date = datetime(end_year, end_month, 1)

while current <= end_date:
    dates.append(current.strftime("%Y-%m"))
    if current.month == 12:
        current = datetime(current.year + 1, 1, 1)
    else:
        current = datetime(current.year, current.month + 1, 1)

# ==============================
# DOWNLOAD + EXTRACT
# ==============================

for date in dates:
    filename = f"BTCUSDT-1m-{date}.zip"
    file_url = base_url + filename
    zip_path = os.path.join(save_path, filename)

    if os.path.exists(zip_path):
        print(f"Already downloaded: {filename}")
    else:
        print(f"Downloading: {filename}")
        response = requests.get(file_url, stream=True)

        if response.status_code == 200:
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            print("Download complete.")
        else:
            print(f"File not found on server: {filename}")
            continue

    # Extract
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(save_path)
        print(f"Extracted: {filename}")
    except zipfile.BadZipFile:
        print(f"Corrupted zip: {filename}")

print("\nALL FILES DOWNLOADED AND EXTRACTED ✅")