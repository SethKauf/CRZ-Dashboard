import json
import os
import pandas as pd
import requests

from pathlib import Path

API_URL = "https://data.ny.gov/resource/t6yz-b64h.json"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "src" / "model_config.json"

def check_for_new_data():
    # 1. Get latest timestamp from NY Open Data
    response = requests.get(
        API_URL,
        params={
            "$select": "toll_10_minute_block",
            "$order": "toll_10_minute_block DESC",
            "$limit": 1
        },
        timeout=30,
    )

    response.raise_for_status()

    results = response.json()

    if not results:
        raise RuntimeError("NY Open Data returned no records.")
    
    if not response.ok:
        print("API error response:", response.text)
        response.raise_for_status()

    source_timestamp = pd.to_datetime(
        results[0]["toll_10_minute_block"]
    )

    # 2. Read last model training timestamp
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    trained_timestamp = pd.to_datetime(
        config["last_trained_timestamp"]
    )

    # 3. Compare timestamps
    new_data_available = source_timestamp > trained_timestamp

    github_output = os.environ.get("GITHUB_OUTPUT")
    
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"new_data_available={str(new_data_available).lower()}\n")

    # 4. Print results
    print(f"Latest source timestamp: {source_timestamp}")
    print(f"Last trained timestamp: {trained_timestamp}")
    print(f"New data available: {new_data_available}")

    return {
        "source_timestamp": source_timestamp.isoformat(),
        "trained_timestamp": trained_timestamp.isoformat(),
        "new_data_available": bool(new_data_available),
    }

if __name__ == "__main__":
    check_for_new_data()