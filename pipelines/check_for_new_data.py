import requests

API_URL = "https://data.ny.gov/resource/t6yz-b64h.json"

def check_for_new_data():
    response = requests.get(
        API_URL,
        params={
            "$select": "toll_10_minute_block",
            "$order": "toll_10_minute_block DESC",
            "$limit": 1
        },
        timeout=30,
    )

    if not response.ok:
        print("API error response:", response.text)
        response.raise_for_status()

    results = response.json()

    if not results:
        raise RuntimeError("NY Open Data returned no records.")

    latest_timestamp = results[0]["toll_10_minute_block"]

    print(f"Latest NY Open Data timestamp: {latest_timestamp}")

    return latest_timestamp

if __name__ == "__main__":
    check_for_new_data()