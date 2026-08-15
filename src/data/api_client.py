import os
import requests
from dotenv import load_dotenv
from datetime import datetime, timezone


# Load API key from .env
load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Peshawar coordinates
LATITUDE = 34.0151
LONGITUDE = 71.5249

# Historical air pollution endpoint
URL = "https://api.openweathermap.org/data/2.5/air_pollution/history"


# Test period: 24 hours
start_date = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
end_date = datetime(2026, 8, 2, 0, 0, tzinfo=timezone.utc)

# Convert dates to Unix timestamps
start_timestamp = int(start_date.timestamp())
end_timestamp = int(end_date.timestamp())


params = {
    "lat": LATITUDE,
    "lon": LONGITUDE,
    "start": start_timestamp,
    "end": end_timestamp,
    "appid": API_KEY
}


response = requests.get(URL, params=params)

print("Status code:", response.status_code)

if response.status_code == 200:

    data = response.json()

    print("Historical request successful!")
    print("Number of records:", len(data.get("list", [])))

else:

    print("Request failed.")
    print("Response:", response.text)