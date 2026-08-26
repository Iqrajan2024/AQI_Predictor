"""
Pearls AQI Predictor
Incremental Open-Meteo Raw Dataset Updater
"""

from pathlib import Path
from datetime import timedelta

import pandas as pd
import requests


# ============================================================
# CONFIGURATION
# ============================================================

LATITUDE = 34.008
LONGITUDE = 71.5785

INITIAL_START_DATE = "2024-08-01"

WEATHER_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
)

AIR_QUALITY_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
)

OUTPUT_PATH = Path(
    "data/raw/peshawar_openmeteo_2024_2026.csv"
)


WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
]


AIR_QUALITY_VARIABLES = [
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
    "us_aqi_pm2_5",
    "us_aqi_pm10",
    "us_aqi_carbon_monoxide",
    "us_aqi_nitrogen_dioxide",
    "us_aqi_sulphur_dioxide",
    "us_aqi_ozone",
]


# ============================================================
# DETERMINE DATE RANGE
# ============================================================

today = pd.Timestamp.now(tz="UTC").normalize()

if OUTPUT_PATH.exists():

    print("=" * 70)
    print("EXISTING RAW DATASET FOUND")
    print("=" * 70)

    existing_df = pd.read_csv(
        OUTPUT_PATH
    )

    existing_df["timestamp"] = pd.to_datetime(
        existing_df["timestamp"],
        utc=True
    )

    latest_timestamp = (
        existing_df["timestamp"]
        .max()
    )

    print(
        "Latest existing timestamp:",
        latest_timestamp
    )

    # Start from the next hour.
    start_timestamp = (
        latest_timestamp
        + timedelta(hours=1)
    )

else:

    print("=" * 70)
    print("INITIAL BACKFILL")
    print("=" * 70)

    existing_df = None

    start_timestamp = pd.Timestamp(
        INITIAL_START_DATE,
        tz="UTC"
    )


# ============================================================
# NOTHING NEW TO DOWNLOAD
# ============================================================

if start_timestamp >= today + timedelta(days=1):

    print()
    print("No new data required.")
    print("Raw dataset is already up to date.")

    raise SystemExit(0)


# ============================================================
# OPEN-METEO USES DATE RANGE
# ============================================================

start_date = (
    start_timestamp
    .strftime("%Y-%m-%d")
)

end_date = (
    today
    .strftime("%Y-%m-%d")
)


print()
print("=" * 70)
print("DOWNLOADING OPEN-METEO DATA")
print("=" * 70)

print("Start date:", start_date)
print("End date:", end_date)


# ============================================================
# WEATHER
# ============================================================

weather_params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": start_date,
    "end_date": end_date,
    "hourly": WEATHER_VARIABLES,
    "timezone": "UTC",
}


weather_response = requests.get(
    WEATHER_URL,
    params=weather_params,
    timeout=120,
)

weather_response.raise_for_status()

weather_data = (
    weather_response.json()
)

weather_df = pd.DataFrame(
    weather_data["hourly"]
)

weather_df["timestamp"] = pd.to_datetime(
    weather_df["time"],
    utc=True
)

weather_df.drop(
    columns=["time"],
    inplace=True
)


# ============================================================
# AIR QUALITY
# ============================================================

air_params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": start_date,
    "end_date": end_date,
    "hourly": AIR_QUALITY_VARIABLES,
    "timezone": "UTC",
}


air_response = requests.get(
    AIR_QUALITY_URL,
    params=air_params,
    timeout=120,
)

air_response.raise_for_status()

air_data = (
    air_response.json()
)

air_df = pd.DataFrame(
    air_data["hourly"]
)

air_df["timestamp"] = pd.to_datetime(
    air_df["time"],
    utc=True
)

air_df.drop(
    columns=["time"],
    inplace=True
)


# ============================================================
# MERGE NEW DATA
# ============================================================

new_df = pd.merge(
    weather_df,
    air_df,
    on="timestamp",
    how="inner",
)

new_df.sort_values(
    "timestamp",
    inplace=True,
)

new_df.reset_index(
    drop=True,
    inplace=True,
)


# ============================================================
# REMOVE ALREADY EXISTING TIMESTAMPS
# ============================================================

if existing_df is not None:

    existing_timestamps = set(
        existing_df["timestamp"]
    )

    new_df = new_df[
        ~new_df["timestamp"].isin(
            existing_timestamps
        )
    ]


# ============================================================
# COMBINE
# ============================================================

if existing_df is not None:

    final_df = pd.concat(
        [
            existing_df,
            new_df,
        ],
        ignore_index=True,
    )

else:

    final_df = new_df


# ============================================================
# CLEAN
# ============================================================

final_df.sort_values(
    "timestamp",
    inplace=True,
)

final_df.drop_duplicates(
    subset=["timestamp"],
    keep="last",
    inplace=True,
)

final_df.reset_index(
    drop=True,
    inplace=True,
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

final_df.to_csv(
    OUTPUT_PATH,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("RAW DATASET UPDATED")
print("=" * 70)

print(
    "New rows downloaded:",
    len(new_df)
)

print(
    "Total rows:",
    len(final_df)
)

print(
    "First timestamp:",
    final_df["timestamp"].min()
)

print(
    "Last timestamp:",
    final_df["timestamp"].max()
)

print(
    "Duplicate timestamps:",
    final_df["timestamp"].duplicated().sum()
)

print(
    "Missing values:",
    final_df.isna().sum().sum()
)

print(
    "Saved:",
    OUTPUT_PATH
)