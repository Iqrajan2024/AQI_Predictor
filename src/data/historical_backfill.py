"""
============================================================
Pearls AQI Predictor
Historical Data Backfill
============================================================

Purpose
-------
Download two years of hourly weather and air-quality data
for Peshawar from Open-Meteo and combine them into one
raw dataset.

Period:
2024-08-01 → 2026-08-01

Data source:
Open-Meteo

Location:
Peshawar
Latitude:  34.008
Longitude: 71.5785
============================================================
"""

import requests
import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

LATITUDE = 34.008
LONGITUDE = 71.5785

START_DATE = "2024-08-01"
END_DATE = "2026-08-01"

WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

AIR_QUALITY_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
)


# ============================================================
# OUTPUT PATH
# ============================================================

OUTPUT_PATH = Path(
    "data/raw/peshawar_openmeteo_2024_2026.csv"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# WEATHER VARIABLES
# ============================================================

WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m"
]


# ============================================================
# AIR QUALITY VARIABLES
# ============================================================

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
    "us_aqi_ozone"
]


# ============================================================
# FETCH WEATHER DATA
# ============================================================

print("=" * 70)
print("PESHAWAR HISTORICAL WEATHER BACKFILL")
print("=" * 70)

weather_params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": WEATHER_VARIABLES,
    "timezone": "UTC"
}

print()
print("Downloading weather data...")
print("Start:", START_DATE)
print("End:", END_DATE)

weather_response = requests.get(
    WEATHER_URL,
    params=weather_params,
    timeout=120
)

print("Status code:", weather_response.status_code)

weather_response.raise_for_status()

weather_data = weather_response.json()

print("Weather request successful!")


# ============================================================
# CONVERT WEATHER TO DATAFRAME
# ============================================================

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
# FETCH AIR QUALITY DATA
# ============================================================

print()
print("=" * 70)
print("PESHAWAR HISTORICAL AIR QUALITY BACKFILL")
print("=" * 70)

air_params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": AIR_QUALITY_VARIABLES,
    "timezone": "UTC"
}

print()
print("Downloading air-quality data...")
print("Start:", START_DATE)
print("End:", END_DATE)

air_response = requests.get(
    AIR_QUALITY_URL,
    params=air_params,
    timeout=120
)

print("Status code:", air_response.status_code)

air_response.raise_for_status()

air_data = air_response.json()

print("Air-quality request successful!")


# ============================================================
# CONVERT AIR QUALITY TO DATAFRAME
# ============================================================

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
# MERGE WEATHER + AIR QUALITY
# ============================================================

print()
print("=" * 70)
print("MERGING DATASETS")
print("=" * 70)

df = pd.merge(
    weather_df,
    air_df,
    on="timestamp",
    how="inner"
)

df.sort_values(
    "timestamp",
    inplace=True
)

df.reset_index(
    drop=True,
    inplace=True
)


# ============================================================
# CHECK DUPLICATES
# ============================================================

duplicate_count = df["timestamp"].duplicated().sum()

print()
print("Duplicate timestamps:", duplicate_count)


# ============================================================
# CHECK MISSING VALUES
# ============================================================

print()
print("Missing values:")

print(
    df.isnull().sum()
)


# ============================================================
# DATASET INFORMATION
# ============================================================

print()
print("=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print()
print("Total records:", len(df))

print(
    "First timestamp:",
    df["timestamp"].min()
)

print(
    "Last timestamp:",
    df["timestamp"].max()
)

print(
    "Number of columns:",
    len(df.columns)
)


# ============================================================
# SAVE RAW DATA
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print()
print("=" * 70)
print("BACKFILL COMPLETE")
print("=" * 70)

print()
print("Saved to:")
print(OUTPUT_PATH)

print()
print("Dataset shape:", df.shape)

print()
print("Columns:")
for column in df.columns:
    print(" -", column)