"""
Pearls AQI Predictor
Incremental Open-Meteo Raw Dataset Updater

Purpose:
    Incrementally update the historical raw dataset using Open-Meteo
    historical weather and air-quality data.

Important:
    This script ONLY updates historical observations.
    It must NOT use forecast data or model predictions.
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

# Keep historical ingestion safely behind the current date.
#
# Open-Meteo historical weather data has a publication delay.
# Five days provides a conservative historical cutoff.
HISTORICAL_DELAY_DAYS = 5

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


REQUIRED_COLUMNS = [
    "timestamp",
    *WEATHER_VARIABLES,
    *AIR_QUALITY_VARIABLES,
]


# ============================================================
# DETERMINE SAFE HISTORICAL CUTOFF
# ============================================================

now_utc = pd.Timestamp.now(tz="UTC")

today = now_utc.normalize()

safe_end_timestamp = (
    today
    - timedelta(days=HISTORICAL_DELAY_DAYS)
)

print("=" * 70)
print("HISTORICAL UPDATE CONFIGURATION")
print("=" * 70)

print("Current UTC:", now_utc)
print("Historical delay:", HISTORICAL_DELAY_DAYS, "days")
print("Safe historical cutoff:", safe_end_timestamp)


# ============================================================
# LOAD EXISTING DATASET
# ============================================================

if OUTPUT_PATH.exists():

    print()
    print("=" * 70)
    print("EXISTING RAW DATASET FOUND")
    print("=" * 70)

    existing_df = pd.read_csv(
        OUTPUT_PATH
    )

    if "timestamp" not in existing_df.columns:
        raise RuntimeError(
            "Existing raw dataset does not contain "
            "the required 'timestamp' column."
        )

    existing_df["timestamp"] = pd.to_datetime(
        existing_df["timestamp"],
        utc=True,
    )

    latest_timestamp = (
        existing_df["timestamp"]
        .max()
    )

    print(
        "Latest existing timestamp:",
        latest_timestamp,
    )

    start_timestamp = (
        latest_timestamp
        + timedelta(hours=1)
    )

else:

    print()
    print("=" * 70)
    print("INITIAL BACKFILL")
    print("=" * 70)

    existing_df = None

    start_timestamp = pd.Timestamp(
        INITIAL_START_DATE,
        tz="UTC",
    )


# ============================================================
# VALIDATE DATE RANGE
# ============================================================

print()
print("=" * 70)
print("HISTORICAL DATE RANGE")
print("=" * 70)

print("Requested start timestamp:", start_timestamp)
print("Safe historical cutoff:", safe_end_timestamp)


if start_timestamp > safe_end_timestamp:

    print()
    print("No new historical data is currently available.")
    print("Existing raw dataset is already up to date.")
    raise SystemExit(0)


# ============================================================
# OPEN-METEO DATE RANGE
# ============================================================

start_date = (
    start_timestamp
    .strftime("%Y-%m-%d")
)

end_date = (
    safe_end_timestamp
    .strftime("%Y-%m-%d")
)

print()
print("Open-Meteo start date:", start_date)
print("Open-Meteo end date:", end_date)


# ============================================================
# WEATHER REQUEST
# ============================================================

print()
print("=" * 70)
print("DOWNLOADING HISTORICAL WEATHER")
print("=" * 70)

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

weather_data = weather_response.json()

if "hourly" not in weather_data:
    raise RuntimeError(
        "Open-Meteo weather response does not contain "
        "'hourly' data."
    )


weather_df = pd.DataFrame(
    weather_data["hourly"]
)

if "time" not in weather_df.columns:
    raise RuntimeError(
        "Open-Meteo weather response does not contain "
        "the 'time' column."
    )

weather_df["timestamp"] = pd.to_datetime(
    weather_df["time"],
    utc=True,
)

weather_df.drop(
    columns=["time"],
    inplace=True,
)


# ============================================================
# AIR QUALITY REQUEST
# ============================================================

print()
print("=" * 70)
print("DOWNLOADING HISTORICAL AIR QUALITY")
print("=" * 70)

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

air_data = air_response.json()

if "hourly" not in air_data:
    raise RuntimeError(
        "Open-Meteo air-quality response does not contain "
        "'hourly' data."
    )


air_df = pd.DataFrame(
    air_data["hourly"]
)

if "time" not in air_df.columns:
    raise RuntimeError(
        "Open-Meteo air-quality response does not contain "
        "the 'time' column."
    )


air_df["timestamp"] = pd.to_datetime(
    air_df["time"],
    utc=True,
)

air_df.drop(
    columns=["time"],
    inplace=True,
)


# ============================================================
# MERGE WEATHER + AIR QUALITY
# ============================================================

print()
print("=" * 70)
print("MERGING WEATHER + AIR QUALITY")
print("=" * 70)

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
# REMOVE DATA OUTSIDE SAFE HISTORICAL WINDOW
# ============================================================

new_df = new_df[
    (new_df["timestamp"] >= start_timestamp)
    & (new_df["timestamp"] <= safe_end_timestamp + timedelta(hours=23))
].copy()

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

    before_dedup = len(new_df)

    new_df = new_df[
        ~new_df["timestamp"].isin(
            existing_timestamps
        )
    ].copy()

    print(
        "Already-existing rows removed:",
        before_dedup - len(new_df),
    )


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

final_df["timestamp"] = pd.to_datetime(
    final_df["timestamp"],
    utc=True,
)

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
# VALIDATE REQUIRED COLUMNS
# ============================================================

missing_columns = [
    column
    for column in REQUIRED_COLUMNS
    if column not in final_df.columns
]

if missing_columns:
    raise RuntimeError(
        "Raw dataset is missing required columns: "
        + ", ".join(missing_columns)
    )


# ============================================================
# VALIDATE TIMESTAMPS
# ============================================================

duplicate_count = (
    final_df["timestamp"]
    .duplicated()
    .sum()
)

if duplicate_count != 0:
    raise RuntimeError(
        f"Duplicate timestamps remain: {duplicate_count}"
    )


if not final_df["timestamp"].is_monotonic_increasing:
    raise RuntimeError(
        "Timestamps are not monotonically increasing."
    )


# ============================================================
# VALIDATE HISTORICAL BOUNDARY
# ============================================================

actual_max_timestamp = (
    final_df["timestamp"]
    .max()
)

if actual_max_timestamp > (
    safe_end_timestamp + timedelta(hours=23)
):

    raise RuntimeError(
        "Raw dataset contains data newer than the "
        "safe historical cutoff."
    )


# ============================================================
# VALIDATE MISSING VALUES
# ============================================================

missing_values = (
    final_df[REQUIRED_COLUMNS]
    .isna()
    .sum()
    .sum()
)

if missing_values != 0:

    raise RuntimeError(
        f"Required raw features contain "
        f"{missing_values} missing values."
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
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("RAW DATASET UPDATED SUCCESSFULLY")
print("=" * 70)

print(
    "New rows downloaded:",
    len(new_df),
)

print(
    "Total rows:",
    len(final_df),
)

print(
    "First timestamp:",
    final_df["timestamp"].min(),
)

print(
    "Last timestamp:",
    final_df["timestamp"].max(),
)

print(
    "Duplicate timestamps:",
    final_df["timestamp"].duplicated().sum(),
)

print(
    "Missing required values:",
    missing_values,
)

print(
    "Saved:",
    OUTPUT_PATH,
)