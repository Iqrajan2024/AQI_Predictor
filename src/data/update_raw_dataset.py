"""
PEARLS AQI PREDICTOR
Incremental Open-Meteo Raw Dataset Updater

FIRST RUN:
    Downloads 2024-08-01 -> today

SUBSEQUENT RUNS:
    Downloads only data after the latest timestamp already present
    in the raw CSV.

Output:
    data/raw/peshawar_openmeteo_2024_2026.csv
"""

from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests


# ============================================================
# CONFIGURATION
# ============================================================

LATITUDE = 34.008
LONGITUDE = 71.5785

INITIAL_START_DATE = "2024-08-01"

RAW_FILE = Path(
    "data/raw/peshawar_openmeteo_2024_2026.csv"
)

WEATHER_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
)

AIR_QUALITY_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
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
# DATE HELPERS
# ============================================================

def get_initial_dates():
    """
    Initial backfill:
        2024-08-01 -> yesterday

    We deliberately stop at yesterday because Open-Meteo's
    archive endpoint is historical data.
    """

    today = datetime.now(timezone.utc).date()
    yesterday = today - timedelta(days=1)

    return (
        datetime.strptime(
            INITIAL_START_DATE,
            "%Y-%m-%d"
        ).date(),
        yesterday,
    )


def get_incremental_dates(df):
    """
    Determine the next date range after the latest stored
    timestamp.
    """

    latest_timestamp = pd.to_datetime(
        df["timestamp"],
        utc=True
    ).max()

    latest_date = latest_timestamp.date()

    start_date = latest_date + timedelta(days=1)

    today = datetime.now(timezone.utc).date()
    yesterday = today - timedelta(days=1)

    return start_date, yesterday


# ============================================================
# OPEN-METEO REQUEST
# ============================================================

def fetch_openmeteo(
    start_date,
    end_date,
):
    print()
    print("=" * 70)
    print("DOWNLOADING OPEN-METEO DATA")
    print("=" * 70)

    print("Start:", start_date)
    print("End:", end_date)

    # --------------------------------------------------------
    # WEATHER
    # --------------------------------------------------------

    weather_params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": WEATHER_VARIABLES,
        "timezone": "UTC",
    }

    print()
    print("Requesting weather data...")

    weather_response = requests.get(
        WEATHER_URL,
        params=weather_params,
        timeout=120,
    )

    print(
        "Weather status:",
        weather_response.status_code
    )

    weather_response.raise_for_status()

    weather_data = weather_response.json()

    weather_df = pd.DataFrame(
        weather_data["hourly"]
    )

    weather_df["timestamp"] = pd.to_datetime(
        weather_df["time"],
        utc=True,
    )

    weather_df.drop(
        columns=["time"],
        inplace=True,
    )

    # --------------------------------------------------------
    # AIR QUALITY
    # --------------------------------------------------------

    air_params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": AIR_QUALITY_VARIABLES,
        "timezone": "UTC",
    }

    print()
    print("Requesting air-quality data...")

    air_response = requests.get(
        AIR_QUALITY_URL,
        params=air_params,
        timeout=120,
    )

    print(
        "Air-quality status:",
        air_response.status_code
    )

    air_response.raise_for_status()

    air_data = air_response.json()

    air_df = pd.DataFrame(
        air_data["hourly"]
    )

    air_df["timestamp"] = pd.to_datetime(
        air_df["time"],
        utc=True,
    )

    air_df.drop(
        columns=["time"],
        inplace=True,
    )

    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------

    df = pd.merge(
        weather_df,
        air_df,
        on="timestamp",
        how="inner",
    )

    df.sort_values(
        "timestamp",
        inplace=True,
    )

    df.drop_duplicates(
        subset=["timestamp"],
        keep="last",
        inplace=True,
    )

    df.reset_index(
        drop=True,
        inplace=True,
    )

    return df


# ============================================================
# INITIAL BACKFILL
# ============================================================

def create_initial_dataset():

    print()
    print("=" * 70)
    print("INITIAL RAW DATASET CREATION")
    print("=" * 70)

    start_date, end_date = get_initial_dates()

    if start_date > end_date:
        print("No historical data required.")
        return

    df = fetch_openmeteo(
        start_date,
        end_date,
    )

    RAW_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        RAW_FILE,
        index=False,
    )

    print()
    print("INITIAL DATASET CREATED")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("First:", df["timestamp"].min())
    print("Last:", df["timestamp"].max())
    print("Saved:", RAW_FILE)


# ============================================================
# INCREMENTAL UPDATE
# ============================================================

def update_existing_dataset():

    print()
    print("=" * 70)
    print("INCREMENTAL RAW DATASET UPDATE")
    print("=" * 70)

    df_existing = pd.read_csv(
        RAW_FILE,
        parse_dates=["timestamp"],
    )

    df_existing["timestamp"] = pd.to_datetime(
        df_existing["timestamp"],
        utc=True,
    )

    start_date, end_date = get_incremental_dates(
        df_existing
    )

    print(
        "Existing latest timestamp:",
        df_existing["timestamp"].max(),
    )

    print(
        "Requested new data:",
        start_date,
        "->",
        end_date,
    )

    # --------------------------------------------------------
    # Nothing new
    # --------------------------------------------------------

    if start_date > end_date:

        print()
        print("NO NEW DATA AVAILABLE.")

        return

    # --------------------------------------------------------
    # Download only new data
    # --------------------------------------------------------

    df_new = fetch_openmeteo(
        start_date,
        end_date,
    )

    print()
    print("New rows:", len(df_new))

    # --------------------------------------------------------
    # Append
    # --------------------------------------------------------

    df = pd.concat(
        [
            df_existing,
            df_new,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Deduplicate
    # --------------------------------------------------------

    df.drop_duplicates(
        subset=["timestamp"],
        keep="last",
        inplace=True,
    )

    df.sort_values(
        "timestamp",
        inplace=True,
    )

    df.reset_index(
        drop=True,
        inplace=True,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        RAW_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print("INCREMENTAL UPDATE COMPLETE")
    print("=" * 70)

    print("Total rows:", len(df))
    print("First:", df["timestamp"].min())
    print("Last:", df["timestamp"].max())
    print("Saved:", RAW_FILE)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PEARLS AQI PREDICTOR")
    print("RAW DATA UPDATE PIPELINE")
    print("=" * 70)

    if not RAW_FILE.exists():

        create_initial_dataset()

    else:

        update_existing_dataset()


if __name__ == "__main__":
    main()