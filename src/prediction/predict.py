"""
============================================================
PEARLS AQI PREDICTOR
NEXT-HOUR PREDICTION PIPELINE
============================================================

Loads the trained XGBoost champion model and predicts the
next-hour AQI using the same 70 model features used during
training.

Model:
    models/champion_model.pkl

Input:
    Recent Open-Meteo weather + air-quality data

Output:
    Next-hour AQI prediction
"""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import requests


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "champion_model.pkl"
)


# ============================================================
# LOCATION
# ============================================================

LATITUDE = 34.008
LONGITUDE = 71.5785


# ============================================================
# OPEN-METEO ENDPOINTS
# ============================================================

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

AIR_QUALITY_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
)


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURE_COLUMNS = [
    # Weather
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",

    # Pollutants
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",

    # Time
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",

    # AQI lags
    "aqi_lag_1",
    "aqi_lag_3",
    "aqi_lag_6",
    "aqi_lag_12",
    "aqi_lag_24",
    "aqi_lag_48",
    "aqi_lag_72",

    # PM2.5 lags
    "pm2_5_lag_1",
    "pm2_5_lag_3",
    "pm2_5_lag_6",
    "pm2_5_lag_24",

    # PM10 lags
    "pm10_lag_1",
    "pm10_lag_3",
    "pm10_lag_6",
    "pm10_lag_24",

    # Carbon monoxide lags
    "carbon_monoxide_lag_1",
    "carbon_monoxide_lag_3",
    "carbon_monoxide_lag_6",
    "carbon_monoxide_lag_24",

    # Nitrogen dioxide lags
    "nitrogen_dioxide_lag_1",
    "nitrogen_dioxide_lag_3",
    "nitrogen_dioxide_lag_6",
    "nitrogen_dioxide_lag_24",

    # Sulphur dioxide lags
    "sulphur_dioxide_lag_1",
    "sulphur_dioxide_lag_3",
    "sulphur_dioxide_lag_6",
    "sulphur_dioxide_lag_24",

    # Ozone lags
    "ozone_lag_1",
    "ozone_lag_3",
    "ozone_lag_6",
    "ozone_lag_24",

    # AQI rolling means
    "aqi_3h_mean",
    "aqi_6h_mean",
    "aqi_12h_mean",
    "aqi_24h_mean",

    # PM2.5 rolling means
    "pm2_5_3h_mean",
    "pm2_5_6h_mean",
    "pm2_5_24h_mean",

    # PM10 rolling means
    "pm10_3h_mean",
    "pm10_6h_mean",
    "pm10_24h_mean",

    # Other pollutant rolling means
    "carbon_monoxide_24h_mean",
    "nitrogen_dioxide_24h_mean",
    "sulphur_dioxide_24h_mean",
    "ozone_24h_mean",

    # AQI changes
    "aqi_change_1h",
    "aqi_change_3h",
    "aqi_change_6h",
    "aqi_change_24h",

    # PM2.5 changes
    "pm2_5_change_1h",
    "pm2_5_change_24h",

    # PM10 changes
    "pm10_change_1h",
    "pm10_change_24h",
]


# ============================================================
# VALIDATION
# ============================================================

if len(FEATURE_COLUMNS) != 70:
    raise ValueError(
        f"Expected 70 features, found {len(FEATURE_COLUMNS)}"
    )


# ============================================================
# PRINT HEADER
# ============================================================

def print_header():

    print("=" * 60)
    print("PEARLS AQI PREDICTOR")
    print("NEXT-HOUR PREDICTION PIPELINE")
    print("=" * 60)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python executable: {sys.executable}")

    print("\nModel file:")
    print(MODEL_FILE)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\n" + "=" * 60)
    print("LOADING CHAMPION MODEL")
    print("=" * 60)

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Champion model not found:\n{MODEL_FILE}"
        )

    model = joblib.load(MODEL_FILE)

    print("✓ Champion model loaded")
    print("Model type:", type(model).__name__)

    return model


# ============================================================
# FETCH WEATHER DATA
# ============================================================

def fetch_weather():

    print("\n" + "=" * 60)
    print("FETCHING WEATHER DATA")
    print("=" * 60)

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,

        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "pressure_msl,"
            "precipitation,"
            "wind_speed_10m,"
            "wind_direction_10m"
        ),

        # Enough history for 72-hour features
        # plus future hours.
        "past_hours": 96,
        "forecast_hours": 24,

        "timezone": "UTC",

        "wind_speed_unit": "kmh",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
    }

    response = requests.get(
        WEATHER_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    if "hourly" not in data:
        raise ValueError(
            "Weather API response does not contain hourly data."
        )

    weather = pd.DataFrame(data["hourly"])

    weather["timestamp"] = pd.to_datetime(
        weather["time"],
        utc=True,
    )

    weather = weather.drop(
        columns=["time"]
    )

    print("✓ Weather data received")
    print("Weather rows:", len(weather))

    return weather


# ============================================================
# FETCH AIR QUALITY DATA
# ============================================================

def fetch_air_quality():

    print("\n" + "=" * 60)
    print("FETCHING AIR-QUALITY DATA")
    print("=" * 60)

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,

        "hourly": (
            "pm2_5,"
            "pm10,"
            "carbon_monoxide,"
            "nitrogen_dioxide,"
            "sulphur_dioxide,"
            "ozone,"
            "us_aqi"
        ),

        "past_hours": 96,
        "forecast_hours": 24,

        "timezone": "UTC",
    }

    response = requests.get(
        AIR_QUALITY_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    if "hourly" not in data:
        raise ValueError(
            "Air-quality API response does not contain hourly data."
        )

    air = pd.DataFrame(data["hourly"])

    air["timestamp"] = pd.to_datetime(
        air["time"],
        utc=True,
    )

    air = air.drop(
        columns=["time"]
    )

    print("✓ Air-quality data received")
    print("Air-quality rows:", len(air))

    return air


# ============================================================
# MERGE DATA
# ============================================================

def merge_data(weather, air):

    print("\n" + "=" * 60)
    print("MERGING WEATHER + AIR QUALITY")
    print("=" * 60)

    df = pd.merge(
        weather,
        air,
        on="timestamp",
        how="inner",
    )

    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    print("Merged dataset shape:", df.shape)

    if df.empty:
        raise ValueError(
            "Merged weather and air-quality dataset is empty."
        )

    print("✓ Data merged successfully")

    return df


# ============================================================
# CREATE FEATURES
# ============================================================

def create_features(df):

    print("\n" + "=" * 60)
    print("CREATING 70 MODEL FEATURES")
    print("=" * 60)

    df = df.copy()

    # --------------------------------------------------------
    # TIME FEATURES
    # --------------------------------------------------------

    df["hour"] = df["timestamp"].dt.hour

    df["day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )

    df["day_of_month"] = (
        df["timestamp"].dt.day
    )

    df["month"] = (
        df["timestamp"].dt.month
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # --------------------------------------------------------
    # AQI LAGS
    # --------------------------------------------------------

    for lag in [1, 3, 6, 12, 24, 48, 72]:

        df[f"aqi_lag_{lag}"] = (
            df["us_aqi"].shift(lag)
        )

    # --------------------------------------------------------
    # POLLUTANT LAGS
    # --------------------------------------------------------

    pollutants = [
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
    ]

    for pollutant in pollutants:

        for lag in [1, 3, 6, 24]:

            df[f"{pollutant}_lag_{lag}"] = (
                df[pollutant].shift(lag)
            )

    # --------------------------------------------------------
    # AQI ROLLING MEANS
    # --------------------------------------------------------

    for window in [3, 6, 12, 24]:

        df[f"aqi_{window}h_mean"] = (
            df["us_aqi"]
            .shift(1)
            .rolling(window=window)
            .mean()
        )

    # --------------------------------------------------------
    # PM2.5 ROLLING MEANS
    # --------------------------------------------------------

    for window in [3, 6, 24]:

        df[f"pm2_5_{window}h_mean"] = (
            df["pm2_5"]
            .shift(1)
            .rolling(window=window)
            .mean()
        )

    # --------------------------------------------------------
    # PM10 ROLLING MEANS
    # --------------------------------------------------------

    for window in [3, 6, 24]:

        df[f"pm10_{window}h_mean"] = (
            df["pm10"]
            .shift(1)
            .rolling(window=window)
            .mean()
        )

    # --------------------------------------------------------
    # OTHER POLLUTANT 24H MEANS
    # --------------------------------------------------------

    for pollutant in [
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
    ]:

        df[f"{pollutant}_24h_mean"] = (
            df[pollutant]
            .shift(1)
            .rolling(window=24)
            .mean()
        )

    # --------------------------------------------------------
    # AQI CHANGE FEATURES
    # --------------------------------------------------------

    df["aqi_change_1h"] = (
        df["us_aqi"] - df["aqi_lag_1"]
    )

    df["aqi_change_3h"] = (
        df["us_aqi"] - df["aqi_lag_3"]
    )

    df["aqi_change_6h"] = (
        df["us_aqi"] - df["aqi_lag_6"]
    )

    df["aqi_change_24h"] = (
        df["us_aqi"] - df["aqi_lag_24"]
    )

    # --------------------------------------------------------
    # PM2.5 CHANGE FEATURES
    # --------------------------------------------------------

    df["pm2_5_change_1h"] = (
        df["pm2_5"] - df["pm2_5_lag_1"]
    )

    df["pm2_5_change_24h"] = (
        df["pm2_5"] - df["pm2_5_lag_24"]
    )

    # --------------------------------------------------------
    # PM10 CHANGE FEATURES
    # --------------------------------------------------------

    df["pm10_change_1h"] = (
        df["pm10"] - df["pm10_lag_1"]
    )

    df["pm10_change_24h"] = (
        df["pm10"] - df["pm10_lag_24"]
    )

    # --------------------------------------------------------
    # VERIFY FEATURES
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing model features:\n"
            + "\n".join(missing_features)
        )

    print("✓ All 70 model features created")

    # --------------------------------------------------------
    # REMOVE ROWS WITHOUT FEATURES
    # --------------------------------------------------------

    valid_df = df.dropna(
        subset=FEATURE_COLUMNS
    ).copy()

    if valid_df.empty:

        raise ValueError(
            "No valid rows remain after feature creation."
        )

    print(
        "Rows available for prediction:",
        len(valid_df),
    )

    return valid_df


# ============================================================
# PREDICT
# ============================================================

def predict_next_hour(model, df):

    print("\n" + "=" * 60)
    print("GENERATING NEXT-HOUR AQI PREDICTION")
    print("=" * 60)

    latest_row = (
        df
        .sort_values("timestamp")
        .iloc[-1]
    )

    X_latest = (
        latest_row[FEATURE_COLUMNS]
        .to_frame()
        .T
    )

    # Make sure all values are numeric.
    X_latest = X_latest.astype(float)

    prediction = model.predict(
        X_latest
    )[0]

    prediction = max(
        0.0,
        float(prediction)
    )

    latest_timestamp = latest_row["timestamp"]

    prediction_timestamp = (
        latest_timestamp
        + pd.Timedelta(hours=1)
    )

    print("\n" + "=" * 60)
    print("AQI PREDICTION")
    print("=" * 60)

    print(
        "Latest data time :",
        latest_timestamp,
    )

    print(
        "Prediction time  :",
        prediction_timestamp,
    )

    print(
        f"Predicted AQI    : {prediction:.2f}"
    )

    return prediction, prediction_timestamp


# ============================================================
# MAIN
# ============================================================

def main():

    print_header()

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # FETCH DATA
    # --------------------------------------------------------

    weather = fetch_weather()

    air = fetch_air_quality()

    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------

    df = merge_data(
        weather,
        air,
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    feature_df = create_features(
        df
    )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    prediction, prediction_time = (
        predict_next_hour(
            model,
            feature_df,
        )
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PREDICTION PIPELINE COMPLETED")
    print("=" * 60)

    print(
        f"Next-hour AQI: {prediction:.2f}"
    )

    print(
        f"Prediction timestamp: {prediction_time}"
    )

    print("\n✓ Prediction completed successfully")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
