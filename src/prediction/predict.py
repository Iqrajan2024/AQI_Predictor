"""
============================================================
PEARLS AQI PREDICTOR
72-HOUR PREDICTION PIPELINE
============================================================

Loads the trained XGBoost champion model and predicts the
next-hour AQI using the same 70 model features used during
training.

Model:
    models/champion_model.pkl

Input:
    Recent Open-Meteo weather + air-quality data

Output:
    Hourly AQI predictions for the next 72 hours
    Daily average AQI values within the forecast window
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
    print("72-HOUR PREDICTION PIPELINE")
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
        "forecast_hours": 72,

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
        "forecast_hours": 72,

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

def create_features(df, verbose=True):
    if verbose:
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

    if verbose:
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

    if verbose:
        print(
            "Rows available for prediction:",
            len(valid_df),
        )

    return valid_df

# ============================================================
# RECURSIVE 72-HOUR PREDICTION
# ============================================================

def predict_next_72_hours(model, df):

    print("\n" + "=" * 60)
    print("GENERATING 72-HOUR AQI FORECAST")
    print("=" * 60)

    working_df = (
        df
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .reset_index(drop=True)
        .copy()
    )

    if len(working_df) < 100:
        raise ValueError(
            "Not enough data returned by Open-Meteo."
        )


    # --------------------------------------------------------
    # SPLIT HISTORY AND FUTURE
    # --------------------------------------------------------
    # Open-Meteo provides AQI forecasts for future timestamps too,
    # so future us_aqi values can also be non-null.
    # --------------------------------------------------------

    HISTORY_HOURS = 96
    FORECAST_HOURS = 72

    total_rows = len(working_df)

    expected_rows = (
        HISTORY_HOURS
        + FORECAST_HOURS
    )

    if total_rows < expected_rows:
        raise ValueError(
            f"Expected at least {expected_rows} rows "
            f"from Open-Meteo, but received {total_rows}."
        )

    # --------------------------------------------------------
    # TAKE THE LAST 168 HOURLY ROWS
    # --------------------------------------------------------

    working_df = (
        working_df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history = (
        working_df
        .iloc[:HISTORY_HOURS]
        .copy()
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # FUTURE FORECAST
    # --------------------------------------------------------

    future_df = (
        working_df
        .iloc[
            HISTORY_HOURS:
            HISTORY_HOURS + FORECAST_HOURS
        ]
        .copy()
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # LATEST OBSERVED TIMESTAMP
    # --------------------------------------------------------

    latest_observed_timestamp = (
        history["timestamp"].iloc[-1]
    )

    print(
        "Latest observed time:",
        latest_observed_timestamp,
    )

    print(
        "Forecast start time:",
        future_df["timestamp"].iloc[0],
    )

    print(
        "Forecast end time:",
        future_df["timestamp"].iloc[-1],
    )

    print(
        "Historical rows:",
        len(history),
    )

    print(
        "Future forecast rows:",
        len(future_df),
    )

    # --------------------------------------------------------
    # VALIDATE FUTURE DATA
    # --------------------------------------------------------

    if len(future_df) != FORECAST_HOURS:
        raise ValueError(
            f"Expected {FORECAST_HOURS} future rows, "
            f"but received {len(future_df)}."
        )

    
    future_df = (
        future_df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    print(
        "Future rows available:",
        len(future_df),
    )

    if len(future_df) < 72:
        raise ValueError(
            f"Expected at least 72 future rows, "
            f"but received {len(future_df)}."
        )

    # Keep exactly 72 hours.
    future_df = future_df.iloc[:72].copy()

    print(
        "Forecast start time:",
        future_df["timestamp"].iloc[0],
    )

    print(
        "Forecast end time:",
        future_df["timestamp"].iloc[-1],
    )

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------
    #
    # Largest lag = 72 hours.
    # Largest rolling window = 24 hours.
    #
    # Keep 96 hours so all features can be constructed.
    # --------------------------------------------------------

    history = (
        working_df
        .iloc[:HISTORY_HOURS]
        .copy()
        .reset_index(drop=True)
    )

    history = (
        history
        .sort_values("timestamp")
        .tail(96)
        .reset_index(drop=True)
    )

    if len(history) < 72:
        raise ValueError(
            "Not enough historical observations "
            "for 72-hour lag features."
        )

    # --------------------------------------------------------
    # RECURSIVE FORECAST
    # --------------------------------------------------------

    predictions = []

    print("\nStarting recursive prediction...")

    for step in range(72):

        # ----------------------------------------------------
        # FUTURE INPUT ROW
        # ----------------------------------------------------
        #
        # This contains weather/pollutant forecasts for the
        # hour we are going to predict.
        # ----------------------------------------------------

        future_row = (
            future_df
            .iloc[step]
            .copy()
        )

        prediction_timestamp = (
            future_row["timestamp"]
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # The trained model predicts:
        #
        #     AQI(T+1) using features(T)
        #
        # Therefore the model input must be constructed from
        # the CURRENT/ANCHOR observation immediately before
        # the prediction timestamp.
        #
        # For the first prediction, this is the latest known
        # observation.
        #
        # For later predictions, this is the previously
        # predicted AQI/weather row.
        # ----------------------------------------------------

        anchor_timestamp = (
            prediction_timestamp
            - pd.Timedelta(hours=1)
        )

        # ----------------------------------------------------
        # FIND ANCHOR ROW
        # ----------------------------------------------------

        anchor_rows = history[
            history["timestamp"]
            == anchor_timestamp
        ]

        if anchor_rows.empty:

            raise ValueError(
                f"Could not find anchor row for "
                f"{prediction_timestamp}. "
                f"Expected anchor timestamp: "
                f"{anchor_timestamp}"
            )

        anchor_row = (
            anchor_rows
            .iloc[-1]
            .copy()
        )

        # ----------------------------------------------------
        # CREATE FEATURES USING ANCHOR ROW
        # ----------------------------------------------------
        #
        # The anchor row contains a REAL AQI:
        #
        #   observed AQI for the first prediction
        #   predicted AQI for subsequent predictions
        #
        # Therefore all 70 features can be calculated.
        # ----------------------------------------------------

        feature_source = history.copy()

        feature_df = create_features(
            feature_source,
            verbose=False,
        )

        target_rows = feature_df[
            feature_df["timestamp"]
            == anchor_timestamp
        ]

        if target_rows.empty:

            raise ValueError(
                f"Could not create features for "
                f"anchor timestamp "
                f"{anchor_timestamp}"
            )

        target_row = (
            target_rows
            .iloc[-1]
            .copy()
        )

        # ----------------------------------------------------
        # MODEL INPUT
        # ----------------------------------------------------

        X = (
            target_row[FEATURE_COLUMNS]
            .to_frame()
            .T
            .astype(float)
        )

        # ----------------------------------------------------
        # NaN CHECK
        # ----------------------------------------------------

        if X.isna().any().any():

            missing_features = (
                X.columns[
                    X.isna().any()
                ]
                .tolist()
            )

            raise ValueError(
                f"Missing features for anchor "
                f"{anchor_timestamp}:\n"
                + "\n".join(missing_features)
            )

        # ----------------------------------------------------
        # PREDICT NEXT-HOUR AQI
        # ----------------------------------------------------

        predicted_aqi = model.predict(X)[0]

        predicted_aqi = max(
            0.0,
            float(predicted_aqi),
        )

        # ----------------------------------------------------
        # STORE PREDICTION
        # ----------------------------------------------------

        predictions.append(
            {
                "timestamp":
                    prediction_timestamp,

                "predicted_aqi":
                    predicted_aqi,
            }
        )

        # ----------------------------------------------------
        # ADD PREDICTED AQI TO HISTORY
        # ----------------------------------------------------
        #
        # The weather and pollutant values for the predicted
        # hour come from Open-Meteo's forecast.
        
        predicted_row = pd.DataFrame(
            [{
                "timestamp":
                    prediction_timestamp,

                "temperature_2m":
                    future_row["temperature_2m"],

                "relative_humidity_2m":
                    future_row["relative_humidity_2m"],

                "pressure_msl":
                    future_row["pressure_msl"],

                "precipitation":
                    future_row["precipitation"],

                "wind_speed_10m":
                    future_row["wind_speed_10m"],

                "wind_direction_10m":
                    future_row["wind_direction_10m"],

                "pm2_5":
                    future_row["pm2_5"],

                "pm10":
                    future_row["pm10"],

                "carbon_monoxide":
                    future_row["carbon_monoxide"],

                "nitrogen_dioxide":
                    future_row["nitrogen_dioxide"],

                "sulphur_dioxide":
                    future_row["sulphur_dioxide"],

                "ozone":
                    future_row["ozone"],

                "us_aqi":
                    predicted_aqi,
            }]
        )

        history = pd.concat(
            [
                history,
                predicted_row,
            ],
            ignore_index=True,
        )

        history = (
            history
            .sort_values("timestamp")
            .tail(96)
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            step < 3
            or (step + 1) % 12 == 0
            or step == 71
        ):

            print(
                f"Hour {step + 1:02d}/72 | "
                f"{prediction_timestamp} | "
                f"AQI: {predicted_aqi:.2f}"
            )

    # ========================================================
    # CREATE FORECAST DATAFRAME
    # ========================================================

    result_df = pd.DataFrame(
        predictions
    )

    # ========================================================
    # FORECAST DAY
    # ========================================================

    result_df["forecast_day"] = (
        result_df["timestamp"].dt.date
    )

    # ========================================================
    # DAILY AVERAGES
    # ========================================================

    daily_forecast = (
        result_df
        .groupby(
            "forecast_day",
            as_index=False,
        )
        .agg(
            predicted_aqi=(
                "predicted_aqi",
                "mean",
            )
        )
    )

    # ========================================================
    # HOURLY RESULTS
    # ========================================================

    print("\n" + "=" * 60)
    print("72-HOUR AQI FORECAST")
    print("=" * 60)

    print(
        result_df[
            [
                "timestamp",
                "predicted_aqi",
            ]
        ].to_string(index=False)
    )

    # ========================================================
    # DAILY RESULTS
    # ========================================================

    print("\n" + "=" * 60)
    print("DAILY AQI AVERAGES WITHIN 72-HOUR FORECAST")
    print("=" * 60)

    for _, row in daily_forecast.iterrows():

        print(
            f"{row['forecast_day']} | "
            f"Average AQI: "
            f"{row['predicted_aqi']:.2f}"
        )

    print("\n" + "=" * 60)
    print("72-HOUR PREDICTION COMPLETED")
    print("=" * 60)

    return result_df, daily_forecast

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
    # 72-HOUR PREDICTION
    #
    # The prediction function creates the required 70 features
    # recursively for each future hour.
    # --------------------------------------------------------

    forecast_df, daily_forecast = (
        predict_next_72_hours(
            model,
            df,
        )
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PREDICTION PIPELINE COMPLETED")
    print("=" * 60)

    print("\nDaily AQI Averages Within 72-Hour Forecast:")

    for _, row in daily_forecast.iterrows():

        print(
            f"{row['forecast_day']} | "
            f"Average AQI: "
            f"{row['predicted_aqi']:.2f}"
        )

    print("\n✓ 72-hour prediction completed successfully")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()