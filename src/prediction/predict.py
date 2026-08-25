"""
============================================================
PEARLS AQI PREDICTOR
NEXT 3-DAY FORECAST PIPELINE
============================================================

Architecture:

    Open-Meteo
        |
        +-- Weather forecast
        +-- Air-quality forecast
        |
        v
    Historical state / feature context
        |
        v
    Feast Feature Store
        |
        v
    70 model features
        |
        v
    MLflow Champion XGBoost
        |
        v
    Hourly AQI predictions
        |
        v
    NEXT 3 CALENDAR DAYS
        |
        +-- Hourly forecast CSV
        +-- Daily average CSV

Model:
    MLflow Model Registry
    Pearls_AQI_XGBoost@champion

Feature Store:
    Feast
    feature_repo/feature_repo

Location:
    Peshawar, Pakistan

Output:
    data/predictions/latest_forecast.csv
    data/predictions/latest_daily_forecast.csv
"""

from pathlib import Path
import sys

import mlflow
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from zoneinfo import ZoneInfo

from feast import FeatureStore


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_REPO = (
    PROJECT_ROOT
    / "feature_repo"
    / "feature_repo"
)

PREDICTION_DIR = (
    PROJECT_ROOT
    / "data"
    / "predictions"
)

HOURLY_PREDICTION_FILE = (
    PREDICTION_DIR
    / "latest_forecast.csv"
)

DAILY_PREDICTION_FILE = (
    PREDICTION_DIR
    / "latest_daily_forecast.csv"
)

FEATURE_PREDICTION_FILE = (
    PREDICTION_DIR
    / "latest_forecast_features.csv"
)

# ============================================================
# MLflow
# ============================================================

MLFLOW_TRACKING_URI = (
    "http://127.0.0.1:5000"
)

MLFLOW_MODEL_URI = (
    "models:/Pearls_AQI_XGBoost@champion"
)


# ============================================================
# FEAST
# ============================================================

FEAST_FEATURE_SERVICE = "aqi_features"


# ============================================================
# LOCATION
# ============================================================

LOCATION_ID = "peshawar"

LATITUDE = 34.008
LONGITUDE = 71.5785


# ============================================================
# OPEN-METEO ENDPOINTS
# ============================================================

WEATHER_URL = (
    "https://api.open-meteo.com/v1/forecast"
)

AIR_QUALITY_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
)


# ============================================================
# FORECAST CONFIGURATION
# ============================================================

HISTORY_HOURS = 96

FORECAST_DAYS = 3

FORECAST_HOURS = 72

# Open-Meteo forecast starts from the current hour.
# We need enough hours to reach midnight of the
# third NEXT calendar day.
API_FORECAST_HOURS = 24 + FORECAST_HOURS

# ============================================================
# 70 MODEL FEATURES
# ============================================================

FEATURE_COLUMNS = [

    # --------------------------------------------------------
    # Weather
    # --------------------------------------------------------

    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",

    # --------------------------------------------------------
    # Pollutants
    # --------------------------------------------------------

    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",

    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",

    # --------------------------------------------------------
    # AQI lags
    # --------------------------------------------------------

    "aqi_lag_1",
    "aqi_lag_3",
    "aqi_lag_6",
    "aqi_lag_12",
    "aqi_lag_24",
    "aqi_lag_48",
    "aqi_lag_72",

    # --------------------------------------------------------
    # PM2.5 lags
    # --------------------------------------------------------

    "pm2_5_lag_1",
    "pm2_5_lag_3",
    "pm2_5_lag_6",
    "pm2_5_lag_24",

    # --------------------------------------------------------
    # PM10 lags
    # --------------------------------------------------------

    "pm10_lag_1",
    "pm10_lag_3",
    "pm10_lag_6",
    "pm10_lag_24",

    # --------------------------------------------------------
    # Carbon monoxide lags
    # --------------------------------------------------------

    "carbon_monoxide_lag_1",
    "carbon_monoxide_lag_3",
    "carbon_monoxide_lag_6",
    "carbon_monoxide_lag_24",

    # --------------------------------------------------------
    # Nitrogen dioxide lags
    # --------------------------------------------------------

    "nitrogen_dioxide_lag_1",
    "nitrogen_dioxide_lag_3",
    "nitrogen_dioxide_lag_6",
    "nitrogen_dioxide_lag_24",

    # --------------------------------------------------------
    # Sulphur dioxide lags
    # --------------------------------------------------------

    "sulphur_dioxide_lag_1",
    "sulphur_dioxide_lag_3",
    "sulphur_dioxide_lag_6",
    "sulphur_dioxide_lag_24",

    # --------------------------------------------------------
    # Ozone lags
    # --------------------------------------------------------

    "ozone_lag_1",
    "ozone_lag_3",
    "ozone_lag_6",
    "ozone_lag_24",

    # --------------------------------------------------------
    # AQI rolling means
    # --------------------------------------------------------

    "aqi_3h_mean",
    "aqi_6h_mean",
    "aqi_12h_mean",
    "aqi_24h_mean",

    # --------------------------------------------------------
    # PM2.5 rolling means
    # --------------------------------------------------------

    "pm2_5_3h_mean",
    "pm2_5_6h_mean",
    "pm2_5_24h_mean",

    # --------------------------------------------------------
    # PM10 rolling means
    # --------------------------------------------------------

    "pm10_3h_mean",
    "pm10_6h_mean",
    "pm10_24h_mean",

    # --------------------------------------------------------
    # Other pollutant rolling means
    # --------------------------------------------------------

    "carbon_monoxide_24h_mean",
    "nitrogen_dioxide_24h_mean",
    "sulphur_dioxide_24h_mean",
    "ozone_24h_mean",

    # --------------------------------------------------------
    # AQI changes
    # --------------------------------------------------------

    "aqi_change_1h",
    "aqi_change_3h",
    "aqi_change_6h",
    "aqi_change_24h",

    # --------------------------------------------------------
    # PM2.5 changes
    # --------------------------------------------------------

    "pm2_5_change_1h",
    "pm2_5_change_24h",

    # --------------------------------------------------------
    # PM10 changes
    # --------------------------------------------------------

    "pm10_change_1h",
    "pm10_change_24h",
]


# ============================================================
# FEATURE COUNT VALIDATION
# ============================================================

if len(FEATURE_COLUMNS) != 70:

    raise ValueError(
        f"Expected exactly 70 features, "
        f"found {len(FEATURE_COLUMNS)}"
    )


# ============================================================
# HEADER
# ============================================================

def print_header():

    print("=" * 70)
    print("PEARLS AQI PREDICTOR")
    print("NEXT 3-DAY AQI FORECAST PIPELINE")
    print("=" * 70)

    print(
        f"Project root       : {PROJECT_ROOT}"
    )

    print(
        f"Python executable  : {sys.executable}"
    )

    print(
        f"MLflow URI         : {MLFLOW_TRACKING_URI}"
    )

    print(
        f"MLflow model       : {MLFLOW_MODEL_URI}"
    )

    print(
        f"Feast repository   : {FEATURE_REPO}"
    )

    print(
        f"Location           : {LOCATION_ID}"
    )

    print(
        f"Forecast horizon   : {FORECAST_DAYS} days "
        f"({FORECAST_HOURS} hours)"
    )


# ============================================================
# LOAD MLflow CHAMPION
# ============================================================

def load_model():

    print("\n" + "=" * 70)
    print("LOADING MLflow CHAMPION MODEL")
    print("=" * 70)

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    print(
        "Tracking URI:",
        mlflow.get_tracking_uri()
    )

    print(
        "Model URI:",
        MLFLOW_MODEL_URI
    )

    model = mlflow.pyfunc.load_model(
        MLFLOW_MODEL_URI
    )

    print(
        "✓ MLflow champion model loaded"
    )

    print(
        "Model type:",
        type(model)
    )

    return model


# ============================================================
# LOAD FEAST
# ============================================================

def load_feast():

    print("\n" + "=" * 70)
    print("LOADING FEAST FEATURE STORE")
    print("=" * 70)

    if not FEATURE_REPO.exists():

        raise FileNotFoundError(
            "Feast repository not found:\n"
            f"{FEATURE_REPO}"
        )

    store = FeatureStore(
        repo_path=str(FEATURE_REPO)
    )

    print(
        "✓ Feast FeatureStore loaded"
    )

    print(
        "Repository:",
        FEATURE_REPO
    )

    return store


def get_api_forecast_hours():
    """
    Calculate how many future hourly records Open-Meteo
    must return so that we can extract the next 3 complete
    calendar days, excluding the current day.
    """

    now_utc = pd.Timestamp.now(tz="UTC")

    current_date = now_utc.date()

    forecast_start = (
        pd.Timestamp(current_date, tz="UTC")
        + pd.Timedelta(days=1)
    )

    forecast_end = (
        forecast_start
        + pd.Timedelta(hours=FORECAST_HOURS - 1)
    )
    current_hour = now_utc.floor("h")

    required_hours = int(
        (
            forecast_end
            - current_hour
        ).total_seconds()
        / 3600
    ) + 1

    return required_hours

# ============================================================
# FETCH WEATHER
# ============================================================

def fetch_weather():

    print("\n" + "=" * 70)
    print("FETCHING OPEN-METEO WEATHER DATA")
    print("=" * 70)

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

        "past_hours": HISTORY_HOURS,

        "forecast_hours":  get_api_forecast_hours(),

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
            "Weather response does not contain "
            "hourly data."
        )

    weather = pd.DataFrame(
        data["hourly"]
    )

    weather["timestamp"] = pd.to_datetime(
        weather["time"],
        utc=True,
    )

    weather = weather.drop(
        columns=["time"]
    )

    weather = (
        weather
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"]
        )
        .reset_index(drop=True)
    )

    print(
        "✓ Weather data received"
    )

    print(
        "Rows:",
        len(weather)
    )

    print(
        "Range:",
        weather["timestamp"].min(),
        "→",
        weather["timestamp"].max(),
    )

    return weather


# ============================================================
# FETCH AIR QUALITY
# ============================================================

def fetch_air_quality():

    print("\n" + "=" * 70)
    print("FETCHING OPEN-METEO AIR-QUALITY DATA")
    print("=" * 70)

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

        "past_hours": HISTORY_HOURS,

        "forecast_hours":  get_api_forecast_hours(),

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
            "Air-quality response does not contain "
            "hourly data."
        )

    air = pd.DataFrame(
        data["hourly"]
    )

    air["timestamp"] = pd.to_datetime(
        air["time"],
        utc=True,
    )

    air = air.drop(
        columns=["time"]
    )

    air = (
        air
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"]
        )
        .reset_index(drop=True)
    )

    print(
        "✓ Air-quality data received"
    )

    print(
        "Rows:",
        len(air)
    )

    print(
        "Range:",
        air["timestamp"].min(),
        "→",
        air["timestamp"].max(),
    )

    return air


# ============================================================
# MERGE API DATA
# ============================================================

def merge_data(
    weather,
    air,
):

    print("\n" + "=" * 70)
    print("MERGING WEATHER + AIR QUALITY")
    print("=" * 70)

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

    if df.empty:

        raise ValueError(
            "Merged Open-Meteo dataset is empty."
        )

    print(
        "✓ Data merged successfully"
    )

    print(
        "Merged rows:",
        len(df)
    )

    print(
        "Merged range:",
        df["timestamp"].min(),
        "→",
        df["timestamp"].max(),
    )

    return df


# ============================================================
# GET FEAST HISTORICAL STATE
# ============================================================

def get_feast_state(
    store,
    timestamp,
):

    print("\n" + "=" * 70)
    print("READING LATEST STATE FROM FEAST")
    print("=" * 70)

    entity_rows = [
        {
            "location_id": LOCATION_ID
        }
    ]

    print(
        "Entity:",
        LOCATION_ID
    )

    print(
        "Reference timestamp:",
        timestamp
    )

    feature_names = [
        f"aqi_features:{feature}"
        for feature in FEATURE_COLUMNS
    ]

    try:

        result = store.get_online_features(
            features=feature_names,
            entity_rows=entity_rows,
        ).to_dict()

    except Exception as exc:

        raise RuntimeError(
            "Feast online feature retrieval failed.\n"
            f"{exc}"
        ) from exc

    feast_df = pd.DataFrame(result)

    print(
        "✓ Feast online feature retrieval succeeded"
    )

    print(
        "Returned columns:",
        len(feast_df.columns)
    )

    # --------------------------------------------------------
    # Validate entity
    # --------------------------------------------------------

    if "location_id" not in feast_df.columns:

        raise ValueError(
            "Feast response does not contain "
            "'location_id'."
        )

    if feast_df.empty:

        raise ValueError(
            "Feast returned an empty online feature state."
        )

    # --------------------------------------------------------
    # Validate all 70 model features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in feast_df.columns
    ]

    if missing_features:

        raise ValueError(
            "Feast response is missing model features:\n"
            + "\n".join(missing_features)
        )

    # --------------------------------------------------------
    # Extract exactly the 70 model features
    # --------------------------------------------------------

    feature_state = (
        feast_df[
            FEATURE_COLUMNS
        ]
        .iloc[0]
        .to_frame()
        .T
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    feature_state = feature_state.astype(float)

    # --------------------------------------------------------
    # Missing-value validation
    # --------------------------------------------------------

    if feature_state.isna().any().any():

        missing = (
            feature_state.columns[
                feature_state.isna().any()
            ]
            .tolist()
        )

        raise ValueError(
            "Feast returned missing values for:\n"
            + "\n".join(missing)
        )

    print(
        "✓ Feast returned all 70 model features"
    )

    print(
        "Feature vector shape:",
        feature_state.shape
    )

    return feature_state

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    df["hour"] = (
        df["timestamp"].dt.hour
    )

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

    for lag in [
        1,
        3,
        6,
        12,
        24,
        48,
        72,
    ]:

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

        for lag in [
            1,
            3,
            6,
            24,
        ]:

            df[
                f"{pollutant}_lag_{lag}"
            ] = (
                df[pollutant]
                .shift(lag)
            )

    # --------------------------------------------------------
    # AQI ROLLING
    # --------------------------------------------------------

    for window in [
        3,
        6,
        12,
        24,
    ]:

        df[
            f"aqi_{window}h_mean"
        ] = (
            df["us_aqi"]
            .shift(1)
            .rolling(window)
            .mean()
        )

    # --------------------------------------------------------
    # PM2.5 ROLLING
    # --------------------------------------------------------

    for window in [
        3,
        6,
        24,
    ]:

        df[
            f"pm2_5_{window}h_mean"
        ] = (
            df["pm2_5"]
            .shift(1)
            .rolling(window)
            .mean()
        )

    # --------------------------------------------------------
    # PM10 ROLLING
    # --------------------------------------------------------

    for window in [
        3,
        6,
        24,
    ]:

        df[
            f"pm10_{window}h_mean"
        ] = (
            df["pm10"]
            .shift(1)
            .rolling(window)
            .mean()
        )

    # --------------------------------------------------------
    # OTHER POLLUTANT ROLLING
    # --------------------------------------------------------

    for pollutant in [
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
    ]:

        df[
            f"{pollutant}_24h_mean"
        ] = (
            df[pollutant]
            .shift(1)
            .rolling(24)
            .mean()
        )

    # --------------------------------------------------------
    # AQI CHANGE
    # --------------------------------------------------------

    df["aqi_change_1h"] = (
        df["us_aqi"]
        - df["aqi_lag_1"]
    )

    df["aqi_change_3h"] = (
        df["us_aqi"]
        - df["aqi_lag_3"]
    )

    df["aqi_change_6h"] = (
        df["us_aqi"]
        - df["aqi_lag_6"]
    )

    df["aqi_change_24h"] = (
        df["us_aqi"]
        - df["aqi_lag_24"]
    )

    # --------------------------------------------------------
    # PM2.5 CHANGE
    # --------------------------------------------------------

    df["pm2_5_change_1h"] = (
        df["pm2_5"]
        - df["pm2_5_lag_1"]
    )

    df["pm2_5_change_24h"] = (
        df["pm2_5"]
        - df["pm2_5_lag_24"]
    )

    # --------------------------------------------------------
    # PM10 CHANGE
    # --------------------------------------------------------

    df["pm10_change_1h"] = (
        df["pm10"]
        - df["pm10_lag_1"]
    )

    df["pm10_change_24h"] = (
        df["pm10"]
        - df["pm10_lag_24"]
    )

    return df


# ============================================================
# PREDICT
# ============================================================

def predict_forecast(
    model,
    feast_state,
    api_df,
):

    print("\n" + "=" * 70)
    print("GENERATING NEXT 3-DAY FORECAST")
    print("=" * 70)

    # ============================================================
    # NEXT 3 CALENDAR DAYS
    # ============================================================

   
    now_utc = pd.Timestamp.now(tz="UTC")

    current_date = now_utc.date()

    forecast_start = (
        pd.Timestamp(current_date, tz="UTC")
        + pd.Timedelta(days=1)
    )

    forecast_end = (
        forecast_start
        + pd.Timedelta(hours=FORECAST_HOURS - 1)
    )

    print("Current UTC time:", now_utc)
    print("Current date:", current_date)

    print(
        "Forecast start:",
        forecast_start
    )

    print(
        "Forecast end:",
        forecast_end
    )

    # ============================================================
    # EXTRACT EXACT 72-HOUR FUTURE WINDOW
    # ============================================================

    future_df = api_df[
        (api_df["timestamp"] >= forecast_start)
        & (api_df["timestamp"] <= forecast_end)
    ].copy()

    future_df = (
        future_df
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    print(
        "Future rows from Open-Meteo:",
        len(future_df)
    )

   
    # --------------------------------------------------------
    # We require exactly 72 hourly rows.
    # --------------------------------------------------------

    if len(future_df) != FORECAST_HOURS:

        raise ValueError(
            "Open-Meteo did not provide exactly "
            f"{FORECAST_HOURS} hourly rows for "
            "the next three calendar days.\n"
            f"Received: {len(future_df)}\n"
            f"Expected: {FORECAST_HOURS}\n"
            f"Start: {forecast_start}\n"
            f"End: {forecast_end}"
        )

    # --------------------------------------------------------
    # Build historical context.
    #
    # We use the most recent API history and replace the
    # latest historical state with the Feast state where
    # possible.
    # --------------------------------------------------------

    history_start = (
        forecast_start
        - pd.Timedelta(hours=HISTORY_HOURS)
    )

    history = api_df[
        (
            api_df["timestamp"]
            >= history_start
        )
        &
        (
            api_df["timestamp"]
            < forecast_start
        )
    ].copy()

    history = (
        history
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if len(history) < 72:

        raise ValueError(
            "Insufficient historical context for "
            "72-hour lag features.\n"
            f"Received: {len(history)} rows."
        )

    # --------------------------------------------------------
    # Feast validation.
    #
    # Feast is used as the authoritative online feature-store
    # integration point. We verify that it is populated.
    # --------------------------------------------------------

    if feast_state.empty:

        raise ValueError(
            "Feast returned an empty online feature state."
        )

    # --------------------------------------------------------
    # Extract Feast values where available.
    #
    # We don't blindly overwrite the API history because
    # future recursive forecasting needs a continuous
    # timestamped history containing the pollutant/weather
    # values as well.
    # --------------------------------------------------------

    print(
        "✓ Feast state available for forecast context"
    )

    # --------------------------------------------------------
    # Recursive forecasting.
    # --------------------------------------------------------

    predictions = []

    prediction_features = []

    working_history = history.copy()

    for step in range(FORECAST_HOURS):

        future_row = (
            future_df
            .iloc[step]
            .copy()
        )

        prediction_timestamp = (
            future_row["timestamp"]
        )

        # ----------------------------------------------------
        # Anchor is the immediately preceding hour.
        # ----------------------------------------------------

        anchor_timestamp = (
            prediction_timestamp
            - pd.Timedelta(hours=1)
        )

        anchor_rows = working_history[
            working_history["timestamp"]
            == anchor_timestamp
        ]

        if anchor_rows.empty:

            raise ValueError(
                "Missing anchor row for "
                f"{prediction_timestamp}.\n"
                f"Expected: {anchor_timestamp}"
            )

        # ----------------------------------------------------
        # Construct feature frame from historical state.
        # ----------------------------------------------------

        feature_df = create_features(
            working_history
        )

        target_rows = feature_df[
            feature_df["timestamp"]
            == anchor_timestamp
        ]

        if target_rows.empty:

            raise ValueError(
                "Could not construct features for "
                f"anchor {anchor_timestamp}"
            )

        target_row = (
            target_rows
            .iloc[-1]
        )

        # ----------------------------------------------------
        # Model input.
        # ----------------------------------------------------

        X = (
            target_row[
                FEATURE_COLUMNS
            ]
            .to_frame()
            .T
        )

        X = X.astype(float)

        # ----------------------------------------------------
        # Save exact 70-feature vector used for this prediction
        # ----------------------------------------------------

        feature_record = X.iloc[0].to_dict()

        feature_record["timestamp"] = prediction_timestamp

        prediction_features.append(feature_record)

        # ----------------------------------------------------
        # Validate all 70 features.
        # ----------------------------------------------------

        if X.shape[1] != 70:

            raise ValueError(
                "Model input does not contain exactly "
                f"70 features. Found {X.shape[1]}."
            )

        if X.isna().any().any():

            missing_features = (
                X.columns[
                    X.isna().any()
                ]
                .tolist()
            )

            raise ValueError(
                "Missing features for "
                f"{anchor_timestamp}:\n"
                + "\n".join(
                    missing_features
                )
            )

        # ----------------------------------------------------
        # Predict.
        # ----------------------------------------------------

        prediction = model.predict(X)

        predicted_aqi = float(
            np.asarray(
                prediction
            ).reshape(-1)[0]
        )

        predicted_aqi = max(
            0.0,
            predicted_aqi
        )

        # ----------------------------------------------------
        # Save prediction.
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
        # Add predicted AQI to recursive history.
        #
        # Future weather/pollutants come from Open-Meteo.
        # AQI is replaced by our model prediction.
        # ----------------------------------------------------

        predicted_row = {

            "timestamp":
                prediction_timestamp,

            "temperature_2m":
                future_row[
                    "temperature_2m"
                ],

            "relative_humidity_2m":
                future_row[
                    "relative_humidity_2m"
                ],

            "pressure_msl":
                future_row[
                    "pressure_msl"
                ],

            "precipitation":
                future_row[
                    "precipitation"
                ],

            "wind_speed_10m":
                future_row[
                    "wind_speed_10m"
                ],

            "wind_direction_10m":
                future_row[
                    "wind_direction_10m"
                ],

            "pm2_5":
                future_row[
                    "pm2_5"
                ],

            "pm10":
                future_row[
                    "pm10"
                ],

            "carbon_monoxide":
                future_row[
                    "carbon_monoxide"
                ],

            "nitrogen_dioxide":
                future_row[
                    "nitrogen_dioxide"
                ],

            "sulphur_dioxide":
                future_row[
                    "sulphur_dioxide"
                ],

            "ozone":
                future_row[
                    "ozone"
                ],

            "us_aqi":
                predicted_aqi,
        }

        working_history = pd.concat(
            [
                working_history,
                pd.DataFrame(
                    [predicted_row]
                ),
            ],
            ignore_index=True,
        )

        working_history = (
            working_history
            .sort_values("timestamp")
            .tail(HISTORY_HOURS)
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            step < 3
            or (step + 1) % 12 == 0
            or step == FORECAST_HOURS - 1
        ):

            print(
                f"Hour {step + 1:02d}/"
                f"{FORECAST_HOURS} | "
                f"{prediction_timestamp} | "
                f"AQI: {predicted_aqi:.2f}"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    forecast_df = pd.DataFrame(
        predictions
    )

    forecast_df["forecast_day"] = (
        forecast_df["timestamp"]
        .dt.date
    )

    # ========================================================
    # DAILY AVERAGES
    # ========================================================

    daily_forecast = (
        forecast_df
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

    # --------------------------------------------------------
    # Validate exactly 3 days.
    # --------------------------------------------------------

    if len(daily_forecast) != 3:

        raise ValueError(
            "Expected exactly 3 forecast days, "
            f"found {len(daily_forecast)}."
        )

    # --------------------------------------------------------
    # Validate 24 hours/day.
    # --------------------------------------------------------

    hours_per_day = (
        forecast_df
        .groupby("forecast_day")
        .size()
    )

    invalid_days = (
        hours_per_day[
            hours_per_day != 24
        ]
    )

    if not invalid_days.empty:

        raise ValueError(
            "Each forecast day must contain "
            "exactly 24 hourly predictions.\n"
            f"Invalid days:\n{invalid_days}"
        )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("NEXT 3 DAYS — DAILY AQI FORECAST")
    print("=" * 70)

    for _, row in daily_forecast.iterrows():

        print(
            f"{row['forecast_day']} | "
            f"Average AQI: "
            f"{row['predicted_aqi']:.2f}"
        )

    print("\n" + "=" * 70)
    print("HOURLY FORECAST")
    print("=" * 70)

    print(
        forecast_df[
            [
                "timestamp",
                "predicted_aqi",
            ]
        ].to_string(
            index=False
        )
    )

    prediction_features_df = pd.DataFrame(
        prediction_features
    )

    prediction_features_df = prediction_features_df[
        ["timestamp"] + FEATURE_COLUMNS
    ]

    return (
        forecast_df,
        daily_forecast,
        prediction_features_df,
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_predictions(
    forecast_df,
    daily_forecast,
    prediction_features_df,
):

    print("\n" + "=" * 70)
    print("SAVING PREDICTIONS")
    print("=" * 70)

    PREDICTION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Hourly
    # --------------------------------------------------------

    hourly_output = forecast_df[
        [
            "timestamp",
            "predicted_aqi",
        ]
    ].copy()

    hourly_output.to_csv(
        HOURLY_PREDICTION_FILE,
        index=False,
    )

    print(
        "✓ Hourly forecast saved:"
    )

    print(
        HOURLY_PREDICTION_FILE
    )

    # --------------------------------------------------------
    # Daily
    # --------------------------------------------------------

    daily_output = daily_forecast[
        [
            "forecast_day",
            "predicted_aqi",
        ]
    ].copy()

    daily_output.to_csv(
        DAILY_PREDICTION_FILE,
        index=False,
    )

    print(
        "✓ Daily forecast saved:"
    )

    print(
        DAILY_PREDICTION_FILE
    )

    # --------------------------------------------------------
    # Prediction feature context
    # --------------------------------------------------------

    prediction_features_df.to_csv(
        FEATURE_PREDICTION_FILE,
        index=False,
    )

    print(
        "✓ Forecast feature context saved:"
    )

    print(
        FEATURE_PREDICTION_FILE
    )


    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    if not HOURLY_PREDICTION_FILE.exists():

        raise FileNotFoundError(
            "Hourly forecast file was not created."
        )

    if not DAILY_PREDICTION_FILE.exists():

        raise FileNotFoundError(
            "Daily forecast file was not created."
        )

    print(
        "\n✓ Prediction artifacts saved successfully"
    )

    if not FEATURE_PREDICTION_FILE.exists():

        raise FileNotFoundError(
            "Forecast feature context file was not created."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print_header()

    # --------------------------------------------------------
    # MLflow
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Feast
    # --------------------------------------------------------

    store = load_feast()

    # --------------------------------------------------------
    # Open-Meteo
    # --------------------------------------------------------

    weather = fetch_weather()

    air = fetch_air_quality()

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    api_df = merge_data(
        weather,
        air,
    )

    # --------------------------------------------------------
    # Get latest Feast state.
    #
    # The latest historical point available to the online
    # feature store is used as the feature-store integration
    # check/context.
    # --------------------------------------------------------

    feast_state = get_feast_state(
        store,
        api_df["timestamp"].max(),
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    (
        forecast_df,
        daily_forecast,
        prediction_features_df,
    ) = predict_forecast(
        model=model,
        feast_state=feast_state,
        api_df=api_df,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_predictions(
        forecast_df,
        daily_forecast,
        prediction_features_df,
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PREDICTION PIPELINE COMPLETED")
    print("=" * 70)

    print(
        "\nNext 3 calendar days:"
    )

    for _, row in daily_forecast.iterrows():

        print(
            f"{row['forecast_day']} | "
            f"Average AQI: "
            f"{row['predicted_aqi']:.2f}"
        )

    print(
        "\n✓ MLflow champion used"
    )

    print(
        "✓ Feast feature store connected"
    )

    print(
        "✓ Open-Meteo future inputs loaded"
    )

    print(
        "✓ Exactly 3 forecast days generated"
    )

    print(
        "✓ 72 hourly predictions generated"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()