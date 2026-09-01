"""
PEARLS AQI PREDICTOR
NEXT 3-DAY FORECAST PIPELINE

Architecture

Open-Meteo
    |
    +-- Weather forecast
    +-- Air-quality forecast
    |
    v
Feast historical context
    |
    +-- 70 model features
    +-- us_aqi recursive state
    |
    v
70 model features
    |
    v
MLflow @champion
    |
    v
Recursive prediction
    |
    +-- bridge to next local midnight
    |
    +-- 72 saved forecast hours
    |
    v
3 daily AQI averages


IMPORTANT CONTRACT

Feast prediction context:
    70 model features + us_aqi = 71 features

MLflow model input:
    exactly 70 model features

Excluded from MLflow:
    target_aqi
    us_aqi

us_aqi:
    retained only as recursive AQI state
"""

from pathlib import Path
import sys

import mlflow
import numpy as np
import pandas as pd
import requests

from feast import FeatureStore


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

FEATURE_REPO = (
    PROJECT_ROOT
    / "feature_repo"
    / "feature_repo"
)

PREDICTION_FEATURE_SERVICE = "aqi_prediction_context"

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
# MLFLOW
# ============================================================

MLFLOW_MODEL_NAME = "Pearls_AQI_XGBoost"
MLFLOW_MODEL_ALIAS = "champion"

MLFLOW_TRACKING_URI = (
    "http://127.0.0.1:5000"
)

MLFLOW_MODEL_URI = (
    f"models:/{MLFLOW_MODEL_NAME}"
    f"@{MLFLOW_MODEL_ALIAS}"
)

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

mlflow.set_registry_uri(
    MLFLOW_TRACKING_URI
)


# ============================================================
# FEAST
# ============================================================

PREDICTION_FEATURE_SERVICE = (
    "aqi_prediction_context"
)


# ============================================================
# LOCATION
# ============================================================

LOCATION_ID = "peshawar"
TIMEZONE = "Asia/Karachi"

LATITUDE = 34.008
LONGITUDE = 71.5785

LOCAL_TIMEZONE = "Asia/Karachi"


# ============================================================
# OPEN-METEO
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

FORECAST_HOURS = (
    FORECAST_DAYS * 24
)


# ============================================================
# 70 MODEL FEATURES
# ============================================================

FEATURE_COLUMNS = [

    # --------------------------------------------------------
    # Current weather
    # --------------------------------------------------------

    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",

    # --------------------------------------------------------
    # Current pollutants
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
    # CO lags
    # --------------------------------------------------------

    "carbon_monoxide_lag_1",
    "carbon_monoxide_lag_3",
    "carbon_monoxide_lag_6",
    "carbon_monoxide_lag_24",

    # --------------------------------------------------------
    # NO2 lags
    # --------------------------------------------------------

    "nitrogen_dioxide_lag_1",
    "nitrogen_dioxide_lag_3",
    "nitrogen_dioxide_lag_6",
    "nitrogen_dioxide_lag_24",

    # --------------------------------------------------------
    # SO2 lags
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
# CONTRACT ASSERTIONS
# ============================================================

assert len(FEATURE_COLUMNS) == 70

assert len(set(FEATURE_COLUMNS)) == 70

assert "target_aqi" not in FEATURE_COLUMNS

assert "us_aqi" not in FEATURE_COLUMNS


# Feast prediction context contains:
#   70 model features
#   + us_aqi recursive state
FEAST_CONTEXT_FEATURES = (
    FEATURE_COLUMNS
    + ["us_aqi"]
)

assert len(FEAST_CONTEXT_FEATURES) == 71


# ============================================================
# GENERIC TIMESTAMP HELPERS
# ============================================================

def utc_now_hour():
    """
    Return current UTC time rounded down to the hour.
    """

    return (
        pd.Timestamp.now(
            tz="UTC"
        ).floor("h")
    )


def next_local_midnight_utc():
    """
    Calculate the next midnight in Asia/Karachi
    and return it as a UTC timestamp.
    """

    now_local = (
        pd.Timestamp.now(
            tz=LOCAL_TIMEZONE
        )
    )

    next_midnight_local = (
        now_local
        .normalize()
        + pd.Timedelta(days=1)
    )

    return (
        next_midnight_local
        .tz_convert("UTC")
    )


# ============================================================
# HEADER
# ============================================================

def print_header():

    print()
    print("=" * 70)
    print("PEARLS AQI PREDICTOR")
    print("NEXT 3-DAY AQI FORECAST")
    print("=" * 70)

    print(
        "Project root:",
        PROJECT_ROOT,
    )

    print(
        "Python:",
        sys.executable,
    )

    print(
        "MLflow model:",
        MLFLOW_MODEL_NAME,
    )

    print(
        "MLflow alias:",
        MLFLOW_MODEL_ALIAS,
    )

    print(
        "MLflow URI:",
        MLFLOW_MODEL_URI,
    )

    print(
        "Feast repository:",
        FEATURE_REPO,
    )

    print(
        "Feast service:",
        PREDICTION_FEATURE_SERVICE,
    )

    print(
        "Model features:",
        len(FEATURE_COLUMNS),
    )

    print(
        "Feast context features:",
        len(FEAST_CONTEXT_FEATURES),
    )

    print(
        "Timezone:",
        LOCAL_TIMEZONE,
    )


# ============================================================
# VALIDATE MLFLOW MODEL
# ============================================================

def load_model():

    print()
    print("=" * 70)
    print("LOADING MLFLOW CHAMPION")
    print("=" * 70)

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_registry_uri(
        MLFLOW_TRACKING_URI
    )

    print(
        "Tracking URI:",
        mlflow.get_tracking_uri(),
    )

    print(
        "Registry URI:",
        mlflow.get_registry_uri(),
    )

    print(
        "Model:",
        MLFLOW_MODEL_NAME,
    )

    print(
        "Alias:",
        MLFLOW_MODEL_ALIAS,
    )

    from mlflow import MlflowClient

    client = MlflowClient()

    try:

        champion = (
            client
            .get_model_version_by_alias(
                MLFLOW_MODEL_NAME,
                MLFLOW_MODEL_ALIAS,
            )
        )

    except Exception as exc:

        raise RuntimeError(
            "\nMLflow champion alias was not found.\n"
            f"Model: {MLFLOW_MODEL_NAME}\n"
            f"Alias: {MLFLOW_MODEL_ALIAS}\n"
            f"Tracking URI: {MLFLOW_TRACKING_URI}\n\n"
            "Start MLflow and make sure the model "
            "has the champion alias."
        ) from exc

    print(
        "Champion version:",
        champion.version,
    )

    model = (
        mlflow.pyfunc.load_model(
            MLFLOW_MODEL_URI
        )
    )

    print(
        "✓ MLflow champion model loaded"
    )

    print(
        "Model URI:",
        MLFLOW_MODEL_URI,
    )

    print(
        "Model type:",
        type(model),
    )

    return model


# ============================================================
# FEAST FEATURE CONTRACT
# ============================================================
 
def validate_prediction_feature_contract(store):
    print()
    print("=" * 70)
    print("FEAST PREDICTION FEATURE CONTRACT")
    print("=" * 70)

    try:
        service = store.get_feature_service(
            "aqi_prediction_context"
        )
    except Exception as exc:
        raise RuntimeError(
            "\nFeast feature service was not found.\n"
            "Expected service: aqi_prediction_context\n\n"
            "Run:\n"
            "  cd feature_repo\\feature_repo\n"
            "  feast apply\n"
        ) from exc

    feast_features = [
        feature.name
        for projection in service.feature_view_projections
        for feature in projection.features
    ]

    expected_model_features = list(FEATURE_COLUMNS)
    expected_context_features = (
        ["us_aqi"] + expected_model_features
    )

    expected = set(expected_context_features)
    actual = set(feast_features)

    print(
        "Expected context features:",
        len(expected),
    )

    print(
        "Feast context features:",
        len(actual),
    )

    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    duplicates = sorted(
        {
            feature
            for feature in feast_features
            if feast_features.count(feature) > 1
        }
    )

    if duplicates:
        raise ValueError(
            "Duplicate Feast prediction-context features:\n"
            + "\n".join(duplicates)
        )

    # --------------------------------------------------------
    # MISSING
    # --------------------------------------------------------

    missing = sorted(
        expected - actual
    )

    if missing:
        raise ValueError(
            "Missing Feast prediction-context features:\n"
            + "\n".join(missing)
        )

    # --------------------------------------------------------
    # EXTRA
    # --------------------------------------------------------

    extra = sorted(
        actual - expected
    )

    if extra:
        raise ValueError(
            "Unexpected Feast prediction-context features:\n"
            + "\n".join(extra)
        )

    # --------------------------------------------------------
    # TARGET EXCLUSION
    # --------------------------------------------------------

    if "target_aqi" in actual:
        raise ValueError(
            "target_aqi MUST NOT be in prediction context."
        )

    # --------------------------------------------------------
    # us_aqi STATE
    # --------------------------------------------------------

    if "us_aqi" not in actual:
        raise ValueError(
            "us_aqi MUST be available in prediction context "
            "as recursive AQI state."
        )

    # --------------------------------------------------------
    # MODEL FEATURES
    # --------------------------------------------------------

    model_actual = [
        feature
        for feature in feast_features
        if feature != "us_aqi"
    ]

    if model_actual != FEATURE_COLUMNS:
        raise ValueError(
            "Feast model feature ordering does not match "
            "FEATURE_COLUMNS."
        )

    print("✓ 70 model features")
    print("✓ us_aqi available as recursive state")
    print("✓ 71 total prediction-context features")
    print("✓ No duplicates")
    print("✓ Exact feature match")
    print("✓ target_aqi excluded")
    print("✓ PREDICTION FEATURE CONTRACT: PASS")

# ============================================================
# LOAD FEAST
# ============================================================

def load_feast():

    print()
    print("=" * 70)
    print("LOADING FEAST")
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
        "Feast project:",
        store.project,
    )

    print(
        "Feast repository:",
        FEATURE_REPO,
    )

    validate_prediction_feature_contract(
        store
    )

    print(
        "✓ Feast FeatureStore loaded"
    )

    return store


# ============================================================
# OPEN-METEO REQUEST HELPER
# ============================================================

def request_open_meteo(
    url,
    params,
    name,
):

    try:

        response = requests.get(
            url,
            params=params,
            timeout=30,
        )

        response.raise_for_status()

    except requests.RequestException as exc:

        raise RuntimeError(
            f"{name} request failed.\n"
            f"URL: {url}\n"
            f"Error: {exc}"
        ) from exc

    try:

        data = response.json()

    except ValueError as exc:

        raise RuntimeError(
            f"{name} returned invalid JSON."
        ) from exc

    if "hourly" not in data:

        raise RuntimeError(
            f"{name} response does not contain "
            "'hourly' data."
        )

    return data


# ============================================================
# FETCH WEATHER
# ============================================================

def fetch_weather(
    forecast_end_utc
):

    print()
    print("=" * 70)
    print("FETCHING OPEN-METEO WEATHER")
    print("=" * 70)

    current_hour = (
        utc_now_hour()
    )

    required_forecast_hours = (
        int(
            (
                forecast_end_utc
                - current_hour
            ).total_seconds()
            / 3600
        )
        + 1
    )

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

        "forecast_hours":
            required_forecast_hours,

        "timezone": "UTC",

        "wind_speed_unit": "kmh",

        "temperature_unit": "celsius",

        "precipitation_unit": "mm",
    }

    data = request_open_meteo(
        WEATHER_URL,
        params,
        "Open-Meteo weather",
    )

    df = pd.DataFrame(
        data["hourly"]
    )

    if "time" not in df.columns:

        raise ValueError(
            "Weather response has no time column."
        )

    df["timestamp"] = (
        pd.to_datetime(
            df["time"],
            utc=True,
        )
        .dt.floor("h")
    )

    df = df.drop(
        columns=["time"]
    )

    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates(
            "timestamp"
        )
        .reset_index(drop=True)
    )

    required_columns = [
        "timestamp",
        "temperature_2m",
        "relative_humidity_2m",
        "pressure_msl",
        "precipitation",
        "wind_speed_10m",
        "wind_direction_10m",
    ]

    missing = sorted(
        set(required_columns)
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Weather data is missing:\n"
            + "\n".join(missing)
        )

    print(
        "Weather rows:",
        len(df),
    )

    print(
        "Weather range:",
        df["timestamp"].min(),
        "->",
        df["timestamp"].max(),
    )

    print(
        "✓ Weather forecast loaded"
    )

    return df


# ============================================================
# FETCH AIR QUALITY
# ============================================================

def fetch_air_quality(
    forecast_end_utc
):

    print()
    print("=" * 70)
    print("FETCHING OPEN-METEO AIR QUALITY")
    print("=" * 70)

    current_hour = (
        utc_now_hour()
    )

    required_forecast_hours = (
        int(
            (
                forecast_end_utc
                - current_hour
            ).total_seconds()
            / 3600
        )
        + 1
    )

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

        "forecast_hours":
            required_forecast_hours,

        "timezone": "UTC",
    }

    data = request_open_meteo(
        AIR_QUALITY_URL,
        params,
        "Open-Meteo air quality",
    )

    df = pd.DataFrame(
        data["hourly"]
    )

    if "time" not in df.columns:

        raise ValueError(
            "Air-quality response has no time column."
        )

    df["timestamp"] = (
        pd.to_datetime(
            df["time"],
            utc=True,
        )
        .dt.floor("h")
    )

    df = df.drop(
        columns=["time"]
    )

    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates(
            "timestamp"
        )
        .reset_index(drop=True)
    )

    required_columns = [
        "timestamp",
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
        "us_aqi",
    ]

    missing = sorted(
        set(required_columns)
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Air-quality data is missing:\n"
            + "\n".join(missing)
        )

    print(
        "Air-quality rows:",
        len(df),
    )

    print(
        "Air-quality range:",
        df["timestamp"].min(),
        "->",
        df["timestamp"].max(),
    )

    print(
        "✓ Air-quality forecast loaded"
    )

    return df


# ============================================================
# MERGE OPEN-METEO
# ============================================================

def merge_api_data(
    weather,
    air,
):

    print()
    print("=" * 70)
    print("MERGING OPEN-METEO DATA")
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
            "timestamp"
        )
        .reset_index(drop=True)
    )

    if df.empty:

        raise ValueError(
            "Open-Meteo merged dataset is empty."
        )

    required_columns = [
        "timestamp",
        "temperature_2m",
        "relative_humidity_2m",
        "pressure_msl",
        "precipitation",
        "wind_speed_10m",
        "wind_direction_10m",
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
        "us_aqi",
    ]

    missing = sorted(
        set(required_columns)
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Merged Open-Meteo data is missing:\n"
            + "\n".join(missing)
        )

    print(
        "Merged rows:",
        len(df),
    )

    print(
        "Merged range:",
        df["timestamp"].min(),
        "->",
        df["timestamp"].max(),
    )

    print(
        "✓ Open-Meteo merge complete"
    )

    return df


# ============================================================
# FEAST HISTORICAL CONTEXT
# ============================================================

def get_historical_context(
    store,
    latest_timestamp,
):

    print()
    print("=" * 70)
    print("LOADING HISTORICAL CONTEXT FROM FEAST")
    print("=" * 70)

    latest_timestamp = (
        pd.to_datetime(
            latest_timestamp,
            utc=True,
        )
        .floor("h")
    )

    history_start = (
        latest_timestamp
        - pd.Timedelta(
            hours=HISTORY_HOURS - 1
        )
    )

    timestamps = pd.date_range(
        start=history_start,
        end=latest_timestamp,
        freq="h",
        tz="UTC",
    )

    if len(timestamps) != HISTORY_HOURS:

        raise ValueError(
            "Incorrect Feast timestamp count.\n"
            f"Expected: {HISTORY_HOURS}\n"
            f"Received: {len(timestamps)}"
        )

    entity_df = pd.DataFrame(
        {
            "location_id": LOCATION_ID,
            "event_timestamp": timestamps,
        }
    )

    print(
        "Feast entity range:",
        history_start,
        "->",
        latest_timestamp,
    )

    print(
        "Requested rows:",
        len(entity_df),
    )

    service = (
        store.get_feature_service(
            PREDICTION_FEATURE_SERVICE
        )
    )

    historical = (
        store
        .get_historical_features(
            entity_df=entity_df,
            features=service,
        )
        .to_df()
    )

    if historical.empty:

        raise ValueError(
            "Feast historical retrieval returned "
            "an empty dataframe."
        )

    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    if "event_timestamp" not in historical.columns:

        raise ValueError(
            "Feast result does not contain "
            "event_timestamp."
        )

    historical["event_timestamp"] = (
        pd.to_datetime(
            historical["event_timestamp"],
            utc=True,
        )
        .dt.floor("h")
    )

    historical = (
        historical
        .sort_values(
            "event_timestamp"
        )
        .drop_duplicates(
            subset=[
                "location_id",
                "event_timestamp",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # ROW COUNT
    # --------------------------------------------------------

    if len(historical) != HISTORY_HOURS:

        raise ValueError(
            "Feast historical retrieval returned "
            f"{len(historical)} rows; "
            f"expected {HISTORY_HOURS}."
        )

    # --------------------------------------------------------
    # TARGET MUST NOT EXIST
    # --------------------------------------------------------

    if "target_aqi" in historical.columns:

        raise ValueError(
            "target_aqi leaked into prediction context."
        )

    # --------------------------------------------------------
    # US AQI MUST EXIST
    # --------------------------------------------------------

    if "us_aqi" not in historical.columns:

        raise ValueError(
            "Feast prediction context does not "
            "contain us_aqi."
        )

    # --------------------------------------------------------
    # REQUIRED COLUMNS
    # --------------------------------------------------------

    expected_columns = [
        "location_id",
        "event_timestamp",
        *FEATURE_COLUMNS,
        "us_aqi",
    ]

    missing = sorted(
        set(expected_columns)
        - set(historical.columns)
    )

    if missing:

        raise ValueError(
            "Feast historical context is missing:\n"
            + "\n".join(missing)
        )

    # --------------------------------------------------------
    # KEEP ONLY CONTEXT
    # --------------------------------------------------------

    historical = historical[
        expected_columns
    ].copy()

    # --------------------------------------------------------
    # NUMERIC
    # --------------------------------------------------------

    numeric_columns = (
        FEATURE_COLUMNS
        + ["us_aqi"]
    )

    historical[numeric_columns] = (
        historical[numeric_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    # --------------------------------------------------------
    # NULLS
    # --------------------------------------------------------

    null_columns = (
        historical[
            numeric_columns
        ]
        .columns[
            historical[
                numeric_columns
            ]
            .isna()
            .any()
        ]
        .tolist()
    )

    if null_columns:

        raise ValueError(
            "Feast historical context contains "
            "missing values:\n"
            + "\n".join(null_columns)
        )

    # --------------------------------------------------------
    # CONTINUITY
    # --------------------------------------------------------

    deltas = (
        historical[
            "event_timestamp"
        ]
        .diff()
        .dropna()
    )

    if not deltas.eq(
        pd.Timedelta(hours=1)
    ).all():

        raise ValueError(
            "Feast historical context contains "
            "non-hourly gaps."
        )

    # --------------------------------------------------------
    # RANGE
    # --------------------------------------------------------

    if (
        historical[
            "event_timestamp"
        ].min()
        != history_start
    ):

        raise ValueError(
            "Feast historical range does not "
            "start at expected timestamp.\n"
            f"Expected: {history_start}\n"
            f"Received: "
            f"{historical['event_timestamp'].min()}"
        )

    if (
        historical[
            "event_timestamp"
        ].max()
        != latest_timestamp
    ):

        raise ValueError(
            "Feast historical range does not "
            "end at expected timestamp.\n"
            f"Expected: {latest_timestamp}\n"
            f"Received: "
            f"{historical['event_timestamp'].max()}"
        )

    print(
        "Historical rows:",
        len(historical),
    )

    print(
        "Historical range:",
        historical[
            "event_timestamp"
        ].min(),
        "->",
        historical[
            "event_timestamp"
        ].max(),
    )

    print("✓ 96 historical Feast rows")
    print("✓ 70 model features")
    print("✓ us_aqi retained as recursive state")
    print("✓ target_aqi excluded")
    print("✓ No Parquet dependency")
    print("✓ No null values")
    print("✓ Hourly continuity verified")
    print("✓ FEAST HISTORICAL CONTEXT: PASS")

    return historical


# ============================================================
# CREATE MODEL FEATURES
# ============================================================

def create_features(df):

    df = df.copy()

    df["timestamp"] = (
        pd.to_datetime(
            df["timestamp"],
            utc=True,
        )
        .dt.floor("h")
    )

    df = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # TIME FEATURES
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

        df[
            f"aqi_lag_{lag}"
        ] = (
            df["us_aqi"]
            .shift(lag)
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
    # AQI ROLLING MEANS
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
            .rolling(
                window=window,
                min_periods=window,
            )
            .mean()
        )

    # --------------------------------------------------------
    # PM2.5 ROLLING MEANS
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
            .rolling(
                window=window,
                min_periods=window,
            )
            .mean()
        )

    # --------------------------------------------------------
    # PM10 ROLLING MEANS
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
            .rolling(
                window=window,
                min_periods=window,
            )
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

        df[
            f"{pollutant}_24h_mean"
        ] = (
            df[pollutant]
            .shift(1)
            .rolling(
                window=24,
                min_periods=24,
            )
            .mean()
        )

    # --------------------------------------------------------
    # AQI CHANGES
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
    # PM2.5 CHANGES
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
    # PM10 CHANGES
    # --------------------------------------------------------

    df["pm10_change_1h"] = (
        df["pm10"]
        - df["pm10_lag_1"]
    )

    df["pm10_change_24h"] = (
        df["pm10"]
        - df["pm10_lag_24"]
    )

    # --------------------------------------------------------
    # FINAL FEATURE CONTRACT
    # --------------------------------------------------------

    missing = sorted(
        set(FEATURE_COLUMNS)
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Feature engineering failed. "
            "Missing features:\n"
            + "\n".join(missing)
        )

    return df


# ============================================================
# VALIDATE MODEL INPUT
# ============================================================

def validate_prediction_features(
    X
):

    if not isinstance(
        X,
        pd.DataFrame,
    ):

        X = pd.DataFrame(X)

    missing = sorted(
        set(FEATURE_COLUMNS)
        - set(X.columns)
    )

    if missing:

        raise ValueError(
            "Prediction input is missing "
            "features:\n"
            + "\n".join(missing)
        )

    extra = sorted(
        set(X.columns)
        - set(FEATURE_COLUMNS)
    )

    if extra:

        raise ValueError(
            "Prediction input contains "
            "unexpected features:\n"
            + "\n".join(extra)
        )

    X = X[
        FEATURE_COLUMNS
    ].copy()

    if X.shape[1] != 70:

        raise ValueError(
            "Prediction requires exactly "
            f"70 features; received "
            f"{X.shape[1]}."
        )

    if "target_aqi" in X.columns:

        raise ValueError(
            "target_aqi leaked into "
            "MLflow prediction input."
        )

    if "us_aqi" in X.columns:

        raise ValueError(
            "us_aqi leaked into "
            "MLflow prediction input."
        )

    X = X.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if X.isna().any().any():

        missing_values = (
            X.columns[
                X.isna().any()
            ]
            .tolist()
        )

        raise ValueError(
            "Prediction input contains "
            "NaN values:\n"
            + "\n".join(missing_values)
        )

    return X


# ============================================================
# SINGLE MODEL PREDICTION
# ============================================================

def predict_one(
    model,
    working_history,
    prediction_timestamp,
):

    prediction_timestamp = (
        pd.to_datetime(
            prediction_timestamp,
            utc=True,
        )
        .floor("h")
    )

    anchor_timestamp = (
        prediction_timestamp
        - pd.Timedelta(hours=1)
    )

    feature_frame = (
        create_features(
            working_history
        )
    )

    anchor_rows = feature_frame[
        feature_frame["timestamp"]
        == anchor_timestamp
    ]

    if anchor_rows.empty:

        raise ValueError(
            "Missing feature anchor.\n"
            f"Prediction timestamp: "
            f"{prediction_timestamp}\n"
            f"Required anchor: "
            f"{anchor_timestamp}"
        )

    row = (
        anchor_rows
        .iloc[-1]
    )

    X = (
        row[
            FEATURE_COLUMNS
        ]
        .to_frame()
        .T
    )

    X = validate_prediction_features(
        X
    )

    if X.shape != (1, 70):

        raise ValueError(
            "Invalid model input shape.\n"
            f"Expected: (1, 70)\n"
            f"Received: {X.shape}"
        )

    # ============================================================
    # MLflow MODEL INPUT DTYPE NORMALIZATION
    # ============================================================

    # MLflow champion signature expects these temporal features
    # as 32-bit integers, not pandas/numpy int64.
    INT32_FEATURES = [
        "hour",
        "day_of_week",
        "day_of_month",
        "month",
    ]

    for col in INT32_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("int32")

    if "is_weekend" in X.columns:
        X["is_weekend"] = X["is_weekend"].astype("int64")

    prediction = (
        model.predict(X)
    )

    predicted_aqi = float(
        np.asarray(
            prediction
        )
        .reshape(-1)[0]
    )

    if not np.isfinite(
        predicted_aqi
    ):

        raise ValueError(
            "MLflow model returned "
            "a non-finite AQI prediction."
        )

    predicted_aqi = max(
        0.0,
        predicted_aqi,
    )

    return (
        predicted_aqi,
        X,
        anchor_timestamp,
    )


# ============================================================
# BUILD RECURSIVE ROW
# ============================================================

def build_recursive_row(
    future_row,
    predicted_aqi,
):

    return {

        "timestamp":
            future_row[
                "timestamp"
            ],

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

        # IMPORTANT:
        # Future AQI comes from our model,
        # not Open-Meteo us_aqi.
        "us_aqi":
            predicted_aqi,
    }


# ============================================================
# VALIDATE API HISTORY
# ============================================================

def validate_api_history(
    api_history,
    expected_start,
    expected_end,
):

    if len(api_history) != HISTORY_HOURS:

        raise ValueError(
            "API historical context must contain "
            f"exactly {HISTORY_HOURS} rows.\n"
            f"Received: {len(api_history)}"
        )

    timestamps = (
        api_history[
            "timestamp"
        ]
        .sort_values()
        .reset_index(drop=True)
    )

    if timestamps.iloc[0] != expected_start:

        raise ValueError(
            "API history start mismatch.\n"
            f"Expected: {expected_start}\n"
            f"Received: {timestamps.iloc[0]}"
        )

    if timestamps.iloc[-1] != expected_end:

        raise ValueError(
            "API history end mismatch.\n"
            f"Expected: {expected_end}\n"
            f"Received: {timestamps.iloc[-1]}"
        )

    deltas = (
        timestamps
        .diff()
        .dropna()
    )

    if not deltas.eq(
        pd.Timedelta(hours=1)
    ).all():

        raise ValueError(
            "API history contains "
            "non-hourly gaps."
        )

    if api_history[
        "us_aqi"
    ].isna().any():

        raise ValueError(
            "API/Feast historical context "
            "contains missing us_aqi."
        )


# ============================================================
# VALIDATE FUTURE API DATA
# ============================================================

def validate_future_api(
    api_df,
    future_start,
    future_end,
):

    future = (
        api_df[
            (
                api_df["timestamp"]
                >= future_start
            )
            &
            (
                api_df["timestamp"]
                <= future_end
            )
        ]
        .copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    expected_hours = (
        int(
            (
                future_end
                - future_start
            ).total_seconds()
            / 3600
        )
        + 1
    )

    if len(future) != expected_hours:

        raise ValueError(
            "Open-Meteo future context is incomplete.\n"
            f"Expected: {expected_hours} rows\n"
            f"Received: {len(future)}\n"
            f"Range: {future_start} -> {future_end}"
        )

    deltas = (
        future[
            "timestamp"
        ]
        .diff()
        .dropna()
    )

    if not deltas.eq(
        pd.Timedelta(hours=1)
    ).all():

        raise ValueError(
            "Open-Meteo future context contains "
            "non-hourly gaps."
        )

    return future


# ============================================================
# FORECAST
# ============================================================

def generate_forecast(
    model,
    feast_history,
    api_df,
    forecast_start,
    forecast_end,
):

    print()
    print("=" * 70)
    print("GENERATING RECURSIVE FORECAST")
    print("=" * 70)

    forecast_start = (
        pd.to_datetime(
            forecast_start,
            utc=True,
        )
        .floor("h")
    )

    forecast_end = (
        pd.to_datetime(
            forecast_end,
            utc=True,
        )
        .floor("h")
    )

    current_hour = (
        utc_now_hour()
    )

    # --------------------------------------------------------
    # Feast history must end at current completed hour.
    # --------------------------------------------------------

    expected_history_end = (
        current_hour
    )

    expected_history_start = (
        expected_history_end
        - pd.Timedelta(
            hours=HISTORY_HOURS - 1
        )
    )

    feast_history = feast_history.copy()

    feast_history[
        "event_timestamp"
    ] = (
        pd.to_datetime(
            feast_history[
                "event_timestamp"
            ],
            utc=True,
        )
        .dt.floor("h")
    )

    feast_history = (
        feast_history
        .sort_values(
            "event_timestamp"
        )
        .reset_index(drop=True)
    )

    if (
        feast_history[
            "event_timestamp"
        ].min()
        != expected_history_start
    ):

        raise ValueError(
            "Feast history does not start "
            "at expected timestamp.\n"
            f"Expected: {expected_history_start}\n"
            f"Received: "
            f"{feast_history['event_timestamp'].min()}"
        )

    if (
        feast_history[
            "event_timestamp"
        ].max()
        != expected_history_end
    ):

        raise ValueError(
            "Feast history does not end "
            "at current hour.\n"
            f"Expected: {expected_history_end}\n"
            f"Received: "
            f"{feast_history['event_timestamp'].max()}"
        )

    # --------------------------------------------------------
    # Build API history.
    #
    # Weather + pollutant state:
    #     Open-Meteo
    #
    # Historical AQI state:
    #     Feast us_aqi
    # --------------------------------------------------------

    api_history = (
        api_df[
            (
                api_df["timestamp"]
                >= expected_history_start
            )
            &
            (
                api_df["timestamp"]
                <= expected_history_end
            )
        ]
        .copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    validate_api_history(
        api_history=api_history,
        expected_start=expected_history_start,
        expected_end=expected_history_end,
    )

    feast_aqi = feast_history[
        [
            "event_timestamp",
            "us_aqi",
        ]
    ].copy()

    feast_aqi = (
        feast_aqi
        .rename(
            columns={
                "event_timestamp":
                    "timestamp"
            }
        )
    )

    if feast_aqi[
        "us_aqi"
    ].isna().any():

        raise ValueError(
            "Feast contains missing historical us_aqi."
        )

    # --------------------------------------------------------
    # Remove Open-Meteo historical AQI.
    # Feast is authoritative for historical AQI.
    # --------------------------------------------------------

    api_history = (
        api_history
        .drop(
            columns=[
                "us_aqi"
            ],
            errors="ignore",
        )
    )

    history = pd.merge(
        api_history,
        feast_aqi,
        on="timestamp",
        how="inner",
    )

    history = (
        history
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if len(history) != HISTORY_HOURS:

        raise ValueError(
            "API + Feast historical alignment failed.\n"
            f"Expected: {HISTORY_HOURS}\n"
            f"Received: {len(history)}"
        )

    if history[
        "us_aqi"
    ].isna().any():

        raise ValueError(
            "Historical prediction state contains "
            "missing us_aqi."
        )

    # --------------------------------------------------------
    # FUTURE API DATA
    #
    # Includes:
    #     bridge hours
    #     + 72 forecast hours
    # --------------------------------------------------------

    future_start = (
        current_hour
        + pd.Timedelta(hours=1)
    )

    future_api = validate_future_api(
        api_df=api_df,
        future_start=future_start,
        future_end=forecast_end,
    )

    # --------------------------------------------------------
    # Bridge hours
    #
    # Current time -> next local midnight.
    #
    # These predictions are NOT saved in the final
    # 72-hour forecast. They only establish recursive
    # state for Day 1.
    # --------------------------------------------------------

    bridge_mask = (
        future_api["timestamp"]
        < forecast_start
    )

    bridge_df = (
        future_api[
            bridge_mask
        ]
        .copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    forecast_df_api = (
        future_api[
            (
                future_api["timestamp"]
                >= forecast_start
            )
            &
            (
                future_api["timestamp"]
                <= forecast_end
            )
        ]
        .copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if len(forecast_df_api) != FORECAST_HOURS:

        raise ValueError(
            "Final forecast API window must "
            f"contain exactly {FORECAST_HOURS} rows.\n"
            f"Received: {len(forecast_df_api)}"
        )

    print(
        "Current UTC hour:",
        current_hour,
    )

    print(
        "Forecast start UTC:",
        forecast_start,
    )

    print(
        "Forecast end UTC:",
        forecast_end,
    )

    print(
        "Bridge hours:",
        len(bridge_df),
    )

    print(
        "Saved forecast hours:",
        len(forecast_df_api),
    )

    # --------------------------------------------------------
    # RECURSIVE STATE
    # --------------------------------------------------------

    working_history = (
        history.copy()
    )

    predictions = []

    feature_records = []

    # ========================================================
    # BRIDGE PREDICTIONS
    # ========================================================

    if not bridge_df.empty:

        print()
        print(
            "Generating bridge predictions:"
        )

        for _, future_row in (
            bridge_df.iterrows()
        ):

            prediction_timestamp = (
                future_row["timestamp"]
            )

            (
                predicted_aqi,
                _,
                _,
            ) = predict_one(
                model=model,
                working_history=working_history,
                prediction_timestamp=
                    prediction_timestamp,
            )

            working_row = (
                build_recursive_row(
                    future_row=future_row,
                    predicted_aqi=
                        predicted_aqi,
                )
            )

            working_history = pd.concat(
                [
                    working_history,
                    pd.DataFrame(
                        [working_row]
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

        print(
            "✓ Bridge predictions:",
            len(bridge_df),
        )

    else:

        print(
            "✓ No bridge hours required."
        )

    # ========================================================
    # FINAL 72-HOUR FORECAST
    # ========================================================

    print()
    print(
        "Generating 72 forecast predictions:"
    )

    for step, (_, future_row) in enumerate(
        forecast_df_api.iterrows(),
        start=1,
    ):

        prediction_timestamp = (
            future_row["timestamp"]
        )

        (
            predicted_aqi,
            X,
            anchor_timestamp,
        ) = predict_one(
            model=model,
            working_history=working_history,
            prediction_timestamp=
                prediction_timestamp,
        )

        # ----------------------------------------------------
        # Save forecast prediction
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
        # Save exact 70-feature model vector
        # ----------------------------------------------------

        feature_record = {
            column:
                float(
                    X.iloc[0][column]
                )
            for column in FEATURE_COLUMNS
        }

        feature_record[
            "timestamp"
        ] = prediction_timestamp

        feature_records.append(
            feature_record
        )

        # ----------------------------------------------------
        # Recursive state
        #
        # Future AQI is now the model prediction.
        # Open-Meteo us_aqi is NOT used here.
        # ----------------------------------------------------

        recursive_row = (
            build_recursive_row(
                future_row=future_row,
                predicted_aqi=
                    predicted_aqi,
            )
        )

        working_history = pd.concat(
            [
                working_history,
                pd.DataFrame(
                    [recursive_row]
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

        if (
            step == 1
            or step % 12 == 0
            or step == FORECAST_HOURS
        ):

            print(
                f"  {step:02d}/{FORECAST_HOURS} "
                f"| {prediction_timestamp} "
                f"| AQI = "
                f"{predicted_aqi:.2f}"
            )

    # ========================================================
    # HOURLY FORECAST DATAFRAME
    # ========================================================

    hourly_forecast = pd.DataFrame(
        predictions
    )

    if len(hourly_forecast) != 72:

        raise ValueError(
            "Final hourly forecast must "
            "contain exactly 72 predictions."
        )

    hourly_forecast[
        "timestamp"
    ] = pd.to_datetime(
        hourly_forecast[
            "timestamp"
        ],
        utc=True,
    )

    hourly_forecast[
        "forecast_day"
    ] = (
        hourly_forecast[
            "timestamp"
        ]
        .dt.tz_convert(
            LOCAL_TIMEZONE
        )
        .dt.strftime("%Y-%m-%d")
    )

    # ========================================================
    # DAILY AVERAGES
    # ========================================================

    daily_forecast = (
        hourly_forecast
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
    # EXACTLY THREE DAYS
    # --------------------------------------------------------

    if len(daily_forecast) != 3:

        raise ValueError(
            "Expected exactly 3 daily forecasts.\n"
            f"Received: {len(daily_forecast)}\n"
            f"Days: "
            f"{daily_forecast['forecast_day'].tolist()}"
        )

    # --------------------------------------------------------
    # EXACTLY 24 HOURS PER DAY
    # --------------------------------------------------------

    daily_counts = (
        hourly_forecast
        .groupby(
            "forecast_day"
        )
        .size()
    )

    if not (
        daily_counts == 24
    ).all():

        raise ValueError(
            "Each forecast day must contain "
            "exactly 24 predictions.\n"
            f"Counts:\n{daily_counts}"
        )

    # ========================================================
    # FEATURE OUTPUT
    # ========================================================

    feature_df = pd.DataFrame(
        feature_records
    )

    feature_df = feature_df[
        [
            "timestamp"
        ]
        + FEATURE_COLUMNS
    ]

    if feature_df.shape != (
        72,
        71,
    ):

        raise ValueError(
            "Forecast feature output has "
            "incorrect shape.\n"
            f"Expected: (72, 71)\n"
            f"Received: {feature_df.shape}"
        )

    # ========================================================
    # FINAL CHECKS
    # ========================================================

    if hourly_forecast[
        "predicted_aqi"
    ].isna().any():

        raise ValueError(
            "Hourly forecast contains "
            "missing predictions."
        )

    if daily_forecast[
        "predicted_aqi"
    ].isna().any():

        raise ValueError(
            "Daily forecast contains "
            "missing predictions."
        )

    print()
    print(
        "✓ 72 recursive forecast predictions"
    )

    print(
        "✓ 3 daily averages"
    )

    print(
        "✓ 24 predictions per day"
    )

    print(
        "✓ 70 model features per prediction"
    )

    print(
        "✓ us_aqi never passed to MLflow"
    )

    print(
        "✓ target_aqi never passed to MLflow"
    )

    # ========================================================
    # 72-HOUR PERSISTENCE BASELINE
    # ========================================================

    hourly_forecast = add_persistence_baseline(
        hourly_forecast=hourly_forecast,
        feast_history=feast_history,
    )

    # ========================================================
    # DAILY PERSISTENCE AVERAGES
    # ========================================================

    daily_persistence = (
        hourly_forecast
        .groupby(
            "forecast_day",
            as_index=False,
        )
        .agg(
            persistence_aqi=(
                "persistence_aqi",
                "mean",
            )
        )
    )

    daily_forecast = daily_forecast.merge(
        daily_persistence,
        on="forecast_day",
        how="left",
    )

    if len(daily_forecast) != 3:
        raise ValueError(
            "Persistence daily forecast must "
            "contain exactly 3 days."
        )

    if hourly_forecast[
        "persistence_aqi"
    ].isna().any():

        raise ValueError(
            "Persistence forecast contains "
            "missing values."
        )

    print()
    print(
        "✓ 72-hour persistence baseline"
    )

    print(
        "✓ 3 daily persistence averages"
    )

    return (
        hourly_forecast,
        daily_forecast,
        feature_df,
    )

# ============================================================
# 72-HOUR PERSISTENCE BASELINE
# ============================================================

def add_persistence_baseline(
    hourly_forecast,
    feast_history,
):
    """
    Create a fixed-origin 72-hour persistence forecast.

    Persistence assumption:
        Every future hour = last observed AQI
        at the forecast origin.

    IMPORTANT:
        Future actual AQI is NOT used.
    """

    hourly_forecast = hourly_forecast.copy()

    feast_history = feast_history.copy()

    feast_history["event_timestamp"] = (
        pd.to_datetime(
            feast_history["event_timestamp"],
            utc=True,
        ).dt.floor("h")
    )

    feast_history = (
        feast_history
        .sort_values("event_timestamp")
        .reset_index(drop=True)
    )

    if feast_history.empty:
        raise ValueError(
            "Cannot create persistence baseline: "
            "Feast history is empty."
        )

    last_observed_aqi = float(
        feast_history.iloc[-1]["us_aqi"]
    )

    if not np.isfinite(last_observed_aqi):
        raise ValueError(
            "Last observed AQI for persistence "
            "baseline is not finite."
        )

    hourly_forecast[
        "persistence_aqi"
    ] = last_observed_aqi

    if len(hourly_forecast) != 72:
        raise ValueError(
            "Persistence baseline must contain "
            f"72 rows. Received {len(hourly_forecast)}."
        )

    print()
    print("=" * 70)
    print("72-HOUR PERSISTENCE BASELINE")
    print("=" * 70)

    print(
        "Persistence origin:",
        feast_history.iloc[-1]["event_timestamp"],
    )

    print(
        "Last observed AQI:",
        f"{last_observed_aqi:.4f}",
    )

    print(
        "Persistence forecast:",
        f"{last_observed_aqi:.4f}",
        "for all 72 hours",
    )

    print(
        "✓ Fixed-origin 72-hour persistence baseline created"
    )

    return hourly_forecast

# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    forecast_df,
    daily_forecast,
    feature_df,
):

    print()
    print("=" * 70)
    print("SAVING PREDICTION OUTPUTS")
    print("=" * 70)

    PREDICTION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # HOURLY
    # --------------------------------------------------------

    hourly_output = (
        forecast_df[
            [
                "timestamp",
                "forecast_day",
                "predicted_aqi",
                "persistence_aqi",
            ]
        ]
        .copy()
    )

    hourly_output.to_csv(
        HOURLY_PREDICTION_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # DAILY
    # --------------------------------------------------------

    daily_forecast[
        [
            "forecast_day",
            "predicted_aqi",
            "persistence_aqi"
        ]
    ].to_csv(
        DAILY_PREDICTION_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    feature_df.to_csv(
        FEATURE_PREDICTION_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # RELOAD VERIFICATION
    # --------------------------------------------------------

    hourly_check = pd.read_csv(
        HOURLY_PREDICTION_FILE
    )

    daily_check = pd.read_csv(
        DAILY_PREDICTION_FILE
    )

    feature_check = pd.read_csv(
        FEATURE_PREDICTION_FILE
    )

    if len(hourly_check) != 72:

        raise ValueError(
            "Saved hourly forecast does not "
            "contain 72 rows."
        )
    
    if "persistence_aqi" not in hourly_check.columns:

        raise ValueError(
            "Saved hourly forecast does not "
            "contain persistence_aqi."
        )

    if len(daily_check) != 3:

        raise ValueError(
            "Saved daily forecast does not "
            "contain 3 rows."
        )

    if "persistence_aqi" not in daily_check.columns:

        raise ValueError(
            "Saved daily forecast does not "
            "contain persistence_aqi."
        )

    if len(feature_check) != 72:

        raise ValueError(
            "Saved feature forecast does not "
            "contain 72 rows."
        )

    if feature_check.shape[1] != 71:

        raise ValueError(
            "Saved feature file must contain "
            "timestamp + 70 features."
        )

    print(
        "Hourly file:",
        HOURLY_PREDICTION_FILE,
    )

    print(
        "Daily file:",
        DAILY_PREDICTION_FILE,
    )

    print(
        "Feature file:",
        FEATURE_PREDICTION_FILE,
    )

    print("✓ Hourly output: 72 rows")
    print("✓ Daily output: 3 rows")
    print("✓ Feature output: 72 rows × 71 columns")


# ============================================================
# MAIN
# ============================================================

def main():

    print_header()

    # ========================================================
    # CURRENT TIME / FORECAST WINDOW
    # ========================================================

    current_hour = (
        utc_now_hour()
    )

    forecast_start = (
        next_local_midnight_utc()
    )

    forecast_end = (
        forecast_start
        + pd.Timedelta(hours=71)
    )

    if forecast_start <= current_hour:

        raise RuntimeError(
            "Invalid forecast window.\n"
            f"Current: {current_hour}\n"
            f"Forecast start: {forecast_start}"
        )

    print()
    print("=" * 70)
    print("FORECAST WINDOW")
    print("=" * 70)

    print(
        "Current UTC:",
        current_hour,
    )

    print(
        "Current local:",
        current_hour.tz_convert(
            LOCAL_TIMEZONE
        ),
    )

    print(
        "Forecast start UTC:",
        forecast_start,
    )

    print(
        "Forecast start local:",
        forecast_start.tz_convert(
            LOCAL_TIMEZONE
        ),
    )

    print(
        "Forecast end UTC:",
        forecast_end,
    )

    print(
        "Forecast end local:",
        forecast_end.tz_convert(
            LOCAL_TIMEZONE
        ),
    )

    # ========================================================
    # LOAD MLFLOW
    # ========================================================

    model = load_model()

    # ========================================================
    # LOAD FEAST
    # ========================================================

    store = load_feast()

    # ========================================================
    # FETCH OPEN-METEO
    # ========================================================

    weather = fetch_weather(
        forecast_end_utc=forecast_end
    )

    air = fetch_air_quality(
        forecast_end_utc=forecast_end
    )

    api_df = merge_api_data(
        weather=weather,
        air=air,
    )

    # ========================================================
    # VALIDATE API COVERAGE
    # ========================================================

    required_api_start = (
        current_hour
        - pd.Timedelta(
            hours=HISTORY_HOURS - 1
        )
    )

    if (
        api_df["timestamp"].min()
        > required_api_start
    ):

        raise ValueError(
            "Open-Meteo does not provide the "
            "required 96-hour historical context.\n"
            f"Required from: {required_api_start}\n"
            f"Available from: "
            f"{api_df['timestamp'].min()}"
        )

    if (
        api_df["timestamp"].max()
        < forecast_end
    ):

        raise ValueError(
            "Open-Meteo does not provide the "
            "complete forecast horizon.\n"
            f"Required through: {forecast_end}\n"
            f"Available through: "
            f"{api_df['timestamp'].max()}"
        )

    # ========================================================
    # FEAST HISTORICAL CONTEXT
    #
    # IMPORTANT:
    # Feast is queried only through the current hour.
    # We never request future Feast timestamps.
    # ========================================================

    feast_history = (
        get_historical_context(
            store=store,
            latest_timestamp=current_hour,
        )
    )

    # ========================================================
    # DEBUG / CONTRACT SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("HISTORICAL CONTEXT VALIDATION")
    print("=" * 70)

    print(
        "Rows:",
        len(feast_history),
    )

    print(
        "Columns:",
        len(feast_history.columns),
    )

    print(
        "Has event_timestamp:",
        "event_timestamp"
        in feast_history.columns,
    )

    print(
        "Timestamp range:",
        feast_history[
            "event_timestamp"
        ].min(),
        "->",
        feast_history[
            "event_timestamp"
        ].max(),
    )

    print(
        "Missing model features:",
        sorted(
            set(FEATURE_COLUMNS)
            - set(feast_history.columns)
        ),
    )

    print(
        "Has target_aqi:",
        "target_aqi"
        in feast_history.columns,
    )

    print(
        "Has us_aqi:",
        "us_aqi"
        in feast_history.columns,
    )

    # ========================================================
    # GENERATE FORECAST
    # ========================================================

    (
        forecast_df,
        daily_forecast,
        feature_df,
    ) = generate_forecast(
        model=model,
        feast_history=feast_history,
        api_df=api_df,
        forecast_start=forecast_start,
        forecast_end=forecast_end,
    )

    
    # ========================================================
    # SAVE
    # ========================================================

    save_outputs(
        forecast_df=forecast_df,
        daily_forecast=daily_forecast,
        feature_df=feature_df,
    )

    # ========================================================
    # FINAL VERIFICATION
    # ========================================================

    print()
    print("=" * 70)
    print("PREDICTION PIPELINE VERIFIED")
    print("=" * 70)

    print(
        "Feast context features:",
        71,
    )

    print(
        "MLflow model features:",
        70,
    )

    print(
        "MLflow champion:",
        MLFLOW_MODEL_URI,
    )

    print(
        "Hourly predictions:",
        len(forecast_df),
    )

    print(
        "Daily averages:",
        len(daily_forecast),
    )

    print(
        "Persistence baseline:",
        "72-hour fixed-origin",
    )

    print()
    print("DAILY AQI COMPARISON")
    print(
        daily_forecast[
            [
                "forecast_day",
                "predicted_aqi",
                "persistence_aqi",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Hourly prediction file:",
        HOURLY_PREDICTION_FILE,
    )

    print(
        "Daily prediction file:",
        DAILY_PREDICTION_FILE,
    )

    print(
        "Feature file:",
        FEATURE_PREDICTION_FILE,
    )

    # --------------------------------------------------------
    # HARD ASSERTIONS
    # --------------------------------------------------------

    assert len(
        forecast_df
    ) == 72

    assert len(
        daily_forecast
    ) == 3

    assert (
        "persistence_aqi"
        in forecast_df.columns
    )

    assert (
        "persistence_aqi"
        in daily_forecast.columns
    )

    assert (
        forecast_df["persistence_aqi"]
        .notna()
        .all()
    )

    assert len(
        feature_df
    ) == 72

    assert (
        feature_df.shape[1]
        == 71
    )

    assert (
        "target_aqi"
        not in feature_df.columns
    )

    assert (
        "us_aqi"
        not in feature_df.columns
    )

    assert (
        set(FEATURE_COLUMNS)
        == set(
            feature_df.columns
        ) - {"timestamp"}
    )

    print()
    print(
        "ALL PREDICTION CHECKS PASSED"
    )
    print(
        "70 model features: PASS"
    )
    print(
        "us_aqi recursive state: PASS"
    )
    print(
        "target_aqi excluded: PASS"
    )
    print(
        "72 hourly predictions: PASS"
    )
    print(
        "3 daily averages: PASS"
    )
    print(
        "24 predictions per day: PASS"
    )
    print(
        "MLflow champion: PASS"
    )
    print(
        "FEAST -> MODEL CONTRACT: PASS"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()