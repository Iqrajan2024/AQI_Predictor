"""
PEARLS AQI PREDICTOR
NEXT 3-DAY FORECAST PIPELINE

Architecture:

Open-Meteo
    |
    +-- Weather forecast
    +-- Air-quality forecast
    |
    v
Feast historical context
    |
    v
70 model features
    |
    v
MLflow @champion
    |
    v
72 hourly recursive predictions
    |
    v
3 daily averages
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

MLFLOW_DATABASE = (
    PROJECT_ROOT / "mlflow.db"
)

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MLFLOW_MODEL_URI = (
    f"models:/{MLFLOW_MODEL_NAME}@{MLFLOW_MODEL_ALIAS}"
)

# ============================================================
# LOCATION
# ============================================================

LOCATION_ID = "peshawar"

LATITUDE = 34.008
LONGITUDE = 71.5785


# ============================================================
# API
# ============================================================

WEATHER_URL = (
    "https://api.open-meteo.com/v1/forecast"
)

AIR_QUALITY_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
)


# ============================================================
# FORECAST
# ============================================================

HISTORY_HOURS = 96

FORECAST_DAYS = 3

FORECAST_HOURS = 72


# ============================================================
# 70 MODEL FEATURES
# ============================================================

FEATURE_COLUMNS = [

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

    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",

    "aqi_lag_1",
    "aqi_lag_3",
    "aqi_lag_6",
    "aqi_lag_12",
    "aqi_lag_24",
    "aqi_lag_48",
    "aqi_lag_72",

    "pm2_5_lag_1",
    "pm2_5_lag_3",
    "pm2_5_lag_6",
    "pm2_5_lag_24",

    "pm10_lag_1",
    "pm10_lag_3",
    "pm10_lag_6",
    "pm10_lag_24",

    "carbon_monoxide_lag_1",
    "carbon_monoxide_lag_3",
    "carbon_monoxide_lag_6",
    "carbon_monoxide_lag_24",

    "nitrogen_dioxide_lag_1",
    "nitrogen_dioxide_lag_3",
    "nitrogen_dioxide_lag_6",
    "nitrogen_dioxide_lag_24",

    "sulphur_dioxide_lag_1",
    "sulphur_dioxide_lag_3",
    "sulphur_dioxide_lag_6",
    "sulphur_dioxide_lag_24",

    "ozone_lag_1",
    "ozone_lag_3",
    "ozone_lag_6",
    "ozone_lag_24",

    "aqi_3h_mean",
    "aqi_6h_mean",
    "aqi_12h_mean",
    "aqi_24h_mean",

    "pm2_5_3h_mean",
    "pm2_5_6h_mean",
    "pm2_5_24h_mean",

    "pm10_3h_mean",
    "pm10_6h_mean",
    "pm10_24h_mean",

    "carbon_monoxide_24h_mean",
    "nitrogen_dioxide_24h_mean",
    "sulphur_dioxide_24h_mean",
    "ozone_24h_mean",

    "aqi_change_1h",
    "aqi_change_3h",
    "aqi_change_6h",
    "aqi_change_24h",

    "pm2_5_change_1h",
    "pm2_5_change_24h",

    "pm10_change_1h",
    "pm10_change_24h",
]


assert len(FEATURE_COLUMNS) == 70


# ============================================================
# HEADER
# ============================================================

def print_header():

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
        "MLflow:",
        MLFLOW_MODEL_URI,
    )

    print(
        "Feast:",
        FEATURE_REPO,
    )

    print(
        "Model features:",
        len(FEATURE_COLUMNS),
    )


# ============================================================
# LOAD MLFLOW CHAMPION
# ============================================================

def load_model():

    print("\n" + "=" * 70)
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
        mlflow.get_tracking_uri()
    )

    print(
        "Registry URI:",
        mlflow.get_registry_uri()
    )

    print(
        "Model:",
        MLFLOW_MODEL_NAME
    )

    print(
        "Alias:",
        MLFLOW_MODEL_ALIAS
    )

    from mlflow import MlflowClient

    client = MlflowClient()

    try:

        registered_model = (
            client.get_registered_model(
                MLFLOW_MODEL_NAME
            )
        )

    except Exception as exc:

        raise RuntimeError(
            "\nMLflow registered model was not found.\n"
            f"Expected model: {MLFLOW_MODEL_NAME}\n"
            f"Expected database: {MLFLOW_DATABASE}\n\n"
            "Run the training pipeline first so the model "
            "is registered and assigned the champion alias."
        ) from exc

    try:

        champion = (
            client.get_model_version_by_alias(
                MLFLOW_MODEL_NAME,
                MLFLOW_MODEL_ALIAS,
            )
        )

    except Exception as exc:

        raise RuntimeError(
            "\nMLflow champion alias was not found.\n"
            f"Model: {MLFLOW_MODEL_NAME}\n"
            f"Alias: {MLFLOW_MODEL_ALIAS}\n"
        ) from exc

    print(
        "Champion version:",
        champion.version
    )

    model = mlflow.pyfunc.load_model(
        MLFLOW_MODEL_URI
    )

    print(
        "✓ MLflow champion model loaded"
    )

    print(
        "Model URI:",
        MLFLOW_MODEL_URI
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
    print("LOADING FEAST")
    print("=" * 70)

    if not FEATURE_REPO.exists():
        raise FileNotFoundError(
            f"Feast repository not found: "
            f"{FEATURE_REPO}"
        )

    store = FeatureStore(
        repo_path=str(FEATURE_REPO)
    )

    return store


# ============================================================
# FETCH WEATHER
# ============================================================

def fetch_weather():

    now_utc = (
        pd.Timestamp.now(tz="UTC")
    )

    current_hour = (
        now_utc.floor("h")
    )

    next_day = (
        current_hour.normalize()
        + pd.Timedelta(days=1)
    )

    forecast_end = (
        next_day
        + pd.Timedelta(hours=71)
    )

    required_hours = int(
        (
            forecast_end
            - current_hour
        ).total_seconds()
        / 3600
    ) + 1

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

        "forecast_hours": required_hours,

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

    df = pd.DataFrame(
        data["hourly"]
    )

    df["timestamp"] = pd.to_datetime(
        df["time"],
        utc=True,
    )

    df = df.drop(
        columns=["time"]
    )

    return (
        df
        .sort_values("timestamp")
        .drop_duplicates(
            "timestamp"
        )
        .reset_index(drop=True)
    )


# ============================================================
# FETCH AIR QUALITY
# ============================================================

def fetch_air_quality():

    now_utc = (
        pd.Timestamp.now(tz="UTC")
    )

    current_hour = (
        now_utc.floor("h")
    )

    next_day = (
        current_hour.normalize()
        + pd.Timedelta(days=1)
    )

    forecast_end = (
        next_day
        + pd.Timedelta(hours=71)
    )

    required_hours = int(
        (
            forecast_end
            - current_hour
        ).total_seconds()
        / 3600
    ) + 1

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

        "forecast_hours": required_hours,

        "timezone": "UTC",
    }

    response = requests.get(
        AIR_QUALITY_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    df = pd.DataFrame(
        data["hourly"]
    )

    df["timestamp"] = pd.to_datetime(
        df["time"],
        utc=True,
    )

    df = df.drop(
        columns=["time"]
    )

    return (
        df
        .sort_values("timestamp")
        .drop_duplicates(
            "timestamp"
        )
        .reset_index(drop=True)
    )


# ============================================================
# MERGE API
# ============================================================

def merge_api_data(
    weather,
    air,
):

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

    return df


# ============================================================
# FEAST HISTORICAL CONTEXT
# ============================================================

def get_feast_historical_context(
    store,
    timestamps,
):

    print("\n" + "=" * 70)
    print("LOADING 70 FEATURES FROM FEAST")
    print("=" * 70)

    entity_df = pd.DataFrame(
        {
            "location_id": [
                LOCATION_ID
            ] * len(timestamps),

            "event_timestamp": (
                pd.to_datetime(
                    timestamps,
                    utc=True,
                )
            ),
        }
    )

    feature_refs = [
        f"aqi_features:{feature}"
        for feature in FEATURE_COLUMNS
    ]

    # Add us_aqi separately because it is
    # historical state used for recursive forecasting,
    # but is NOT a model input.
    feature_refs.append(
        "aqi_features:us_aqi"
    )

    feast_df = (
        store
        .get_historical_features(
            entity_df=entity_df,
            features=feature_refs,
        )
        .to_df()
    )

    feast_df["event_timestamp"] = (
        pd.to_datetime(
            feast_df["event_timestamp"],
            utc=True,
        )
    )

    missing = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in feast_df.columns
    ]

    if missing:
        raise ValueError(
            "Feast is missing model features:\n"
            + "\n".join(missing)
        )

    if "us_aqi" not in feast_df.columns:
        raise ValueError(
            "Feast did not return us_aqi."
        )

    if len(FEATURE_COLUMNS) != 70:
        raise ValueError(
            "Feature count is not 70."
        )

    feast_df = (
        feast_df
        .sort_values(
            "event_timestamp"
        )
        .reset_index(drop=True)
    )

    if feast_df.empty:
        raise ValueError(
            "Feast returned no historical rows."
        )

    print(
        "Feast historical rows:",
        len(feast_df),
    )

    print(
        "Feast model features:",
        len(FEATURE_COLUMNS),
    )

    print(
        "Feast historical feature loading: PASS"
    )

    return feast_df


# ============================================================
# CREATE FUTURE FEATURES
# ============================================================

def create_features(df):

    df = df.copy()

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

    df["pm2_5_change_1h"] = (
        df["pm2_5"]
        - df["pm2_5_lag_1"]
    )

    df["pm2_5_change_24h"] = (
        df["pm2_5"]
        - df["pm2_5_lag_24"]
    )

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
# FORECAST
# ============================================================

def generate_forecast(
    model,
    feast_history,
    api_df,
):

    now_utc = (
        pd.Timestamp.now(tz="UTC")
    )

    current_date = (
        now_utc.date()
    )

    forecast_start = (
        pd.Timestamp(
            current_date,
            tz="UTC",
        )
        + pd.Timedelta(days=1)
    )

    forecast_end = (
        forecast_start
        + pd.Timedelta(hours=71)
    )

    future_df = (
        api_df[
            (
                api_df["timestamp"]
                >= forecast_start
            )
            &
            (
                api_df["timestamp"]
                <= forecast_end
            )
        ]
        .copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if len(future_df) != 72:

        raise ValueError(
            f"Expected 72 future rows, "
            f"received {len(future_df)}"
        )

    # ========================================================
    # BUILD API HISTORY
    # ========================================================

    history_start = (
        forecast_start
        - pd.Timedelta(hours=96)
    )

    api_history = (
        api_df[
            (
                api_df["timestamp"]
                >= history_start
            )
            &
            (
                api_df["timestamp"]
                < forecast_start
            )
        ]
        .copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # ========================================================
    # REPLACE API AQI HISTORY WITH FEAST AQI
    # ========================================================

    feast_aqi = feast_history[
        [
            "event_timestamp",
            "us_aqi",
        ]
    ].copy()

    feast_aqi = feast_aqi.rename(
        columns={
            "event_timestamp":
                "timestamp"
        }
    )

    history = pd.merge(
        api_history.drop(
            columns=["us_aqi"],
            errors="ignore",
        ),
        feast_aqi,
        on="timestamp",
        how="inner",
    )

    history = (
        history
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if len(history) < 72:
        raise ValueError(
            "Insufficient Feast historical "
            "context for recursive forecast."
        )

    predictions = []

    feature_records = []

    working_history = (
        history.copy()
    )

    # ========================================================
    # RECURSIVE 72 HOURS
    # ========================================================

    for step in range(72):

        future_row = (
            future_df
            .iloc[step]
        )

        prediction_timestamp = (
            future_row["timestamp"]
        )

        # ----------------------------------------------
        # Feature construction
        # ----------------------------------------------
        
        anchor_timestamp = (
                    prediction_timestamp
                    - pd.Timedelta(hours=1)
                )
        if step == 0:

            X = feast_history[
                FEATURE_COLUMNS
            ].copy()

        else:

            feature_frame = create_features(
                working_history
            )

            anchor_rows = feature_frame[
                feature_frame["timestamp"]
                == anchor_timestamp
            ]
        

            if anchor_rows.empty:
                raise ValueError(
                    "Missing feature anchor: "
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
                .astype(float)
            )

        # ----------------------------------------------
        # EXACT 70 FEATURE CHECK
        # ----------------------------------------------

        X = X.astype(float)

        assert X.shape[1] == 70


        if X.isna().any().any():

            missing = (
                X.columns[
                    X.isna().any()
                ]
                .tolist()
            )

            raise ValueError(
                "Missing prediction features:\n"
                + "\n".join(missing)
            )

        # ----------------------------------------------
        # Predict
        # ----------------------------------------------

        prediction = model.predict(X)

        predicted_aqi = float(
            np.asarray(
                prediction
            )
            .reshape(-1)[0]
        )

        predicted_aqi = max(
            0.0,
            predicted_aqi,
        )

        predictions.append(
            {
                "timestamp":
                    prediction_timestamp,
                "predicted_aqi":
                    predicted_aqi,
            }
        )

        # ----------------------------------------------
        # Save exact feature vector
        # ----------------------------------------------

        feature_record = {
            column:
                float(X.iloc[0][column])
            for column in FEATURE_COLUMNS
        }

        feature_record[
            "timestamp"
        ] = prediction_timestamp

        feature_records.append(
            feature_record
        )

        # ----------------------------------------------
        # Recursive state
        # ----------------------------------------------

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

    forecast_df = (
        pd.DataFrame(
            predictions
        )
    )

    forecast_df[
        "forecast_day"
    ] = (
        forecast_df[
            "timestamp"
        ].dt.date
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

    if len(
        forecast_df
    ) != 72:
        raise ValueError(
            "Expected 72 predictions."
        )

    if len(
        daily_forecast
    ) != 3:
        raise ValueError(
            "Expected 3 daily averages."
        )

    counts = (
        forecast_df
        .groupby(
            "forecast_day"
        )
        .size()
    )

    if not (
        counts == 24
    ).all():

        raise ValueError(
            "Every forecast day must "
            "contain exactly 24 predictions."
        )

    feature_df = (
        pd.DataFrame(
            feature_records
        )
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
            "Forecast feature output must "
            "contain 72 rows + timestamp + "
            "70 features."
        )

    return (
        forecast_df,
        daily_forecast,
        feature_df,
    )


# ============================================================
# SAVE
# ============================================================

def save_outputs(
    forecast_df,
    daily_forecast,
    feature_df,
):

    PREDICTION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecast_df[
        [
            "timestamp",
            "predicted_aqi",
        ]
    ].to_csv(
        HOURLY_PREDICTION_FILE,
        index=False,
    )

    daily_forecast[
        [
            "forecast_day",
            "predicted_aqi",
        ]
    ].to_csv(
        DAILY_PREDICTION_FILE,
        index=False,
    )

    feature_df.to_csv(
        FEATURE_PREDICTION_FILE,
        index=False,
    )

    assert (
        len(
            pd.read_csv(
                HOURLY_PREDICTION_FILE
            )
        )
        == 72
    )

    assert (
        len(
            pd.read_csv(
                DAILY_PREDICTION_FILE
            )
        )
        == 3
    )

    assert (
        len(
            pd.read_csv(
                FEATURE_PREDICTION_FILE
            )
        )
        == 72
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print_header()

    # ========================================================
    # MLflow champion
    # ========================================================

    model = load_model()

    # ========================================================
    # Feast
    # ========================================================

    store = load_feast()

    # ========================================================
    # Open-Meteo
    # ========================================================

    weather = fetch_weather()

    air = fetch_air_quality()

    api_df = merge_api_data(
        weather,
        air,
    )

    # ========================================================
    # LAST 96 HOURS FOR FEAST
    # ========================================================

    now_utc = (
        pd.Timestamp.now(tz="UTC")
    )

    forecast_start = (
        pd.Timestamp(
            now_utc.date(),
            tz="UTC",
        )
        + pd.Timedelta(days=1)
    )

    history_start = (
        forecast_start
        - pd.Timedelta(hours=96)
    )

    history_timestamps = (
        api_df[
            (
                api_df["timestamp"]
                >= history_start
            )
            &
            (
                api_df["timestamp"]
                < forecast_start
            )
        ]
        [
            "timestamp"
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    feast_history = (
        get_feast_historical_context(
            store,
            history_timestamps,
        )
    )

    # ========================================================
    # FORECAST
    # ========================================================

    (
        forecast_df,
        daily_forecast,
        feature_df,
    ) = generate_forecast(
        model=model,
        feast_history=feast_history,
        api_df=api_df,
    )

    # ========================================================
    # SAVE
    # ========================================================

    save_outputs(
        forecast_df,
        daily_forecast,
        feature_df,
    )

    # ========================================================
    # FINAL VERIFICATION
    # ========================================================

    print("\n" + "=" * 70)
    print("PREDICTION PIPELINE VERIFIED")
    print("=" * 70)

    print(
        "Feast model features loaded:",
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
        "\nDAILY AQI"
    )

    print(
        daily_forecast.to_string(
            index=False
        )
    )

    print(
        "\nHourly prediction file:",
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

    assert len(
        forecast_df
    ) == 72

    assert len(
        daily_forecast
    ) == 3

    assert len(
        feature_df
    ) == 72

    assert (
        feature_df.shape[1]
        == 71
    )

    print(
        "\nALL PREDICTION CHECKS PASSED"
    )


if __name__ == "__main__":
    main()