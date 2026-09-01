"""
============================================================
PEARLS AQI PREDICTOR
FASTAPI DASHBOARD BACKEND
============================================================

Provides:

    /health
    /current
    /current/trend
    /forecast
    /forecast/{day}
    /shap/{day}
    /dashboard

Data sources:

    Open-Meteo
        -> current AQI
        -> current pollutants
        -> 24-hour AQI trend

    Prediction artifacts
        -> latest_forecast.csv
        -> latest_daily_forecast.csv
        -> latest_forecast_features.csv

    MLflow
        -> champion model
        -> model RMSE
        -> SHAP explanations

    Feast
        -> already integrated into prediction pipeline

============================================================
"""

from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
import requests
import shap

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTION_DIR = (
    PROJECT_ROOT
    / "data"
    / "predictions"
)

HOURLY_FORECAST_FILE = (
    PREDICTION_DIR
    / "latest_forecast.csv"
)

DAILY_FORECAST_FILE = (
    PREDICTION_DIR
    / "latest_daily_forecast.csv"
)

FORECAST_FEATURES_FILE = (
    PREDICTION_DIR
    / "latest_forecast_features.csv"
)


# ============================================================
# MLflow
# ============================================================

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

MLFLOW_MODEL_URI = (
    "models:/Pearls_AQI_XGBoost@champion"
)


# ============================================================
# LOCATION
# ============================================================

LOCATION_ID = "peshawar"

LATITUDE = 34.008
LONGITUDE = 71.5785

TIMEZONE = ZoneInfo("Asia/Karachi")

# ============================================================
# OPEN-METEO
# ============================================================

AIR_QUALITY_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
)


# ============================================================
# MODEL FEATURES
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


if len(FEATURE_COLUMNS) != 70:
    raise RuntimeError(
        f"Expected 70 features, "
        f"found {len(FEATURE_COLUMNS)}"
    )


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Pearls AQI Predictor API",
    description=(
        "Backend API for the Pearls AQI Predictor "
        "Streamlit dashboard."
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# GLOBAL MODEL
# ============================================================

MODEL = None
NATIVE_XGB_MODEL = None

# ============================================================
# AQI CATEGORY
# ============================================================

def get_aqi_category(aqi: float) -> str:

    if aqi <= 50:
        return "Good"

    if aqi <= 100:
        return "Moderate"

    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"

    if aqi <= 200:
        return "Unhealthy"

    if aqi <= 300:
        return "Very Unhealthy"

    return "Hazardous"

# ============================================================
# AQI HEALTH ALERT
# ============================================================

def get_aqi_health_alert(aqi: float) -> dict:

    if aqi <= 50:
        return {
            "alert": False,
            "level": "Good",
            "message": (
                "Air quality is considered satisfactory "
                "and poses little or no risk."
            ),
            "recommendation": (
                "No special precautions are needed."
            ),
        }

    if aqi <= 100:
        return {
            "alert": False,
            "level": "Moderate",
            "message": (
                "Air quality is acceptable, although "
                "unusually sensitive people may experience "
                "some health effects."
            ),
            "recommendation": (
                "Unusually sensitive individuals should "
                "consider reducing prolonged or heavy outdoor activity if they notice "
                "symptoms."
            ),
        }

    if aqi <= 150:
        return {
            "alert": True,
            "level": "Unhealthy for Sensitive Groups",
            "message": (
                "Sensitive groups may "
                "experience health effects."
            ),
            "recommendation": (
                "Sensitive individuals should consider "
                "reducing prolonged or heavy outdoor exertion."
            ),
        }

    if aqi <= 200:
        return {
            "alert": True,
            "level": "Unhealthy",
            "message": (
                "Everyone may begin to experience "
                "health effects."
            ),
            "recommendation": (
                "Consider reducing prolonged or heavy exertion."
            ),
        }

    if aqi <= 300:
        return {
            "alert": True,
            "level": "Very Unhealthy",
            "message": (
                "Everyone may experience "
                "more serious health effects."
            ),
            "recommendation": (
                "Avoid prolonged or "
                "heavy outdoor exertion and consider "
                "limiting time outdoors."
            ),
        }

    return {
        "alert": True,
        "level": "Hazardous",
        "message": (
            "Health warning: emergency conditions. "
            "The predicted air quality represents a serious health risk."
        ),
        "recommendation": (
            "Avoid outdoor exposure as much as possible  "
            "and follow local health guidance."
        ),
    }

# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    global MODEL

    if MODEL is not None:
        return MODEL

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    try:

        MODEL = mlflow.pyfunc.load_model(
            MLFLOW_MODEL_URI
        )

    except Exception as exc:

        raise RuntimeError(
            "Unable to load MLflow champion model.\n"
            f"MLflow URI: {MLFLOW_TRACKING_URI}\n"
            f"Model URI: {MLFLOW_MODEL_URI}\n"
            f"Error: {exc}"
        ) from exc

    return MODEL

# ============================================================
# LOAD NATIVE XGBOOST MODEL FOR SHAP
# ============================================================

def load_native_xgb_model():
    """
    Load the champion model using MLflow's native XGBoost
    flavor instead of the generic PyFunc wrapper.

    SHAP TreeExplainer requires the actual XGBoost model.
    """

    global NATIVE_XGB_MODEL

    if NATIVE_XGB_MODEL is not None:
        return NATIVE_XGB_MODEL

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    try:

        NATIVE_XGB_MODEL = (
            mlflow.xgboost.load_model(
                MLFLOW_MODEL_URI
            )
        )

    except Exception as exc:

        raise RuntimeError(
            "Unable to load native XGBoost model "
            "for SHAP.\n"
            f"MLflow URI: {MLFLOW_TRACKING_URI}\n"
            f"Model URI: {MLFLOW_MODEL_URI}\n"
            f"Error: {exc}"
        ) from exc

    return NATIVE_XGB_MODEL

# ============================================================
# LOAD PREDICTION FILES
# ============================================================

def load_hourly_forecast():

    if not HOURLY_FORECAST_FILE.exists():

        raise FileNotFoundError(
            f"Missing forecast file:\n"
            f"{HOURLY_FORECAST_FILE}"
        )

    df = pd.read_csv(
        HOURLY_FORECAST_FILE
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    return df


def load_daily_forecast():

    if not DAILY_FORECAST_FILE.exists():

        raise FileNotFoundError(
            f"Missing daily forecast file:\n"
            f"{DAILY_FORECAST_FILE}"
        )

    df = pd.read_csv(
        DAILY_FORECAST_FILE
    )

    df["forecast_day"] = pd.to_datetime(
        df["forecast_day"]
    ).dt.date

    return df


def load_forecast_features():

    if not FORECAST_FEATURES_FILE.exists():

        raise FileNotFoundError(
            f"Missing forecast feature file:\n"
            f"{FORECAST_FEATURES_FILE}"
        )

    df = pd.read_csv(
        FORECAST_FEATURES_FILE
    )

    if "timestamp" not in df.columns:

        raise ValueError(
            "latest_forecast_features.csv "
            "does not contain timestamp."
        )

    # --------------------------------------------------------
    # Parse timestamps as UTC
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    if df["timestamp"].isna().any():

        raise ValueError(
            "Invalid timestamp values found in "
            "latest_forecast_features.csv."
        )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Create LOCAL forecast day
    #
    # IMPORTANT:
    # Dashboard dates are Asia/Karachi dates.
    # --------------------------------------------------------

    df["local_timestamp"] = (
        df["timestamp"]
        .dt.tz_convert(TIMEZONE)
    )

    df["forecast_day"] = (
        df["local_timestamp"]
        .dt.date
    )

    return df

# ============================================================
# CURRENT AIR QUALITY
# ============================================================

def fetch_current_air_quality():

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

        "past_hours": 24,

        "forecast_hours": 1,

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
            "Open-Meteo response does not contain "
            "hourly air-quality data."
        )

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

    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"]
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# CURRENT AQI
# ============================================================

@app.get("/current")
def current_aqi():

    try:

        df = fetch_current_air_quality()

    except Exception as exc:

        raise HTTPException(
            status_code=502,
            detail=str(exc),
        )

    if df.empty:

        raise HTTPException(
            status_code=404,
            detail="No current AQI data available.",
        )

    latest = df.iloc[-1]

    aqi = float(
        latest["us_aqi"]
    )

    health_alert = get_aqi_health_alert(aqi)

    return {

        "location": LOCATION_ID,

        "timestamp": (
            latest["timestamp"]
            .isoformat()
        ),

        "aqi": round(aqi, 2),

        "category": get_aqi_category(aqi),

        "health_alert": health_alert,

        "pollutants": {

            "pm2_5": float(
                latest["pm2_5"]
            ),

            "pm10": float(
                latest["pm10"]
            ),

            "ozone": float(
                latest["ozone"]
            ),

            "nitrogen_dioxide": float(
                latest["nitrogen_dioxide"]
            ),

            "sulphur_dioxide": float(
                latest["sulphur_dioxide"]
            ),

            "carbon_monoxide": float(
                latest["carbon_monoxide"]
            ),
        },
    }


# ============================================================
# CURRENT 24-HOUR AQI TREND
# ============================================================

@app.get("/current/trend")
def current_aqi_trend():

    try:

        df = fetch_current_air_quality()

    except Exception as exc:

        raise HTTPException(
            status_code=502,
            detail=str(exc),
        )

    trend = df[
        [
            "timestamp",
            "us_aqi",
        ]
    ].copy()

    trend = trend.rename(
        columns={
            "us_aqi": "aqi"
        }
    )

    trend["aqi"] = trend["aqi"].astype(float)

    return {

        "location": LOCATION_ID,

        "hours": trend.to_dict(
            orient="records"
        ),
    }


# ============================================================
# MODEL RMSE
# ============================================================

def get_model_rmse():

    try:

        mlflow.set_tracking_uri(
            MLFLOW_TRACKING_URI
        )

        client = mlflow.tracking.MlflowClient()

        model_versions = client.get_model_version_by_alias(
            "Pearls_AQI_XGBoost",
            "champion",
        )

        run_id = model_versions.run_id

        if not run_id:
            return None

        run = client.get_run(
            run_id
        )

        metrics = run.data.metrics

        # Prefer test RMSE.
        candidates = [
            "test_rmse",
            "rmse",
            "validation_rmse",
            "val_rmse",
        ]

        for metric in candidates:

            if metric in metrics:

                return float(
                    metrics[metric]
                )

        # Case-insensitive fallback.
        for key, value in metrics.items():

            if "rmse" in key.lower():

                return float(value)

    except Exception:
        pass

    return None


# ============================================================
# FORECAST
# ============================================================

@app.get("/forecast")
def forecast():

    try:

        hourly = load_hourly_forecast()
        daily = load_daily_forecast()

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    rmse = get_model_rmse()

    daily_records = []

    for _, row in daily.iterrows():

        aqi = float(
            row["predicted_aqi"]
        )

        health_alert = get_aqi_health_alert(aqi)

        daily_records.append({

            "date": str(
                row["forecast_day"]
            ),

            "aqi": round(
                aqi,
                2,
            ),

            "category": get_aqi_category(
                aqi
            ),

            "health_alert": health_alert,

            "model_rmse": rmse,
        })

    hourly_records = []

    for _, row in hourly.iterrows():

        aqi = float(
            row["predicted_aqi"]
        )

        health_alert = get_aqi_health_alert(aqi)

        hourly_records.append({

            "timestamp": (
                row["timestamp"]
                .isoformat()
            ),

            "aqi": round(
                aqi,
                2,
            ),

            "category": get_aqi_category(
                aqi
            ),

            "health_alert": health_alert,
        })

    return {

        "location": LOCATION_ID,

        "forecast_days": 3,

        "forecast_hours": 72,

        "model": "Pearls_AQI_XGBoost",

        "model_alias": "champion",

        "model_rmse": rmse,

        "daily": daily_records,

        "hourly": hourly_records,
    }


# ============================================================
# SINGLE FORECAST DAY
# ============================================================

@app.get("/forecast/{forecast_day}")
def forecast_day(
    forecast_day: str,
):

    try:

        hourly = load_hourly_forecast()

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    try:

        requested_date = pd.to_datetime(
            forecast_day
        ).date()

    except Exception:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid date. "
                "Use YYYY-MM-DD."
            ),
        )
    
    hourly["local_timestamp"] = (
        hourly["timestamp"]
        .dt.tz_convert(TIMEZONE)
    )

    hourly["forecast_day"] = (
        hourly["local_timestamp"]
        .dt.date
    )

    result = hourly[
        hourly["forecast_day"]
        == requested_date
    ].copy()

    if result.empty:

        raise HTTPException(
            status_code=404,
            detail=(
                f"No forecast available for "
                f"{forecast_day}."
            ),
        )

    mean_aqi = float(
        result["predicted_aqi"].mean()
    )

    health_alert = get_aqi_health_alert(mean_aqi)

    return {

        "date": forecast_day,

        "aqi": round(
            mean_aqi,
            2,
        ),

        "category": get_aqi_category(
            mean_aqi
        ),

        "health_alert": health_alert,

        "model_rmse": get_model_rmse(),

        "hourly": [

            {
                "timestamp": (
                    row["local_timestamp"]
                    .isoformat()
                ),

                "aqi": round(
                    float(
                        row["predicted_aqi"]
                    ),
                    2,
                ),

                "category": get_aqi_category(
                    float(
                        row["predicted_aqi"]
                    )
                ),

                "health_alert": get_aqi_health_alert(
                    float(
                        row["predicted_aqi"]
                    )
                ),
            }

            for _, row in result.iterrows()
        ],
    }


# ============================================================
# SHAP EXPLANATION
# ============================================================

def calculate_shap_explanation(
    shap_model,
    forecast_features,
    feature_columns,
):
    """
    Calculate SHAP explanations for forecast rows
    using the native XGBoost model.

    Input:
        forecast_features:
            timestamp + 70 model features

    Output:
        One SHAP explanation per forecast hour.
    """

    # --------------------------------------------------------
    # Prepare model input
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in forecast_features.columns
    ]

    if missing_features:

        raise ValueError(
            "SHAP input is missing model features:\n"
            + "\n".join(missing_features)
        )

    X = (
        forecast_features[
            feature_columns
        ]
        .copy()
        .astype(float)
    )

    # --------------------------------------------------------
    # Validate feature count
    # --------------------------------------------------------

    if X.shape[1] != 70:

        raise ValueError(
            "Expected exactly 70 model features "
            f"for SHAP, received {X.shape[1]}"
        )

    # --------------------------------------------------------
    # Validate missing values
    # --------------------------------------------------------

    if X.isna().any().any():

        missing = (
            X.columns[
                X.isna().any()
            ]
            .tolist()
        )

        raise ValueError(
            "Missing values found in SHAP input:\n"
            + "\n".join(missing)
        )

    # --------------------------------------------------------
    # TreeExplainer
    # --------------------------------------------------------

    explainer = shap.TreeExplainer(
        shap_model
    )

    shap_values = explainer.shap_values(
        X
    )

    shap_values = np.asarray(
        shap_values
    )

    # --------------------------------------------------------
    # Handle unexpected dimensions
    # --------------------------------------------------------

    if shap_values.ndim == 1:

        shap_values = shap_values.reshape(
            1,
            -1
        )

    if shap_values.shape[1] != 70:

        raise ValueError(
            "Unexpected SHAP output shape: "
            f"{shap_values.shape}"
        )

    # --------------------------------------------------------
    # Base value
    # --------------------------------------------------------

    expected_value = explainer.expected_value

    if isinstance(
        expected_value,
        np.ndarray
    ):

        expected_value = float(
            expected_value.reshape(-1)[0]
        )

    else:

        expected_value = float(
            expected_value
        )

    # --------------------------------------------------------
    # Build explanations
    # --------------------------------------------------------

    explanations = []

    for row_index in range(
        len(X)
    ):

        values = shap_values[
            row_index
        ]

        contributions = []

        for feature, value in zip(
            feature_columns,
            values,
        ):

            contributions.append(
                {
                    "feature": feature,

                    "shap_value": round(
                        float(value),
                        6,
                    ),

                    "abs_shap_value": round(
                        float(abs(value)),
                        6,
                    ),
                }
            )

        # ----------------------------------------------------
        # Most influential features first
        # ----------------------------------------------------

        contributions = sorted(
            contributions,
            key=lambda item:
                item["abs_shap_value"],
            reverse=True,
        )

        explanations.append(
            {
                "row": row_index,

                "base_value": round(
                    expected_value,
                    6,
                ),

                "features": contributions,
            }
        )

    return explanations

# ============================================================
# SHAP FOR ONE FORECAST DAY
# ============================================================

@app.get("/shap/{forecast_day}")
def shap_day(
    forecast_day: str,
):

    # --------------------------------------------------------
    # Validate requested date
    # --------------------------------------------------------

    try:

        requested_date = (
            pd.to_datetime(
                forecast_day
            ).date()
        )

    except Exception:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid date. "
                "Use YYYY-MM-DD."
            ),
        )

    # --------------------------------------------------------
    # Load forecast features
    # --------------------------------------------------------

    try:

        features = load_forecast_features()

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    # --------------------------------------------------------
    # Make absolutely sure local forecast day exists
    # --------------------------------------------------------

    if "forecast_day" not in features.columns:

        features["local_timestamp"] = (
            features["timestamp"]
            .dt.tz_convert(TIMEZONE)
        )

        features["forecast_day"] = (
            features["local_timestamp"]
            .dt.date
        )

    # --------------------------------------------------------
    # Select requested LOCAL calendar day
    # --------------------------------------------------------

    day_features = (
        features[
            features["forecast_day"]
            == requested_date
        ]
        .copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # No data
    # --------------------------------------------------------

    if day_features.empty:

        available_days = sorted(
            features["forecast_day"]
            .dropna()
            .unique()
            .tolist()
        )

        raise HTTPException(
            status_code=404,
            detail={
                "message": (
                    f"No forecast features found "
                    f"for {forecast_day}."
                ),
                "requested_date": str(
                    requested_date
                ),
                "available_dates": [
                    str(day)
                    for day in available_days
                ],
            },
        )

    # --------------------------------------------------------
    # EXACTLY 24 HOURS REQUIRED
    # --------------------------------------------------------

    if len(day_features) != 24:

        local_times = (
            day_features["local_timestamp"]
            .dt.strftime(
                "%Y-%m-%d %H:%M:%S %Z"
            )
            .tolist()
        )

        raise HTTPException(
            status_code=500,
            detail={
                "message": (
                    f"Expected 24 forecast rows "
                    f"for {forecast_day}, "
                    f"found {len(day_features)}."
                ),
                "requested_date": str(
                    requested_date
                ),
                "rows_found": len(
                    day_features
                ),
                "timestamps": local_times,
            },
        )

    # --------------------------------------------------------
    # Validate hourly continuity
    # --------------------------------------------------------

    timestamp_diffs = (
        day_features["timestamp"]
        .diff()
        .dropna()
    )

    expected_delta = pd.Timedelta(
        hours=1
    )

    if not (
        timestamp_diffs
        == expected_delta
    ).all():

        raise HTTPException(
            status_code=500,
            detail=(
                f"Forecast rows for {forecast_day} "
                "are not hourly-continuous."
            ),
        )

    # --------------------------------------------------------
    # Validate local-day boundaries
    # --------------------------------------------------------

    first_local = (
        day_features.iloc[0]
        ["local_timestamp"]
    )

    last_local = (
        day_features.iloc[-1]
        ["local_timestamp"]
    )

    expected_start = pd.Timestamp(
        requested_date,
        tz=TIMEZONE,
    )

    expected_end = (
        expected_start
        + pd.Timedelta(hours=23)
    )

    if first_local != expected_start:

        raise HTTPException(
            status_code=500,
            detail={
                "message": (
                    "Forecast day does not start "
                    "at local midnight."
                ),
                "expected_start": (
                    expected_start.isoformat()
                ),
                "actual_start": (
                    first_local.isoformat()
                ),
            },
        )

    if last_local != expected_end:

        raise HTTPException(
            status_code=500,
            detail={
                "message": (
                    "Forecast day does not end "
                    "at local 23:00."
                ),
                "expected_end": (
                    expected_end.isoformat()
                ),
                "actual_end": (
                    last_local.isoformat()
                ),
            },
        )

    # --------------------------------------------------------
    # Load native XGBoost model
    # --------------------------------------------------------

    try:

        native_model = (
            load_native_xgb_model()
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    # --------------------------------------------------------
    # Calculate SHAP
    # --------------------------------------------------------

    try:

        explanations = (
            calculate_shap_explanation(
                shap_model=native_model,
                forecast_features=day_features,
                feature_columns=FEATURE_COLUMNS,
            )
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to calculate SHAP "
                f"explanations: {exc}"
            ),
        )

    # --------------------------------------------------------
    # Build response
    # --------------------------------------------------------

    results = []

    for index, explanation in enumerate(explanations):

        timestamp = day_features.iloc[index]["timestamp"]

        # Each forecast hour gets its OWN most influential feature
        top_feature = explanation["features"][0]

        results.append(
            {
                "timestamp": timestamp.isoformat(),

                "base_value": explanation["base_value"],

                "top_feature": top_feature["feature"],

                "top_shap_value": top_feature["shap_value"],

                "top_abs_shap_value": top_feature["abs_shap_value"],

                "features": explanation["features"],
            }
        )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {

        "date": str(
            requested_date
        ),

        "hours": len(
            results
        ),

        "model": (
            "Pearls_AQI_XGBoost"
        ),

        "model_alias": "champion",

        "explanations": results,
    }


# ============================================================
# COMPLETE DASHBOARD RESPONSE
# ============================================================

@app.get("/dashboard")
def dashboard():

    try:

        current = current_aqi()

        trend = current_aqi_trend()

        forecasts = forecast()

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    return {

        "current": current,

        "current_trend":
            trend["hours"],

        "forecast": forecasts,
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {

        "status": "healthy",

        "service":
            "Pearls AQI Predictor API",

        "location":
            LOCATION_ID,

        "forecast_hours":
            72,

        "forecast_days":
            3,

        "forecast_features_exists":
            FORECAST_FEATURES_FILE.exists(),

        "hourly_forecast_exists":
            HOURLY_FORECAST_FILE.exists(),

        "daily_forecast_exists":
            DAILY_FORECAST_FILE.exists(),

        "timestamp":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }