from datetime import timedelta
from pathlib import Path

from feast import (
    Entity,
    FeatureView,
    Field,
    FileSource,
    FeatureService,
)

from feast.types import (
    Int32,
    Float64,
)

from feast.value_type import ValueType

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "aqi_features.parquet"
)


# ============================================================
# ENTITY
# ============================================================

location_id = Entity(
    name="location_id",
    value_type= ValueType.STRING,
    description="Single AQI monitoring location",
)


# ============================================================
# SOURCE
# ============================================================

aqi_features_source = FileSource(
    name="aqi_features_source",
    path=str(FEATURE_FILE),
    timestamp_field="timestamp",
)


# ============================================================
# 70 MODEL FEATURES + RECURSIVE STATE + TRAINING TARGET
# ============================================================

aqi_features = FeatureView(
    name="aqi_features",
    entities=[location_id],
    ttl=timedelta(days=30),
    online=True,
    source=aqi_features_source,

    schema=[
        # ----------------------------------------------------
        # WEATHER
        # ----------------------------------------------------

        Field(name="temperature_2m", dtype=Float64),
        Field(name="relative_humidity_2m", dtype=Float64),
        Field(name="pressure_msl", dtype=Float64),
        Field(name="precipitation", dtype=Float64),
        Field(name="wind_speed_10m", dtype=Float64),
        Field(name="wind_direction_10m", dtype=Float64),

        # ----------------------------------------------------
        # POLLUTANTS
        # ----------------------------------------------------

        Field(name="pm2_5", dtype=Float64),
        Field(name="pm10", dtype=Float64),
        Field(name="carbon_monoxide", dtype=Float64),
        Field(name="nitrogen_dioxide", dtype=Float64),
        Field(name="sulphur_dioxide", dtype=Float64),
        Field(name="ozone", dtype=Float64),

        # ----------------------------------------------------
        # TIME
        # ----------------------------------------------------

        Field(name="hour", dtype=Int32),
        Field(name="day_of_week", dtype=Int32),
        Field(name="day_of_month", dtype=Int32),
        Field(name="month", dtype=Int32),
        Field(name="is_weekend", dtype=Int32),

        # ----------------------------------------------------
        # AQI LAGS
        # ----------------------------------------------------

        Field(name="aqi_lag_1", dtype=Float64),
        Field(name="aqi_lag_3", dtype=Float64),
        Field(name="aqi_lag_6", dtype=Float64),
        Field(name="aqi_lag_12", dtype=Float64),
        Field(name="aqi_lag_24", dtype=Float64),
        Field(name="aqi_lag_48", dtype=Float64),
        Field(name="aqi_lag_72", dtype=Float64),

        # ----------------------------------------------------
        # PM2.5 LAGS
        # ----------------------------------------------------

        Field(name="pm2_5_lag_1", dtype=Float64),
        Field(name="pm2_5_lag_3", dtype=Float64),
        Field(name="pm2_5_lag_6", dtype=Float64),
        Field(name="pm2_5_lag_24", dtype=Float64),

        # ----------------------------------------------------
        # PM10 LAGS
        # ----------------------------------------------------

        Field(name="pm10_lag_1", dtype=Float64),
        Field(name="pm10_lag_3", dtype=Float64),
        Field(name="pm10_lag_6", dtype=Float64),
        Field(name="pm10_lag_24", dtype=Float64),

        # ----------------------------------------------------
        # CO LAGS
        # ----------------------------------------------------

        Field(name="carbon_monoxide_lag_1", dtype=Float64),
        Field(name="carbon_monoxide_lag_3", dtype=Float64),
        Field(name="carbon_monoxide_lag_6", dtype=Float64),
        Field(name="carbon_monoxide_lag_24", dtype=Float64),

        # ----------------------------------------------------
        # NO2 LAGS
        # ----------------------------------------------------

        Field(name="nitrogen_dioxide_lag_1", dtype=Float64),
        Field(name="nitrogen_dioxide_lag_3", dtype=Float64),
        Field(name="nitrogen_dioxide_lag_6", dtype=Float64),
        Field(name="nitrogen_dioxide_lag_24", dtype=Float64),

        # ----------------------------------------------------
        # SO2 LAGS
        # ----------------------------------------------------

        Field(name="sulphur_dioxide_lag_1", dtype=Float64),
        Field(name="sulphur_dioxide_lag_3", dtype=Float64),
        Field(name="sulphur_dioxide_lag_6", dtype=Float64),
        Field(name="sulphur_dioxide_lag_24", dtype=Float64),

        # ----------------------------------------------------
        # OZONE LAGS
        # ----------------------------------------------------

        Field(name="ozone_lag_1", dtype=Float64),
        Field(name="ozone_lag_3", dtype=Float64),
        Field(name="ozone_lag_6", dtype=Float64),
        Field(name="ozone_lag_24", dtype=Float64),

        # ----------------------------------------------------
        # AQI ROLLING MEANS
        # ----------------------------------------------------

        Field(name="aqi_3h_mean", dtype=Float64),
        Field(name="aqi_6h_mean", dtype=Float64),
        Field(name="aqi_12h_mean", dtype=Float64),
        Field(name="aqi_24h_mean", dtype=Float64),

        # ----------------------------------------------------
        # PM2.5 ROLLING MEANS
        # ----------------------------------------------------

        Field(name="pm2_5_3h_mean", dtype=Float64),
        Field(name="pm2_5_6h_mean", dtype=Float64),
        Field(name="pm2_5_24h_mean", dtype=Float64),

        # ----------------------------------------------------
        # PM10 ROLLING MEANS
        # ----------------------------------------------------

        Field(name="pm10_3h_mean", dtype=Float64),
        Field(name="pm10_6h_mean", dtype=Float64),
        Field(name="pm10_24h_mean", dtype=Float64),

        # ----------------------------------------------------
        # OTHER POLLUTANT ROLLING MEANS
        # ----------------------------------------------------

        Field(name="carbon_monoxide_24h_mean", dtype=Float64),
        Field(name="nitrogen_dioxide_24h_mean", dtype=Float64),
        Field(name="sulphur_dioxide_24h_mean", dtype=Float64),
        Field(name="ozone_24h_mean", dtype=Float64),

        # ----------------------------------------------------
        # AQI CHANGES
        # ----------------------------------------------------

        Field(name="aqi_change_1h", dtype=Float64),
        Field(name="aqi_change_3h", dtype=Float64),
        Field(name="aqi_change_6h", dtype=Float64),
        Field(name="aqi_change_24h", dtype=Float64),

        # ----------------------------------------------------
        # PM2.5 CHANGES
        # ----------------------------------------------------

        Field(name="pm2_5_change_1h", dtype=Float64),
        Field(name="pm2_5_change_24h", dtype=Float64),

        # ----------------------------------------------------
        # PM10 CHANGES
        # ----------------------------------------------------

        Field(name="pm10_change_1h", dtype=Float64),
        Field(name="pm10_change_24h", dtype=Float64),

        # ----------------------------------------------------
        # RECURSIVE AQI STATE
        # NOT A MODEL INPUT
        # ----------------------------------------------------

        Field(name="us_aqi", dtype=Float64),

        # ----------------------------------------------------
        # TRAINING TARGET
        # NOT A MODEL INPUT
        # NOT A PREDICTION CONTEXT FEATURE
        # ----------------------------------------------------

        Field(name="target_aqi", dtype=Float64),
    ],
)

# ============================================================
# MODEL FEATURE SERVICE
# EXACTLY 70 MODEL FEATURES
#
# Excludes:
#   - us_aqi      -> recursive prediction state
#   - target_aqi  -> training target
# ============================================================

MODEL_FEATURES = [
    # WEATHER
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",

    # POLLUTANTS
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",

    # TIME
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",

    # AQI LAGS
    "aqi_lag_1",
    "aqi_lag_3",
    "aqi_lag_6",
    "aqi_lag_12",
    "aqi_lag_24",
    "aqi_lag_48",
    "aqi_lag_72",

    # PM2.5 LAGS
    "pm2_5_lag_1",
    "pm2_5_lag_3",
    "pm2_5_lag_6",
    "pm2_5_lag_24",

    # PM10 LAGS
    "pm10_lag_1",
    "pm10_lag_3",
    "pm10_lag_6",
    "pm10_lag_24",

    # CO LAGS
    "carbon_monoxide_lag_1",
    "carbon_monoxide_lag_3",
    "carbon_monoxide_lag_6",
    "carbon_monoxide_lag_24",

    # NO2 LAGS
    "nitrogen_dioxide_lag_1",
    "nitrogen_dioxide_lag_3",
    "nitrogen_dioxide_lag_6",
    "nitrogen_dioxide_lag_24",

    # SO2 LAGS
    "sulphur_dioxide_lag_1",
    "sulphur_dioxide_lag_3",
    "sulphur_dioxide_lag_6",
    "sulphur_dioxide_lag_24",

    # OZONE LAGS
    "ozone_lag_1",
    "ozone_lag_3",
    "ozone_lag_6",
    "ozone_lag_24",

    # AQI ROLLING MEANS
    "aqi_3h_mean",
    "aqi_6h_mean",
    "aqi_12h_mean",
    "aqi_24h_mean",

    # PM2.5 ROLLING MEANS
    "pm2_5_3h_mean",
    "pm2_5_6h_mean",
    "pm2_5_24h_mean",

    # PM10 ROLLING MEANS
    "pm10_3h_mean",
    "pm10_6h_mean",
    "pm10_24h_mean",

    # OTHER POLLUTANT ROLLING MEANS
    "carbon_monoxide_24h_mean",
    "nitrogen_dioxide_24h_mean",
    "sulphur_dioxide_24h_mean",
    "ozone_24h_mean",

    # AQI CHANGES
    "aqi_change_1h",
    "aqi_change_3h",
    "aqi_change_6h",
    "aqi_change_24h",

    # PM2.5 CHANGES
    "pm2_5_change_1h",
    "pm2_5_change_24h",

    # PM10 CHANGES
    "pm10_change_1h",
    "pm10_change_24h",
]


# ============================================================
# VALIDATE MODEL FEATURE COUNT
# ============================================================

assert len(MODEL_FEATURES) == 70, (
    f"MODEL_FEATURES contains {len(MODEL_FEATURES)} features; "
    "expected exactly 70"
)

assert "us_aqi" not in MODEL_FEATURES
assert "target_aqi" not in MODEL_FEATURES


# ============================================================
# MODEL FEATURE SERVICE
# EXACTLY 70 FEATURES
# ============================================================

aqi_model_features = FeatureService(
    name="aqi_model_features",
    features=[
        aqi_features[MODEL_FEATURES],
    ],
)


# ============================================================
# PREDICTION CONTEXT SERVICE
#
# 70 MODEL FEATURES
# + us_aqi
# = 71 CONTEXT FEATURES
#
# target_aqi IS NOT INCLUDED.
# ============================================================

PREDICTION_CONTEXT_FEATURES = MODEL_FEATURES + [
    "us_aqi",
]


assert len(PREDICTION_CONTEXT_FEATURES) == 71

assert "target_aqi" not in PREDICTION_CONTEXT_FEATURES


aqi_prediction_context = FeatureService(
    name="aqi_prediction_context",
    features=[
        aqi_features[PREDICTION_CONTEXT_FEATURES],
    ],
)



