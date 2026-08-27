from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String
from feast.value_type import ValueType


# ============================================================
# ENTITY
# ============================================================

aqi_entity = Entity(
    name="aqi_location",
    join_keys=["location_id"],
    value_type=ValueType.STRING,
    description="Single AQI monitoring location",
)


# ============================================================
# DATA SOURCE
# ============================================================

aqi_source = FileSource(
    name="aqi_features_source",
    path="D:/Internship/pearls-aqi-predictor/data/processed/aqi_features.parquet",
    timestamp_field="timestamp",
    
)


# ============================================================
# 70 MODEL FEATURES
# ============================================================

MODEL_FEATURES = [

    # Weather
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",

    # Current pollution
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

    # CO lags
    "carbon_monoxide_lag_1",
    "carbon_monoxide_lag_3",
    "carbon_monoxide_lag_6",
    "carbon_monoxide_lag_24",

    # NO2 lags
    "nitrogen_dioxide_lag_1",
    "nitrogen_dioxide_lag_3",
    "nitrogen_dioxide_lag_6",
    "nitrogen_dioxide_lag_24",

    # SO2 lags
    "sulphur_dioxide_lag_1",
    "sulphur_dioxide_lag_3",
    "sulphur_dioxide_lag_6",
    "sulphur_dioxide_lag_24",

    # Ozone lags
    "ozone_lag_1",
    "ozone_lag_3",
    "ozone_lag_6",
    "ozone_lag_24",

    # Rolling AQI
    "aqi_3h_mean",
    "aqi_6h_mean",
    "aqi_12h_mean",
    "aqi_24h_mean",

    # Rolling PM2.5
    "pm2_5_3h_mean",
    "pm2_5_6h_mean",
    "pm2_5_24h_mean",

    # Rolling PM10
    "pm10_3h_mean",
    "pm10_6h_mean",
    "pm10_24h_mean",

    # Rolling gases
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


assert len(MODEL_FEATURES) == 70


# ============================================================
# FEATURE VIEW
# ============================================================

aqi_features = FeatureView(
    name="aqi_features",
    entities=[aqi_entity],
    ttl=timedelta(days=3650),

    schema=[

        # ====================================================
        # MODEL INPUTS
        # ====================================================

        # Weather
        Field(name="temperature_2m", dtype=Float32),
        Field(name="relative_humidity_2m", dtype=Float32),
        Field(name="pressure_msl", dtype=Float32),
        Field(name="precipitation", dtype=Float32),
        Field(name="wind_speed_10m", dtype=Float32),
        Field(name="wind_direction_10m", dtype=Float32),

        # Current pollution
        Field(name="pm2_5", dtype=Float32),
        Field(name="pm10", dtype=Float32),
        Field(name="carbon_monoxide", dtype=Float32),
        Field(name="nitrogen_dioxide", dtype=Float32),
        Field(name="sulphur_dioxide", dtype=Float32),
        Field(name="ozone", dtype=Float32),

        # Time
        Field(name="hour", dtype=Int64),
        Field(name="day_of_week", dtype=Int64),
        Field(name="day_of_month", dtype=Int64),
        Field(name="month", dtype=Int64),
        Field(name="is_weekend", dtype=Int64),

        # AQI lags
        Field(name="aqi_lag_1", dtype=Float32),
        Field(name="aqi_lag_3", dtype=Float32),
        Field(name="aqi_lag_6", dtype=Float32),
        Field(name="aqi_lag_12", dtype=Float32),
        Field(name="aqi_lag_24", dtype=Float32),
        Field(name="aqi_lag_48", dtype=Float32),
        Field(name="aqi_lag_72", dtype=Float32),

        # PM2.5 lags
        Field(name="pm2_5_lag_1", dtype=Float32),
        Field(name="pm2_5_lag_3", dtype=Float32),
        Field(name="pm2_5_lag_6", dtype=Float32),
        Field(name="pm2_5_lag_24", dtype=Float32),

        # PM10 lags
        Field(name="pm10_lag_1", dtype=Float32),
        Field(name="pm10_lag_3", dtype=Float32),
        Field(name="pm10_lag_6", dtype=Float32),
        Field(name="pm10_lag_24", dtype=Float32),

        # CO lags
        Field(name="carbon_monoxide_lag_1", dtype=Float32),
        Field(name="carbon_monoxide_lag_3", dtype=Float32),
        Field(name="carbon_monoxide_lag_6", dtype=Float32),
        Field(name="carbon_monoxide_lag_24", dtype=Float32),

        # NO2 lags
        Field(name="nitrogen_dioxide_lag_1", dtype=Float32),
        Field(name="nitrogen_dioxide_lag_3", dtype=Float32),
        Field(name="nitrogen_dioxide_lag_6", dtype=Float32),
        Field(name="nitrogen_dioxide_lag_24", dtype=Float32),

        # SO2 lags
        Field(name="sulphur_dioxide_lag_1", dtype=Float32),
        Field(name="sulphur_dioxide_lag_3", dtype=Float32),
        Field(name="sulphur_dioxide_lag_6", dtype=Float32),
        Field(name="sulphur_dioxide_lag_24", dtype=Float32),

        # Ozone lags
        Field(name="ozone_lag_1", dtype=Float32),
        Field(name="ozone_lag_3", dtype=Float32),
        Field(name="ozone_lag_6", dtype=Float32),
        Field(name="ozone_lag_24", dtype=Float32),

        # Rolling AQI
        Field(name="aqi_3h_mean", dtype=Float32),
        Field(name="aqi_6h_mean", dtype=Float32),
        Field(name="aqi_12h_mean", dtype=Float32),
        Field(name="aqi_24h_mean", dtype=Float32),

        # Rolling PM2.5
        Field(name="pm2_5_3h_mean", dtype=Float32),
        Field(name="pm2_5_6h_mean", dtype=Float32),
        Field(name="pm2_5_24h_mean", dtype=Float32),

        # Rolling PM10
        Field(name="pm10_3h_mean", dtype=Float32),
        Field(name="pm10_6h_mean", dtype=Float32),
        Field(name="pm10_24h_mean", dtype=Float32),

        # Rolling gases
        Field(name="carbon_monoxide_24h_mean", dtype=Float32),
        Field(name="nitrogen_dioxide_24h_mean", dtype=Float32),
        Field(name="sulphur_dioxide_24h_mean", dtype=Float32),
        Field(name="ozone_24h_mean", dtype=Float32),

        # AQI changes
        Field(name="aqi_change_1h", dtype=Float32),
        Field(name="aqi_change_3h", dtype=Float32),
        Field(name="aqi_change_6h", dtype=Float32),
        Field(name="aqi_change_24h", dtype=Float32),

        # PM2.5 changes
        Field(name="pm2_5_change_1h", dtype=Float32),
        Field(name="pm2_5_change_24h", dtype=Float32),

        # PM10 changes
        Field(name="pm10_change_1h", dtype=Float32),
        Field(name="pm10_change_24h", dtype=Float32),

        
    ],

    source=aqi_source,
)


# ============================================================
# MODEL FEATURE SERVICE
# EXACTLY 70 FEATURES
# ============================================================

aqi_model_features = FeatureService(
    name="aqi_model_features",
    features=[aqi_features],
)