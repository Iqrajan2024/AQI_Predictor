from datetime import timedelta

from feast import Entity, Feature, FeatureView, FileSource, FeatureService
from feast.types import Float32, Int64, String


# ============================================================
# ENTITY
# ============================================================

aqi_location = Entity(
    name="aqi_location",
    description="Single AQI monitoring location",
    value_type=String,
)


# ============================================================
# SOURCE
# ============================================================

aqi_features_source = FileSource(
    name="aqi_features_source",
    path="../../data/processed/aqi_features.parquet",
    timestamp_field="timestamp",
)


# ============================================================
# 70 MODEL FEATURES + RECURSIVE STATE + TRAINING TARGET
# ============================================================

aqi_features = FeatureView(
    name="aqi_features",
    entities=[aqi_location],
    ttl=timedelta(days=30),
    online=True,
    source=aqi_features_source,

    schema=[
        # ----------------------------------------------------
        # WEATHER
        # ----------------------------------------------------

        Feature(name="temperature_2m", dtype=Float32),
        Feature(name="relative_humidity_2m", dtype=Float32),
        Feature(name="pressure_msl", dtype=Float32),
        Feature(name="precipitation", dtype=Float32),
        Feature(name="wind_speed_10m", dtype=Float32),
        Feature(name="wind_direction_10m", dtype=Float32),

        # ----------------------------------------------------
        # POLLUTANTS
        # ----------------------------------------------------

        Feature(name="pm2_5", dtype=Float32),
        Feature(name="pm10", dtype=Float32),
        Feature(name="carbon_monoxide", dtype=Float32),
        Feature(name="nitrogen_dioxide", dtype=Float32),
        Feature(name="sulphur_dioxide", dtype=Float32),
        Feature(name="ozone", dtype=Float32),

        # ----------------------------------------------------
        # TIME
        # ----------------------------------------------------

        Feature(name="hour", dtype=Int64),
        Feature(name="day_of_week", dtype=Int64),
        Feature(name="day_of_month", dtype=Int64),
        Feature(name="month", dtype=Int64),
        Feature(name="is_weekend", dtype=Int64),

        # ----------------------------------------------------
        # AQI LAGS
        # ----------------------------------------------------

        Feature(name="aqi_lag_1", dtype=Float32),
        Feature(name="aqi_lag_3", dtype=Float32),
        Feature(name="aqi_lag_6", dtype=Float32),
        Feature(name="aqi_lag_12", dtype=Float32),
        Feature(name="aqi_lag_24", dtype=Float32),
        Feature(name="aqi_lag_48", dtype=Float32),
        Feature(name="aqi_lag_72", dtype=Float32),

        # ----------------------------------------------------
        # PM2.5 LAGS
        # ----------------------------------------------------

        Feature(name="pm2_5_lag_1", dtype=Float32),
        Feature(name="pm2_5_lag_3", dtype=Float32),
        Feature(name="pm2_5_lag_6", dtype=Float32),
        Feature(name="pm2_5_lag_24", dtype=Float32),

        # ----------------------------------------------------
        # PM10 LAGS
        # ----------------------------------------------------

        Feature(name="pm10_lag_1", dtype=Float32),
        Feature(name="pm10_lag_3", dtype=Float32),
        Feature(name="pm10_lag_6", dtype=Float32),
        Feature(name="pm10_lag_24", dtype=Float32),

        # ----------------------------------------------------
        # CO LAGS
        # ----------------------------------------------------

        Feature(name="carbon_monoxide_lag_1", dtype=Float32),
        Feature(name="carbon_monoxide_lag_3", dtype=Float32),
        Feature(name="carbon_monoxide_lag_6", dtype=Float32),
        Feature(name="carbon_monoxide_lag_24", dtype=Float32),

        # ----------------------------------------------------
        # NO2 LAGS
        # ----------------------------------------------------

        Feature(name="nitrogen_dioxide_lag_1", dtype=Float32),
        Feature(name="nitrogen_dioxide_lag_3", dtype=Float32),
        Feature(name="nitrogen_dioxide_lag_6", dtype=Float32),
        Feature(name="nitrogen_dioxide_lag_24", dtype=Float32),

        # ----------------------------------------------------
        # SO2 LAGS
        # ----------------------------------------------------

        Feature(name="sulphur_dioxide_lag_1", dtype=Float32),
        Feature(name="sulphur_dioxide_lag_3", dtype=Float32),
        Feature(name="sulphur_dioxide_lag_6", dtype=Float32),
        Feature(name="sulphur_dioxide_lag_24", dtype=Float32),

        # ----------------------------------------------------
        # OZONE LAGS
        # ----------------------------------------------------

        Feature(name="ozone_lag_1", dtype=Float32),
        Feature(name="ozone_lag_3", dtype=Float32),
        Feature(name="ozone_lag_6", dtype=Float32),
        Feature(name="ozone_lag_24", dtype=Float32),

        # ----------------------------------------------------
        # AQI ROLLING MEANS
        # ----------------------------------------------------

        Feature(name="aqi_3h_mean", dtype=Float32),
        Feature(name="aqi_6h_mean", dtype=Float32),
        Feature(name="aqi_12h_mean", dtype=Float32),
        Feature(name="aqi_24h_mean", dtype=Float32),

        # ----------------------------------------------------
        # PM2.5 ROLLING MEANS
        # ----------------------------------------------------

        Feature(name="pm2_5_3h_mean", dtype=Float32),
        Feature(name="pm2_5_6h_mean", dtype=Float32),
        Feature(name="pm2_5_24h_mean", dtype=Float32),

        # ----------------------------------------------------
        # PM10 ROLLING MEANS
        # ----------------------------------------------------

        Feature(name="pm10_3h_mean", dtype=Float32),
        Feature(name="pm10_6h_mean", dtype=Float32),
        Feature(name="pm10_24h_mean", dtype=Float32),

        # ----------------------------------------------------
        # OTHER POLLUTANT ROLLING MEANS
        # ----------------------------------------------------

        Feature(name="carbon_monoxide_24h_mean", dtype=Float32),
        Feature(name="nitrogen_dioxide_24h_mean", dtype=Float32),
        Feature(name="sulphur_dioxide_24h_mean", dtype=Float32),
        Feature(name="ozone_24h_mean", dtype=Float32),

        # ----------------------------------------------------
        # AQI CHANGES
        # ----------------------------------------------------

        Feature(name="aqi_change_1h", dtype=Float32),
        Feature(name="aqi_change_3h", dtype=Float32),
        Feature(name="aqi_change_6h", dtype=Float32),
        Feature(name="aqi_change_24h", dtype=Float32),

        # ----------------------------------------------------
        # PM2.5 CHANGES
        # ----------------------------------------------------

        Feature(name="pm2_5_change_1h", dtype=Float32),
        Feature(name="pm2_5_change_24h", dtype=Float32),

        # ----------------------------------------------------
        # PM10 CHANGES
        # ----------------------------------------------------

        Feature(name="pm10_change_1h", dtype=Float32),
        Feature(name="pm10_change_24h", dtype=Float32),

        # ----------------------------------------------------
        # RECURSIVE AQI STATE
        # NOT A MODEL INPUT
        # ----------------------------------------------------

        Feature(name="us_aqi", dtype=Float32),

        # ----------------------------------------------------
        # TRAINING TARGET
        # NOT A MODEL INPUT
        # NOT A PREDICTION CONTEXT FEATURE
        # ----------------------------------------------------

        Feature(name="target_aqi", dtype=Float32),
    ],
)


# ============================================================
# MODEL FEATURE SERVICE
# EXACTLY 70 MODEL FEATURES
# ============================================================

aqi_model_features = FeatureService(
    name="aqi_model_features",
    features=[
        aqi_features,
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

aqi_prediction_context = FeatureService(
    name="aqi_prediction_context",
    features=[
        aqi_features[
            [
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
                "us_aqi",
            ]
        ],
    ],
)