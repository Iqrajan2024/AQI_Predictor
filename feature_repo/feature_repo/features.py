from datetime import timedelta

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
    String,
)

# ============================================================
# ENTITY
# ============================================================

location_id = Entity(
    name="location_id",
    description="Single AQI monitoring location",
)


# ============================================================
# SOURCE
# ============================================================

aqi_features_source = FileSource(
    name="aqi_features_source",
    path=r"D:\Internship\pearls-aqi-predictor\data\processed\aqi_features.parquet",
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