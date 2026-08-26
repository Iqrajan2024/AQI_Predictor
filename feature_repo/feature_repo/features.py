from datetime import timedelta

from feast import Entity, FeatureView, Field
from feast.types import Float32, Int64, String
from feast.value_type import ValueType
from feast.infra.offline_stores.file_source import FileSource


# ============================================================
# ENTITY
# ============================================================

aqi_entity = Entity(
    name="aqi_location",
    join_keys=["location_id"],
    value_type=ValueType.STRING,
    description="Single AQI monitoring location"
)


# ============================================================
# DATA SOURCE
# ============================================================

aqi_source = FileSource(
    name="aqi_features_source",
    path="../../data/processed/aqi_features.parquet",
    timestamp_field="timestamp",

)


# ============================================================
# FEATURE VIEW
# ============================================================

aqi_features = FeatureView(
    name="aqi_features",
    entities=[aqi_entity],
    ttl=timedelta(days=3650),
    schema=[
        
        # Weather
        Field(name="temperature_2m", dtype=Float32),
        Field(name="relative_humidity_2m", dtype=Int64),
        Field(name="pressure_msl", dtype=Float32),
        Field(name="precipitation", dtype=Float32),
        Field(name="wind_speed_10m", dtype=Float32),
        Field(name="wind_direction_10m", dtype=Int64),

        # Current pollution
        Field(name="pm2_5", dtype=Float32),
        Field(name="pm10", dtype=Float32),
        Field(name="carbon_monoxide", dtype=Float32),
        Field(name="nitrogen_dioxide", dtype=Float32),
        Field(name="sulphur_dioxide", dtype=Float32),
        Field(name="ozone", dtype=Float32),

        Field(name="us_aqi", dtype=Float32),


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

        # Changes
        Field(name="aqi_change_1h", dtype=Float32),
        Field(name="aqi_change_3h", dtype=Float32),
        Field(name="aqi_change_6h", dtype=Float32),
        Field(name="aqi_change_24h", dtype=Float32),

        Field(name="pm2_5_change_1h", dtype=Float32),
        Field(name="pm2_5_change_24h", dtype=Float32),

        Field(name="pm10_change_1h", dtype=Float32),
        Field(name="pm10_change_24h", dtype=Float32),

        Field(name="target_aqi", dtype=Float32),
    ],
    source=aqi_source,
)