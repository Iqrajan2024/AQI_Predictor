from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "aqi_features.parquet"
)


# ============================================================
# CONFIGURATION
# ============================================================

N_ROWS = 100


# ============================================================
# EXACT 70 MODEL FEATURES
# ============================================================

MODEL_FEATURES = [
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


# ============================================================
# VALIDATE FEATURE COUNT
# ============================================================

assert len(MODEL_FEATURES) == 70


# ============================================================
# CREATE SYNTHETIC DATA
# ============================================================

rng = np.random.default_rng(42)

timestamps = pd.date_range(
    start="2026-01-01 00:00:00",
    periods=N_ROWS,
    freq="h",
    tz="UTC",
)

df = pd.DataFrame(
    {
        "timestamp": timestamps,
        "us_aqi": rng.uniform(20, 200, N_ROWS),
        "target_aqi": rng.uniform(20, 200, N_ROWS),
    }
)


# ============================================================
# CREATE ALL 70 FEATURES
# ============================================================

for feature in MODEL_FEATURES:

    if feature == "hour":
        df[feature] = df["timestamp"].dt.hour

    elif feature == "day_of_week":
        df[feature] = df["timestamp"].dt.dayofweek

    elif feature == "day_of_month":
        df[feature] = df["timestamp"].dt.day

    elif feature == "month":
        df[feature] = df["timestamp"].dt.month

    elif feature == "is_weekend":
        df[feature] = (
            df["timestamp"].dt.dayofweek >= 5
        ).astype(int)

    else:
        df[feature] = rng.uniform(0, 100, N_ROWS)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_parquet(
    OUTPUT_FILE,
    index=False,
)

print("=" * 60)
print("CI TEST FEATURE DATASET CREATED")
print("=" * 60)
print("Output:", OUTPUT_FILE)
print("Rows:", len(df))
print("Model features:", len(MODEL_FEATURES))
print("Total columns:", len(df.columns))
print("Missing values:", int(df.isna().sum().sum()))