"""
============================================================
PEARLS AQI PREDICTOR
FEATURE ENGINEERING PIPELINE
============================================================

Builds the same 70 model features used by the trained
XGBoost champion model.

Input:
    data/raw/peshawar_openmeteo_2024_2026.csv

Output:
    data/processed/aqi_features.parquet
"""

from pathlib import Path
import sys

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "peshawar_openmeteo_2024_2026.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "aqi_features.parquet"
)


# ============================================================
# CHAMPION MODEL FEATURES
# ============================================================

FEATURE_COLUMNS = [
    # Weather
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",

    # Pollutants
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

    # Pollutant lags
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

    # AQI rolling means
    "aqi_3h_mean",
    "aqi_6h_mean",
    "aqi_12h_mean",
    "aqi_24h_mean",

    # PM2.5 rolling means
    "pm2_5_3h_mean",
    "pm2_5_6h_mean",
    "pm2_5_24h_mean",

    # PM10 rolling means
    "pm10_3h_mean",
    "pm10_6h_mean",
    "pm10_24h_mean",

    # Other pollutant rolling means
    "carbon_monoxide_24h_mean",
    "nitrogen_dioxide_24h_mean",
    "sulphur_dioxide_24h_mean",
    "ozone_24h_mean",

    # Change features
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
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("PEARLS AQI PREDICTOR")
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python executable: {sys.executable}")

    print("\nRaw file:")
    print(RAW_FILE)

    print("\nOutput file:")
    print(OUTPUT_FILE)

    # --------------------------------------------------------
    # CHECK INPUT
    # --------------------------------------------------------

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw dataset not found:\n{RAW_FILE}"
        )

    # --------------------------------------------------------
    # LOAD RAW DATA
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("LOADING RAW DATA")
    print("=" * 60)

    df = pd.read_csv(RAW_FILE)

    print("Raw dataset shape:", df.shape)

    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    if "timestamp" not in df.columns:
        raise ValueError(
            "Required column 'timestamp' is missing."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # REQUIRED BASE COLUMNS
    # --------------------------------------------------------

    required_columns = [
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

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required raw columns: {missing}"
        )

    print("\nRequired raw columns verified.")

    # ========================================================
    # TIME FEATURES
    # ========================================================

    print("\n" + "=" * 60)
    print("CREATING TIME FEATURES")
    print("=" * 60)

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["day_of_month"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    print("✓ Time features created")

    # ========================================================
    # AQI LAG FEATURES
    # ========================================================

    print("\n" + "=" * 60)
    print("CREATING AQI LAG FEATURES")
    print("=" * 60)

    lag_hours = [1, 3, 6, 12, 24, 48, 72]

    for lag in lag_hours:
        df[f"aqi_lag_{lag}"] = (
            df["us_aqi"].shift(lag)
        )

    print("✓ AQI lag features created")

    # ========================================================
    # NEXT-HOUR TARGET
    # ========================================================

    df["target_aqi"] = (
        df["us_aqi"].shift(-1)
    )

    print("✓ Next-hour target created")

    # ========================================================
    # POLLUTANT LAG FEATURES
    # ========================================================

    print("\n" + "=" * 60)
    print("CREATING POLLUTANT LAG FEATURES")
    print("=" * 60)

    pollutants = [
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
    ]

    pollution_lags = [1, 3, 6, 24]

    for pollutant in pollutants:

        for lag in pollution_lags:

            df[f"{pollutant}_lag_{lag}"] = (
                df[pollutant].shift(lag)
            )

    print("✓ Pollutant lag features created")

    # ========================================================
    # ROLLING FEATURES
    # ========================================================

    print("\n" + "=" * 60)
    print("CREATING ROLLING FEATURES")
    print("=" * 60)

    # --------------------------------------------------------
    # AQI rolling means
    # --------------------------------------------------------

    df["aqi_3h_mean"] = (
        df["us_aqi"]
        .shift(1)
        .rolling(window=3)
        .mean()
    )

    df["aqi_6h_mean"] = (
        df["us_aqi"]
        .shift(1)
        .rolling(window=6)
        .mean()
    )

    df["aqi_12h_mean"] = (
        df["us_aqi"]
        .shift(1)
        .rolling(window=12)
        .mean()
    )

    df["aqi_24h_mean"] = (
        df["us_aqi"]
        .shift(1)
        .rolling(window=24)
        .mean()
    )

    # --------------------------------------------------------
    # PM2.5 rolling means
    # --------------------------------------------------------

    df["pm2_5_3h_mean"] = (
        df["pm2_5"]
        .shift(1)
        .rolling(window=3)
        .mean()
    )

    df["pm2_5_6h_mean"] = (
        df["pm2_5"]
        .shift(1)
        .rolling(window=6)
        .mean()
    )

    df["pm2_5_24h_mean"] = (
        df["pm2_5"]
        .shift(1)
        .rolling(window=24)
        .mean()
    )

    # --------------------------------------------------------
    # PM10 rolling means
    # --------------------------------------------------------

    df["pm10_3h_mean"] = (
        df["pm10"]
        .shift(1)
        .rolling(window=3)
        .mean()
    )

    df["pm10_6h_mean"] = (
        df["pm10"]
        .shift(1)
        .rolling(window=6)
        .mean()
    )

    df["pm10_24h_mean"] = (
        df["pm10"]
        .shift(1)
        .rolling(window=24)
        .mean()
    )

    # --------------------------------------------------------
    # Other pollutant 24-hour rolling means
    # --------------------------------------------------------

    rolling_pollutants = [
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
    ]

    for pollutant in rolling_pollutants:

        df[f"{pollutant}_24h_mean"] = (
            df[pollutant]
            .shift(1)
            .rolling(window=24)
            .mean()
        )

    print("✓ Rolling features created")

    # ========================================================
    # CHANGE FEATURES
    # ========================================================

    print("\n" + "=" * 60)
    print("CREATING CHANGE FEATURES")
    print("=" * 60)

    # AQI changes

    df["aqi_change_1h"] = (
        df["us_aqi"] - df["aqi_lag_1"]
    )

    df["aqi_change_3h"] = (
        df["us_aqi"] - df["aqi_lag_3"]
    )

    df["aqi_change_6h"] = (
        df["us_aqi"] - df["aqi_lag_6"]
    )

    df["aqi_change_24h"] = (
        df["us_aqi"] - df["aqi_lag_24"]
    )

    # PM2.5 changes

    df["pm2_5_change_1h"] = (
        df["pm2_5"] - df["pm2_5_lag_1"]
    )

    df["pm2_5_change_24h"] = (
        df["pm2_5"] - df["pm2_5_lag_24"]
    )

    # PM10 changes

    df["pm10_change_1h"] = (
        df["pm10"] - df["pm10_lag_1"]
    )

    df["pm10_change_24h"] = (
        df["pm10"] - df["pm10_lag_24"]
    )

    print("✓ Change features created")

    # ========================================================
    # FEATURE VERIFICATION
    # ========================================================

    print("\n" + "=" * 60)
    print("VERIFYING 70 MODEL FEATURES")
    print("=" * 60)

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing model features:\n"
            + "\n".join(missing_features)
        )

    print("✓ All 70 model features are present")

    # ========================================================
    # REMOVE ROWS WITH UNAVAILABLE FEATURES
    # ========================================================

    before_rows = len(df)

    df = df.dropna(
        subset=FEATURE_COLUMNS + ["target_aqi"]
    ).reset_index(drop=True)

    removed_rows = before_rows - len(df)

    print("\nRows before feature cleanup:", before_rows)
    print("Rows removed:", removed_rows)
    print("Rows after cleanup:", len(df))

    # ========================================================
    # FINAL VERIFICATION
    # ========================================================

    print("\n" + "=" * 60)
    print("FINAL DATASET VERIFICATION")
    print("=" * 60)

    feature_matrix = df[FEATURE_COLUMNS]

    print("Feature matrix shape:", feature_matrix.shape)
    print("Expected features:", len(FEATURE_COLUMNS))

    assert feature_matrix.shape[1] == 70

    missing_values = (
        feature_matrix
        .isna()
        .sum()
        .sum()
    )

    print("Missing feature values:", missing_values)

    assert missing_values == 0

    assert df["target_aqi"].isna().sum() == 0

    assert df["timestamp"].is_monotonic_increasing

    print("✓ 70 features")
    print("✓ No missing feature values")
    print("✓ No missing target values")
    print("✓ Chronologically ordered")

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 60)
    print("FEATURE DATASET SAVED")
    print("=" * 60)

    print("Path:", OUTPUT_FILE)
    print("Shape:", df.shape)
    print(
        "Date range:",
        df["timestamp"].min(),
        "→",
        df["timestamp"].max()
    )

    print("\nFeature columns:", len(FEATURE_COLUMNS))

    print("\n✓ FEATURE PIPELINE COMPLETED SUCCESSFULLY")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()