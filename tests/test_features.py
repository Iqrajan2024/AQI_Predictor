import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "aqi_features.parquet"
)


# ============================================================
# EXPECTED MODEL FEATURES
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
# TEST 1 — FEATURE FILE EXISTS
# ============================================================

def test_feature_file_exists():
    """Verify that the feature pipeline created the Parquet file."""

    assert FEATURE_FILE.exists(), (
        f"Feature file not found: {FEATURE_FILE}"
    )


# ============================================================
# TEST 2 — DATASET STRUCTURE
# ============================================================

def test_feature_dataset_structure():
    """Verify the generated feature dataset has the expected structure."""

    df = pd.read_parquet(FEATURE_FILE)

    # Exactly 70 model features
    assert len(MODEL_FEATURES) == 70

    # Required columns
    assert "timestamp" in df.columns
    assert "us_aqi" in df.columns
    assert "target_aqi" in df.columns

    # Every expected model feature exists
    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in df.columns
    ]

    assert not missing_features, (
        f"Missing model features: {missing_features}"
    )


# ============================================================
# TEST 3 — FEATURE VALUES
# ============================================================

def test_feature_values_have_no_missing_values():
    """Verify that model features contain no missing values."""

    df = pd.read_parquet(FEATURE_FILE)

    missing_values = (
        df[MODEL_FEATURES]
        .isna()
        .sum()
        .sum()
    )

    assert missing_values == 0, (
        f"Found {missing_values} missing feature values"
    )


# ============================================================
# TEST 4 — TARGET
# ============================================================

def test_target_has_no_missing_values():
    """Verify that the prediction target contains no missing values."""

    df = pd.read_parquet(FEATURE_FILE)

    assert "target_aqi" in df.columns

    assert df["target_aqi"].isna().sum() == 0


# ============================================================
# TEST 5 — TIMESTAMP ORDER
# ============================================================

def test_timestamp_is_chronological():
    """Verify that timestamps are chronological."""

    df = pd.read_parquet(FEATURE_FILE)

    timestamps = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    assert timestamps.is_monotonic_increasing