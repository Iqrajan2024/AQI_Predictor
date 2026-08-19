import pandas as pd
from pathlib import Path


# ============================================================
# TEST: PROCESSED FEATURE DATASET
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "aqi_features.parquet"
)


def test_feature_file_exists():
    """Verify that the processed feature dataset exists."""

    assert FEATURE_FILE.exists(), (
        f"Feature file not found: {FEATURE_FILE}"
    )


def test_feature_dataset_structure():
    """Verify the processed dataset has the expected structure."""

    df = pd.read_parquet(FEATURE_FILE)

    # The model uses exactly 70 features
    expected_feature_count = 70

    # Target column
    assert "target_aqi" in df.columns

    # Current AQI
    assert "us_aqi" in df.columns

    # Timestamp
    assert "timestamp" in df.columns

    # Verify exactly 70 model features
    model_features = [
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

    assert len(model_features) == expected_feature_count

    # Every model feature must exist
    missing_features = [
        feature
        for feature in model_features
        if feature not in df.columns
    ]

    assert not missing_features, (
        f"Missing model features: {missing_features}"
    )


def test_feature_values_have_no_missing_values():
    """Verify that the 70 model features contain no missing values."""

    df = pd.read_parquet(FEATURE_FILE)

    model_features = [
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

    assert len(model_features) == 70

    missing_values = df[model_features].isna().sum().sum()

    assert missing_values == 0, (
        f"Found {missing_values} missing feature values"
    )

def test_target_has_no_missing_values():
    """Verify that the AQI prediction target has no missing values."""

    df = pd.read_parquet(FEATURE_FILE)

    assert df["target_aqi"].isna().sum() == 0


def test_timestamp_is_chronological():
    """Verify that timestamps are sorted chronologically."""

    df = pd.read_parquet(FEATURE_FILE)

    timestamps = pd.to_datetime(df["timestamp"], utc=True)

    assert timestamps.is_monotonic_increasing