from pathlib import Path

from feast import FeatureStore


FEATURE_REPO = Path(__file__).resolve().parent


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
    "us_aqi",
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


def test_feature_store_config_exists():
    config_file = FEATURE_REPO / "feature_store.yaml"

    assert config_file.exists(), (
        f"Feast configuration not found: {config_file}"
    )


def test_feature_store_loads():
    store = FeatureStore(
        repo_path=str(FEATURE_REPO)
    )

    assert store is not None


def test_us_aqi_and_target_are_registered():
    store = FeatureStore(
        repo_path=str(FEATURE_REPO)
    )

    view = store.get_feature_view("aqi_features")

    feature_names = {
        field.name
        for field in view.schema
    }

    assert "us_aqi" in feature_names
    assert "target_aqi" in feature_names


def test_exactly_70_model_features():
    assert len(MODEL_FEATURES) == 70

    assert "us_aqi" in MODEL_FEATURES
    assert "target_aqi" not in MODEL_FEATURES