from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from feast import FeatureStore

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_REPO = (
    PROJECT_ROOT
    / "feature_repo"
    / "feature_repo"
)


# ============================================================
# CONSTANTS
# ============================================================

LOCATION_ID = "peshawar"

MODEL_NAME = "Pearls_AQI_XGBoost"

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


assert len(MODEL_FEATURES) == 70
assert "target_aqi" not in MODEL_FEATURES


# ============================================================
# FEAST
# ============================================================

store = FeatureStore(
    repo_path=str(FEATURE_REPO)
)

FEATURE_REFS = [
    f"aqi_features:{feature}"
    for feature in MODEL_FEATURES
]

FEATURE_REFS.append(
    "aqi_features:target_aqi"
)


# ============================================================
# BUILD ENTITY DATAFRAME
# ============================================================

source = store.get_data_source(
    "aqi_features_source"
)

entity_sql = f"""
SELECT
    location_id,
    timestamp AS event_timestamp
FROM {source.get_table_query_string()}
WHERE location_id = '{LOCATION_ID}'
"""


# ============================================================
# HISTORICAL RETRIEVAL THROUGH FEAST
# ============================================================

print("=" * 60)
print("RETRIEVING HISTORICAL TRAINING DATA FROM FEAST")
print("=" * 60)

historical = (
    store
    .get_historical_features(
        entity_df=entity_sql,
        features=FEATURE_REFS,
    )
    .to_df()
)


historical.columns = [
    column.split(":")[-1]
    for column in historical.columns
]


historical["timestamp"] = pd.to_datetime(
    historical["event_timestamp"],
    utc=True,
)

historical = (
    historical
    .sort_values("timestamp")
    .reset_index(drop=True)
)


required = (
    MODEL_FEATURES
    + [
        "target_aqi",
        "timestamp",
    ]
)


missing = [
    column
    for column in required
    if column not in historical.columns
]

if missing:
    raise ValueError(
        f"Feast historical retrieval missing: {missing}"
    )


historical = historical.dropna(
    subset=required
).reset_index(drop=True)


if historical.empty:
    raise ValueError(
        "Feast returned no historical training data."
    )


print("FEAST TRAINING DATA READY")
print("Rows:", len(historical))
print("70 model features:", len(MODEL_FEATURES))
print("Target: target_aqi")
print("First:", historical["timestamp"].min())
print("Last:", historical["timestamp"].max())


# ============================================================
# TRAIN / VALIDATION / TEST
# ============================================================

X = historical[MODEL_FEATURES]
y = historical["target_aqi"]

n = len(historical)

train_end = int(n * 0.70)
validation_end = int(n * 0.85)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_validation = X.iloc[
    train_end:validation_end
]
y_validation = y.iloc[
    train_end:validation_end
]

X_test = X.iloc[validation_end:]
y_test = y.iloc[validation_end:]


# ============================================================
# MODELS
# ============================================================

models = {
    "Ridge": Ridge(alpha=1.0),

    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    ),

    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=2,
    ),
}


# ============================================================
# MLFLOW
# ============================================================

MLFLOW_TRACKING_URI = (
    __import__("os").environ["MLFLOW_TRACKING_URI"]
)

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

mlflow.set_experiment(
    "Pearls_AQI_Training"
)


# ============================================================
# TRAIN
# ============================================================

results = []

for name, model in models.items():

    print("=" * 60)
    print(f"TRAINING {name}")
    print("=" * 60)

    with mlflow.start_run(
        run_name=f"{name}_daily_training"
    ):

        model.fit(
            X_train,
            y_train,
        )

        validation_pred = model.predict(
            X_validation
        )

        test_pred = model.predict(
            X_test
        )

        validation_rmse = mean_squared_error(
            y_validation,
            validation_pred,
        ) ** 0.5

        test_rmse = mean_squared_error(
            y_test,
            test_pred,
        ) ** 0.5

        validation_mae = mean_absolute_error(
            y_validation,
            validation_pred,
        )

        test_mae = mean_absolute_error(
            y_test,
            test_pred,
        )

        test_r2 = r2_score(
            y_test,
            test_pred,
        )

        mlflow.log_param(
            "model",
            name,
        )

        mlflow.log_metric(
            "validation_rmse",
            validation_rmse,
        )

        mlflow.log_metric(
            "validation_mae",
            validation_mae,
        )

        mlflow.log_metric(
            "test_rmse",
            test_rmse,
        )

        mlflow.log_metric(
            "test_mae",
            test_mae,
        )

        mlflow.log_metric(
            "test_r2",
            test_r2,
        )

        mlflow.log_param(
            "feature_count",
            70,
        )

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
        )

        results.append({
            "name": name,
            "model": model,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            "test_r2": test_r2,
        })


# ============================================================
# SELECT WINNER
# ============================================================

results.sort(
    key=lambda x: x["test_rmse"]
)

winner = results[0]

winner_name = winner["name"]
winner_model = winner["model"]


print("=" * 60)
print("CHAMPION")
print("=" * 60)
print(winner_name)
print("RMSE:", winner["test_rmse"])
print("MAE:", winner["test_mae"])
print("R2:", winner["test_r2"])


# ============================================================
# REGISTER WINNER IN MLFLOW MODEL REGISTRY
# ============================================================

with mlflow.start_run(
    run_name="champion_registration"
) as run:

    mlflow.log_param(
        "champion_model",
        winner_name,
    )

    mlflow.log_metric(
        "champion_test_rmse",
        winner["test_rmse"],
    )

    model_info = mlflow.sklearn.log_model(
        winner_model,
        artifact_path="champion_model",
        registered_model_name=MODEL_NAME,
    )

    run_id = run.info.run_id


# ============================================================
# ASSIGN CHAMPION ALIAS
# ============================================================

client = mlflow.MlflowClient()

model_version = client.get_model_version_by_run_id(
    run_id
)

client.set_registered_model_alias(
    MODEL_NAME,
    "champion",
    model_version.version,
)

client.set_model_version_tag(
    MODEL_NAME,
    model_version.version,
    "model_type",
    winner_name,
)

client.set_model_version_tag(
    MODEL_NAME,
    model_version.version,
    "feature_count",
    "70",
)

print("=" * 60)
print("MLFLOW MODEL REGISTRY UPDATED")
print("=" * 60)
print("Model:", MODEL_NAME)
print("Version:", model_version.version)
print("Alias: champion")
print("Winner:", winner_name)