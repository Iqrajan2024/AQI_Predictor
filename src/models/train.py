from pathlib import Path
import json

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd

from feast import FeatureStore

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import XGBRegressor


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

FEATURE_REPO = (
    PROJECT_ROOT
    / "feature_repo"
    / "feature_repo"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# MLflow
# ============================================================

MLFLOW_DATABASE = (
    PROJECT_ROOT / "mlflow.db"
)

MLFLOW_TRACKING_URI = (
    f"sqlite:///{MLFLOW_DATABASE.as_posix()}"
)

MODEL_NAME = "Pearls_AQI_XGBoost"
MODEL_ALIAS = "champion"

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

mlflow.set_registry_uri(
    MLFLOW_TRACKING_URI
)

store = FeatureStore(
    repo_path=str(FEATURE_REPO)
)

EXPERIMENT_NAME = "Pearls_AQI_Training"

MODEL_NAME = "Pearls_AQI_Model"

mlflow.set_experiment(
    EXPERIMENT_NAME
)


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [

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


assert len(FEATURE_COLUMNS) == 70


# ============================================================
# LOAD TRAINING DATA THROUGH FEAST
# ============================================================

def load_training_data():

    print("=" * 70)
    print("LOADING HISTORICAL TRAINING DATA THROUGH FEAST")
    print("=" * 70)

    store = FeatureStore(
        repo_path=str(FEATURE_REPO)
    )

    source = store.get_data_source(
        "aqi_features_source"
    )

    source_query = (
        source.get_table_query_string()
    )

    entity_sql = f"""
        SELECT
            location_id,
            timestamp AS event_timestamp,
            target_aqi
        FROM {source_query}
        WHERE target_aqi IS NOT NULL
    """

    print("Using Feast historical retrieval...")

    training_df = (
        store
        .get_historical_features(
            entity_df=entity_sql,
            features=store.get_feature_service(
                "aqi_model_features"
            ),
        )
        .to_df()
    )

    if training_df.empty:
        raise ValueError(
            "Feast returned an empty training dataset."
        )

    training_df["event_timestamp"] = (
        pd.to_datetime(
            training_df["event_timestamp"],
            utc=True,
        )
    )

    training_df = (
        training_df
        .sort_values("event_timestamp")
        .reset_index(drop=True)
    )

    required = set(
        FEATURE_COLUMNS
        + [
            "location_id",
            "event_timestamp",
            "target_aqi",
        ]
    )

    missing = (
        required
        .difference(training_df.columns)
    )

    if missing:
        raise ValueError(
            "Feast training data is missing:\n"
            + "\n".join(sorted(missing))
        )

    # ========================================================
    # CRITICAL:
    # target_aqi MUST NOT be in X.
    # ========================================================

    X = training_df[
        FEATURE_COLUMNS
    ].copy()

    y = training_df[
        "target_aqi"
    ].copy()

    # Ensure target is numeric.
    y = pd.to_numeric(
        y,
        errors="coerce",
    )

    valid = (
        y.notna()
        & ~X.isna().any(axis=1)
    )

    X = X.loc[valid].reset_index(
        drop=True
    )

    y = y.loc[valid].reset_index(
        drop=True
    )

    timestamps = (
        training_df.loc[
            valid,
            "event_timestamp",
        ]
        .reset_index(drop=True)
    )

    if len(X) < 100:
        raise ValueError(
            f"Insufficient training rows: {len(X)}"
        )

    if X.shape[1] != 70:
        raise ValueError(
            f"Expected 70 features, "
            f"received {X.shape[1]}"
        )

    if "target_aqi" in X.columns:
        raise ValueError(
            "target_aqi leaked into model inputs."
        )

    if "us_aqi" in X.columns:
        raise ValueError(
            "us_aqi must not be used as a separate "
            "model input because it is not part of "
            "the 70-feature model schema."
        )

    print(
        "Feast training rows:",
        len(X),
    )

    print(
        "Feast model features:",
        X.shape[1],
    )

    print(
        "First timestamp:",
        timestamps.min(),
    )

    print(
        "Last timestamp:",
        timestamps.max(),
    )

    return X, y, timestamps


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(
    X,
    y,
    timestamps,
):

    n = len(X)

    train_end = int(
        n * 0.70
    )

    val_end = int(
        n * 0.85
    )

    X_train = X.iloc[
        :train_end
    ]

    y_train = y.iloc[
        :train_end
    ]

    X_val = X.iloc[
        train_end:val_end
    ]

    y_val = y.iloc[
        train_end:val_end
    ]

    X_test = X.iloc[
        val_end:
    ]

    y_test = y.iloc[
        val_end:
    ]

    print("=" * 70)
    print("CHRONOLOGICAL SPLIT")
    print("=" * 70)

    print(
        "Train:",
        X_train.shape,
        timestamps.iloc[0],
        "->",
        timestamps.iloc[train_end - 1],
    )

    print(
        "Validation:",
        X_val.shape,
        timestamps.iloc[train_end],
        "->",
        timestamps.iloc[val_end - 1],
    )

    print(
        "Test:",
        X_test.shape,
        timestamps.iloc[val_end],
        "->",
        timestamps.iloc[-1],
    )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )

    mae = float(
        mean_absolute_error(
            y_true,
            y_pred,
        )
    )

    r2 = float(
        r2_score(
            y_true,
            y_pred,
        )
    )

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


# ============================================================
# MODELS
# ============================================================

def build_models():

    return {

        "ridge": Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    Ridge(
                        alpha=1.0
                    ),
                ),
            ]
        ),

        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=18,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),

        "xgboost": XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=42,
            n_jobs=2,
        ),
    }


# ============================================================
# TRAIN
# ============================================================

def train():

    (
        X,
        y,
        timestamps,
    ) = load_training_data()

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = chronological_split(
        X,
        y,
        timestamps,
    )

    models = build_models()

    results = []

    trained_models = {}

    # ========================================================
    # TRAIN CANDIDATES
    # ========================================================

    for model_name, model in models.items():

        print("\n" + "=" * 70)
        print(
            f"TRAINING: {model_name.upper()}"
        )
        print("=" * 70)

        with mlflow.start_run(
            run_name=f"{model_name}_candidate"
        ) as run:

            model.fit(
                X_train,
                y_train,
            )

            val_pred = model.predict(
                X_val
            )

            test_pred = model.predict(
                X_test
            )

            val_metrics = calculate_metrics(
                y_val,
                val_pred,
            )

            test_metrics = calculate_metrics(
                y_test,
                test_pred,
            )

            mlflow.log_param(
                "model_name",
                model_name,
            )

            mlflow.log_param(
                "feature_count",
                70,
            )

            mlflow.log_param(
                "target",
                "target_aqi",
            )

            mlflow.log_metrics(
                {
                    "validation_rmse":
                        val_metrics["rmse"],
                    "validation_mae":
                        val_metrics["mae"],
                    "validation_r2":
                        val_metrics["r2"],
                    "test_rmse":
                        test_metrics["rmse"],
                    "test_mae":
                        test_metrics["mae"],
                    "test_r2":
                        test_metrics["r2"],
                }
            )

            if model_name == "xgboost":

                mlflow.xgboost.log_model(
                    model,
                    name="model",
                    input_example=X_train.head(2),
                )

            else:

                mlflow.sklearn.log_model(
                    model,
                    name="model",
                    input_example=X_train.head(2),
                )

            results.append(
                {
                    "model": model_name,
                    "validation_rmse":
                        val_metrics["rmse"],
                    "validation_mae":
                        val_metrics["mae"],
                    "validation_r2":
                        val_metrics["r2"],
                    "test_rmse":
                        test_metrics["rmse"],
                    "test_mae":
                        test_metrics["mae"],
                    "test_r2":
                        test_metrics["r2"],
                    "run_id":
                        run.info.run_id,
                }
            )

            trained_models[
                model_name
            ] = model

    # ========================================================
    # SELECT WINNER
    # ========================================================

    results_df = (
        pd.DataFrame(results)
        .sort_values(
            "validation_rmse"
        )
        .reset_index(drop=True)
    )

    winner = (
        results_df.iloc[0]
    )

    winner_name = winner[
        "model"
    ]

    winner_run_id = winner[
        "run_id"
    ]

    winner_model = trained_models[
        winner_name
    ]

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df[
            [
                "model",
                "validation_rmse",
                "validation_mae",
                "validation_r2",
                "test_rmse",
                "test_mae",
                "test_r2",
            ]
        ].to_string(index=False)
    )

    print("\nWINNER:", winner_name)

    # ========================================================
    # REGISTER WINNER
    # ========================================================

    print("\n" + "=" * 70)
    print("REGISTERING WINNING MODEL")
    print("=" * 70)

    model_uri = (
        f"runs:/{winner_run_id}/model"
    )

    registered = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    version = str(
        registered.version
    )

    client = mlflow.MlflowClient()

    client.set_registered_model_alias(
        MODEL_NAME,
        "champion",
        version,
    )

    client.set_model_version_tag(
        MODEL_NAME,
        version,
        "status",
        "champion",
    )

    client.set_model_version_tag(
        MODEL_NAME,
        version,
        "model_type",
        winner_name,
    )

    client.set_model_version_tag(
        MODEL_NAME,
        version,
        "feature_count",
        "70",
    )

    client.set_model_version_tag(
        MODEL_NAME,
        version,
        "target",
        "target_aqi",
    )

    print(
        "Registered model:",
        MODEL_NAME,
    )

    print(
        "Champion version:",
        version,
    )

    print(
        "Champion model type:",
        winner_name,
    )

    # ========================================================
    # SAVE LOCAL CHAMPION COPY
    # ========================================================

    champion_path = (
        MODEL_DIR
        / "champion_model.pkl"
    )

    metadata_path = (
        MODEL_DIR
        / "champion_metadata.json"
    )

    import joblib

    joblib.dump(
        winner_model,
        champion_path,
    )

    metadata = {
        "model_name": MODEL_NAME,
        "model_type": winner_name,
        "mlflow_run_id": winner_run_id,
        "mlflow_version": version,
        "mlflow_alias": "champion",
        "feature_count": 70,
        "features": FEATURE_COLUMNS,
        "target": "target_aqi",
        "validation_rmse": float(
            winner["validation_rmse"]
        ),
        "validation_mae": float(
            winner["validation_mae"]
        ),
        "validation_r2": float(
            winner["validation_r2"]
        ),
        "test_rmse": float(
            winner["test_rmse"]
        ),
        "test_mae": float(
            winner["test_mae"]
        ),
        "test_r2": float(
            winner["test_r2"]
        ),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    print(
        "Local champion saved:",
        champion_path,
    )

    print(
        "Champion metadata saved:",
        metadata_path,
    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    champion_info = (
        client.get_model_version_by_alias(
            MODEL_NAME,
            "champion",
        )
    )

    assert str(
        champion_info.version
    ) == version

    assert (
        len(FEATURE_COLUMNS) == 70
    )

    assert (
        "target_aqi"
        not in FEATURE_COLUMNS
    )

    assert (
        "us_aqi"
        not in FEATURE_COLUMNS
    )

    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETED")
    print("=" * 70)

    print(
        "Feast historical retrieval: PASS"
    )

    print(
        "Model features: 70"
    )

    print(
        "target_aqi excluded from X: PASS"
    )

    print(
        "MLflow registration: PASS"
    )

    print(
        "MLflow champion alias: PASS"
    )

    print(
        "Champion:",
        winner_name,
        "version",
        version,
    )


if __name__ == "__main__":
    train()