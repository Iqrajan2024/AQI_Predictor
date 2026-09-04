from pathlib import Path
import json
import joblib
import os

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_REPO = (
    PROJECT_ROOT
    / "feature_repo"
    / "feature_repo"
)

DATA_DIR = PROJECT_ROOT / "data" / "processed"

FEATURE_FILE = DATA_DIR / "aqi_features.parquet"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MLflow
# ============================================================

MODEL_NAME = "Pearls_AQI_XGBoost"
MODEL_ALIAS = "champion"
EXPERIMENT_NAME = "Pearls_AQI_Training"

DEFAULT_MLFLOW_DB = PROJECT_ROOT / "mlflow.db" 

MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{DEFAULT_MLFLOW_DB.as_posix()}", 
) 

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI) 

# Use the same backend for the Model Registry. 
mlflow.set_registry_uri(MLFLOW_TRACKING_URI)

mlflow.set_experiment(EXPERIMENT_NAME)

# ============================================================
# FEAST
# ============================================================

store = FeatureStore(
    repo_path=str(FEATURE_REPO)
)


# ============================================================
# EXACT 70 MODEL FEATURES
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

assert "target_aqi" not in FEATURE_COLUMNS
assert "us_aqi" not in FEATURE_COLUMNS


# ============================================================
# VALIDATE FEAST FEATURE SERVICE
# ============================================================

def validate_feature_contract():

    print("=" * 70)
    print("FEAST FEATURE CONTRACT")
    print("=" * 70)

    service = store.get_feature_service(
        "aqi_model_features"
    )

    feast_features = [
        feature.name
        for projection in service.feature_view_projections
        for feature in projection.features
    ]

    print(
        "Expected feature count:",
        len(FEATURE_COLUMNS),
    )

    print(
        "Feast feature count:",
        len(feast_features),
    )

    duplicates = sorted(
        {
            feature
            for feature in feast_features
            if feast_features.count(feature) > 1
        }
    )

    if duplicates:
        raise ValueError(
            f"Duplicate Feast features: {duplicates}"
        )

    missing = sorted(
        set(FEATURE_COLUMNS)
        - set(feast_features)
    )

    extra = sorted(
        set(feast_features)
        - set(FEATURE_COLUMNS)
    )

    if missing or extra:
        raise ValueError(
            "Feast/model feature contract mismatch.\n"
            f"Missing: {missing}\n"
            f"Extra: {extra}"
        )

    if "target_aqi" in feast_features:
        raise ValueError(
            "target_aqi must NOT be a Feast model feature."
        )

    if "us_aqi" in feast_features:
        raise ValueError(
            "us_aqi must NOT be a Feast model feature."
        )

    print("✓ Feature count: 70")
    print("✓ No duplicate features")
    print("✓ Exact feature-name match")
    print("✓ target_aqi excluded")
    print("✓ us_aqi excluded")
    print("✓ FEATURE CONTRACT: PASS")


# ============================================================
# LOAD TRAINING DATA
#
# IMPORTANT:
# Do NOT call Feast get_historical_features() here.
#
# Your Feast FileSource + DuckDB + Ibis point-in-time join
# is what is causing the OutOfMemoryException.
#
# The parquet file already contains the engineered feature
# rows and target_aqi. We therefore load only the exact
# training columns directly from the same Feast batch source.
#
# Feast remains responsible for the feature contract and
# online serving.
# ============================================================

def load_training_data():

    print("=" * 70)
    print("LOADING HISTORICAL TRAINING DATA")
    print("=" * 70)

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature parquet does not exist:\n"
            f"{FEATURE_FILE}"
        )

    validate_feature_contract()

    print("=" * 70)
    print("LOADING SOURCE DATA")
    print("=" * 70)

    required_columns = [
        "location_id",
        "timestamp",
        "target_aqi",
    ] + FEATURE_COLUMNS

    # Read only the required columns.
    # This avoids loading the complete 80-column parquet.
    df = pd.read_parquet(
        FEATURE_FILE,
        columns=required_columns,
        engine="pyarrow",
    )

    if df.empty:
        raise ValueError(
            "Feature parquet is empty."
        )

    print(
        "Source training shape:",
        df.shape,
    )

    print(
        "Locations:",
        df["location_id"].nunique(),
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = (
        df
        .sort_values(
            ["location_id", "timestamp"]
        )
        .reset_index(drop=True)
    )

    print(
        "First timestamp:",
        df["timestamp"].min(),
    )

    print(
        "Last timestamp:",
        df["timestamp"].max(),
    )

    print(
        "target_aqi available:",
        df["target_aqi"].notna().sum(),
    )

    # --------------------------------------------------------
    # HARD CONTRACT CHECKS
    # --------------------------------------------------------

    if "target_aqi" not in df.columns:
        raise ValueError(
            "target_aqi is missing from training source."
        )

    if "us_aqi" in FEATURE_COLUMNS:
        raise ValueError(
            "us_aqi must not be used as a model feature."
        )

    if "target_aqi" in FEATURE_COLUMNS:
        raise ValueError(
            "target_aqi must not be used as a model feature."
        )

    # --------------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------------

    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df["target_aqi"] = pd.to_numeric(
        df["target_aqi"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # REMOVE INVALID ROWS
    # --------------------------------------------------------

    before = len(df)

    valid_mask = (
        df["target_aqi"].notna()
        & ~df[FEATURE_COLUMNS]
            .isna()
            .any(axis=1)
    )

    df = df.loc[
        valid_mask
    ].reset_index(drop=True)

    removed = before - len(df)

    print(
        "Rows removed because of NaN:",
        removed,
    )

    if len(df) < 100:
        raise ValueError(
            f"Insufficient training rows: {len(df)}"
        )

    # --------------------------------------------------------
    # FINAL X / y
    # --------------------------------------------------------

    X = df[
        FEATURE_COLUMNS
    ].copy()

    y = df[
        "target_aqi"
    ].copy()

    timestamps = df[
        "timestamp"
    ].copy()

    locations = df[
        "location_id"
    ].copy()

    # --------------------------------------------------------
    # FINAL SAFETY CHECKS
    # --------------------------------------------------------

    if X.shape[1] != 70:
        raise ValueError(
            f"Expected 70 model features; "
            f"received {X.shape[1]}"
        )

    if "target_aqi" in X.columns:
        raise ValueError(
            "CRITICAL: target_aqi leaked into X."
        )

    if "us_aqi" in X.columns:
        raise ValueError(
            "CRITICAL: us_aqi leaked into X."
        )

    if X.isna().any().any():
        raise ValueError(
            "X still contains NaN values."
        )

    if y.isna().any():
        raise ValueError(
            "y still contains NaN values."
        )

    print(
        "Training rows:",
        len(X),
    )

    print(
        "Model features:",
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

    print(
        "✓ X contains exactly 70 features"
    )

    print(
        "✓ target_aqi is not in X"
    )

    print(
        "✓ us_aqi is not in X"
    )

    print(
        "✓ Historical training data loaded"
    )

    return (
        X,
        y,
        timestamps,
        locations,
    )


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
    ].copy()

    y_train = y.iloc[
        :train_end
    ].copy()

    X_val = X.iloc[
        train_end:val_end
    ].copy()

    y_val = y.iloc[
        train_end:val_end
    ].copy()

    X_test = X.iloc[
        val_end:
    ].copy()

    y_test = y.iloc[
        val_end:
    ].copy()

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
# PERSISTENCE BASELINE
# ============================================================

def calculate_persistence_baseline(
    y_train,
    y_test,
):
    """
    Persistence baseline:
    predict the next AQI using the most recently observed AQI.

    For the test period, the baseline starts with the last
    observed AQI from the training period and then updates
    using the previous actual AQI at each test timestep.
    """

    print("=" * 70)
    print("PERSISTENCE BASELINE")
    print("=" * 70)

    # First prediction uses the final observed training AQI.
    previous_aqi = float(
        y_train.iloc[-1]
    )

    persistence_predictions = []

    for actual_aqi in y_test:

        # Predict current AQI using previous observed AQI.
        persistence_predictions.append(
            previous_aqi
        )

        # After the prediction, the actual value becomes
        # available and is used for the next prediction.
        previous_aqi = float(actual_aqi)

    persistence_predictions = np.array(
        persistence_predictions
    )

    metrics = calculate_metrics(
        y_test,
        persistence_predictions,
    )

    print(
        f"Persistence Test RMSE: "
        f"{metrics['rmse']:.4f}"
    )

    print(
        f"Persistence Test MAE:  "
        f"{metrics['mae']:.4f}"
    )

    print(
        f"Persistence Test R²:   "
        f"{metrics['r2']:.4f}"
    )

    return (
        metrics,
        persistence_predictions,
    )

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

    print("=" * 70)
    print("PEARLS AQI MODEL TRAINING")
    print("=" * 70)

    print(
        "MLflow:",
        MLFLOW_TRACKING_URI,
    )

    print(
        "Feature repository:",
        FEATURE_REPO,
    )

    print(
        "Feature file:",
        FEATURE_FILE,
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    (
        X,
        y,
        timestamps,
        locations,
    ) = load_training_data()

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PERSISTENCE BASELINE
    # --------------------------------------------------------

    (
        persistence_metrics,
        persistence_predictions,
    ) = calculate_persistence_baseline(
        y_train,
        y_test,
    )

    # --------------------------------------------------------
    # BUILD MODELS
    # --------------------------------------------------------

    models = build_models()

    results = []

    trained_models = {}

    # --------------------------------------------------------
    # TRAIN CANDIDATES
    # --------------------------------------------------------

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

            # ------------------------------------------------
            # LOG PARAMETERS
            # ------------------------------------------------

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

            mlflow.log_param(
                "data_source",
                "data/processed/aqi_features.parquet",
            )

            mlflow.log_param(
                "feast_feature_service",
                "aqi_model_features",
            )

            mlflow.log_param(
                "split",
                "70/15/15 chronological",
            )

            # ------------------------------------------------
            # LOG METRICS
            # ------------------------------------------------

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

            # ------------------------------------------------
            # LOG MODEL
            # ------------------------------------------------

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

            # ------------------------------------------------
            # STORE RESULT
            # ------------------------------------------------

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

            print(
                "Validation RMSE:",
                val_metrics["rmse"],
            )

            print(
                "Validation MAE:",
                val_metrics["mae"],
            )

            print(
                "Validation R²:",
                val_metrics["r2"],
            )

            print(
                "Test RMSE:",
                test_metrics["rmse"],
            )

            print(
                "Test MAE:",
                test_metrics["mae"],
            )

            print(
                "Test R²:",
                test_metrics["r2"],
            )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    results_df = (
        pd.DataFrame(results)
        .sort_values(
            "validation_rmse"
        )
        .reset_index(drop=True)
    )

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

    winner = results_df.iloc[0]

    winner_name = winner[
        "model"
    ]

    winner_run_id = winner[
        "run_id"
    ]

    winner_model = trained_models[
        winner_name
    ]

    
    print(
        "\nWINNER:",
        winner_name,
    )

    # ========================================================
    # CHAMPION VS PERSISTENCE BASELINE
    # ========================================================

    champion_test_rmse = float(
        winner["test_rmse"]
    )

    champion_test_mae = float(
        winner["test_mae"]
    )

    champion_test_r2 = float(
        winner["test_r2"]
    )

    persistence_rmse = persistence_metrics[
        "rmse"
    ]

    persistence_mae = persistence_metrics[
        "mae"
    ]

    persistence_r2 = persistence_metrics[
        "r2"
    ]

    rmse_improvement = (
        (
            persistence_rmse
            - champion_test_rmse
        )
        / persistence_rmse
    ) * 100

    mae_improvement = (
        (
            persistence_mae
            - champion_test_mae
        )
        / persistence_mae
    ) * 100

    # ========================================================
    # LOG PERSISTENCE BASELINE TO WINNING RUN
    # ========================================================

    with mlflow.start_run(
        run_id=winner_run_id
    ):

        mlflow.log_metrics(
            {
                "persistence_test_rmse":
                    persistence_rmse,

                "persistence_test_mae":
                    persistence_mae,

                "persistence_test_r2":
                    persistence_r2,

                "rmse_improvement_vs_persistence_pct":
                    rmse_improvement,

                "mae_improvement_vs_persistence_pct":
                    mae_improvement,
            }
        )


    print("\n" + "=" * 70)
    print("CHAMPION VS PERSISTENCE BASELINE")
    print("=" * 70)

    comparison_df = pd.DataFrame(
        [
            {
                "model": "Persistence",
                "RMSE": persistence_rmse,
                "MAE": persistence_mae,
                "R2": persistence_r2,
            },
            {
                "model": f"Champion {winner_name}",
                "RMSE": champion_test_rmse,
                "MAE": champion_test_mae,
                "R2": champion_test_r2,
            },
        ]
    )

    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n" + "-" * 70)

    print(
        f"RMSE improvement: "
        f"{rmse_improvement:.2f}%"
    )

    print(
        f"MAE improvement:  "
        f"{mae_improvement:.2f}%"
    )

    print("-" * 70)

    # ========================================================
    # REGISTER WINNER
    # ========================================================

    print("\n" + "=" * 70)
    print("REGISTERING WINNING MODEL")
    print("=" * 70)

    # register the LoggedModel created by log_model(),
    # rather than reconstructing runs:/<run_id>/model.


    logged_models = mlflow.search_logged_models(
        filter_string=f"source_run_id = '{winner_run_id}'"
    )

    if logged_models.empty:
        raise RuntimeError(
            f"No MLflow LoggedModel found for winner run: "
            f"{winner_run_id}"
        )

    # The winning run contains exactly one model because
    # each candidate logs its model with name="model".
    winner_logged_model = logged_models.iloc[0]

    print(
        "Winner LoggedModel ID:",
        winner_logged_model.model_id,
    )

    print(
        "Winner LoggedModel URI:",
        winner_logged_model.model_uri,
    )

    registered = mlflow.register_model(
        model_uri=winner_logged_model.model_uri,
        name=MODEL_NAME,
    )

    version = str(
        registered.version
    )

    client = mlflow.MlflowClient()

    # --------------------------------------------------------
    # CHAMPION ALIAS
    # --------------------------------------------------------

    client.set_registered_model_alias(
        MODEL_NAME,
        MODEL_ALIAS,
        version,
    )

    # --------------------------------------------------------
    # TAGS
    # --------------------------------------------------------

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

    client.set_model_version_tag(
        MODEL_NAME,
        version,
        "feast_feature_service",
        "aqi_model_features",
    )

    client.set_model_version_tag(
        MODEL_NAME,
        version,
        "us_aqi_used_as_input",
        "false",
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

    print(
        "Champion alias:",
        f"{MODEL_NAME}@{MODEL_ALIAS}",
    )

    # ========================================================
    # SAVE LOCAL CHAMPION
    # ========================================================

    champion_path = (
        MODEL_DIR
        / "champion_model.pkl"
    )

    metadata_path = (
        MODEL_DIR
        / "champion_metadata.json"
    )

    joblib.dump(
        winner_model,
        champion_path,
    )

    metadata = {
        "model_name":
            MODEL_NAME,

        "model_type":
            winner_name,

        "mlflow_tracking_uri":
            MLFLOW_TRACKING_URI,

        "mlflow_run_id":
            winner_run_id,

        "mlflow_version":
            version,

        "mlflow_alias":
            MODEL_ALIAS,

        "feature_count":
            70,

        "features":
            FEATURE_COLUMNS,

        "target":
            "target_aqi",

        "us_aqi_used_as_input":
            False,

        "feast_feature_service":
            "aqi_model_features",

        "validation_rmse":
            float(
                winner[
                    "validation_rmse"
                ]
            ),

        "validation_mae":
            float(
                winner[
                    "validation_mae"
                ]
            ),

        "validation_r2":
            float(
                winner[
                    "validation_r2"
                ]
            ),

        "test_rmse":
            float(
                winner[
                    "test_rmse"
                ]
            ),

        "test_mae":
            float(
                winner[
                    "test_mae"
                ]
            ),

        "test_r2":
            float(
                winner[
                    "test_r2"
                ]
            ),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
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
    # FINAL REGISTRY VALIDATION
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL MODEL REGISTRY VALIDATION")
    print("=" * 70)

    champion_info = (
        client.get_model_version_by_alias(
            MODEL_NAME,
            MODEL_ALIAS,
        )
    )

    print(
        "Registered version:",
        champion_info.version,
    )

    print(
        "Registered run:",
        champion_info.run_id,
    )

    print(
        "Registered tags:",
        champion_info.tags,
    )

    assert (
        str(champion_info.version)
        == version
    )

    assert (
        champion_info.run_id
        == winner_run_id
    )

    assert (
        len(FEATURE_COLUMNS)
        == 70
    )

    assert (
        "target_aqi"
        not in FEATURE_COLUMNS
    )

    assert (
        "us_aqi"
        not in FEATURE_COLUMNS
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETED")
    print("=" * 70)

    print(
        "Historical data loading: PASS"
    )

    print(
        "Feast feature contract: PASS"
    )

    print(
        "Model features: 70"
    )

    print(
        "target_aqi excluded from X: PASS"
    )

    print(
        "us_aqi excluded from X: PASS"
    )

    print(
        "Chronological split: PASS"
    )

    print(
        "Candidate training: PASS"
    )

    print(
        "Winner:",
        winner_name,
    )

    print(
        "MLflow registration: PASS"
    )

    print(
        "MLflow champion alias: PASS"
    )

    print(
        "Champion:",
        f"{MODEL_NAME}@{MODEL_ALIAS}",
    )

    print(
        "Version:",
        version,
    )

    print(
        "Run:",
        winner_run_id,
    )

    print(
        "Training pipeline: PASS"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    train()