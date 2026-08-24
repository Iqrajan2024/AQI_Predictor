"""
============================================================
PEARLS AQI PREDICTOR
DAILY MODEL TRAINING PIPELINE
============================================================

Loads the processed AQI feature dataset, trains multiple
regression models, evaluates them, selects the best model,
and saves the trained models.

Input:
    data/processed/aqi_features.parquet

Output:
    models/ridge_aqi_model.pkl
    models/ridge_scaler.pkl
    models/random_forest_aqi.pkl
    models/xgboost_aqi.pkl
    models/champion_model.pkl
    models/champion_metadata.json
"""

from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import XGBRegressor


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "aqi_features.parquet"
)

MODEL_DIR = PROJECT_ROOT / "models"

RIDGE_MODEL_FILE = MODEL_DIR / "ridge_aqi_model.pkl"
RIDGE_SCALER_FILE = MODEL_DIR / "ridge_scaler.pkl"

RF_MODEL_FILE = MODEL_DIR / "random_forest_aqi.pkl"

XGB_MODEL_FILE = MODEL_DIR / "xgboost_aqi.pkl"

CHAMPION_MODEL_FILE = MODEL_DIR / "champion_model.pkl"

CHAMPION_METADATA_FILE = (
    MODEL_DIR / "champion_metadata.json"
)


# ============================================================
# MODEL FEATURES
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

TARGET_COLUMN = "target_aqi"


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X, y, dataset_name):
    """
    Evaluate a regression model using MAE, RMSE and R².
    """

    predictions = model.predict(X)

    mae = mean_absolute_error(
        y,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y,
            predictions
        )
    )

    r2 = r2_score(
        y,
        predictions
    )

    print(f"\n{dataset_name}")
    print("-" * 40)
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("PEARLS AQI PREDICTOR")
    print("DAILY MODEL TRAINING PIPELINE")
    print("=" * 60)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python executable: {sys.executable}")

    print("\nFeature file:")
    print(FEATURE_FILE)

    # --------------------------------------------------------
    # CHECK INPUT
    # --------------------------------------------------------

    if not FEATURE_FILE.exists():

        raise FileNotFoundError(
            f"Feature dataset not found:\n{FEATURE_FILE}"
        )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("LOADING FEATURE DATASET")
    print("=" * 60)

    df = pd.read_parquet(
        FEATURE_FILE
    )

    print("Dataset shape:", df.shape)

    # --------------------------------------------------------
    # VERIFY COLUMNS
    # --------------------------------------------------------

    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing model features:\n"
            + "\n".join(missing_features)
        )

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            f"Missing target column: {TARGET_COLUMN}"
        )

    print(
        f"✓ All {len(FEATURE_COLUMNS)} model features found"
    )

    print(
        f"✓ Target column '{TARGET_COLUMN}' found"
    )

    # --------------------------------------------------------
    # SORT CHRONOLOGICALLY
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
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # FINAL DATA CHECK
    # --------------------------------------------------------

    feature_data = df[FEATURE_COLUMNS]

    target_data = df[TARGET_COLUMN]

    if feature_data.isna().sum().sum() != 0:

        raise ValueError(
            "Feature dataset contains missing values."
        )

    if target_data.isna().sum() != 0:

        raise ValueError(
            "Target dataset contains missing values."
        )

    if not df["timestamp"].is_monotonic_increasing:

        raise ValueError(
            "Dataset is not chronologically ordered."
        )

    print("✓ No missing feature values")
    print("✓ No missing target values")
    print("✓ Dataset is chronologically ordered")

    # ========================================================
    # CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
    # ========================================================

    print("\n" + "=" * 60)
    print("CREATING CHRONOLOGICAL DATA SPLIT")
    print("=" * 60)

    X = df[FEATURE_COLUMNS]

    y = df[TARGET_COLUMN]

    total_rows = len(df)

    train_end = int(
        total_rows * 0.70
    )

    validation_end = int(
        total_rows * 0.85
    )

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_val = X.iloc[
        train_end:validation_end
    ]

    y_val = y.iloc[
        train_end:validation_end
    ]

    X_test = X.iloc[
        validation_end:
    ]

    y_test = y.iloc[
        validation_end:
    ]

    print(
        "Training shape:",
        X_train.shape
    )

    print(
        "Validation shape:",
        X_val.shape
    )

    print(
        "Testing shape:",
        X_test.shape
    )

    # ========================================================
    # CREATE MODEL DIRECTORY
    # ========================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # RIDGE REGRESSION
    # ========================================================

    print("\n" + "=" * 60)
    print("RIDGE REGRESSION TRAINING")
    print("=" * 60)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_val_scaled = scaler.transform(
        X_val
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    ridge_model = Ridge(
        alpha=1.0
    )

    print("\nTraining Ridge Regression...")

    ridge_model.fit(
        X_train_scaled,
        y_train
    )

    print(
        "Ridge training completed."
    )

    ridge_train_results = evaluate_model(
        ridge_model,
        X_train_scaled,
        y_train,
        "Training"
    )

    ridge_val_results = evaluate_model(
        ridge_model,
        X_val_scaled,
        y_val,
        "Validation"
    )

    ridge_test_results = evaluate_model(
        ridge_model,
        X_test_scaled,
        y_test,
        "Testing"
    )

    joblib.dump(
        ridge_model,
        RIDGE_MODEL_FILE
    )

    joblib.dump(
        scaler,
        RIDGE_SCALER_FILE
    )

    print(
        "\nRidge model saved:",
        RIDGE_MODEL_FILE
    )

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    print("\n" + "=" * 60)
    print("RANDOM FOREST REGRESSION TRAINING")
    print("=" * 60)

    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining Random Forest...")

    rf_model.fit(
        X_train,
        y_train
    )

    print(
        "Random Forest training completed."
    )

    rf_train_results = evaluate_model(
        rf_model,
        X_train,
        y_train,
        "Training"
    )

    rf_val_results = evaluate_model(
        rf_model,
        X_val,
        y_val,
        "Validation"
    )

    rf_test_results = evaluate_model(
        rf_model,
        X_test,
        y_test,
        "Testing"
    )

    joblib.dump(
        rf_model,
        RF_MODEL_FILE
    )

    print(
        "\nRandom Forest model saved:",
        RF_MODEL_FILE
    )

    # ========================================================
    # XGBOOST
    # ========================================================

    print("\n" + "=" * 60)
    print("XGBOOST REGRESSION TRAINING")
    print("=" * 60)

    xgb_model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining XGBoost...")

    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_val, y_val)
        ],
        verbose=False
    )

    print(
        "XGBoost training completed."
    )

    xgb_train_results = evaluate_model(
        xgb_model,
        X_train,
        y_train,
        "Training"
    )

    xgb_val_results = evaluate_model(
        xgb_model,
        X_val,
        y_val,
        "Validation"
    )

    xgb_test_results = evaluate_model(
        xgb_model,
        X_test,
        y_test,
        "Testing"
    )

    joblib.dump(
        xgb_model,
        XGB_MODEL_FILE
    )

    print(
        "\nXGBoost model saved:",
        XGB_MODEL_FILE
    )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    model_results = {

        "Ridge": {
            "model": ridge_model,
            "test": ridge_test_results,
            "model_file": str(
                RIDGE_MODEL_FILE
            ),
        },

        "Random Forest": {
            "model": rf_model,
            "test": rf_test_results,
            "model_file": str(
                RF_MODEL_FILE
            ),
        },

        "XGBoost": {
            "model": xgb_model,
            "test": xgb_test_results,
            "model_file": str(
                XGB_MODEL_FILE
            ),
        },
    }

    comparison_rows = []

    for model_name, result in model_results.items():

        comparison_rows.append({
            "Model": model_name,
            "MAE": result["test"]["MAE"],
            "RMSE": result["test"]["RMSE"],
            "R2": result["test"]["R2"],
        })

    comparison = pd.DataFrame(
        comparison_rows
    )

    comparison = comparison.sort_values(
        "RMSE",
        ascending=True
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    # ========================================================
    # SELECT CHAMPION
    # ========================================================

    champion_name = (
        comparison.iloc[0]["Model"]
    )

    champion_model = model_results[
        champion_name
    ]["model"]

    champion_model_path = (
        model_results[
            champion_name
        ]["model_file"]
    )

    print("\n" + "=" * 60)
    print("CHAMPION MODEL")
    print("=" * 60)

    print(
        "Champion:",
        champion_name
    )

    print(
        "Test MAE:",
        f"{comparison.iloc[0]['MAE']:.4f}"
    )

    print(
        "Test RMSE:",
        f"{comparison.iloc[0]['RMSE']:.4f}"
    )

    print(
        "Test R²:",
        f"{comparison.iloc[0]['R2']:.4f}"
    )

    # --------------------------------------------------------
    # SAVE CHAMPION MODEL
    # --------------------------------------------------------

    joblib.dump(
        champion_model,
        CHAMPION_MODEL_FILE
    )

    print(
        "\nChampion model copied to:",
        CHAMPION_MODEL_FILE
    )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    if champion_name == "XGBoost":

        importance = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "importance": (
                xgb_model.feature_importances_
            )
        })

        importance = (
            importance
            .sort_values(
                "importance",
                ascending=False
            )
        )

        print("\n" + "=" * 60)
        print("TOP 20 XGBOOST FEATURES")
        print("=" * 60)

        print(
            importance
            .head(20)
            .to_string(index=False)
        )

    # ========================================================
    # SAVE METADATA
    # ========================================================

    metadata = {

        "champion_model": champion_name,

        "target": TARGET_COLUMN,

        "number_of_features": len(
            FEATURE_COLUMNS
        ),

        "features": FEATURE_COLUMNS,

        "test_metrics": {
            "MAE": float(
                comparison.iloc[0]["MAE"]
            ),
            "RMSE": float(
                comparison.iloc[0]["RMSE"]
            ),
            "R2": float(
                comparison.iloc[0]["R2"]
            ),
        },

        "models": {

            name: {
                "MAE": result["test"]["MAE"],
                "RMSE": result["test"]["RMSE"],
                "R2": result["test"]["R2"],
                "path": result["model_file"],
            }

            for name, result
            in model_results.items()
        },

        "training_rows": len(
            X_train
        ),

        "validation_rows": len(
            X_val
        ),

        "testing_rows": len(
            X_test
        ),
    }

    with open(
        CHAMPION_METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print(
        "\nChampion metadata saved:",
        CHAMPION_METADATA_FILE
    )

    # ========================================================
    # FINAL
    # ========================================================

    print("\n" + "=" * 60)
    print("DAILY TRAINING PIPELINE COMPLETED")
    print("=" * 60)

    print(
        "Champion model:",
        champion_name
    )

    print(
        "Champion model path:",
        CHAMPION_MODEL_FILE
    )

    print(
        "Training rows:",
        len(X_train)
    )

    print(
        "Validation rows:",
        len(X_val)
    )

    print(
        "Testing rows:",
        len(X_test)
    )

    print("\n Training pipeline completed successfully")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()