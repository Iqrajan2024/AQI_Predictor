from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "aqi_features.parquet"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "champion_model.pkl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FORECAST_OUTPUT_FILE = (
    OUTPUT_DIR
    / "historical_72h_forecasts.csv"
)

DAILY_OUTPUT_FILE = (
    OUTPUT_DIR
    / "historical_72h_daily_metrics.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "historical_72h_summary.json"
)


# ============================================================
# SETTINGS
# ============================================================

FORECAST_HOURS = 72
HISTORY_HOURS = 96

TIMESTAMP_COLUMN = "timestamp"
TARGET_COLUMN = "us_aqi"


# ============================================================
# EXACT PRODUCTION MODEL CONTRACT
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

    # PM2.5 lags
    "pm2_5_lag_1",
    "pm2_5_lag_3",
    "pm2_5_lag_6",
    "pm2_5_lag_24",

    # PM10 lags
    "pm10_lag_1",
    "pm10_lag_3",
    "pm10_lag_6",
    "pm10_lag_24",

    # Carbon monoxide lags
    "carbon_monoxide_lag_1",
    "carbon_monoxide_lag_3",
    "carbon_monoxide_lag_6",
    "carbon_monoxide_lag_24",

    # Nitrogen dioxide lags
    "nitrogen_dioxide_lag_1",
    "nitrogen_dioxide_lag_3",
    "nitrogen_dioxide_lag_6",
    "nitrogen_dioxide_lag_24",

    # Sulphur dioxide lags
    "sulphur_dioxide_lag_1",
    "sulphur_dioxide_lag_3",
    "sulphur_dioxide_lag_6",
    "sulphur_dioxide_lag_24",

    # Ozone lags
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

    # 24-hour pollutant rolling means
    "carbon_monoxide_24h_mean",
    "nitrogen_dioxide_24h_mean",
    "sulphur_dioxide_24h_mean",
    "ozone_24h_mean",

    # AQI changes
    "aqi_change_1h",
    "aqi_change_3h",
    "aqi_change_6h",
    "aqi_change_24h",

    # PM2.5 changes
    "pm2_5_change_1h",
    "pm2_5_change_24h",

    # PM10 changes
    "pm10_change_1h",
    "pm10_change_24h",
]


# ============================================================
# CONTRACT CHECKS
# ============================================================

assert len(FEATURE_COLUMNS) == 70
assert len(set(FEATURE_COLUMNS)) == 70
assert "target_aqi" not in FEATURE_COLUMNS
assert "us_aqi" not in FEATURE_COLUMNS


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print()
    print("=" * 70)
    print("LOADING HISTORICAL AQI DATA")
    print("=" * 70)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Historical feature file not found:\n{DATA_FILE}"
        )

    df = pd.read_parquet(DATA_FILE)

    required_columns = {
        TIMESTAMP_COLUMN,
        TARGET_COLUMN,
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    df[TIMESTAMP_COLUMN] = (
        pd.to_datetime(
            df[TIMESTAMP_COLUMN],
            utc=True,
        )
        .dt.floor("h")
    )

    df = (
        df
        .sort_values(TIMESTAMP_COLUMN)
        .drop_duplicates(
            subset=[TIMESTAMP_COLUMN],
            keep="last",
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Verify hourly continuity
    # --------------------------------------------------------

    expected = pd.date_range(
        start=df[TIMESTAMP_COLUMN].iloc[0],
        end=df[TIMESTAMP_COLUMN].iloc[-1],
        freq="h",
        tz="UTC",
    )

    actual = pd.DatetimeIndex(
        df[TIMESTAMP_COLUMN]
    )

    if not actual.equals(expected):

        raise ValueError(
            "Historical dataset is not hourly continuous."
        )

    if df[TARGET_COLUMN].isna().any():

        raise ValueError(
            "Historical us_aqi contains null values."
        )

    # --------------------------------------------------------
    # Verify all model source columns
    # --------------------------------------------------------

    source_columns = [
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

    missing_source = [
        column
        for column in source_columns
        if column not in df.columns
    ]

    if missing_source:

        raise ValueError(
            "Missing source columns: "
            f"{missing_source}"
        )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        "Range:",
        df[TIMESTAMP_COLUMN].min(),
        "->",
        df[TIMESTAMP_COLUMN].max(),
    )

    print(
        "✓ Hourly continuity verified"
    )

    print(
        "✓ No null AQI values"
    )

    return df


# ============================================================
# LOAD CHAMPION MODEL
# ============================================================

def load_model():

    print()
    print("=" * 70)
    print("LOADING CHAMPION MODEL")
    print("=" * 70)

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Champion model not found:\n{MODEL_FILE}"
        )

    model = joblib.load(
        MODEL_FILE
    )

    print(
        "Model:",
        MODEL_FILE,
    )

    print(
        "✓ Champion model loaded"
    )

    return model


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(
    history_df,
):
    """
    Recreate the exact 70-feature production contract.

    IMPORTANT:
    The final row is the row for which the model prediction
    is being generated.

    All lag/rolling/change features use information available
    before that prediction timestamp.
    """

    df = history_df.copy()

    df = (
        df
        .sort_values(TIMESTAMP_COLUMN)
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # TIME FEATURES
    # --------------------------------------------------------

    timestamp = df[
        TIMESTAMP_COLUMN
    ]

    df["hour"] = timestamp.dt.hour

    df["day_of_week"] = (
        timestamp.dt.dayofweek
    )

    df["day_of_month"] = (
        timestamp.dt.day
    )

    df["month"] = (
        timestamp.dt.month
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # --------------------------------------------------------
    # AQI LAGS
    # --------------------------------------------------------

    df["aqi_lag_1"] = (
        df["us_aqi"].shift(1)
    )

    df["aqi_lag_3"] = (
        df["us_aqi"].shift(3)
    )

    df["aqi_lag_6"] = (
        df["us_aqi"].shift(6)
    )

    df["aqi_lag_12"] = (
        df["us_aqi"].shift(12)
    )

    df["aqi_lag_24"] = (
        df["us_aqi"].shift(24)
    )

    df["aqi_lag_48"] = (
        df["us_aqi"].shift(48)
    )

    df["aqi_lag_72"] = (
        df["us_aqi"].shift(72)
    )

    # --------------------------------------------------------
    # POLLUTANT LAGS
    # --------------------------------------------------------

    pollutant_lag_columns = [
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
    ]

    for column in pollutant_lag_columns:

        for lag in [
            1,
            3,
            6,
            24,
        ]:

            df[
                f"{column}_lag_{lag}"
            ] = (
                df[column]
                .shift(lag)
            )

    # --------------------------------------------------------
    # AQI ROLLING MEANS
    # --------------------------------------------------------

    df["aqi_3h_mean"] = (
        df["us_aqi"]
        .shift(1)
        .rolling(3)
        .mean()
    )

    df["aqi_6h_mean"] = (
        df["us_aqi"]
        .shift(1)
        .rolling(6)
        .mean()
    )

    df["aqi_12h_mean"] = (
        df["us_aqi"]
        .shift(1)
        .rolling(12)
        .mean()
    )

    df["aqi_24h_mean"] = (
        df["us_aqi"]
        .shift(1)
        .rolling(24)
        .mean()
    )

    # --------------------------------------------------------
    # PM2.5 ROLLING MEANS
    # --------------------------------------------------------

    df["pm2_5_3h_mean"] = (
        df["pm2_5"]
        .shift(1)
        .rolling(3)
        .mean()
    )

    df["pm2_5_6h_mean"] = (
        df["pm2_5"]
        .shift(1)
        .rolling(6)
        .mean()
    )

    df["pm2_5_24h_mean"] = (
        df["pm2_5"]
        .shift(1)
        .rolling(24)
        .mean()
    )

    # --------------------------------------------------------
    # PM10 ROLLING MEANS
    # --------------------------------------------------------

    df["pm10_3h_mean"] = (
        df["pm10"]
        .shift(1)
        .rolling(3)
        .mean()
    )

    df["pm10_6h_mean"] = (
        df["pm10"]
        .shift(1)
        .rolling(6)
        .mean()
    )

    df["pm10_24h_mean"] = (
        df["pm10"]
        .shift(1)
        .rolling(24)
        .mean()
    )

    # --------------------------------------------------------
    # 24-HOUR POLLUTANT MEANS
    # --------------------------------------------------------

    df["carbon_monoxide_24h_mean"] = (
        df["carbon_monoxide"]
        .shift(1)
        .rolling(24)
        .mean()
    )

    df["nitrogen_dioxide_24h_mean"] = (
        df["nitrogen_dioxide"]
        .shift(1)
        .rolling(24)
        .mean()
    )

    df["sulphur_dioxide_24h_mean"] = (
        df["sulphur_dioxide"]
        .shift(1)
        .rolling(24)
        .mean()
    )

    df["ozone_24h_mean"] = (
        df["ozone"]
        .shift(1)
        .rolling(24)
        .mean()
    )

    # --------------------------------------------------------
    # AQI CHANGES
    # --------------------------------------------------------

    df["aqi_change_1h"] = (
        df["us_aqi"]
        - df["us_aqi"].shift(1)
    )

    df["aqi_change_3h"] = (
        df["us_aqi"]
        - df["us_aqi"].shift(3)
    )

    df["aqi_change_6h"] = (
        df["us_aqi"]
        - df["us_aqi"].shift(6)
    )

    df["aqi_change_24h"] = (
        df["us_aqi"]
        - df["us_aqi"].shift(24)
    )

    # --------------------------------------------------------
    # PM2.5 CHANGES
    # --------------------------------------------------------

    df["pm2_5_change_1h"] = (
        df["pm2_5"]
        - df["pm2_5"].shift(1)
    )

    df["pm2_5_change_24h"] = (
        df["pm2_5"]
        - df["pm2_5"].shift(24)
    )

    # --------------------------------------------------------
    # PM10 CHANGES
    # --------------------------------------------------------

    df["pm10_change_1h"] = (
        df["pm10"]
        - df["pm10"].shift(1)
    )

    df["pm10_change_24h"] = (
        df["pm10"]
        - df["pm10"].shift(24)
    )

    # --------------------------------------------------------
    # CONTRACT VALIDATION
    # --------------------------------------------------------

    missing = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Feature engineering failed. "
            f"Missing features: {missing}"
        )

    features = df[
        FEATURE_COLUMNS
    ].copy()

    return features


# ============================================================
# ONE 72-HOUR RECURSIVE FORECAST
# ============================================================

def recursive_forecast(
    df,
    model,
    origin_index,
):
    """
    Historical 72-hour recursive forecast.

    At the origin:
        actual AQI is known.

    Hour 1:
        predict using origin information.

    Hour 2:
        previous prediction becomes AQI state.

    ...

    Hour 72:
        recursively uses previous predictions.

    Future actual AQI is NEVER used as recursive input.
    """

    origin_timestamp = df.iloc[
        origin_index
    ][TIMESTAMP_COLUMN]

    last_observed_aqi = float(
        df.iloc[
            origin_index
        ][TARGET_COLUMN]
    )

    if not np.isfinite(
        last_observed_aqi
    ):
        raise ValueError(
            "Origin AQI is not finite."
        )

    # --------------------------------------------------------
    # Working data.
    #
    # Weather/pollutants remain historical future values.
    # AQI will be overwritten recursively.
    # --------------------------------------------------------

    working = df.copy()

    predictions = []

    previous_prediction = (
        last_observed_aqi
    )

    for step in range(
        1,
        FORECAST_HOURS + 1,
    ):

        future_index = (
            origin_index + step
        )

        if future_index >= len(
            working
        ):

            raise ValueError(
                "Not enough historical data "
                "for complete 72-hour horizon."
            )

        # ----------------------------------------------------
        # CRITICAL:
        #
        # Never use actual future AQI.
        #
        # For every future step, replace AQI with the
        # previous predicted AQI.
        #
        # For step 1, the origin AQI remains the known state.
        # ----------------------------------------------------

        if step == 1:

            working.loc[
                future_index,
                TARGET_COLUMN,
            ] = last_observed_aqi

        else:

            working.loc[
                future_index,
                TARGET_COLUMN,
            ] = previous_prediction

        # ----------------------------------------------------
        # Only use history through current prediction row.
        # ----------------------------------------------------

        history = working.iloc[
            : future_index + 1
        ].copy()

        # We need enough historical rows for lag 72.
        if len(history) < 73:

            raise ValueError(
                "Insufficient history for "
                "72-hour lag features."
            )

        features = create_features(
            history
        )

        feature_row = features.iloc[
            -1
        ].copy()

        if feature_row.isna().any():

            missing_features = list(
                feature_row[
                    feature_row.isna()
                ].index
            )

            raise ValueError(
                "NaN model features at "
                f"step {step}: "
                f"{missing_features}"
            )

        X = pd.DataFrame(
            [feature_row.values],
            columns=FEATURE_COLUMNS,
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = float(
            model.predict(X)[0]
        )

        if not np.isfinite(
            prediction
        ):

            raise ValueError(
                f"Non-finite prediction at "
                f"step {step}."
            )

        prediction = max(
            0.0,
            prediction,
        )

        actual_aqi = float(
            df.iloc[
                future_index
            ][TARGET_COLUMN]
        )

        predictions.append(
            {
                "origin_timestamp":
                    origin_timestamp,

                "forecast_timestamp":
                    df.iloc[
                        future_index
                    ][TIMESTAMP_COLUMN],

                "lead_hour":
                    step,

                "predicted_aqi":
                    prediction,

                "actual_aqi":
                    actual_aqi,

                "persistence_aqi":
                    last_observed_aqi,
            }
        )

        previous_prediction = (
            prediction
        )

    return pd.DataFrame(
        predictions
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    actual,
    predicted,
):

    rmse = float(
        np.sqrt(
            mean_squared_error(
                actual,
                predicted,
            )
        )
    )

    mae = float(
        mean_absolute_error(
            actual,
            predicted,
        )
    )

    r2 = float(
            r2_score(
                actual,
                predicted,
            )
        )

    return rmse, mae, r2


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("HISTORICAL 72-HOUR ROLLING EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()

    model = load_model()

    print()
    print(
        "✓ Model feature contract:",
        len(FEATURE_COLUMNS),
        "features",
    )

    # --------------------------------------------------------
    # Determine origins
    #
    # Need:
    #   96 historical hours
    #   + 72 future hours
    #
    # Origins are separated by 24 hours.
    # --------------------------------------------------------

    first_origin = (
        HISTORY_HOURS - 1
    )

    last_origin = (
        len(df)
        - FORECAST_HOURS
        - 1
    )

    if last_origin <= first_origin:

        raise ValueError(
            "Not enough data for historical "
            "72-hour evaluation."
        )

    origins = list(
        range(
            first_origin,
            last_origin + 1,
            24,
        )
    )

    print()
    print("=" * 70)
    print("EVALUATION SETUP")
    print("=" * 70)

    print(
        "History required:",
        HISTORY_HOURS,
        "hours",
    )

    print(
        "Forecast horizon:",
        FORECAST_HOURS,
        "hours",
    )

    print(
        "Origin frequency:",
        "24 hours",
    )

    print(
        "Number of origins:",
        len(origins),
    )

    print(
        "First origin:",
        df.iloc[
            first_origin
        ][TIMESTAMP_COLUMN],
    )

    print(
        "Last origin:",
        df.iloc[
            last_origin
        ][TIMESTAMP_COLUMN],
    )

    # --------------------------------------------------------
    # Rolling evaluation
    # --------------------------------------------------------

    all_forecasts = []

    for number, origin_index in enumerate(
        origins,
        start=1,
    ):

        origin_timestamp = df.iloc[
            origin_index
        ][TIMESTAMP_COLUMN]

        print()
        print(
            f"[{number:03d}/{len(origins):03d}] "
            f"{origin_timestamp}"
        )

        forecast = recursive_forecast(
            df=df,
            model=model,
            origin_index=origin_index,
        )

        all_forecasts.append(
            forecast
        )

        # ----------------------------------------------------
        # Quick per-origin metrics
        # ----------------------------------------------------

        xgb_rmse, xgb_mae, xgb_r2 = (
            calculate_metrics(
                forecast["actual_aqi"],
                forecast["predicted_aqi"],
            )
        )

        persistence_rmse, persistence_mae, persistence_r2 = (
            calculate_metrics(
                forecast["actual_aqi"],
                forecast["persistence_aqi"],
            )
        )

        rmse_improvement = (
            (
                persistence_rmse
                - xgb_rmse
            )
            / persistence_rmse
            * 100
        )

        mae_improvement = (
            (
                persistence_mae
                - xgb_mae
            )
            / persistence_mae
            * 100
        )

        print(
            f"XGB RMSE: {xgb_rmse:.4f} | "
            f"Persistence RMSE: "
            f"{persistence_rmse:.4f}"
        )

        print(
            f"XGB MAE:  {xgb_mae:.4f} | "
            f"Persistence MAE: "
            f"{persistence_mae:.4f}"
        )

        print(
            f"XGB R²:   {xgb_r2:.4f} | "
            f"Persistence R²: "
            f"{persistence_r2:.4f}"
        )

        print(
            f"RMSE improvement: "
            f"{rmse_improvement:.2f}%"
        )

        print(
            f"MAE improvement: "
            f"{mae_improvement:.2f}%"
        )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    results = pd.concat(
        all_forecasts,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    actual = results[
        "actual_aqi"
    ].to_numpy()

    predicted = results[
        "predicted_aqi"
    ].to_numpy()

    persistence = results[
        "persistence_aqi"
    ].to_numpy()

    xgb_rmse, xgb_mae, xgb_r2 = (
        calculate_metrics(
            actual,
            predicted,
        )
    )

    persistence_rmse, persistence_mae, persistence_r2 = (
        calculate_metrics(
            actual,
            persistence,
        )
    )

    rmse_improvement = (
        (
            persistence_rmse
            - xgb_rmse
        )
        / persistence_rmse
        * 100
    )

    mae_improvement = (
        (
            persistence_mae
            - xgb_mae
        )
        / persistence_mae
        * 100
    )

    # --------------------------------------------------------
    # Day 1 / Day 2 / Day 3
    # --------------------------------------------------------

    daily_rows = []

    for day in [
        1,
        2,
        3,
    ]:

        start_hour = (
            (day - 1) * 24 + 1
        )

        end_hour = (
            day * 24
        )

        subset = results[
            results[
                "lead_hour"
            ].between(
                start_hour,
                end_hour,
            )
        ]

        actual_day = subset[
            "actual_aqi"
        ].to_numpy()

        predicted_day = subset[
            "predicted_aqi"
        ].to_numpy()

        persistence_day = subset[
            "persistence_aqi"
        ].to_numpy()

        day_xgb_rmse, day_xgb_mae, day_xgb_r2 = (
            calculate_metrics(
                actual_day,
                predicted_day,
            )
        )

        (
            day_persistence_rmse,
            day_persistence_mae,
            day_persistence_r2,
        ) = calculate_metrics(
            actual_day,
            persistence_day,
        )

        day_rmse_improvement = (
            (
                day_persistence_rmse
                - day_xgb_rmse
            )
            / day_persistence_rmse
            * 100
        )

        day_mae_improvement = (
            (
                day_persistence_mae
                - day_xgb_mae
            )
            / day_persistence_mae
            * 100
        )

        daily_rows.append(
            {
                "day": day,
                "lead_hours": (
                    f"{start_hour}-{end_hour}"
                ),
                "xgb_rmse":
                    day_xgb_rmse,
                "xgb_mae":
                    day_xgb_mae,
                "xgb_r2":
                    day_xgb_r2,
                "persistence_rmse":
                    day_persistence_rmse,
                "persistence_mae":
                    day_persistence_mae,
                "persistence_r2":
                    day_persistence_r2,
                "rmse_improvement_percent":
                    day_rmse_improvement,
                "mae_improvement_percent":
                    day_mae_improvement,
            }
        )

    daily_results = pd.DataFrame(
        daily_rows
    )

    # --------------------------------------------------------
    # Save forecast-level results
    # --------------------------------------------------------

    results.to_csv(
        FORECAST_OUTPUT_FILE,
        index=False,
    )

    daily_results.to_csv(
        DAILY_OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = {
        "evaluation_type":
            "historical_72_hour_recursive",

        "number_of_origins":
            len(origins),

        "origin_frequency_hours":
            24,

        "history_hours":
            HISTORY_HOURS,

        "forecast_horizon_hours":
            FORECAST_HOURS,

        "model":
            "Pearls_AQI_XGBoost",

        "feature_count":
            len(FEATURE_COLUMNS),

        "persistence_type":
            "fixed_origin_72_hour",

        "overall": {
            "xgb_rmse":
                xgb_rmse,

            "xgb_mae":
                xgb_mae,

            "xgb_r2":
                xgb_r2,

            "persistence_rmse":
                persistence_rmse,

            "persistence_mae":
                persistence_mae,

            "persistence_r2":
                persistence_r2,

            "rmse_improvement_percent":
                rmse_improvement,

            "mae_improvement_percent":
                mae_improvement,
        },

        "daily": daily_results.to_dict(
            orient="records"
        ),

        "oracle_exogenous":
            True,

        "oracle_exogenous_note":
            (
                "Historical evaluation uses actual "
                "future weather and pollutant values "
                "available in the historical feature "
                "dataset. These future exogenous values "
                "may not represent the forecasts that "
                "would have been available operationally "
                "at each historical forecast origin."
            ),
    }

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL 72-HOUR EVALUATION RESULTS")
    print("=" * 70)

    print()
    print(
        f"Forecast origins: "
        f"{len(origins)}"
    )

    print()
    print("OVERALL")
    print("-" * 70)

    print(
        f"XGBoost RMSE:        "
        f"{xgb_rmse:.4f}"
    )

    print(
        f"XGBoost MAE:         "
        f"{xgb_mae:.4f}"
    )

    print(
        f"XGBoost R²:          "
        f"{xgb_r2:.4f}"
    )

    print(
        f"Persistence RMSE:   "
        f"{persistence_rmse:.4f}"
    )

    print(
        f"Persistence MAE:    "
        f"{persistence_mae:.4f}"
    )

    print(
        f"Persistence R²:     "
        f"{persistence_r2:.4f}"
    )

    print(
        f"RMSE improvement:    "
        f"{rmse_improvement:.2f}%"
    )

    print(
        f"MAE improvement:     "
        f"{mae_improvement:.2f}%"
    )

    print()
    print("DAY-BY-DAY")
    print("-" * 70)

    print(
        daily_results.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)

    print(
        "Forecast-level:"
    )
    print(
        FORECAST_OUTPUT_FILE
    )

    print()
    print(
        "Daily metrics:"
    )
    print(
        DAILY_OUTPUT_FILE
    )

    print()
    print(
        "Summary:"
    )
    print(
        SUMMARY_FILE
    )

    print()
    print("=" * 70)
    print(
        "✓ HISTORICAL 72-HOUR EVALUATION COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()