"""
PEARLS AQI PREDICTOR
Streamlit Dashboard

Run from the project root:
    streamlit run src/dashboard/app.py

Backend:
    FastAPI -> http://127.0.0.1:8000

The dashboard consumes the existing FastAPI API. It does not bypass
the Feast feature-store / MLflow model-registry prediction pipeline.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import textwrap

import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Pearls AQI Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONFIG
# ============================================================

DEFAULT_API_URL = "http://127.0.0.1:8000"

EVALUATION_SUMMARY_FILE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "evaluation"
    / "historical_72h_summary.json"
)

EVALUATION_DAILY_METRICS_FILE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "evaluation"
    / "historical_72h_daily_metrics.csv"
)

# ============================================================
# CSS
# ============================================================

st.markdown(
    textwrap.dedent(
        """
        <style>
            .main-title {
                font-size: 2.25rem;
                font-weight: 800;
                margin-bottom: 0.15rem;
            }

            .subtitle {
                opacity: 0.70;
                font-size: 1rem;
                margin-bottom: 1.5rem;
            }

            /* Smaller RMSE metric values ONLY inside forecast cards */
            .st-key-forecast_card_1 [data-testid="stMetricValue"],
            .st-key-forecast_card_2 [data-testid="stMetricValue"],
            .st-key-forecast_card_3 [data-testid="stMetricValue"] {
                font-size: 1.25rem !important;
            }

            .aqi-card {
                padding: 1.25rem 1.35rem;
                border-radius: 18px;
                border: 1px solid rgba(128,128,128,0.20);
                min-height: 150px;
            }

            .aqi-value {
                font-size: 3rem;
                font-weight: 800;
                line-height: 1;
                margin: 0.45rem 0;
            }

            .aqi-category {
                font-size: 1.05rem;
                font-weight: 650;
            }

            .metric-card {
                padding: 1rem;
                border-radius: 16px;
                border: 1px solid rgba(128,128,128,0.20);
                min-height: 110px;
            }

            .metric-label {
                opacity: 0.70;
                font-size: 0.85rem;
            }

            .metric-value {
                font-size: 1.65rem;
                font-weight: 750;
                margin-top: 0.3rem;
            }

        
            .section-gap {
                margin-top: 1rem;
            }

            .shap-note {
                opacity: 0.70;
                font-size: 0.88rem;
            }
        </style>
        """
        ),
        unsafe_allow_html=True,
    )


# ============================================================
# SESSION STATE
# ============================================================

if "api_url" not in st.session_state:
    st.session_state.api_url = DEFAULT_API_URL.rstrip("/")


# ============================================================
# HELPERS
# ============================================================

def api_get(endpoint: str, timeout: int = 45) -> Any:
    """GET one FastAPI endpoint and return decoded JSON."""
    url = f"{st.session_state.api_url}{endpoint}"

    try:
        response = requests.get(url, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Unable to connect to FastAPI at {st.session_state.api_url}. "
            f"Make sure the API is running. Error: {exc}"
        ) from exc

    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text

        raise RuntimeError(
            f"FastAPI returned HTTP {response.status_code}: {detail}"
        )

    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"FastAPI returned invalid JSON from {url}."
        ) from exc


def get_daily_rmse_vs_persistence() -> dict[int, float]:
    """
    Load day-by-day RMSE improvement versus persistence
    from the historical 72-hour evaluation CSV.

    Returns:
        {
            1: Day 1 improvement percentage,
            2: Day 2 improvement percentage,
            3: Day 3 improvement percentage,
        }

    Positive = champion model has lower/better RMSE
    than persistence.
    """

    try:

        if not EVALUATION_DAILY_METRICS_FILE.exists():
            return {}

        daily_df = pd.read_csv(
            EVALUATION_DAILY_METRICS_FILE
        )

        if daily_df.empty:
            return {}

        results = {}

        for _, row in daily_df.iterrows():

            lead_hours = str(
                row.get("lead_hours", "")
            ).strip()

            improvement = row.get(
                "rmse_improvement_percent"
            )

            if pd.isna(improvement):
                continue

            # Map lead-hour ranges to forecast days.
            if lead_hours == "1-24":
                day_number = 1

            elif lead_hours == "25-48":
                day_number = 2

            elif lead_hours == "49-72":
                day_number = 3

            else:
                continue

            results[day_number] = float(
                improvement
            )

        return results

    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
    ):
        return {}


@st.cache_data(ttl=0, show_spinner=False)
def get_dashboard_data(api_url: str) -> dict:
    """Load current AQI, current trend, and forecast data."""
    base = api_url.rstrip("/")

    def get(path: str) -> Any:
        response = requests.get(f"{base}{path}", timeout=45)
        response.raise_for_status()
        return response.json()

    return {
        "current": get("/current"),
        "current_trend": get("/current/trend"),
        "forecast": get("/forecast"),
    }


@st.cache_data(ttl=0, show_spinner=False) 
def get_shap_data(api_url: str, forecast_day: str) -> dict: 
    """ Load fresh SHAP explanations for one forecast day. 
    ttl=0 ensures that dashboard reruns do not keep an old 
    SHAP response for the same date. """ 
    base = api_url.rstrip("/") 
    response = requests.get( 
        f"{base}/shap/{forecast_day}", 
        timeout=90, 
    ) 
    response.raise_for_status() 
    return response.json()

# ============================================================ 
# SHAP FEATURE AGGREGATION  
# ============================================================
def aggregate_daily_shap(explanations: list[dict]) -> pd.DataFrame:

    """ Aggregate the 24 hourly SHAP explanations for one day. 
    Each technical feature remains separate. 
    mean_shap: 
    Average signed SHAP contribution across the 24 hours. 
    mean_abs_shap: Average magnitude of the SHAP contribution across 
    the 24 hours. Used for importance ranking. 
    """
    rows = []

    for explanation in explanations:
        timestamp = explanation.get("timestamp")
        for feature_data in explanation.get("features", []):
            feature = feature_data.get("feature")

            if not feature:
                continue

            try:
                shap_value = float(
                    feature_data.get(
                        "shap_value",
                        0.0,
                    )
                )
            except (TypeError, ValueError):
                continue

            rows.append(
                {
                    "timestamp": timestamp,
                    "feature": str(feature),
                    "shap_value": shap_value,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "feature",
                "mean_shap",
                "mean_abs_shap",
            ]
        )

    raw_df = pd.DataFrame(rows)

    aggregate_df = (
        raw_df
        .groupby("feature", as_index=False)
        .agg(
            mean_shap=(
                "shap_value",
                "mean",
            ),
            mean_abs_shap=(
                "shap_value",
                lambda x: x.abs().mean(),
            ),
        )
    )

    return aggregate_df

# ============================================================ 
# SHAP EXPLANATION TEXT 
# ============================================================ 

def build_shap_explanation( 
        aggregate_df: pd.DataFrame, 
        forecast_date: str, 
) -> tuple[str, str]: 
    """ Build the natural-language explanation from the actual 
    strongest SHAP contributors for this specific forecast day. 
    
    Returns: message_type, message 
    """ 
    if aggregate_df.empty: 
        return ( 
            "info",
            f"No meaningful SHAP explanation is available " 
            f"for {forecast_date}.", 
        ) 



def format_number(value: Any, decimals: int = 1) -> str:
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def friendly_feature_name(feature: str) -> str:
    """
    Convert technical ML feature names into explanations
    that a normal dashboard user can understand.
    """

    names = {
        "temperature_2m": "Temperature",
        "relative_humidity_2m": "Humidity",
        "pressure_msl": "Air Pressure",
        "precipitation": "Rainfall",
        "wind_speed_10m": "Wind Speed",
        "wind_direction_10m": "Wind Direction",

        "pm2_5": "PM2.5",
        "pm10": "PM10",
        "carbon_monoxide": "Carbon Monoxide",
        "nitrogen_dioxide": "Nitrogen Dioxide",
        "sulphur_dioxide": "Sulphur Dioxide",
        "ozone": "Ozone",

        "hour": "Time of Day",
        "day_of_week": "Day of Week",
        "day_of_month": "Day of Month",
        "month": "Month",
        "is_weekend": "Weekend",

        "aqi_lag_1": "AQI — Previous Hour",
        "aqi_lag_3": "AQI — 3 Hours Earlier",
        "aqi_lag_6": "AQI — 6 Hours Earlier",
        "aqi_lag_12": "AQI — 12 Hours Earlier",
        "aqi_lag_24": "AQI — Previous Day",
        "aqi_lag_48": "AQI — 2 Days Earlier",
        "aqi_lag_72": "AQI — 3 Days Earlier",

        "pm2_5_lag_1": "PM2.5 — Previous Hour",
        "pm2_5_lag_3": "PM2.5 — 3 Hours Earlier",
        "pm2_5_lag_6": "PM2.5 — 6 Hours Earlier",
        "pm2_5_lag_24": "PM2.5 — Previous Day",

        "pm10_lag_1": "PM10 — Previous Hour",
        "pm10_lag_3": "PM10 — 3 Hours Earlier",
        "pm10_lag_6": "PM10 — 6 Hours Earlier",
        "pm10_lag_24": "PM10 — Previous Day",

        "carbon_monoxide_lag_1": "Carbon Monoxide — Previous Hour",
        "carbon_monoxide_lag_3": "Carbon Monoxide — 3 Hours Earlier",
        "carbon_monoxide_lag_6": "Carbon Monoxide — 6 Hours Earlier",
        "carbon_monoxide_lag_24": "Carbon Monoxide — Previous Day",

        "nitrogen_dioxide_lag_1": "Nitrogen Dioxide — Previous Hour",
        "nitrogen_dioxide_lag_3": "Nitrogen Dioxide — 3 Hours Earlier",
        "nitrogen_dioxide_lag_6": "Nitrogen Dioxide — 6 Hours Earlier",
        "nitrogen_dioxide_lag_24": "Nitrogen Dioxide — Previous Day",

        "sulphur_dioxide_lag_1": "Sulphur Dioxide — Previous Hour",
        "sulphur_dioxide_lag_3": "Sulphur Dioxide — 3 Hours Earlier",
        "sulphur_dioxide_lag_6": "Sulphur Dioxide — 6 Hours Earlier",
        "sulphur_dioxide_lag_24": "Sulphur Dioxide — Previous Day",

        "ozone_lag_1": "Ozone — Previous Hour",
        "ozone_lag_3": "Ozone — 3 Hours Earlier",
        "ozone_lag_6": "Ozone — 6 Hours Earlier",
        "ozone_lag_24": "Ozone — Previous Day",

        "aqi_3h_mean": "Average AQI — Last 3 Hours",
        "aqi_6h_mean": "Average AQI — Last 6 Hours",
        "aqi_12h_mean": "Average AQI — Last 12 Hours",
        "aqi_24h_mean": "Average AQI — Last 24 Hours",

        "pm2_5_3h_mean": "Average PM2.5 — Last 3 Hours",
        "pm2_5_6h_mean": "Average PM2.5 — Last 6 Hours",
        "pm2_5_24h_mean": "Average PM2.5 — Last 24 Hours",

        "pm10_3h_mean": "Average PM10 — Last 3 Hours",
        "pm10_6h_mean": "Average PM10 — Last 6 Hours",
        "pm10_24h_mean": "Average PM10 — Last 24 Hours",

        "carbon_monoxide_24h_mean": "Average Carbon Monoxide — Last 24 Hours",
        "nitrogen_dioxide_24h_mean": "Average Nitrogen Dioxide — Last 24 Hours",
        "sulphur_dioxide_24h_mean": "Average Sulphur Dioxide — Last 24 Hours",
        "ozone_24h_mean": "Average Ozone — Last 24 Hours",

        "aqi_change_1h": "AQI Change — Last Hour",
        "aqi_change_3h": "AQI Change — Last 3 Hours",
        "aqi_change_6h": "AQI Change — Last 6 Hours",
        "aqi_change_24h": "AQI Change — Last 24 Hours",

        "pm2_5_change_1h": "PM2.5 Change — Last Hour",
        "pm2_5_change_24h": "PM2.5 Change — Last 24 Hours",

        "pm10_change_1h": "PM10 Change — Last Hour",
        "pm10_change_24h": "PM10 Change — Last 24 Hours",
    }

    return names.get(
        feature,
        str(feature).replace("_", " ").title(),
    )

def category_badge(category: str) -> str:
    category_lower = str(category).lower()

    if category_lower == "good":
        return "🟢"
    if category_lower == "moderate":
        return "🟡"
    if "sensitive" in category_lower:
        return "🟠"
    if category_lower == "unhealthy":
        return "🔴"
    if "very unhealthy" in category_lower:
        return "🟣"
    return "⚫"

def forecast_category_class(category: str) -> str:
    """
    Return a CSS class for the AQI category indicator.
    """

    category_lower = str(category).strip().lower()

    if category_lower == "good":
        return "forecast-good"

    if category_lower == "moderate":
        return "forecast-moderate"

    if "sensitive" in category_lower:
        return "forecast-sensitive"

    if category_lower == "unhealthy":
        return "forecast-unhealthy"

    if "very unhealthy" in category_lower:
        return "forecast-very-unhealthy"

    if "hazardous" in category_lower:
        return "forecast-hazardous"

    return "forecast-unknown"

def display_timestamp(timestamp: str) -> str:
    """
    Convert UTC timestamp from FastAPI to
    Pakistan Standard Time (Peshawar).
    """
    try:
        dt = pd.to_datetime(timestamp, utc=True)

        # Convert UTC -> Pakistan Standard Time
        dt = dt.tz_convert("Asia/Karachi")

        return dt.strftime("%d %b %Y, %I:%M %p PKT")

    except Exception:
        return str(timestamp)

def theme_colors() -> tuple[str, str, str]:
    """
    Use a neutral palette that remains readable under either Streamlit
    light or dark theme.
    """
    return "#4F46E5", "#EF4444", "#64748B"


def build_line_chart(
    dataframe: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    y_title: str = "AQI",
) -> go.Figure:
    primary, warning, muted = theme_colors()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dataframe[x_col],
            y=dataframe[y_col],
            mode="lines+markers",
            name="AQI",
            line=dict(width=3, color=primary),
            marker=dict(size=5),
            hovertemplate=(
                "%{x}<br>"
                "AQI: %{y:.1f}"
                "<extra></extra>"
            ),
        )
    )

    # AQI reference thresholds.
    thresholds = [
        (50, "Good"),
        (100, "Moderate"),
        (150, "Unhealthy for Sensitive Groups"),
        (200, "Unhealthy"),
        (300, "Very Unhealthy"),
    ]

    for threshold, label in thresholds:
        fig.add_hline(
            y=threshold,
            line_width=1,
            line_dash="dot",
            line_color=muted,
            annotation_text=label,
            annotation_position="top right",
            opacity=0.55,
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time (PKT)",
        yaxis_title=y_title,
        height=420,
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode="x unified",
        template="plotly_white",
    )

    return fig


def build_shap_chart(
    contributions: list[dict],
    title: str,
    top_n: int = 10,
) -> go.Figure:
    primary, warning, muted = theme_colors()

    df = pd.DataFrame(contributions)

    if df.empty:
        return go.Figure()

    df = (
        df.sort_values("abs_shap_value", ascending=False)
        .head(top_n)
        .sort_values("shap_value")
    )

    colors = [
        warning if value > 0 else primary
        for value in df["shap_value"]
    ]

    fig = go.Figure(
        go.Bar(
            x=df["shap_value"],
            y=df["feature"],
            orientation="h",
            marker_color=colors,
            hovertemplate=(
                "%{y}<br>"
                "SHAP: %{x:.4f}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_vline(
        x=0,
        line_width=1,
        line_color=muted,
    )

    fig.update_layout(
        title=title,
        xaxis_title="SHAP value",
        yaxis_title="Feature",
        height=430,
        margin=dict(l=10, r=10, t=60, b=10),
        template="plotly_white",
    )

    return fig

def normalize_forecast_days(forecast: dict) -> list[dict]:
    """
    Return exactly the three daily forecast records supplied by FastAPI.

    The prediction pipeline produces the next three calendar days,
    excluding the current day.
    """
    days = forecast.get("daily", [])

    normalized = []

    for index, record in enumerate(days[:3], start=1):
        normalized.append(
            {
                "day_number": index,
                "date": record.get("date"),
                "aqi": record.get("aqi"),
                "category": record.get(
                    "category",
                    "Unknown",
                ),
                "health_alert": record.get(
                    "health_alert",
                    {},
                ),
                
            }
        )

    return normalized


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Dashboard Settings")

    api_url = st.text_input(
        "FastAPI URL",
        value=st.session_state.api_url,
    ).rstrip("/")

    st.session_state.api_url = api_url

    st.divider()

    st.subheader("Theme")

    st.info(
        "Use Streamlit's Settings menu (⋮ → Settings) to switch "
        "between the configured light and dark themes."
    )

    st.divider()

    refresh = st.button(
        "🔄 Refresh dashboard",
        use_container_width=True,
    )

    if refresh:
        st.cache_data.clear()
        st.rerun()

    st.caption(
        "Pearls AQI Predictor • Peshawar"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title"> Pearls AQI Predictor</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Peshawar Air Quality Monitoring & 3-Day Forecast"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DASHBOARD
# ============================================================

try:
    dashboard_data = get_dashboard_data(
        st.session_state.api_url
    )
except Exception as exc:
    st.error(str(exc))
    st.info(
        "Start the FastAPI backend first, for example:\n\n"
        "`uvicorn src.api.main:app --reload --port 8000`"
    )
    st.stop()


current = dashboard_data["current"]
current_trend = dashboard_data["current_trend"]
forecast = dashboard_data["forecast"]


# ============================================================
# CURRENT AQI
# ============================================================

st.subheader(" Current Air Quality")

current_aqi = current.get("aqi")
current_category = current.get("category", "Unknown")
current_timestamp = current.get("timestamp", "")

col_aqi, col_update = st.columns([2, 1])

with col_aqi:
    st.markdown(
        f"""
        <div class="aqi-card">
            <div>Current AQI</div>
            <div class="aqi-value">{format_number(current_aqi, 0)}</div>
            <div class="aqi-category">
                {category_badge(current_category)}
                {current_category}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_update:
    st.markdown(
        f"""
        <div class="aqi-card">
            <div>Last Updated</div>
            <div style="font-size:1.25rem;font-weight:700;margin-top:0.8rem;">
                {display_timestamp(current_timestamp)}
            </div>
            <div style="opacity:0.7;margin-top:0.6rem;">
                Location: Peshawar
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# POLLUTANTS
# ============================================================

st.markdown("###  Current Pollutants")

pollutants = current.get("pollutants", {})

pollutant_definitions = [
    ("PM2.5", "pm2_5", "μg/m³"),
    ("PM10", "pm10", "μg/m³"),
    ("O₃", "ozone", "μg/m³"),
    ("NO₂", "nitrogen_dioxide", "μg/m³"),
    ("SO₂", "sulphur_dioxide", "μg/m³"),
    ("CO", "carbon_monoxide", "μg/m³"),
]

pollutant_cols = st.columns(6)

for column, (label, key, unit) in zip(
    pollutant_cols,
    pollutant_definitions,
):
    with column:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">
                    {format_number(pollutants.get(key), 1)}
                </div>
                <div class="metric-label">{unit}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# CURRENT 24-HOUR TREND
# ============================================================

st.markdown("###  Current 24-Hour AQI Trend")

trend_hours = current_trend.get("hours", [])

if trend_hours:
    trend_df = pd.DataFrame(trend_hours)
    trend_df["timestamp"] = (
        pd.to_datetime(
        trend_df["timestamp"],
        utc=True,
    )
    .dt.tz_convert("Asia/Karachi")
    )
    trend_df["aqi"] = pd.to_numeric(
        trend_df["aqi"],
        errors="coerce",
    )
    trend_df = trend_df.dropna(
        subset=["timestamp", "aqi"]
    )

    if len(trend_df) > 24:
        trend_df = trend_df.tail(24)

    st.plotly_chart(
        build_line_chart(
            trend_df,
            "timestamp",
            "aqi",
            "AQI — Last 24 Hours",
        ),
        width="stretch",
    )
else:
    st.warning("No current 24-hour AQI trend is available.")


# ============================================================
# THREE-DAY FORECAST
# ============================================================

st.divider()

st.subheader("Next 3-Day AQI Forecast for Peshawar")

daily_rmse_vs_persistence = (
    get_daily_rmse_vs_persistence()
)

forecast_days = normalize_forecast_days(forecast)

if not forecast_days:

    st.warning("No forecast days were returned by FastAPI.")

else:

    # ========================================================
    # THREE EQUAL FORECAST CARDS
    # ========================================================

    cards = st.columns(
        3,
        gap="medium",
        vertical_alignment="top",
    )

    for card, item in zip(cards, forecast_days):

        with card:

            # ------------------------------------------------
            # CARD CONTAINER
            # ------------------------------------------------

            with st.container(
                border=True,
                key=f"forecast_card_{item['day_number']}",
                gap="small",
            ):

                date_text = item.get("date") or "Unknown date"

                aqi = item.get("aqi")

                category = (
                    item.get("category")
                    or "Unknown"
                )

                health_alert = item.get("health_alert") or {}
                alert_active = bool(health_alert.get("alert", False))
                alert_level = health_alert.get("level", category)
                alert_message = health_alert.get("message", "")
                alert_recommendation = health_alert.get("recommendation", "")

                # ------------------------------------------------
                # DAY
                # ------------------------------------------------

                st.markdown(
                    f"**Day {item['day_number']}**"
                )

                # ------------------------------------------------
                # DATE
                # ------------------------------------------------

                st.caption(
                    f"📅 {date_text}"
                )

                # ------------------------------------------------
                # PREDICTED AQI
                # ------------------------------------------------

                st.markdown(
                    '<div style="'
                    'font-size:0.82rem;'
                    'color:rgba(226,232,240,0.70);'
                    'margin-top:0.45rem;'
                    'margin-bottom:0.15rem;'
                    '">'
                    'Predicted AQI'
                    '</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f'<div style="'
                    f'font-size:2.15rem;'
                    f'font-weight:700;'
                    f'line-height:1.1;'
                    f'margin-bottom:0.65rem;'
                    f'">'
                    f'{format_number(aqi, 1)}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # ------------------------------------------------
                # CATEGORY
                # ------------------------------------------------

                category_emoji = category_badge(category)

                st.markdown(
                    f'<div style="'
                    f'display:flex;'
                    f'align-items:center;'
                    f'gap:0.45rem;'
                    f'font-size:0.88rem;'
                    f'font-weight:600;'
                    f'line-height:1.3;'
                    f'min-height:2.3rem;'
                    f'margin-bottom:0.7rem;'
                    f'">'
                    f'<span style="font-size:0.95rem;">'
                    f'{category_emoji}'
                    f'</span>'
                    f'<span>{category}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # ------------------------------------------------
                # RMSE VS PERSISTENCE
                # ------------------------------------------------

                day_number = item["day_number"]

                rmse_vs_persistence = (
                    daily_rmse_vs_persistence.get(day_number)
                )

                if rmse_vs_persistence is None:

                    st.metric(
                        label="Champion RMSE vs Persistence",
                        value="N/A",
                        delta=None,
                        border=True,
                    )

                else:

                    st.metric(
                        label="Champion RMSE vs Persistence",
                        value=f"{rmse_vs_persistence:+.2f}%",
                        delta=None,
                        border=True,
                        help=(
                            "Percentage improvement in RMSE compared with "
                            "the fixed-origin persistence forecast for this "
                            "forecast day. Positive means the champion model "
                            "has lower RMSE."
                        ),
                    )

                # ------------------------------------------------
                # HEALTH ALERT
                # ------------------------------------------------
                
                category_lower = str(category).strip().lower()

                if alert_message or alert_recommendation:

                    if category_lower == "good":
                        alert_function = st.success

                    elif category_lower == "moderate":
                        alert_function = st.warning

                    elif "sensitive" in category_lower:
                        alert_function = st.warning

                    elif category_lower == "unhealthy":
                        alert_function = st.error

                    elif "very unhealthy" in category_lower:
                        alert_function = st.error

                    elif "hazardous" in category_lower:
                        alert_function = st.error

                    else:
                        alert_function = st.info

                    message = (
                        f"**{alert_level} — Health Alert**\n\n"
                        f"{alert_message}"
                    )

                    if alert_recommendation:
                        message += (
                            f"\n\n**Recommendation:** "
                            f"{alert_recommendation}"
                        )

                    alert_function(message)
                            
                            
# ============================================================
# 72-HOUR FORECAST TREND
# ============================================================

st.markdown("###  Predicted AQI — Next 72 Hours")

hourly_forecast = forecast.get("hourly", [])

if hourly_forecast:
    hourly_df = pd.DataFrame(hourly_forecast)

    hourly_df["timestamp"] = pd.to_datetime(
        hourly_df["timestamp"],
        utc=True,
        errors="coerce",
    )

    hourly_df["aqi"] = pd.to_numeric(
        hourly_df["aqi"],
        errors="coerce",
    )

    hourly_df = hourly_df.dropna(
        subset=["timestamp", "aqi"]
    )

    # Keep exactly the 72-hour prediction horizon if the API
    # returns more data.
    hourly_df = hourly_df.sort_values("timestamp").head(72)

    fig = build_line_chart(
        hourly_df,
        "timestamp",
        "aqi",
        "Predicted AQI — Next 72 Hours",
    )

    
    fig.update_xaxes(
        type="date",
        range=[
            hourly_df["timestamp"].min(),
            hourly_df["timestamp"].max()
        ],
        autorange=False,
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

    st.caption(
        "The forecast shown here represents the next 72 hours "
        "(3 days), excluding the current day."
    )
else:
    st.warning("No 72-hour forecast data is available.")



# ============================================================
# WHY IS THE AQI EXPECTED TO BE THIS WAY?
# ============================================================

st.divider()

st.subheader(" Why Is the AQI Expected to Be This Way?")

st.markdown(
    """
    The charts below show the main factors that influenced each
    day's predicted AQI.

    **🟠 Orange bars** indicate factors that generally pushed the
    predicted AQI higher.

    **🔵 Blue bars** indicate factors that generally pushed the
    predicted AQI lower.

    The longer the bar, the stronger the influence on the prediction.
    """
)

if not forecast_days:

    st.info(
        "Daily explanations cannot be displayed because "
        "no forecast days are available."
    )

else:

    for item in forecast_days:

        forecast_date = item["date"]

        if not forecast_date:
            continue

        with st.expander(
            f" Day {item['day_number']} — {forecast_date}",
            expanded=(item["day_number"] == 1),
        ):

            try:

                # ------------------------------------------------
                # LOAD SHAP DATA
                # ------------------------------------------------

                shap_data = get_shap_data(
                    st.session_state.api_url,
                    forecast_date,
                )

                explanations = shap_data.get(
                    "explanations",
                    [],
                )

                if not explanations:

                    st.warning(
                        "No explanation data is available "
                        f"for {forecast_date}."
                    )

                    continue

                # ------------------------------------------------
                # AGGREGATE ALL 24 HOURS
                # ------------------------------------------------

                aggregate_df = aggregate_daily_shap(
                    explanations
                )

                if aggregate_df.empty:
                    st.warning(
                        "No meaningful SHAP feature contributions "
                        f"are available for {forecast_date}."
                    )

                    continue

                # ------------------------------------------------
                # CONVERT TO USER-FRIENDLY NAMES
                # ------------------------------------------------

                aggregate_df["friendly_name"] = (
                    aggregate_df["feature"]
                    .apply(
                        friendly_feature_name
                    )
                )

                # ==================================================== 
                # TOP 8 FEATURES  
                # ==================================================== 
                
                chart_df = ( 
                    aggregate_df 
                    .sort_values( 
                        "mean_abs_shap", 
                        ascending=False, 
                    ) 
                    .head(8) 
                    .sort_values( 
                        "mean_shap", 
                        ascending=True, 
                    ) .copy() 
                )

                # ------------------------------------------------
                # DETERMINE BAR COLORS
                # ------------------------------------------------

                primary, warning, muted = theme_colors()

                colors = [
                    warning
                    if value > 0
                    else primary
                    for value in chart_df[
                        "mean_shap"
                    ]
                ]

                # ------------------------------------------------
                # CREATE USER-FRIENDLY CHART
                # ------------------------------------------------

                fig = go.Figure()

                fig.add_trace(
                    go.Bar(
                        x=chart_df[
                            "mean_shap"
                        ],
                        y=chart_df[
                            "friendly_name"
                        ],
                        orientation="h",
                        marker_color=colors,

                        customdata=chart_df[
                            [
                                "feature",
                                "mean_abs_shap",
                            ]
                        ].values,

                        hovertemplate=(
                            "<b>%{y}</b><br>"
                            "Influence: %{x:.2f}"
                            "<extra></extra>"
                        ),
                    )
                )

                fig.add_vline(
                    x=0,
                    line_width=1,
                    line_color=muted,
                )

                fig.update_layout(
                    title=(
                        f"Main Factors Influencing "
                        f"the {forecast_date} Forecast"
                    ),
                    xaxis_title=(
                        "Influence on Predicted AQI"
                    ),
                    yaxis_title="",
                    height=450,
                    margin=dict(
                        l=10,
                        r=10,
                        t=70,
                        b=10,
                    ),
                    template="plotly_white",
                    showlegend=False,
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                    key=f"shap_chart_{forecast_date}",
                )

                # ====================================================  
                # TOP POSITIVE CONTRIBUTORS  
                # ==================================================== 
                
                positive_df = ( 
                    aggregate_df[ 
                        aggregate_df["mean_shap"] > 0 
                    ] 
                    .sort_values( 
                        "mean_abs_shap", 
                        ascending=False, 
                    ) 
                    .head(3) 
                )

                # ====================================================  
                # TOP NEGATIVE CONTRIBUTORS  
                # ==================================================== 
                
                negative_df = ( 
                    aggregate_df[ 
                        aggregate_df["mean_shap"] < 0 
                        ] 
                        .sort_values( 
                            "mean_abs_shap", 
                            ascending=False, 
                        ) 
                        .head(3) 
                    )
                
                # ====================================================  
                # DAILY EXPLANATION  
                # ==================================================== 
                
                st.markdown( 
                    "### What influenced this forecast?" 
                ) 

                explanation_col1, explanation_col2 = st.columns(2)

                # ----------------------------------------------------  
                # HIGHER AQI CONTRIBUTORS  
                # ---------------------------------------------------- 
                
                with explanation_col1: 
                    st.markdown( 
                        "**🟠 Factors pushing AQI higher**" 
                    ) 
                    if positive_df.empty: 
                        st.caption( 
                            "No consistently positive " 
                            "contributors were identified." 
                        ) 
                    else: 
                        for _, row in positive_df.iterrows(): 
                            feature_name = ( 
                                friendly_feature_name( 
                                    row["feature"] 
                                ) 
                            ) 

                            st.markdown( 
                                f"- **{feature_name}** " 
                                f"({row['mean_shap']:+.2f})" 
                            )

                # ----------------------------------------------------  
                # LOWER AQI CONTRIBUTORS  
                # ---------------------------------------------------- 
                
                with explanation_col2: 
                    st.markdown( 
                        "**🔵 Factors pushing AQI lower**" 
                    ) 

                    if negative_df.empty: 
                        st.caption( 
                            "No consistently negative " 
                            "contributors were identified." 
                        ) 
                    else: 
                        for _, row in negative_df.iterrows(): 
                            feature_name = ( 
                                friendly_feature_name( 
                                    row["feature"] 
                                ) 
                            ) 

                            st.markdown( 
                                f"- **{feature_name}** " 
                                f"({row['mean_shap']:+.2f})" 
                            )

                # ====================================================
                # STRONGEST FEATURE EXPLANATION
                # ====================================================

                # Features that should not dominate the natural-language
                # explanation every time because they are historical AQI lags.
                EXCLUDED_FROM_PRIMARY_EXPLANATION = {
                    "aqi_lag_1",
                    "aqi_lag_3",
                    "aqi_lag_6",
                    "aqi_lag_12",
                    "aqi_lag_24",
                    "aqi_lag_48",
                    "aqi_lag_72",
                }

                # First try to find the strongest non-AQI-lag feature.
                meaningful_df = aggregate_df[
                    ~aggregate_df["feature"].isin(
                        EXCLUDED_FROM_PRIMARY_EXPLANATION
                    )
                ].copy()

                # If there are meaningful non-lag features available,
                # use the strongest one.
                if not meaningful_df.empty:

                    strongest = (
                        meaningful_df
                        .sort_values(
                            "mean_abs_shap",
                            ascending=False,
                        )
                        .iloc[0]
                    )

                else:

                    # Fallback: if every available feature is an AQI lag,
                    # use the strongest actual feature.
                    strongest = (
                        aggregate_df
                        .sort_values(
                            "mean_abs_shap",
                            ascending=False,
                        )
                        .iloc[0]
                    )


                strongest_feature = str(
                    strongest["feature"]
                )

                strongest_name = friendly_feature_name(
                    strongest_feature
                )

                strongest_value = float(
                    strongest["mean_shap"]
                )


                # ====================================================
                # DISPLAY NATURAL-LANGUAGE EXPLANATION
                # ====================================================

                if strongest_value > 0:

                    st.error(
                        f"**What this means:** "
                        f"{strongest_name} was one of the strongest factors "
                        f"pushing the predicted AQI higher on "
                        f"{forecast_date}."
                    )

                elif strongest_value < 0:

                    st.success(
                        f"**What this means:** "
                        f"{strongest_name} was one of the strongest factors "
                        f"helping to keep the predicted AQI lower on "
                        f"{forecast_date}."
                    )

                else:

                    st.info(
                        f"**What this means:** "
                        f"{strongest_name} had the strongest overall "
                        f"influence on the prediction for "
                        f"{forecast_date}."
                    )

                # ==================================================== 
                # TECHNICAL VALIDATION  
                # ==================================================== 
                
                with st.expander( 
                    "View SHAP feature details" 
                ): 
                    details_df = ( 
                        aggregate_df 
                        .sort_values( 
                            "mean_abs_shap", 
                            ascending=False, 
                        ) 
                        .copy() 
                    ) 

                    details_df["friendly_name"] = ( 
                        details_df["feature"] 
                        .apply( 
                            friendly_feature_name 
                        ) 
                    ) 

                    details_df = details_df[ 
                        [ 
                            "feature", 
                            "friendly_name", 
                            "mean_shap", 
                            "mean_abs_shap", 
                        ] 
                    ] 

                    details_df = details_df.rename( 
                        columns={ 
                            "feature": 
                                "Technical Feature", 

                            "friendly_name": 
                                "Dashboard Feature", 

                            "mean_shap": 
                                "Mean SHAP", 

                            "mean_abs_shap": 
                                "Mean Absolute SHAP", 
                        } 
                    ) 

                    st.dataframe( 
                        details_df, 
                        width="stretch", 
                        hide_index=True, 
                    ) 

                st.caption( 
                        "These explanations summarize the factors " 
                        "considered by the prediction model. They do " 
                        "not mean that one factor alone determines AQI." 
                    ) 

            except requests.HTTPError as exc: 

                st.error( 
                        f"Unable to load the explanation for " 
                        f"{forecast_date}: {exc}" 
                ) 

            except Exception as exc: 

                st.error( 
                        f"Unable to display the explanation for " 
                        f"{forecast_date}: {exc}" 
                )         
                # ------------------------------------------------
                # SIMPLE LEGEND
                # ------------------------------------------------

                st.markdown(
                    """
                   
                    These explanations summarize the factors considered
                    by the prediction model; they do not mean that one
                    factor alone determines the AQI.
                    """
                )

            except requests.HTTPError as exc:

                st.error(
                    f"Unable to load the explanation for "
                    f"{forecast_date}: {exc}"
                )

            except Exception as exc:

                st.error(
                    f"Unable to display the explanation for "
                    f"{forecast_date}: {exc}"
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    f"Pearls AQI Predictor - Peshawar • FastAPI backend: "
    f"Dashboard generated {datetime.now().strftime('%d %b %Y %H:%M')}"
)