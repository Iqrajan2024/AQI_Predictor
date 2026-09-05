# Pearls AQI Predictor for PESHAWAR

### End-to-End MLOps System for 72-Hour Air Quality Index Forecasting

An end-to-end machine learning and MLOps system for forecasting the **Air Quality Index (AQI) for Peshawar, Pakistan**. The system combines real-time and historical weather and air-quality data, automated feature engineering, a Feast Feature Store, machine-learning model training and evaluation, MLflow model tracking and registry, a FastAPI prediction backend, SHAP explainability, GitHub Actions automation, and an interactive Streamlit dashboard.

## Deployed Application

**Live Dashboard:**
https://aqi-predictor-axaefuyojztztpjgazhdcq.streamlit.app/

**GitHub Repository:**
https://github.com/Iqrajan2024/AQI_Predictor

---

## Project Overview

The Pearls AQI Predictor provides:

* Current AQI monitoring
* AQI trend visualization
* 3-day / 72-hour AQI forecasting
* Automated feature engineering
* Feature Store integration using Feast
* MLflow model tracking and champion model management
* SHAP-based model explainability
* FastAPI backend
* Streamlit dashboard
* Automated data, feature, and training pipelines using GitHub Actions

The prediction target is the **next-hour AQI**, and 72-hour forecasts are generated recursively.

---

## Implementation Steps

The project follows this pipeline:

1. **Collect Data**

   * Historical and recent weather and air-quality data are collected from Open-Meteo.
   * Data is maintained for Peshawar.

2. **Feature Engineering**

   * Weather, pollutant, AQI lag, rolling average, and time-based features are generated.
   * The final model uses 70 features.

3. **Feature Store**

   * Features are managed using Feast for consistent training and prediction data.

4. **Model Training**

   * Ridge Regression
   * Random Forest
   * XGBoost


5. **Model Evaluation**

   * Models are evaluated using RMSE, MAE, and R².
   * The best-performing model is XGBoost. 

6. **Forecasting**

   * The champion model generates recursive 72-hour AQI predictions.

7. **Explainability**

   * SHAP is used to explain the factors influencing predictions.

8. **Deployment**

   * FastAPI provides prediction APIs.
   * Streamlit provides the user dashboard.
   * GitHub Actions automates feature processing and model training.

---

## Tech Stack

| Category            | Technologies           |
| ------------------- | ---------------------- |
| Language            | Python 3.13            |
| ML                  | Scikit-learn, Tensorflow|
| Data Processing     | Pandas, NumPy          |
| Feature Store       | Feast                  |
| Experiment Tracking | MLflow                 |
| Explainability      | SHAP                   |
| Backend             | FastAPI, Uvicorn       |
| Dashboard           | Streamlit              |
| Data Source         | Open-Meteo             |
| Automation          | GitHub Actions         |
| Testing             | Pytest                 |
| Version Control     | Git, GitHub            |

---

## Key Features

### 72-Hour AQI Forecasting

Generates hourly AQI predictions for the next three days using recursive forecasting.

### Multiple ML Models

Compares Ridge Regression, Random Forest, BiLSTM, and XGBoost to identify the best-performing model.

### Champion Model

MLflow maintains the production XGBoost model using the `champion` alias.

### Feature Engineering

Uses AQI history, pollutant lags, rolling statistics, weather variables, and temporal features.

### Explainable Predictions

SHAP explanations help identify which features influence AQI predictions.

### Dashboard

The Streamlit dashboard displays:

* Current AQI
* AQI category
* AQI trend
* Health alerts
* 3-day forecast
* Hourly predictions
* Model information
* Prediction explanations

### Automated MLOps

GitHub Actions automates the feature pipeline and model training workflow.

---

## Project Structure

```text
AQI_Predictor/
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── feature_pipeline.yml
│       └── training_pipeline.yml
│
├── data/
│   ├── evaluation/
│   └── predictions/
│
├── feature_repo/
│   └── feature_repo/
│       ├── features.py
│       ├── feature_store.cloud.yaml
│       └── feature_store.local.yaml
│
├── models/
│   ├── champion_model.pkl
│   ├── champion_metadata.json
│   └── xgboost_aqi.pkl
│
├── notebooks/
│   ├── eda.ipynb
│   ├── model_training.ipynb
│   └── three_day_forecasting.ipynb
│
├── scripts/
│   └── configure_feast.py
│
├── src/
│   ├── api/
│   ├── dashboard/
│   ├── data/
│   ├── evaluation/
│   ├── features/
│   ├── models/
│   └── prediction/
│
├── tests/
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/Iqrajan2024/AQI_Predictor.git
cd AQI_Predictor
```

### 2. Create and Activate Environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file or configure the required environment variables for:

* Open-Meteo/data services
* MLflow
* Feast
* API configuration


### 5. Run the FastAPI Backend

```bash
uvicorn src.api.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

### 6. Run the Streamlit Dashboard

In a second terminal:

```bash
streamlit run src/dashboard/app.py
```

The dashboard will open in your browser.

---


## Limitations

* Predictions depend on the quality and availability of external weather and air-quality data.
* Forecast accuracy can decrease as the prediction horizon increases because the 72-hour forecast is recursive.
* The current system is focused on Peshawar.
* AQI behavior can be affected by sudden events such as dust storms, fires, construction, or unusual weather conditions that may not be fully captured by historical features.
* External services such as MLflow, Feast, and Open-Meteo can affect system availability.

---

## Future Improvements

* Support AQI forecasting for multiple cities.
* Add additional environmental and meteorological data sources.
* Improve long-horizon forecasting models.
* Introduce automated model performance monitoring and retraining triggers.
* Improve dashboard customization and historical analysis.

---

## Summary

Pearls AQI Predictor is an end-to-end machine learning and MLOps system for AQI forecasting. It combines automated data collection, feature engineering, Feast feature store, multiple regression models, MLflow model management, SHAP explainability, FastAPI, Streamlit, and GitHub Actions into a complete forecasting pipeline.

The deployed system provides a practical interface for monitoring current air quality and forecasting AQI for the next 72 hours.


### Live Demo

 https://aqi-predictor-axaefuyojztztpjgazhdcq.streamlit.app/

### Repository

 https://github.com/Iqrajan2024/AQI_Predictor
