from feast import FeatureStore

store = FeatureStore(repo_path=".")

features = [
    "aqi_features:temperature_2m",
    "aqi_features:relative_humidity_2m",
    "aqi_features:pressure_msl",
    "aqi_features:precipitation",
    "aqi_features:wind_speed_10m",
    "aqi_features:wind_direction_10m",
    "aqi_features:pm2_5",
    "aqi_features:pm10",
    "aqi_features:carbon_monoxide",
    "aqi_features:nitrogen_dioxide",
    "aqi_features:sulphur_dioxide",
    "aqi_features:ozone",

    "aqi_features:hour",
    "aqi_features:day_of_week",
    "aqi_features:day_of_month",
    "aqi_features:month",
    "aqi_features:is_weekend",

    "aqi_features:aqi_lag_1",
    "aqi_features:aqi_lag_3",
    "aqi_features:aqi_lag_6",
    "aqi_features:aqi_lag_12",
    "aqi_features:aqi_lag_24",
    "aqi_features:aqi_lag_48",
    "aqi_features:aqi_lag_72",

    "aqi_features:pm2_5_lag_1",
    "aqi_features:pm2_5_lag_3",
    "aqi_features:pm2_5_lag_6",
    "aqi_features:pm2_5_lag_24",

    "aqi_features:pm10_lag_1",
    "aqi_features:pm10_lag_3",
    "aqi_features:pm10_lag_6",
    "aqi_features:pm10_lag_24",

    "aqi_features:carbon_monoxide_lag_1",
    "aqi_features:carbon_monoxide_lag_3",
    "aqi_features:carbon_monoxide_lag_6",
    "aqi_features:carbon_monoxide_lag_24",

    "aqi_features:nitrogen_dioxide_lag_1",
    "aqi_features:nitrogen_dioxide_lag_3",
    "aqi_features:nitrogen_dioxide_lag_6",
    "aqi_features:nitrogen_dioxide_lag_24",

    "aqi_features:sulphur_dioxide_lag_1",
    "aqi_features:sulphur_dioxide_lag_3",
    "aqi_features:sulphur_dioxide_lag_6",
    "aqi_features:sulphur_dioxide_lag_24",

    "aqi_features:ozone_lag_1",
    "aqi_features:ozone_lag_3",
    "aqi_features:ozone_lag_6",
    "aqi_features:ozone_lag_24",

    "aqi_features:aqi_3h_mean",
    "aqi_features:aqi_6h_mean",
    "aqi_features:aqi_12h_mean",
    "aqi_features:aqi_24h_mean",

    "aqi_features:pm2_5_3h_mean",
    "aqi_features:pm2_5_6h_mean",
    "aqi_features:pm2_5_24h_mean",

    "aqi_features:pm10_3h_mean",
    "aqi_features:pm10_6h_mean",
    "aqi_features:pm10_24h_mean",

    "aqi_features:carbon_monoxide_24h_mean",
    "aqi_features:nitrogen_dioxide_24h_mean",
    "aqi_features:sulphur_dioxide_24h_mean",
    "aqi_features:ozone_24h_mean",

    "aqi_features:aqi_change_1h",
    "aqi_features:aqi_change_3h",
    "aqi_features:aqi_change_6h",
    "aqi_features:aqi_change_24h",

    "aqi_features:pm2_5_change_1h",
    "aqi_features:pm2_5_change_24h",

    "aqi_features:pm10_change_1h",
    "aqi_features:pm10_change_24h",
]

result = store.get_online_features(
    features=features,
    entity_rows=[
        {"location_id": "peshawar"}
    ],
).to_dict()

print("=" * 60)
print("FEAST ONLINE FEATURE RETRIEVAL")
print("=" * 60)

for key, value in result.items():
    print(f"{key}: {value}")