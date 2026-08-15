import requests


# ============================================================
# PESHAWAR LOCATION
# ============================================================

LATITUDE = 34.008
LONGITUDE = 71.5785


# ============================================================
# DATE RANGE FOR TEST
# ============================================================

START_DATE = "2026-08-01"
END_DATE = "2026-08-04"


# ============================================================
# OPEN-METEO WEATHER API
# ============================================================

WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

weather_params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,

    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
        "pressure_msl",
        "precipitation",
        "wind_speed_10m",
        "wind_direction_10m"
    ],

    "timezone": "UTC"
}


# ============================================================
# REQUEST WEATHER DATA
# ============================================================

weather_response = requests.get(
    WEATHER_URL,
    params=weather_params,
    timeout=30
)

print("=" * 60)
print("OPEN-METEO WEATHER TEST")
print("=" * 60)

print("Status code:", weather_response.status_code)


if weather_response.status_code == 200:

    weather_data = weather_response.json()

    print("Weather request successful!")

    print()
    print("Latitude:", weather_data["latitude"])
    print("Longitude:", weather_data["longitude"])

    print()
    print("Hourly variables:")
    print(
        weather_data["hourly"].keys()
    )

    print(
        "Number of hourly records:",
        len(weather_data["hourly"]["time"])
    )

else:

    print("Weather request failed.")
    print(weather_response.text)


# ============================================================
# OPEN-METEO AIR QUALITY API
# ============================================================

AIR_QUALITY_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
)

air_quality_params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,

    "hourly": [
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",

        "us_aqi",
        "us_aqi_pm2_5",
        "us_aqi_pm10",
        "us_aqi_carbon_monoxide",
        "us_aqi_nitrogen_dioxide",
        "us_aqi_sulphur_dioxide",
        "us_aqi_ozone"
    ],

    "timezone": "UTC"
}


# ============================================================
# REQUEST AIR QUALITY DATA
# ============================================================

air_response = requests.get(
    AIR_QUALITY_URL,
    params=air_quality_params,
    timeout=30
)

print()
print("=" * 60)
print("OPEN-METEO AIR QUALITY TEST")
print("=" * 60)

print("Status code:", air_response.status_code)


if air_response.status_code == 200:

    air_data = air_response.json()

    print()
    print("=" * 60)
    print("AIR QUALITY UNITS")
    print("=" * 60)

    print(
        air_data["hourly_units"]
    )

    print("Air-quality request successful!")

    print()
    print("Latitude:", air_data["latitude"])
    print("Longitude:", air_data["longitude"])

    print()
    print("Hourly variables:")
    print(
        air_data["hourly"].keys()
    )

    print(
        "Number of hourly records:",
        len(air_data["hourly"]["time"])
    )

else:

    print("Air-quality request failed.")
    print(air_response.text)

print()
print("=" * 60)
print("SAMPLE WEATHER DATA")
print("=" * 60)

for i in range(5):

    print(
        weather_data["hourly"]["time"][i],
        "| Temperature:",
        weather_data["hourly"]["temperature_2m"][i],
        "| Humidity:",
        weather_data["hourly"]["relative_humidity_2m"][i],
        "| Pressure:",
        weather_data["hourly"]["pressure_msl"][i],
        "| Wind:",
        weather_data["hourly"]["wind_speed_10m"][i]
    )


print()
print("=" * 60)
print("SAMPLE AIR QUALITY DATA")
print("=" * 60)

for i in range(5):

    print(
        air_data["hourly"]["time"][i],
        "| PM2.5:",
        air_data["hourly"]["pm2_5"][i],
        "| PM10:",
        air_data["hourly"]["pm10"][i],
        "| CO:",
        air_data["hourly"]["carbon_monoxide"][i],
        "| NO2:",
        air_data["hourly"]["nitrogen_dioxide"][i],
        "| SO2:",
        air_data["hourly"]["sulphur_dioxide"][i],
        "| O3:",
        air_data["hourly"]["ozone"][i]
    )