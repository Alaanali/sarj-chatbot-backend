import os
from typing import Any, Dict

import requests

WEATHER_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")

BASE_URL = "http://api.openweathermap.org/data/2.5"


def get_current_weather(city: str, units: str = "celsius") -> Dict[str, Any]:

    try:
        api_units = "metric" if units == "celsius" else "imperial"

        url = f"{BASE_URL}/weather"
        params = {"q": city, "appid": WEATHER_API_KEY, "units": api_units}

        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            return {"error": f"Weather API error: {response.status_code}", "error_code": "api_error"}

        data = response.json()
        return {
            "city": data["name"],
            "country": data["sys"]["country"],
            "temperature": round(data["main"]["temp"], 1),
            "feels_like": round(data["main"]["feels_like"], 1),
            "description": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "wind_speed": data.get("wind", {}).get("speed", 0),
            "wind_direction": data.get("wind", {}).get("deg", 0),
            "visibility": data.get("visibility", 0) / 1000,  # Convert to km
            "units": api_units,
            "icon": data["weather"][0]["icon"],
            "timestamp": data["dt"],
        }

    except Exception as e:
        return {"error": f"Unexpected error getting weather data: {str(e)}", "error_code": "unknown_error"}


def get_weather_forecast(city: str, days: int = 5) -> Dict[str, Any]:

    try:
        days = max(1, min(days, 5))

        url = f"{BASE_URL}/forecast"
        params = {"q": city, "appid": WEATHER_API_KEY, "units": "metric", "cnt": days * 8}  # 8 forecasts per day (every 3 hours)

        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            return {"error": f"Forecast API error: {response.status_code}", "error_code": "api_error"}

        data = response.json()

        forecasts = []
        current_date = None
        daily_data = {}

        for item in data["list"]:
            # Extract date
            date_str = item["dt_txt"].split(" ")[0]

            if date_str != current_date:
                # Save previous day data
                if current_date and daily_data:
                    forecasts.append(daily_data)

                # Start new day
                current_date = date_str
                daily_data = {
                    "date": date_str,
                    "high_temp": item["main"]["temp_max"],
                    "low_temp": item["main"]["temp_min"],
                    "description": item["weather"][0]["description"],
                    "icon": item["weather"][0]["icon"],
                    "humidity": item["main"]["humidity"],
                    "wind_speed": item["wind"]["speed"],
                }
            else:
                # Update daily data with min/max temps
                daily_data["high_temp"] = max(daily_data["high_temp"], item["main"]["temp_max"])
                daily_data["low_temp"] = min(daily_data["low_temp"], item["main"]["temp_min"])

        # Add the last day
        if daily_data:
            forecasts.append(daily_data)

        # Round temperatures
        for forecast in forecasts:
            forecast["high_temp"] = round(forecast["high_temp"], 1)
            forecast["low_temp"] = round(forecast["low_temp"], 1)

        return {
            "city": data["city"]["name"],
            "country": data["city"]["country"],
            "forecast": forecasts[:days],
            "days_requested": days,
            "days_returned": len(forecasts[:days]),
        }

    except Exception as e:
        return {"error": f"Error getting weather forecast: {str(e)}", "error_code": "unknown_error"}
