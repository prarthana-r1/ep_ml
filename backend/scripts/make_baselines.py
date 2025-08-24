import requests
import json
import pandas as pd
from datetime import datetime

# Open-Meteo API docs: https://open-meteo.com/en/docs
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Example locations (lat, lon). You can replace with subdivisions.json data.
LOCATIONS = {
    "delhi": (28.6139, 77.2090),
    "bengaluru": (12.9716, 77.5946),
    "mumbai": (19.0760, 72.8777),
}

# Parameters we want from Open-Meteo
PARAMS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "soil_moisture_0_to_7cm",
]

START_DATE = "2000-01-01"
END_DATE = "2020-12-31"


def fetch_weather(lat, lon, start=START_DATE, end=END_DATE):
    """Fetch historical weather for given lat/lon"""
    url = (
        f"{BASE_URL}?latitude={lat}&longitude={lon}"
        f"&start_date={start}&end_date={end}"
        f"&daily={','.join(PARAMS)}&timezone=auto"
    )
    print(f"Fetching {url}")
    r = requests.get(url)
    r.raise_for_status()
    return r.json()


def compute_baseline(data):
    """Compute long-term means (baseline) from daily weather"""
    df = pd.DataFrame(data["daily"])
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)

    # Monthly averages → better seasonal baseline
    monthly_means = df.groupby(df.index.month).mean().to_dict()

    return monthly_means


def main():
    baselines = {}

    for name, (lat, lon) in LOCATIONS.items():
        raw = fetch_weather(lat, lon)
        baseline = compute_baseline(raw)
        baselines[name] = baseline

    # Save results
    out_path = "../data/weather_baselines.json"
    with open(out_path, "w") as f:
        json.dump(baselines, f, indent=2)

    print(f"✅ Baselines saved to {out_path}")


if __name__ == "__main__":
    main()
