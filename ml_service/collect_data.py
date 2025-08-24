import requests
import pandas as pd
import numpy as np
import time
import os

# ------------------ Step 1: Fetch Historical Weather ------------------
def fetch_historical(name, lat, lon, start_date, end_date):
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&"
        f"hourly=temperature_2m,relativehumidity_2m,windspeed_10m,precipitation&timezone=auto"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"❌ Error fetching {lat},{lon} from {start_date} to {end_date}: {e}")
        return pd.DataFrame()

    hourly = data.get("hourly", {})
    if not hourly:
        print(f"⚠️ No data for {lat},{lon} {start_date}–{end_date}")
        return pd.DataFrame()

    df = pd.DataFrame({
        "time": hourly.get("time", []),
        "temp": hourly.get("temperature_2m", []),
        "humidity": hourly.get("relativehumidity_2m", []),
        "wind": hourly.get("windspeed_10m", []),
        "rain": hourly.get("precipitation", [])
    })

    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])
        df["lat"] = lat
        df["lon"] = lon
        df["place"] = name

    return df

# ------------------ Step 2: Collect for Multiple Locations ------------------
def collect_history(locations, start_year=2016, end_year=2024):
    all_dfs = []
    for name, lat, lon in locations:
        print(f"📍 Collecting {name} ({lat},{lon})")
        for year in range(start_year, end_year+1):
            df = fetch_historical(name, lat, lon, f"{year}-01-01", f"{year}-12-31")
            if not df.empty:
                all_dfs.append(df)
            time.sleep(1)
    if not all_dfs:
        print("❌ No data collected")
        return pd.DataFrame()
    big_df = pd.concat(all_dfs)
    big_df.set_index("time", inplace=True)

    daily = big_df.resample("D").agg({
        "temp": "mean",
        "humidity": "mean",
        "wind": "mean",
        "rain": "sum",
        "lat": "first",
        "lon": "first",
        "place": "first"
    }).reset_index()

    return daily

# ------------------ Step 3: Compute Derived Features ------------------
def compute_derived_features(df):
    df = df.sort_values(['place', 'time']).copy()

    # Flood: 3-day rolling rainfall
    df['rain_3d'] = df.groupby('place')['rain'].rolling(3, min_periods=1).sum().reset_index(0, drop=True)

    # Wildfire: consecutive dry days
    def dry_days_counter(x):
        dry = (x == 0).astype(int)
        count = dry * (dry.groupby((dry != dry.shift()).cumsum()).cumcount() + 1)
        return count

    df['dry_days'] = df.groupby('place')['rain'].transform(dry_days_counter)

    return df


# ------------------ Step 4: Merge Static Features ------------------
def merge_static_features(df, static_file="static_features.csv"):
    """
    static_features.csv: place, elevation, river_distance, soil_moisture, ndvi
    """
    if os.path.exists(static_file):
        static_df = pd.read_csv(static_file)
        df = df.merge(static_df, on='place', how='left')
    else:
        print(f"⚠️ Static file {static_file} not found. Skipping merge.")
        df['elevation'] = np.nan
        df['river_distance'] = np.nan
        df['soil_moisture'] = np.nan
        df['ndvi'] = np.nan
    return df

# ------------------ Step 5: Merge Disaster Labels ------------------
def merge_disaster_labels(df, flood_file="flood_events.csv", wildfire_file="wildfire_events.csv"):
    # Flood
    if os.path.exists(flood_file):
        flood_df = pd.read_csv(flood_file)
        flood_df['time'] = pd.to_datetime(flood_df['date'])
        df = df.merge(flood_df[['place','time','flood_flag']], on=['place','time'], how='left')
        df['flood'] = df['flood_flag'].fillna(0).astype(int)
        df.drop(columns=['flood_flag'], inplace=True)
    else:
        df['flood'] = 0
    # Wildfire
    if os.path.exists(wildfire_file):
        wildfire_df = pd.read_csv(wildfire_file)
        wildfire_df['time'] = pd.to_datetime(wildfire_df['date'])
        df = df.merge(wildfire_df[['place','time','wildfire_flag']], on=['place','time'], how='left')
        df['wildfire'] = df['wildfire_flag'].fillna(0).astype(int)
        df.drop(columns=['wildfire_flag'], inplace=True)
    else:
        df['wildfire'] = 0
    return df

# ------------------ Step 6: Main ------------------
if __name__ == "__main__":
    locations = [
        ("Delhi", 28.6, 77.2),
        ("NewYork", 40.7, -74.0),
        ("Sydney", -33.9, 151.2),
        ("Bangkok", 13.7, 100.5)
    ]

    print("📥 Collecting historical weather data...")
    daily_df = collect_history(locations)

    print("⚙️ Computing derived features...")
    daily_df = compute_derived_features(daily_df)

    print("🌄 Merging static features (soil moisture, NDVI, elevation, river distance)...")
    daily_df = merge_static_features(daily_df, static_file="static_features.csv")

    print("📜 Merging disaster labels...")
    daily_df = merge_disaster_labels(daily_df, flood_file="flood_events.csv", wildfire_file="wildfire_events.csv")

    daily_df.to_csv("risk_dataset.csv", index=False)
    print(f"✅ Saved final risk dataset with {len(daily_df)} rows to risk_dataset.csv")
