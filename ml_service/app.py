import requests
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import joblib
import datetime

app = Flask(__name__)

# ------------------ MODELS ------------------
INPUT_DIM = 10  # temp, humidity, wind, rain, rain_3d, dry_days, soil_moisture, elevation, river_distance, ndvi

class RiskNet(nn.Module):
    def __init__(self, input_dim):
        super(RiskNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.layers(x)

# Load models
wildfire_model = RiskNet(INPUT_DIM)
wildfire_model.load_state_dict(torch.load("wildfire_model.pth", map_location="cpu"))
wildfire_model.eval()

flood_model = RiskNet(INPUT_DIM)
flood_model.load_state_dict(torch.load("flood_model.pth", map_location="cpu"))
flood_model.eval()

# Load scaler
scaler = joblib.load("risk_scaler.pkl")

# ------------------ GEOLOCATION ------------------
def geocode_location(location_name: str):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={location_name}&count=1"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if "results" not in data or len(data["results"]) == 0:
        raise ValueError("Location not found")
    result = data["results"][0]
    return result["latitude"], result["longitude"]

# ------------------ WEATHER FETCH ------------------
def fetch_current_weather(lat, lon):
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        "&current_weather=true&hourly=temperature_2m,relativehumidity_2m,windspeed_10m,precipitation&timezone=auto"
    )
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    # current weather (temperature, windspeed, etc.)
    current = data.get("current_weather", {})

    # latest humidity from hourly data
    humidity = 0
    if "hourly" in data and "relativehumidity_2m" in data["hourly"]:
        humidity = data["hourly"]["relativehumidity_2m"][-1]  # latest hourly value

    # latest precipitation from hourly data
    rain = 0
    if "hourly" in data and "precipitation" in data["hourly"]:
        rain = data["hourly"]["precipitation"][-1]  # latest hourly value in mm

    return {
        "temp": current.get("temperature", 0),
        "humidity": humidity,  # now correct
        "wind": current.get("windspeed", 0),
        "rain": rain,  # now in mm
    }


# ------------------ PLACEHOLDER FEATURES ------------------
def get_rain_3d(lat, lon):
    # Implement actual 3-day rainfall sum from historical API
    return 0.0

def get_dry_days(lat, lon):
    # Implement actual consecutive dry days calculation
    return 0

def get_soil_moisture(lat, lon):
    # Implement satellite soil moisture retrieval
    return 0.3

def get_elevation(lat, lon):
    # Implement DEM/elevation lookup
    return 200

def get_river_distance(lat, lon):
    # Implement nearest river distance calculation
    return 1.5

def get_ndvi(lat, lon):
    # Implement NDVI retrieval from satellite
    return 0.45

# ------------------ RISK LABELING ------------------
def risk_label(score):
    if score > 0.75:
        return "High"
    elif score > 0.4:
        return "Moderate"
    else:
        return "Low"

# ------------------ PREDICT ------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    location = data.get("location")
    if not location:
        return jsonify({"error": "Please provide a location name"}), 400

    try:
        # Get latitude and longitude
        lat, lon = geocode_location(location)

        # Fetch current weather
        weather = fetch_current_weather(lat, lon)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch location/weather: {str(e)}"}), 500

    # ------------------ Derived / Placeholder Features ------------------
    # Replace these with real API/satellite values later
    rain_3d = 0         # Sum of last 3 days rainfall
    dry_days = 10        # Days since last rain
    soil_moisture = 0.2  # Soil moisture index
    elevation = 350      # meters
    river_distance = 2   # km
    ndvi = 0.45          # Vegetation dryness index

    # ------------------ Build Input Vector ------------------
    X = torch.tensor([[
        weather["temp"],
        weather["humidity"],
        weather["wind"],
        weather["rain"],
        rain_3d,
        dry_days,
        soil_moisture,
        elevation,
        river_distance,
        ndvi
    ]], dtype=torch.float32)

    # Scale input (must match training)
    X_scaled = torch.tensor(scaler.transform(X), dtype=torch.float32)

    # ------------------ Predict ------------------
    wildfire_score = wildfire_model(X_scaled).item()
    flood_score = flood_model(X_scaled).item()

    # ------------------ Risk Labels ------------------
    def risk_label(score):
        if score > 0.75:
            return "High"
        elif score > 0.4:
            return "Moderate"
        else:
            return "Low"

    return jsonify({
        "location": location,
        "latitude": lat,
        "longitude": lon,
        "weather": weather,
        "flood_score": round(flood_score, 3),
        "wildfire_score": round(wildfire_score, 3),
        "flood_risk": risk_label(flood_score),
        "wildfire_risk": risk_label(wildfire_score)
    })


# ------------------ RUN ------------------
if __name__ == "__main__":
    app.run(debug=True)
