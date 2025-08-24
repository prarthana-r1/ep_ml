import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("risk_dataset.csv")

# Fill missing values (NaN / inf) with 0
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# -----------------------------
# 2. Features and Labels
# -----------------------------
features = [
    "temp", "humidity", "wind", "rain",
    "rain_3d", "dry_days", "soil_moisture",
    "elevation", "river_distance", "ndvi"
]

X = df[features].values
y_flood = df["flood"].values
y_wildfire = df["wildfire"].values

# -----------------------------
# 3. Preprocess
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_flood_train, y_flood_test, y_wildfire_train, y_wildfire_test = train_test_split(
    X_scaled, y_flood, y_wildfire, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_flood_train = torch.tensor(y_flood_train, dtype=torch.float32).unsqueeze(1)
y_flood_test = torch.tensor(y_flood_test, dtype=torch.float32).unsqueeze(1)

y_wildfire_train = torch.tensor(y_wildfire_train, dtype=torch.float32).unsqueeze(1)
y_wildfire_test = torch.tensor(y_wildfire_test, dtype=torch.float32).unsqueeze(1)

# -----------------------------
# 4. Model Definition
# -----------------------------
INPUT_DIM = len(features)

class RiskNet(nn.Module):
    def __init__(self, input_dim):
        super(RiskNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # ensures output in [0,1]
        )

    def forward(self, x):
        return self.layers(x)

flood_model = RiskNet(INPUT_DIM)
wildfire_model = RiskNet(INPUT_DIM)

criterion = nn.BCELoss()
flood_optimizer = optim.Adam(flood_model.parameters(), lr=0.001)
wildfire_optimizer = optim.Adam(wildfire_model.parameters(), lr=0.001)

# -----------------------------
# 5. Training Function
# -----------------------------
def train_model(model, optimizer, X_train, y_train, epochs=50):
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)

        # Clip outputs to avoid any floating-point issues
        outputs = torch.clamp(outputs, 0, 1)

        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            acc = ((outputs > 0.5).float() == y_train).float().mean()
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f} Acc: {acc.item():.4f}")

# -----------------------------
# 6. Train Flood Model
# -----------------------------
print("🌊 Training Flood Risk Model...")
train_model(flood_model, flood_optimizer, X_train, y_flood_train)

# -----------------------------
# 7. Train Wildfire Model
# -----------------------------
print("🔥 Training Wildfire Risk Model...")
train_model(wildfire_model, wildfire_optimizer, X_train, y_wildfire_train)

# -----------------------------
# 8. Save Models and Scaler
# -----------------------------
torch.save(flood_model.state_dict(), "flood_model.pth")
torch.save(wildfire_model.state_dict(), "wildfire_model.pth")
joblib.dump(scaler, "risk_scaler.pkl")

print("✅ Flood and Wildfire models trained and saved.")
