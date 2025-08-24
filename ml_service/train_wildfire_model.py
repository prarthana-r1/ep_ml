import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("multi_location_history.csv")

features = ["temp", "humidity", "wind", "rain"]
X = df[features].values

# Dummy wildfire labels (replace with real fire records if available)
# Rule: wildfire risk if temp > 35C, humidity < 30%, and rain < 2mm
y = ((df["temp"] > 35) & (df["humidity"] < 30) & (df["rain"] < 2)).astype(int).values

# -----------------------------
# 2. Preprocess
# -----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# -----------------------------
# 3. Model
# -----------------------------
class WildfireNet(nn.Module):
    def __init__(self):
        super(WildfireNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.layers(x)

model = WildfireNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# 4. Train
# -----------------------------
epochs = 20
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 5 == 0:
        acc = ((outputs > 0.5).float() == y_train).float().mean()
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f} Acc: {acc.item():.4f}")

# -----------------------------
# 5. Save
# -----------------------------
torch.save(model.state_dict(), "wildfire_model.pth")
joblib.dump(scaler, "wildfire_scaler.pkl")

print("✅ Wildfire model trained and saved.")
