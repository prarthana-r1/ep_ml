import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv("disaster_dataset.csv")

X = data.drop(["flood_label","wildfire_label"], axis=1).values
y_flood = data["flood_label"].values
y_wildfire = data["wildfire_label"].values

# Normalize inputs
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_flood_train, y_flood_test = train_test_split(X, y_flood, test_size=0.2)
X_train, X_test, y_wildfire_train, y_wildfire_test = train_test_split(X, y_wildfire, test_size=0.2)

# Define model
class RiskNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.layers(x)

input_dim = X.shape[1]
flood_model = RiskNet(input_dim)
wildfire_model = RiskNet(input_dim)

criterion = nn.BCELoss()
optimizer_flood = optim.Adam(flood_model.parameters(), lr=0.001)
optimizer_wildfire = optim.Adam(wildfire_model.parameters(), lr=0.001)

# Training loop (example for flood)
for epoch in range(50):
    flood_model.train()
    inputs = torch.tensor(X_train, dtype=torch.float32)
    labels = torch.tensor(y_flood_train, dtype=torch.float32).view(-1,1)

    optimizer_flood.zero_grad()
    outputs = flood_model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer_flood.step()

print("Flood model trained ✅")

torch.save(flood_model.state_dict(), "flood_model.pth")

# Training loop (for wildfire)
for epoch in range(50):
    wildfire_model.train()
    inputs = torch.tensor(X_train, dtype=torch.float32)
    labels = torch.tensor(y_wildfire_train, dtype=torch.float32).view(-1,1)

    optimizer_wildfire.zero_grad()
    outputs = wildfire_model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer_wildfire.step()

print("Wildfire model trained ✅")

# Save
torch.save(wildfire_model.state_dict(), "wildfire_model.pth")
