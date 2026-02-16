#!/usr/bin/env python3
"""MS-QCR: True Quantum Circuit Execution"""
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pennylane as qml
import matplotlib.pyplot as plt
import time

# Config
N_QUBITS = 6
SHORT_W, LONG_W = 5, 20
EPOCHS = 30
BATCH = 32  # Smaller for quantum
N_SAMPLES = 200  # Smaller for speed
N_LAYERS = 1  # Shallow entanglement

print(f"=== TRUE QUANTUM: {N_QUBITS} Qubits, {N_LAYERS} Layer ===")

# Data
def make_data(n=200):
    import pandas as pd
    df = pd.read_csv('/home/ubuntu/.openclaw/workspace/hybrid-cnn-qrc/data/energy_demand.csv')
    return df['Demand'].values.astype(np.float32)[:n]

class DS(Dataset):
    def __init__(self, d):
        s, l, y = [], [], []
        for i in range(max(SHORT_W, LONG_W), len(d)-1):
            s.append(d[i-SHORT_W:i]); l.append(d[i-LONG_W:i]); y.append(d[i+1])
        m, s_std = np.mean(d), np.std(d)+1e-8
        self.s = (np.array(s)-m)/s_std
        self.l = (np.array(l)-m)/s_std
        self.y = (np.array(y)-m)/s_std
    def __len__(self): return len(self.y)
    def __getitem__(self,i): 
        return (torch.tensor(self.s[i], dtype=torch.float32).unsqueeze(0),
                torch.tensor(self.l[i], dtype=torch.float32).unsqueeze(0),
                torch.tensor([self.y[i]], dtype=torch.float32))

# Quantum Device
dev = qml.device("lightning.gpu", wires=N_QUBITS)

@qml.qnode(dev, diff_method="adjoint")
def quantum_circuit(inputs, weights):
    """True quantum circuit with parameterized entanglement"""
    # Angle embedding
    for i in range(min(N_QUBITS, len(inputs))):
        qml.RY(np.pi * inputs[i], wires=i)
    
    # Parameterized entanglement layers
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    
    # Measurement
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

# Pre-compute quantum features
print("Pre-computing quantum features...")
d = make_data(N_SAMPLES)
ds = DS(d)

# Extract features from CNN
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.cs = nn.Sequential(nn.Conv1d(1, 8, 3, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(8*SHORT_W, 3))
        self.cl = nn.Sequential(nn.Conv1d(1, 8, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2), nn.Flatten(), nn.Linear(8*LONG_W//2, 3))
    
    def forward(self, xs, xl):
        return torch.cat([self.cs(xs), self.cl(xl)], dim=1)

extractor = FeatureExtractor()
extractor.eval()

# Generate quantum features
q_features = []
with torch.no_grad():
    for i in range(len(ds)):
        xs, xl, _ = ds[i]
        xs, xl = xs.unsqueeze(0), xl.unsqueeze(0)
        feats = extractor(xs, xl).squeeze().numpy()
        
        # Pad to N_QUBITS
        padded = np.pad(feats, (0, N_QUBITS - len(feats)), mode='constant')
        
        # Quantum execution
        weights = np.random.randn(N_LAYERS, N_QUBITS, 3)
        q_out = quantum_circuit(padded, weights)
        q_features.append(q_out)

q_features = np.array(q_features)
print(f"Quantum features shape: {q_features.shape}")

# Classical model on quantum features
class QuantumReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(N_QUBITS, 8), nn.ReLU(), nn.Linear(8, 1))
    
    def forward(self, q):
        return self.fc(q)

# Train
X = torch.tensor(q_features, dtype=torch.float32)
y = torch.tensor(ds.y, dtype=torch.float32).unsqueeze(1)

split = int(0.8 * len(X))
X_tr, X_va = X[:split], X[split:]
y_tr, y_va = y[:split], y[split:]

model = QuantumReadout()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

print("\n=== Training Readout on Quantum Features ===")
for e in range(EPOCHS):
    model.train()
    opt.zero_grad()
    pred = model(X_tr)
    loss = criterion(pred, y_tr)
    loss.backward()
    opt.step()
    
    model.eval()
    val_loss = criterion(model(X_va), y_va).item()
    if e % 10 == 0:
        print(f"E{e+1:2d} | Train: {loss.item():.4f} | Val: {val_loss:.4f}")

# Final
model.eval()
val_mse = criterion(model(X_va), y_va).item()
print(f"\n=== TRUE QUANTUM FINAL Val MSE: {val_mse:.4f} ===")

# Plot
preds = model(X_va).squeeze().detach().numpy()
actual = y_va.squeeze().numpy()

plt.figure(figsize=(10,4))
plt.plot(actual, label='Actual')
plt.plot(preds, label='Predicted')
plt.title(f"True Quantum Circuit ({N_QUBITS}Q, {N_LAYERS}L)")
plt.legend()
plt.savefig(f'/home/ubuntu/.openclaw/workspace/hybrid-cnn-qrc/scripts/forecast_true_quantum.png', dpi=150)
print("Saved: forecast_true_quantum.png")
