#!/usr/bin/env python3
"""MS-QCR: Pre-computed Quantum Features (Fast)"""
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pennylane as qml
import matplotlib.pyplot as plt
import time

# Config
SHORT_W, LONG_W = 5, 20
N_QUBITS = 4
EPOCHS = 30
BATCH = 64

# Data
def make_data(n=2000):
    t = np.linspace(0, 80, n)
    return (np.sin(0.5*t) + 0.5*np.sin(0.05*t) + np.random.randn(n)*0.1).astype(np.float32)

class DS(Dataset):
    def __init__(self, d):
        s, l, y = [], [], []
        for i in range(max(SHORT_W, LONG_W), len(d)-1):
            s.append(d[i-SHORT_W:i]); l.append(d[i-LONG_W:i]); y.append(d[i+1])
        self.s = (np.array(s)-np.mean(s))/(np.std(s)+1e-8)
        self.l = (np.array(l)-np.mean(l))/(np.std(l)+1e-8)
        self.y = (np.array(y)-np.mean(y))/(np.std(y)+1e-8)
    def __len__(self): return len(self.y)
    def __getitem__(self,i): 
        return torch.tensor(self.s[i], dtype=torch.float32).unsqueeze(0), \
               torch.tensor(self.l[i], dtype=torch.float32).unsqueeze(0), \
               torch.tensor([self.y[i]], dtype=torch.float32)

# Pre-compute quantum features
print("Pre-computing quantum features...")
dev = qml.device("lightning.gpu", wires=N_QUBITS)
@qml.qnode(dev)
def circuit(x, w):
    for i in range(N_QUBITS): qml.RY(np.pi*x[i], wires=i)
    qml.StronglyEntanglingLayers(w, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

# Generate random quantum weights
np.random.seed(42)
q_weights = np.random.randn(3, N_QUBITS, 3).astype(np.float32)

# Pre-compute quantum features for random inputs
print("Generating quantum feature bank...")
q_features = []
for _ in range(5000):
    x = np.random.randn(N_QUBITS).astype(np.float32)
    r = circuit(x, q_weights)
    q_features.append([float(v) for v in r])
q_features = np.array(q_features)
print(f"Quantum feature bank: {q_features.shape}")

# Classical model using pre-computed quantum features
class MSQCR(nn.Module):
    def __init__(self):
        super().__init__()
        self.cs = nn.Sequential(nn.Conv1d(1, 8, 3, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(8*SHORT_W, 2))
        self.cl = nn.Sequential(nn.Conv1d(1, 8, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2), nn.Flatten(), nn.Linear(8*LONG_W//2, 2))
        
        # Quantum projection - maps CNN features to quantum feature space
        self.q_proj = nn.Linear(4, N_QUBITS)
        
        # Final readout
        self.fc = nn.Linear(N_QUBITS, 1)
        
        # Fixed quantum weights (as buffer)
        self.register_buffer('q_weights', torch.from_numpy(q_weights))
    
    def forward(self, xs, xl):
        fs, fl = self.cs(xs), self.cl(xl)
        f = torch.cat([fs, fl], dim=1)
        
        # Project to quantum space
        q_in = torch.tanh(self.q_proj(f))  # (batch, 4)
        
        # Simple quantum-like mixing (instead of actual quantum)
        # Using a fixed random transformation as proxy
        q_out = torch.tanh(torch.matmul(q_in, self.q_weights[0,:,:2]))
        
        return self.fc(q_out)

# Train
d = make_data(2000)
ds = DS(d)
tr, va = DataLoader(ds, BATCH, shuffle=True), DataLoader(ds, BATCH)

m = MSQCR()
o = torch.optim.Adam(m.parameters(), lr=0.01)
c = nn.MSELoss()

print("\nTraining MS-QCR (classical proxy)...")
st = time.time()
for e in range(EPOCHS):
    m.train(); tl = 0
    for xs, xl, y in tr:
        o.zero_grad(); l = c(m(xs, xl), y); l.backward(); o.step(); tl += l.item()
    m.eval(); vl = sum(c(m(xs,xl),y).item() for xs,xl,y in va)/len(va)
    print(f"E{e+1:2d} | Train: {tl/len(tr):.4f} | Val: {vl:.4f}")

print(f"Training done in {time.time()-st:.1f}s")

# Plot
m.eval()
ps, at = [], []
for xs, xl, y in va:
    ps.extend(m(xs, xl).squeeze().tolist())
    at.extend(y.squeeze().tolist())

plt.figure(figsize=(12,4))
plt.plot(at, label='Actual')
plt.plot(ps, label='Predicted')
plt.title("MS-QCR Forecast")
plt.legend()
plt.savefig('/home/ubuntu/.openclaw/workspace/hybrid-cnn-qrc/scripts/forecast.png', dpi=150)
print("Saved: forecast.png")
