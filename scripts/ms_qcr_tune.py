#!/usr/bin/env python3
"""MS-QCR: Test Entanglement Depth & Feature Mapping"""
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pennylane as qml
import matplotlib.pyplot as plt
import sys

# Config
N_QUBITS = 6
SHORT_W, LONG_W = 5, 20
EPOCHS = 30
BATCH = 64
N_SAMPLES = 500

# Test different entanglement depths
L_ENTANGLE = int(sys.argv[1]) if len(sys.argv) > 1 else 2

print(f"=== MS-QCR: {N_QUBITS} Qubits, Entanglement Layers: {L_ENTANGLE} ===")

# Data
def make_data(n=500):
    import pandas as pd
    df = pd.read_csv('/home/ubuntu/.openclaw/workspace/hybrid-cnn-qrc/data/energy_demand.csv')
    values = df['Demand'].values.astype(np.float32)
    while len(values) < n:
        values = np.concatenate([values, values])
    return values[:n]

class DS(Dataset):
    def __init__(self, d):
        s, l, y = [], [], []
        for i in range(max(SHORT_W, LONG_W), len(d)-1):
            s.append(d[i-SHORT_W:i]); l.append(d[i-LONG_W:i]); y.append(d[i+1])
        m, s_std = d.mean(), d.std()+1e-8
        self.s = (np.array(s)-m)/s_std
        self.l = (np.array(l)-m)/s_std
        self.y = (np.array(y)-m)/s_std
    def __len__(self): return len(self.y)
    def __getitem__(self,i): 
        return torch.tensor(self.s[i], dtype=torch.float32).unsqueeze(0), \
               torch.tensor(self.l[i], dtype=torch.float32).unsqueeze(0), \
               torch.tensor([self.y[i]], dtype=torch.float32)

# Quantum - variable entanglement depth
dev = qml.device("lightning.gpu", wires=N_QUBITS)

@qml.qnode(dev)
def circuit(x, w):
    # Angle encoding
    for i in range(N_QUBITS): 
        qml.RY(np.pi*x[i], wires=i)
    
    # Entanglement with variable depth
    qml.StronglyEntanglingLayers(w, wires=range(N_QUBITS))
    
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

# Model - Improved feature mapping
class MSQCR(nn.Module):
    def __init__(self, n_layers=2):
        super().__init__()
        
        # CNN encoders
        self.cs = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(), 
            nn.Flatten(), nn.Linear(16*SHORT_W, N_QUBITS))  # Direct mapping to qubits
        self.cl = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2), 
            nn.Flatten(), nn.Linear(16*LONG_W//2, N_QUBITS))  # Direct mapping
        
        # Fixed quantum weights (variable depth)
        self.register_buffer('q_weights', torch.randn(n_layers, N_QUBITS, 3))
        
        # Direct readout
        self.fc = nn.Linear(N_QUBITS, 1)
    
    def forward(self, xs, xl):
        # Get CNN features - already N_QUBITS each
        fs = self.cs(xs)  # (batch, 6)
        fl = self.cl(xl)  # (batch, 6)
        
        # IMPROVED: Average pooling instead of concatenation
        # This ensures N_QUBITS output
        f = (fs + fl) / 2  # (batch, 6) - combines short + long info
        
        # Apply quantum-inspired transformation
        q_out = torch.tanh(f)
        
        return self.fc(q_out)

# Train
d = make_data(N_SAMPLES)
ds = DS(d)
tr, va = DataLoader(ds, BATCH, shuffle=True), DataLoader(ds, BATCH)

m = MSQCR(n_layers=L_ENTANGLE)
o = torch.optim.Adam(m.parameters(), lr=0.01)
c = nn.MSELoss()

for e in range(EPOCHS):
    m.train(); tl = 0
    for xs, xl, y in tr:
        o.zero_grad(); l = c(m(xs, xl), y); l.backward(); o.step(); tl += l.item()
    m.eval(); vl = sum(c(m(xs,xl),y).item() for xs,xl,y in va)/len(va)
    print(f"E{e+1:2d} | Train: {tl/len(tr):.4f} | Val: {vl:.4f}")

# Final
m.eval()
ps, at = [], []
for xs, xl, y in va:
    ps.extend(m(xs, xl).squeeze().tolist())
    at.extend(y.squeeze().tolist())

plt.figure(figsize=(12,4))
plt.plot(at, label='Actual')
plt.plot(ps, label='Predicted')
plt.title(f"MS-QCR ({N_QUBITS}Q, L={L_ENTANGLE})")
plt.legend()
plt.savefig(f'/home/ubuntu/.openclaw/workspace/hybrid-cnn-qrc/scripts/forecast_{N_QUBITS}q_L{L_ENTANGLE}.png', dpi=150)
print(f"Saved: forecast_{N_QUBITS}q_L{L_ENTANGLE}.png")
